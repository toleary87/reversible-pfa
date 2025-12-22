
"""
Refactored, single-file PFA simulation pipeline aligned to the project goal:
computationally explore PFA dosing near Koch's triangle to create zones of
reversible (RE) and irreversible electroporation (IRE) in the slow pathway (SP)
while minimizing conduction-system risk (AV node / His).

This file intentionally does NOT modify your existing model.py/main.py/
visualize_field.py; it provides an improved "v2" implementation in one place.

Main upgrades:
- Regions are the source of truth: the same geometry defines conductivity AND ROI readouts.
- His bundle rectangle is centered on his_bundle_pos.
- Finite-area electrode patches (Dirichlet) instead of single-node point electrodes.
- Cached stiffness matrix per conductivity map (fast parameter sweeps).
- Robust ROI metrics: p95/p99 and fraction above RE/IRE thresholds.
- Clinically explicit, mutually exclusive outcome taxonomy.
- Optional robustness assessment (position/contact/sigma perturbations) for candidate selection.

Run:
  python pfa_fem_refactor_codex_v2.py

Dependencies: numpy, matplotlib, scipy, scikit-fem (see requirements.txt)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

from skfem import MeshTri, Basis, ElementTriP1, BilinearForm, asm, condense, solve
from skfem.helpers import dot, grad
from skfem.element import ElementTriP0

try:
    from scipy.interpolate import LinearNDInterpolator
except Exception:
    LinearNDInterpolator = None


# -----------------------------
# Configuration data structures
# -----------------------------

@dataclass(frozen=True)
class PulseConfig:
    label: str
    ire_threshold_v_cm: float
    color: str = "k"

    @property
    def re_threshold_v_cm(self) -> float:
        # Default assumption retained: RE at 50% of IRE.
        return 0.5 * self.ire_threshold_v_cm


@dataclass(frozen=True)
class GeometryConfig:
    domain_width_mm: float = 40.0
    domain_height_mm: float = 40.0
    resolution_mm: float = 0.5

    his_pos: Tuple[float, float] = (0.0, 10.0)
    sp_pos: Tuple[float, float] = (0.0, -5.0)
    av_node_pos: Tuple[float, float] = (0.0, 12.0)

    av_node_radius_mm: float = 2.0
    his_width_mm: float = 2.0
    his_length_mm: float = 4.0
    sp_radius_mm: float = 2.0

    blood_y_gt_mm: float = 10.0


@dataclass(frozen=True)
class ElectrodeConfig:
    spacing_mm: float = 4.0
    radius_mm: float = 0.75  # finite-area patch radius


# -----------------------------
# Outcome taxonomy
# -----------------------------

class Outcome:
    """Discrete outcomes for operating-window maps (mutually exclusive)."""

    INEFFECTIVE = 0
    HIS_RE_ONLY = 1
    DESIRED_MAPPING = 2
    DESIRED_ABLATION = 3
    UNSAFE_HIS_IRE = 4
    ABLATION_WITH_HIS_RE = 5
    MAPPING_WITH_HIS_RE = 6


OUTCOME_LABELS: Dict[int, str] = {
    Outcome.INEFFECTIVE: "Ineffective",
    Outcome.HIS_RE_ONLY: "His RE only",
    Outcome.DESIRED_MAPPING: "Desired mapping (SP RE, His < RE)",
    Outcome.DESIRED_ABLATION: "Desired ablation (SP IRE, His < RE)",
    Outcome.UNSAFE_HIS_IRE: "Unsafe (His IRE)",
    Outcome.ABLATION_WITH_HIS_RE: "Ablation with His RE risk",
    Outcome.MAPPING_WITH_HIS_RE: "Mapping with His RE",
}


_ALLOWED_METRICS = {"mean", "max", "p95", "p99"}


def _validate_metric(metric: str) -> str:
    if metric not in _ALLOWED_METRICS:
        raise ValueError(f"Invalid metric '{metric}'. Expected one of {_ALLOWED_METRICS}.")
    return metric


def classify_outcome(e_his_metric: float, e_sp_metric: float, re_thresh: float, ire_thresh: float) -> int:
    """Classify a setting using a single chosen E-metric (typically ROI p99).

    Parameters
    ----------
    e_his_metric : float
        Representative field metric in His ROI (e.g., p99 of |E|).
    e_sp_metric : float
        Representative field metric in SP ROI.
    re_thresh : float
        Reversible electroporation threshold in V/cm.
    ire_thresh : float
        Irreversible electroporation threshold in V/cm.

    Returns
    -------
    int : Outcome code.
    """
    his_re = e_his_metric >= re_thresh
    his_ire = e_his_metric >= ire_thresh
    sp_re = e_sp_metric >= re_thresh
    sp_ire = e_sp_metric >= ire_thresh

    if his_ire:
        return Outcome.UNSAFE_HIS_IRE
    if sp_ire and not his_re:
        return Outcome.DESIRED_ABLATION
    if sp_re and (not sp_ire) and not his_re:
        return Outcome.DESIRED_MAPPING
    if sp_ire and his_re:
        return Outcome.ABLATION_WITH_HIS_RE
    if sp_re and his_re:
        return Outcome.MAPPING_WITH_HIS_RE
    if his_re:
        return Outcome.HIS_RE_ONLY
    return Outcome.INEFFECTIVE


# -----------------------------
# FEM model (v2)
# -----------------------------

class PFAModelV2:
    """2D FEM Laplace solver with heterogeneous conductivity and authoritative ROIs."""

    def __init__(self, width_mm: float = 40.0, height_mm: float = 40.0, resolution_mm: float = 0.5):
        self.width_mm = float(width_mm)
        self.height_mm = float(height_mm)
        self.resolution_mm = float(resolution_mm)

        self.mesh = MeshTri.init_tensor(
            np.linspace(-self.width_mm / 2.0, self.width_mm / 2.0, int(self.width_mm / self.resolution_mm) + 1),
            np.linspace(-self.height_mm / 2.0, self.height_mm / 2.0, int(self.height_mm / self.resolution_mm) + 1),
        )
        self.basis = Basis(self.mesh, ElementTriP1())

        # Default conductivities (S/m)
        self.sigma_myocardium = 0.16
        self.sigma_blood = 0.50
        self.sigma_av_node = 0.05
        self.sigma_his = 0.15

        self.conductivities = np.zeros(self.basis.nelems, dtype=float) + self.sigma_myocardium

        # Element centroids (2, nelems)
        self.centroids = self.mesh.p[:, self.mesh.t].mean(axis=1)

        # Region bookkeeping
        self.region_masks: Dict[str, np.ndarray] = {}
        self.region_ids: Dict[str, np.ndarray] = {}

        # Cached system
        self._A = None
        self._sigma_field = None

        # Solution fields
        self.potential = None
        self.E_field_v_cm = None

    def invalidate_system(self) -> None:
        self._A = None
        self._sigma_field = None

    def define_regions(self, geom: GeometryConfig) -> None:
        """Define tissue regions and assign conductivities.

        Regions created:
          - blood (y > geom.blood_y_gt_mm)
          - av_node (circle)
          - his (centered rectangle)
          - sp (circle)
        """
        x, y = self.centroids[0], self.centroids[1]

        self.conductivities[:] = self.sigma_myocardium
        self.region_masks = {}

        # Blood half-plane
        blood_mask = y > geom.blood_y_gt_mm
        self.conductivities[blood_mask] = self.sigma_blood
        self.region_masks["blood"] = blood_mask

        # AV node
        av = geom.av_node_pos
        av_dist = np.sqrt((x - av[0]) ** 2 + (y - av[1]) ** 2)
        av_mask = av_dist < geom.av_node_radius_mm
        self.conductivities[av_mask] = self.sigma_av_node
        self.region_masks["av_node"] = av_mask

        # His bundle (CENTERED rectangle)
        hx, hy = geom.his_pos
        half_w = geom.his_width_mm / 2.0
        half_l = geom.his_length_mm / 2.0
        his_mask = (
            (x > hx - half_w)
            & (x < hx + half_w)
            & (y > hy - half_l)
            & (y < hy + half_l)
        )
        self.conductivities[his_mask] = self.sigma_his
        self.region_masks["his"] = his_mask

        # Slow pathway
        sp = geom.sp_pos
        sp_dist = np.sqrt((x - sp[0]) ** 2 + (y - sp[1]) ** 2)
        sp_mask = sp_dist < geom.sp_radius_mm
        self.conductivities[sp_mask] = self.sigma_av_node
        self.region_masks["sp"] = sp_mask

        self.region_ids = {k: np.where(v)[0] for k, v in self.region_masks.items()}

        self.invalidate_system()

    def assemble_system(self):
        """Assemble and cache stiffness matrix A for the current conductivity map."""

        @BilinearForm
        def laplace(u, v, w):
            return w["sigma"] * dot(grad(u), grad(v))

        basis0 = self.basis.with_element(ElementTriP0())
        self._sigma_field = basis0.interpolate(self.conductivities)
        self._A = asm(laplace, self.basis, sigma=self._sigma_field)
        return self._A

    def solve(
        self,
        voltage_v: float,
        electrode1_pos: Tuple[float, float],
        electrode2_pos: Tuple[float, float],
        electrode_radius_mm: float,
    ):
        """Solve Laplace with finite-area Dirichlet patches for electrodes."""
        if self._A is None:
            self.assemble_system()

        nodes = self.mesh.p  # (2, nnodes)

        e1x, e1y = electrode1_pos
        e2x, e2y = electrode2_pos

        dist1 = np.sqrt((nodes[0] - e1x) ** 2 + (nodes[1] - e1y) ** 2)
        dist2 = np.sqrt((nodes[0] - e2x) ** 2 + (nodes[1] - e2y) ** 2)

        dofs_e1 = np.where(dist1 <= electrode_radius_mm)[0]
        dofs_e2 = np.where(dist2 <= electrode_radius_mm)[0]

        if dofs_e1.size == 0 or dofs_e2.size == 0:
            raise ValueError(
                f"Electrode radius too small for mesh resolution. radius={electrode_radius_mm} mm; "
                f"captured nodes: e1={dofs_e1.size}, e2={dofs_e2.size}"
            )

        overlap = np.intersect1d(dofs_e1, dofs_e2)
        if overlap.size > 0:
            raise ValueError("Electrode patches overlap; reduce radius or increase spacing.")

        D = np.unique(np.concatenate([dofs_e1, dofs_e2]))
        x = np.zeros(self.basis.N)
        x[dofs_e1] = float(voltage_v)
        x[dofs_e2] = 0.0

        self.potential = solve(*condense(self._A, x=x, D=D))
        return self.potential

    def calculate_field(self) -> np.ndarray:
        """Compute element-wise |E| in V/cm."""
        if self.potential is None:
            raise RuntimeError("Call solve(...) before calculate_field().")

        out = self.basis.interpolate(self.potential)
        grad_u = out.grad  # (2, nqp, nelems)
        E_mag = np.sqrt(grad_u[0] ** 2 + grad_u[1] ** 2)
        # scikit-fem versions differ in gradient layout; reduce along quadrature axis.
        if E_mag.ndim != 2:
            raise RuntimeError(f"Unexpected E-field shape {E_mag.shape}; expected 2D array.")
        if E_mag.shape[0] == self.basis.nelems:
            E_elem = np.mean(E_mag, axis=1)
        elif E_mag.shape[1] == self.basis.nelems:
            E_elem = np.mean(E_mag, axis=0)
        else:
            raise RuntimeError(
                f"Unexpected E-field shape {E_mag.shape}; cannot align with nelems={self.basis.nelems}."
            )

        # Convert V/mm to V/cm
        self.E_field_v_cm = E_elem * 10.0
        return self.E_field_v_cm

    def roi_metrics(
        self,
        E_field_v_cm: np.ndarray,
        region_name: str,
        thresholds: Optional[Dict[str, float]] = None,
        q: Tuple[float, float] = (0.95, 0.99),
    ) -> Dict[str, float]:
        """Compute stable ROI metrics for a named region."""
        if region_name not in self.region_masks:
            raise KeyError(f"Unknown region '{region_name}'. Available: {list(self.region_masks.keys())}")

        mask = self.region_masks[region_name]
        vals = E_field_v_cm[mask]
        if vals.size == 0:
            return {"n": 0}

        out = {
            "n": int(vals.size),
            "mean": float(np.mean(vals)),
            "max": float(np.max(vals)),
            "p95": float(np.quantile(vals, q[0])),
            "p99": float(np.quantile(vals, q[1])),
        }

        if thresholds:
            re = thresholds.get("re")
            ire = thresholds.get("ire")
            if re is not None:
                out["frac_ge_re"] = float(np.mean(vals >= re))
            if ire is not None:
                out["frac_ge_ire"] = float(np.mean(vals >= ire))

        return out

    def apply_conductivity_scaling(self, scale_by_region: Dict[str, float], clip: Tuple[float, float] = (1e-6, 10.0)):
        """Scale conductivities in-place for specified regions and invalidate the cached system.

        scale_by_region: e.g. {"blood": 1.2, "his": 0.8}
        """
        lo, hi = clip
        for region, s in scale_by_region.items():
            if region not in self.region_masks:
                continue
            self.conductivities[self.region_masks[region]] *= float(s)

        self.conductivities = np.clip(self.conductivities, lo, hi)
        self.invalidate_system()


# -----------------------------
# Parameter sweep + plots
# -----------------------------

@dataclass
class SimulationPoint:
    voltage_v: float
    distance_from_his_mm: float
    y_center_mm: float
    e1_pos: Tuple[float, float]
    e2_pos: Tuple[float, float]


def _electrode_positions_from_distance(geom: GeometryConfig, elec: ElectrodeConfig, distance_from_his_mm: float) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """Return electrode positions and center y for a given inferior distance from His."""
    y_center = geom.his_pos[1] - float(distance_from_his_mm)
    e1 = (-elec.spacing_mm / 2.0, y_center)
    e2 = (elec.spacing_mm / 2.0, y_center)
    return e1, e2, y_center


def run_parameter_sweep_v2(
    pulse_configs: List[PulseConfig],
    voltages_v: np.ndarray,
    distances_mm: np.ndarray,
    geom: GeometryConfig,
    elec: ElectrodeConfig,
    metric: str = "p99",
    verbose: bool = True,
) -> Dict[str, Dict[float, Dict[float, Dict[str, Dict[str, float]]]]]:
    """Run FEM simulations once per (V, d) and compute ROI metrics for each pulse config.

    Returns a nested dict:
      results[pulse_label][V][d] = {
         "his": {metrics...}, "sp": {metrics...}, "av_node": {metrics...},
         "metric_used": {"his": float, "sp": float, "outcome": int}
      }

    Notes
    -----
    - Conductivity regions are defined once and reused.
    - The stiffness matrix is assembled once per conductivity map and reused across voltages.
    """
    metric = _validate_metric(metric)

    model = PFAModelV2(width_mm=geom.domain_width_mm, height_mm=geom.domain_height_mm, resolution_mm=geom.resolution_mm)
    model.define_regions(geom)
    model.assemble_system()

    results: Dict[str, Dict[float, Dict[float, Dict[str, Dict[str, float]]]]] = {pc.label: {} for pc in pulse_configs}

    for V in voltages_v:
        if verbose:
            print(f"Simulating Voltage: {V:.0f} V")

        for pc in pulse_configs:
            results[pc.label].setdefault(float(V), {})

        for d in distances_mm:
            e1, e2, y_center = _electrode_positions_from_distance(geom, elec, float(d))

            model.solve(voltage_v=float(V), electrode1_pos=e1, electrode2_pos=e2, electrode_radius_mm=elec.radius_mm)
            E = model.calculate_field()

            # Compute per-pulse-config ROI metrics including fractions relative to that pulse's thresholds.
            for pc in pulse_configs:
                thr = {"re": pc.re_threshold_v_cm, "ire": pc.ire_threshold_v_cm}
                his_m = model.roi_metrics(E, "his", thresholds=thr)
                sp_m = model.roi_metrics(E, "sp", thresholds=thr)
                av_m = model.roi_metrics(E, "av_node", thresholds=thr)

                e_his = float(his_m.get(metric, 0.0))
                e_sp = float(sp_m.get(metric, 0.0))

                outcome = classify_outcome(e_his, e_sp, thr["re"], thr["ire"])

                results[pc.label][float(V)][float(d)] = {
                    "his": his_m,
                    "sp": sp_m,
                    "av_node": av_m,
                    "metric_used": {"metric": metric, "his": e_his, "sp": e_sp, "outcome": outcome},
                    "electrodes": {"e1": e1, "e2": e2, "y_center": float(y_center)},
                }

    return results


def plot_operating_window(
    pulse_config: PulseConfig,
    results_for_pulse: Dict[float, Dict[float, Dict[str, Dict[str, float]]]],
    voltages_v: np.ndarray,
    distances_mm: np.ndarray,
    out_png: str,
) -> None:
    """Plot operating window as a voltage-distance heatmap for a single pulse config."""

    outcome_grid = np.zeros((len(voltages_v), len(distances_mm)), dtype=int)

    for i, V in enumerate(voltages_v):
        for j, d in enumerate(distances_mm):
            outcome_grid[i, j] = int(results_for_pulse[float(V)][float(d)]["metric_used"]["outcome"])

    d_diffs = np.diff(distances_mm)
    if not (np.all(d_diffs > 0) or np.all(d_diffs < 0)):
        raise ValueError("distances_mm must be strictly monotonic for plotting.")

    # Color mapping for outcomes 0..6
    colors = [
        "lightgray",   # 0 Ineffective
        "gold",        # 1 His RE only
        "lightgreen",  # 2 Desired mapping
        "dodgerblue",  # 3 Desired ablation
        "salmon",      # 4 Unsafe His IRE
        "orange",      # 5 Ablation with His RE
        "khaki",       # 6 Mapping with His RE
    ]

    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, 7.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(11, 8))

    plt.imshow(
        outcome_grid,
        aspect="auto",
        origin="lower",
        extent=[distances_mm[0] - 0.5, distances_mm[-1] + 0.5, voltages_v[0] - 50, voltages_v[-1] + 50],
        cmap=cmap,
        norm=norm,
    )

    patches = [
        mpatches.Patch(color=colors[Outcome.INEFFECTIVE], label=OUTCOME_LABELS[Outcome.INEFFECTIVE]),
        mpatches.Patch(color=colors[Outcome.HIS_RE_ONLY], label=OUTCOME_LABELS[Outcome.HIS_RE_ONLY]),
        mpatches.Patch(color=colors[Outcome.DESIRED_MAPPING], label=OUTCOME_LABELS[Outcome.DESIRED_MAPPING]),
        mpatches.Patch(color=colors[Outcome.MAPPING_WITH_HIS_RE], label=OUTCOME_LABELS[Outcome.MAPPING_WITH_HIS_RE]),
        mpatches.Patch(color=colors[Outcome.DESIRED_ABLATION], label=OUTCOME_LABELS[Outcome.DESIRED_ABLATION]),
        mpatches.Patch(color=colors[Outcome.ABLATION_WITH_HIS_RE], label=OUTCOME_LABELS[Outcome.ABLATION_WITH_HIS_RE]),
        mpatches.Patch(color=colors[Outcome.UNSAFE_HIS_IRE], label=OUTCOME_LABELS[Outcome.UNSAFE_HIS_IRE]),
    ]

    plt.legend(handles=patches, loc="upper right", fontsize=8)
    plt.xlabel("Distance from His bundle (mm)")
    plt.ylabel("Applied Voltage (V)")
    plt.title(
        f"Operating Window ({pulse_config.label})\n"
        f"IRE={pulse_config.ire_threshold_v_cm:.0f} V/cm, RE={pulse_config.re_threshold_v_cm:.0f} V/cm"
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_safe_distance_summary(
    pulse_configs: List[PulseConfig],
    results: Dict[str, Dict[float, Dict[float, Dict[str, Dict[str, float]]]]],
    voltages_v: np.ndarray,
    distances_mm: np.ndarray,
    out_png: str,
    safety_metric: str = "his",
    use_threshold: str = "ire",
) -> None:
    """Plot minimum 'safe' distance vs voltage for each pulse width.

    By default, "safe" means: His metric < IRE threshold.
    """

    plt.figure(figsize=(10, 6))

    for pc in pulse_configs:
        safe_dists: List[float] = []
        thr = pc.ire_threshold_v_cm if use_threshold == "ire" else pc.re_threshold_v_cm

        if safety_metric not in {"his", "sp", "av_node"}:
            raise ValueError(f"Invalid safety_metric '{safety_metric}'. Expected 'his', 'sp', or 'av_node'.")

        for V in voltages_v:
            safe_candidates: List[float] = []

            for d in distances_mm:
                entry = results[pc.label][float(V)][float(d)]
                metric_name = entry["metric_used"]["metric"]

                if safety_metric in {"his", "sp"}:
                    metric_val = entry["metric_used"].get(safety_metric)
                else:
                    metric_val = entry[safety_metric].get(metric_name)

                if metric_val is None:
                    raise ValueError(f"Missing metric '{metric_name}' for region '{safety_metric}'.")

                if float(metric_val) < thr:
                    safe_candidates.append(float(d))

            if safe_candidates:
                safe_dists.append(float(np.min(safe_candidates)))
            else:
                safe_dists.append(float("nan"))

        plt.plot(
            voltages_v,
            safe_dists,
            marker="o",
            linestyle="-",
            color=pc.color,
            label=f"{pc.label} (thr={thr:.0f} V/cm)",
        )

    plt.xlabel("Applied Voltage (V)")
    plt.ylabel("Minimum safe distance (mm)")
    plt.title("Safe distance from His vs voltage (by pulse width)")
    plt.grid(True, alpha=0.3)
    plt.xticks(voltages_v, rotation=45)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# -----------------------------
# Robustness assessment + shortlist
# -----------------------------

@dataclass(frozen=True)
class RobustnessConfig:
    n_samples: int = 50
    jitter_x_mm: Tuple[float, float] = (-1.0, 1.0)
    jitter_y_mm: Tuple[float, float] = (-2.0, 2.0)
    electrode_radius_mm: Tuple[float, float] = (0.5, 1.0)
    sigma_scale_range: Tuple[float, float] = (0.8, 1.2)  # per region scaling
    rng_seed: Optional[int] = 0


def robustness_assessment(
    model: PFAModelV2,
    geom: GeometryConfig,
    elec: ElectrodeConfig,
    pulse: PulseConfig,
    voltage_v: float,
    distance_from_his_mm: float,
    metric: str = "p99",
    rconf: RobustnessConfig = RobustnessConfig(),
    perturb_conductivity: bool = True,
) -> Dict[str, float]:
    """Estimate outcome probabilities for a given setting under perturbations."""
    metric = _validate_metric(metric)

    if rconf.rng_seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(rconf.rng_seed)

    # Base regions assumed already defined on model
    # We'll copy baseline conductivities to restore after each perturbation.
    base_sigma = model.conductivities.copy()

    thr = {"re": pulse.re_threshold_v_cm, "ire": pulse.ire_threshold_v_cm}

    counts = {
        "p_desired_ablation": 0,
        "p_desired_mapping": 0,
        "p_his_re": 0,
        "p_his_ire": 0,
        "p_sp_re": 0,
        "p_sp_ire": 0,
    }

    for _ in range(int(rconf.n_samples)):
        dx = float(rng.uniform(*rconf.jitter_x_mm))
        dy = float(rng.uniform(*rconf.jitter_y_mm))
        r_e = float(rng.uniform(*rconf.electrode_radius_mm))

        e1, e2, y_center = _electrode_positions_from_distance(geom, elec, float(distance_from_his_mm))
        e1 = (e1[0] + dx, e1[1] + dy)
        e2 = (e2[0] + dx, e2[1] + dy)

        # Perturb conductivities by region (simple isotropic scaling)
        if perturb_conductivity:
            model.conductivities[:] = base_sigma
            scale = lambda: float(rng.uniform(*rconf.sigma_scale_range))
            model.apply_conductivity_scaling(
                {
                    "blood": scale(),
                    "his": scale(),
                    "av_node": scale(),
                    "sp": scale(),
                }
            )
            model.assemble_system()

        model.solve(voltage_v=float(voltage_v), electrode1_pos=e1, electrode2_pos=e2, electrode_radius_mm=r_e)
        E = model.calculate_field()

        his_m = model.roi_metrics(E, "his", thresholds=thr)
        sp_m = model.roi_metrics(E, "sp", thresholds=thr)

        e_his = float(his_m.get(metric, 0.0))
        e_sp = float(sp_m.get(metric, 0.0))

        his_re = e_his >= thr["re"]
        his_ire = e_his >= thr["ire"]
        sp_re = e_sp >= thr["re"]
        sp_ire = e_sp >= thr["ire"]

        outcome = classify_outcome(e_his, e_sp, thr["re"], thr["ire"])

        counts["p_his_re"] += int(his_re)
        counts["p_his_ire"] += int(his_ire)
        counts["p_sp_re"] += int(sp_re)
        counts["p_sp_ire"] += int(sp_ire)
        counts["p_desired_ablation"] += int(outcome == Outcome.DESIRED_ABLATION)
        counts["p_desired_mapping"] += int(outcome == Outcome.DESIRED_MAPPING)

    # restore
    model.conductivities[:] = base_sigma
    model.invalidate_system()
    model.assemble_system()

    n = float(rconf.n_samples)
    return {k: v / n for k, v in counts.items()}


def shortlist_candidates(
    results_for_pulse: Dict[float, Dict[float, Dict[str, Dict[str, float]]]],
    voltages_v: np.ndarray,
    distances_mm: np.ndarray,
    desired: str = "ablation",
    top_k: int = 10,
) -> List[Tuple[float, float, int, float, float]]:
    """Create a simple deterministic shortlist based on desired category.

    Returns list of tuples:
      (V, d, outcome, E_sp_metric, E_his_metric)

    You can then run robustness_assessment on these points.
    """

    target_outcome = Outcome.DESIRED_ABLATION if desired == "ablation" else Outcome.DESIRED_MAPPING

    cands = []
    for V in voltages_v:
        for d in distances_mm:
            md = results_for_pulse[float(V)][float(d)]["metric_used"]
            if int(md["outcome"]) == target_outcome:
                cands.append((float(V), float(d), int(md["outcome"]), float(md["sp"]), float(md["his"])))

    # Sort: prioritize larger SP metric and smaller His metric
    cands.sort(key=lambda t: (-t[3], t[4]))
    return cands[: int(top_k)]


# -----------------------------
# Detailed visualization (optional)
# -----------------------------

def visualize_field_detailed_v2(
    voltage_v: float,
    distance_from_his_mm: float,
    pulse: PulseConfig,
    geom: GeometryConfig,
    elec: ElectrodeConfig,
    out_png: str,
    metric: str = "p99",
) -> Dict[str, float]:
    """Detailed visualization similar to your original visualize_field.py, but ROI-consistent."""
    metric = _validate_metric(metric)

    model = PFAModelV2(width_mm=geom.domain_width_mm, height_mm=geom.domain_height_mm, resolution_mm=0.25)
    model.define_regions(geom)
    model.assemble_system()

    e1, e2, y_center = _electrode_positions_from_distance(geom, elec, float(distance_from_his_mm))

    model.solve(voltage_v=float(voltage_v), electrode1_pos=e1, electrode2_pos=e2, electrode_radius_mm=elec.radius_mm)
    E = model.calculate_field()

    thr = {"re": pulse.re_threshold_v_cm, "ire": pulse.ire_threshold_v_cm}
    his_m = model.roi_metrics(E, "his", thresholds=thr)
    sp_m = model.roi_metrics(E, "sp", thresholds=thr)
    av_m = model.roi_metrics(E, "av_node", thresholds=thr)

    e_his = float(his_m.get(metric, 0.0))
    e_sp = float(sp_m.get(metric, 0.0))
    outcome = classify_outcome(e_his, e_sp, thr["re"], thr["ire"])

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1) Potential
    ax1 = fig.add_subplot(gs[0, 0])
    pot_plot = ax1.tripcolor(model.mesh.p[0], model.mesh.p[1], model.mesh.t.T, model.potential, shading="flat", cmap="RdBu_r")
    plt.colorbar(pot_plot, ax=ax1, label="Potential (V)")
    ax1.plot(*e1, "r*", markersize=14, label=f"Anode (+{voltage_v:.0f}V)")
    ax1.plot(*e2, "k*", markersize=14, label="Cathode (0V)")
    ax1.set_title("Potential")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.2)
    ax1.legend(fontsize=8)

    # 2) E-field
    ax2 = fig.add_subplot(gs[0, 1])
    vmax = min(3000.0, float(np.max(E)))
    e_plot = ax2.tripcolor(model.mesh.p[0], model.mesh.p[1], model.mesh.t.T, E, shading="flat", cmap="inferno", vmin=0.0, vmax=vmax)
    plt.colorbar(e_plot, ax=ax2, label="|E| (V/cm)")
    ax2.plot(*e1, "r*", markersize=14)
    ax2.plot(*e2, "w*", markersize=14)
    ax2.set_title(f"|E| (metric={metric})")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.2, color="white")

    # 3) Conductivity
    ax3 = fig.add_subplot(gs[0, 2])
    c_plot = ax3.tripcolor(model.mesh.p[0], model.mesh.p[1], model.mesh.t.T, model.conductivities, shading="flat", cmap="viridis")
    plt.colorbar(c_plot, ax=ax3, label="Conductivity (S/m)")
    ax3.set_title("Conductivity map")
    ax3.set_aspect("equal")
    ax3.grid(True, alpha=0.2)

    # 4) Midline profile x=0
    ax4 = fig.add_subplot(gs[1, 0])
    y_line = np.linspace(-20, 20, 300)
    x_line = np.zeros_like(y_line)

    if LinearNDInterpolator is not None:
        x_c, y_c = model.centroids[0], model.centroids[1]
        interp = LinearNDInterpolator(list(zip(x_c, y_c)), E)
        E_line = interp(x_line, y_line)
        # Robust NaN handling
        if np.any(np.isnan(E_line)):
            E_line = np.nan_to_num(E_line, nan=np.nanmin(E))
    else:
        # Fallback: nearest centroid by y (coarse)
        x_c, y_c = model.centroids[0], model.centroids[1]
        E_line = np.array([E[np.argmin((x_c - 0.0) ** 2 + (y_c - yy) ** 2)] for yy in y_line])

    ax4.plot(y_line, E_line, linewidth=2)
    ax4.axhline(thr["re"], linestyle="--", linewidth=2, label=f"RE {thr['re']:.0f}")
    ax4.axhline(thr["ire"], linestyle="-", linewidth=2, label=f"IRE {thr['ire']:.0f}")
    ax4.axvline(geom.his_pos[1], linestyle=":", alpha=0.6, label="His")
    ax4.axvline(geom.sp_pos[1], linestyle=":", alpha=0.6, label="SP")
    ax4.axvline(geom.av_node_pos[1], linestyle=":", alpha=0.6, label="AV node")
    ax4.axvline(y_center, linestyle=":", alpha=0.6, label="Catheter center")
    ax4.set_xlabel("Y position (mm)")
    ax4.set_ylabel("|E| (V/cm)")
    ax4.set_title("Midline profile (x=0)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.2)

    # 5) Electrode-line profile y=y_center
    ax5 = fig.add_subplot(gs[1, 1])
    x_line_h = np.linspace(-20, 20, 300)
    y_line_h = np.ones_like(x_line_h) * y_center

    if LinearNDInterpolator is not None:
        E_line_h = interp(x_line_h, y_line_h)
        if np.any(np.isnan(E_line_h)):
            E_line_h = np.nan_to_num(E_line_h, nan=np.nanmin(E))
    else:
        x_c, y_c = model.centroids[0], model.centroids[1]
        E_line_h = np.array([E[np.argmin((x_c - xx) ** 2 + (y_c - y_center) ** 2)] for xx in x_line_h])

    ax5.plot(x_line_h, E_line_h, linewidth=2)
    ax5.axhline(thr["re"], linestyle="--", linewidth=2, label=f"RE {thr['re']:.0f}")
    ax5.axhline(thr["ire"], linestyle="-", linewidth=2, label=f"IRE {thr['ire']:.0f}")
    ax5.axvline(e1[0], linestyle=":", alpha=0.7, label="Anode")
    ax5.axvline(e2[0], linestyle=":", alpha=0.7, label="Cathode")
    ax5.set_xlabel("X position (mm)")
    ax5.set_ylabel("|E| (V/cm)")
    ax5.set_title(f"Profile through electrodes (y={y_center:.1f} mm)")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.2)

    # 6) Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    summary = [
        ["Parameter", "Value"],
        ["Applied voltage", f"{voltage_v:.0f} V"],
        ["Pulse width", pulse.label],
        ["Electrode spacing", f"{elec.spacing_mm:.2f} mm"],
        ["Electrode radius", f"{elec.radius_mm:.2f} mm"],
        ["Distance from His", f"{distance_from_his_mm:.1f} mm"],
        ["RE threshold", f"{thr['re']:.0f} V/cm"],
        ["IRE threshold", f"{thr['ire']:.0f} V/cm"],
        ["His metric", f"{e_his:.1f} V/cm"],
        ["SP metric", f"{e_sp:.1f} V/cm"],
        ["AV metric", f"{float(av_m.get(metric, 0.0)):.1f} V/cm"],
        ["Outcome", OUTCOME_LABELS[int(outcome)]],
    ]

    table = ax6.table(cellText=summary, cellLoc="left", loc="center", colWidths=[0.48, 0.52])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    ax6.set_title("Summary", fontsize=12, fontweight="bold")

    fig.suptitle(
        f"PFA Field Analysis (v2): {voltage_v:.0f}V @ {pulse.label} | d={distance_from_his_mm:.1f} mm",
        fontsize=14,
        fontweight="bold",
    )

    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "his_metric": e_his,
        "sp_metric": e_sp,
        "av_metric": float(av_m.get(metric, 0.0)),
        "outcome": int(outcome),
    }


# -----------------------------
# Script entry point
# -----------------------------

def main():
    pulse_configs = [
        PulseConfig(label="0.1 us", ire_threshold_v_cm=3000, color="m"),
        PulseConfig(label="10 us", ire_threshold_v_cm=1200, color="c"),
        PulseConfig(label="100 us", ire_threshold_v_cm=750, color="g"),
        PulseConfig(label="1000 us", ire_threshold_v_cm=500, color="b"),
    ]

    voltages = np.arange(100, 2100, 100)
    distances = np.arange(20, 0, -1)  # 20, 19, ..., 1

    geom = GeometryConfig(
        domain_width_mm=40,
        domain_height_mm=40,
        resolution_mm=0.5,
        his_pos=(0, 10),
        sp_pos=(0, -5),
        av_node_pos=(0, 12),
        av_node_radius_mm=2.0,
        his_width_mm=2.0,
        his_length_mm=4.0,
        sp_radius_mm=2.0,
        blood_y_gt_mm=10.0,
    )

    elec = ElectrodeConfig(spacing_mm=4.0, radius_mm=0.75)

    print("Starting v2 parameter sweep...")
    print(f"His: {geom.his_pos} | SP: {geom.sp_pos} | AVN: {geom.av_node_pos}")

    results = run_parameter_sweep_v2(
        pulse_configs=pulse_configs,
        voltages_v=voltages,
        distances_mm=distances,
        geom=geom,
        elec=elec,
        metric="p99",
        verbose=True,
    )

    # Operating window plots
    for pc in pulse_configs:
        out_png = f"Operating_Window_v2_{pc.label.replace(' ', '_')}.png"
        plot_operating_window(pc, results[pc.label], voltages, distances, out_png)
        print(f"Saved {out_png}")

    # Safe distance summary
    plot_safe_distance_summary(
        pulse_configs=pulse_configs,
        results=results,
        voltages_v=voltages,
        distances_mm=distances,
        out_png="Safe_Distance_vs_Voltage_v2.png",
    )
    print("Saved Safe_Distance_vs_Voltage_v2.png")

    # Deterministic shortlist + optional robustness (example: 100 us)
    example_pc = pulse_configs[2]  # 100 us
    cands = shortlist_candidates(results[example_pc.label], voltages, distances, desired="ablation", top_k=8)

    if len(cands) > 0:
        print("\nTop deterministic ablation candidates (100 us):")
        for (V, d, oc, e_sp, e_his) in cands:
            print(f"  V={V:.0f} d={d:.0f} outcome={OUTCOME_LABELS[oc]} | SP={e_sp:.1f} His={e_his:.1f}")

        # Run robustness on the top 3
        base_model = PFAModelV2(width_mm=geom.domain_width_mm, height_mm=geom.domain_height_mm, resolution_mm=geom.resolution_mm)
        base_model.define_regions(geom)
        base_model.assemble_system()

        rconf = RobustnessConfig(n_samples=50, rng_seed=0)
        print("\nRobustness assessment (top 3):")
        for (V, d, _, _, _) in cands[:3]:
            probs = robustness_assessment(
                model=base_model,
                geom=geom,
                elec=elec,
                pulse=example_pc,
                voltage_v=V,
                distance_from_his_mm=d,
                metric="p99",
                rconf=rconf,
                perturb_conductivity=True,
            )
            print(
                f"  V={V:.0f} d={d:.0f} | "
                f"P(desired ablation)={probs['p_desired_ablation']:.2f}, "
                f"P(His RE)={probs['p_his_re']:.2f}, P(His IRE)={probs['p_his_ire']:.2f}"
            )

    # One example detailed visualization
    vis_png = "Electric_Field_Detail_v2_2000V_100us_d15mm.png"
    visualize_field_detailed_v2(
        voltage_v=2000,
        distance_from_his_mm=15,
        pulse=example_pc,
        geom=geom,
        elec=elec,
        out_png=vis_png,
    )
    print(f"Saved {vis_png}")


if __name__ == "__main__":
    main()
