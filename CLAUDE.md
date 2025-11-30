# PFA FEM Simulation for Cardiac Ablation

A finite element method (FEM) simulation tool for analyzing Pulsed Field Ablation (PFA) in cardiac electrophysiology procedures, specifically for AV nodal reentrant tachycardia (AVNRT) treatment.

## Overview

This project simulates the electric field distributions during PFA procedures to evaluate safe operating windows for slow pathway ablation while protecting the His bundle. The simulation uses scikit-fem to solve Laplace's equation for electric potential in cardiac tissue with heterogeneous conductivities.

## Clinical Context

**Application**: Slow pathway ablation for AVNRT treatment

**Key Anatomical Structures**:
- **AV Node**: Located at (0, 12 mm)
- **His Bundle**: Located at (0, 10 mm) - critical structure to preserve
- **Slow Pathway (SP)**: Located at (0, -5 mm) - ablation target (15 mm inferior to His)

**Clinical Objectives**:
1. **Safe Mapping**: Stun SP for identification while keeping His safe (reversible electroporation, RE)
2. **Effective Ablation**: Ablate SP while preserving His function (irreversible electroporation, IRE)
3. **Safety**: Avoid His bundle damage at all voltage/distance combinations

## Features

### 1. FEM-Based Electric Field Simulation
- 2D triangular mesh with configurable resolution
- Heterogeneous tissue conductivities:
  - Myocardium: 0.16 S/m
  - Blood: 0.50 S/m
  - AV Node/SP: 0.05 S/m
  - His Bundle: 0.15 S/m
- Bipolar electrode configuration with adjustable spacing (default: 4 mm)

### 2. Parameter Sweep Analysis
The simulation sweeps two critical parameters:

**Voltage Range**: 100V to 2000V (100V increments)

**Electrode Distance**: 1-20 mm inferior to His bundle

### 3. Pulse Width Analysis (Strength-Duration Relationship)
Evaluates four pulse widths with corresponding IRE thresholds:
- **0.1 μs**: 3000 V/cm threshold
- **10 μs**: 1200 V/cm threshold  
- **100 μs**: 750 V/cm threshold
- **1000 μs**: 500 V/cm threshold

*Note: RE threshold is defined as 50% of IRE threshold*

### 4. Clinical Outcome Classification

The simulation categorizes each voltage-distance combination into:

| Outcome | Color | Definition |
|---------|-------|------------|
| **Ineffective** | Gray | E_SP < RE threshold and E_His < RE threshold |
| **His Stunned Only** | Yellow | E_His ≥ RE threshold, E_SP < RE threshold |
| **Warning Zone** | Orange | E_His ≥ RE threshold and RE ≤ E_SP < IRE |
| **Safe Mapping** | Green | RE ≤ E_SP < IRE and E_His < RE |
| **Effective Ablation** | Blue | E_SP ≥ IRE and E_His < IRE |
| **Unsafe** | Red | E_His ≥ IRE threshold |

### 5. Visualization Outputs

**Operating Window Heatmaps** (`Operating_Window_*.png`):
- 2D heatmaps showing clinical outcomes across voltage-distance space
- One plot per pulse width configuration
- Color-coded by clinical outcome category

**Safe Distance Analysis** (`Safe_Distance_vs_Voltage_Microseconds.png`):
- Multi-line plot showing minimum safe distance vs. voltage
- Demonstrates strength-duration relationship
- Compares all four pulse width configurations

**Individual Simulations** (`PFA_*.png`, `Safety_Profile_*.png`):
- Electric potential and field distributions
- Single-voltage simulation results

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

**Dependencies**:
- scikit-fem
- numpy
- matplotlib
- scipy

### Python Version
Tested with Python 3.8+

## Usage

### Run Complete Parameter Sweep

```bash
python main.py
```

This will:
1. Execute 800+ simulations (20 distances × 20 voltages × 4 pulse widths)
2. Generate operating window heatmaps for each pulse width
3. Create a summary safe distance plot
4. Save all visualizations to PNG files

### Expected Runtime
- **Total simulations**: ~800
- **Estimated time**: 5-15 minutes (depends on mesh resolution and hardware)
- **Output files**: 5 PNG files

### Customize Simulation Parameters

Modify parameters in `main.py`:

```python
# Voltage range
voltages = np.arange(100, 2100, 100)

# Distance range (mm inferior to His)
distances = np.arange(20, 0, -1)

# Electrode spacing
electrode_spacing = 4.0  # mm

# Mesh resolution
model = PFAModel(width=40, height=40, resolution=0.5)

# Pulse configurations
pulse_configs = [
    {"width": "0.1 us", "threshold": 3000, "color": "m"},
    # Add more configurations...
]
```

## Code Structure

### `model.py`: PFAModel Class
Core FEM simulation engine

**Key Methods**:
- `__init__(width, height, resolution)`: Initialize mesh and domain
- `define_regions(...)`: Define anatomical structures with conductivities
- `solve(voltage, electrode1_pos, electrode2_pos)`: Solve Laplace equation
- `calculate_field()`: Compute electric field magnitude (V/cm)
- `plot_results(title)`: Visualize potential and field distributions

### `main.py`: Parameter Sweep and Analysis
Main execution script

**Key Functions**:
- `run_parameter_sweep()`: Complete multi-parameter analysis workflow
  1. Execute voltage-distance sweep
  2. Classify clinical outcomes for each pulse width
  3. Generate operating window visualizations
  4. Create safe distance summary plots

## Physical Model

### Governing Equation
Laplace's equation for quasi-static electric potential:

```
∇ · (σ ∇φ) = 0
```

where:
- `σ` = tissue conductivity (S/m)
- `φ` = electric potential (V)

### Boundary Conditions
- Dirichlet conditions at electrode positions
- Electrode 1 (anode): φ = V_applied
- Electrode 2 (cathode): φ = 0V

### Electric Field
```
E = -∇φ
|E| = sqrt(Ex² + Ey²)
```

Field strength reported in **V/cm** (clinical standard)

## Output Files

| File | Description |
|------|-------------|
| `Operating_Window_0.1_us.png` | Operating window for 0.1 μs pulses |
| `Operating_Window_10_us.png` | Operating window for 10 μs pulses |
| `Operating_Window_100_us.png` | Operating window for 100 μs pulses |
| `Operating_Window_1000_us.png` | Operating window for 1000 μs pulses |
| `Safe_Distance_vs_Voltage_Microseconds.png` | Safe distance summary across all pulse widths |

*Note: Additional PFA_*.png and Safety_Profile_*.png files may be generated from previous runs*

## Clinical Interpretation

### Reading Operating Windows
- **X-axis**: Distance from His Bundle (mm) - further = safer
- **Y-axis**: Applied Voltage (V) - higher = stronger field
- **Colors**: Clinical outcome classification (see table above)

### Key Clinical Insights
1. **Green zones** (Safe Mapping): Optimal for SP identification without His damage
2. **Blue zones** (Effective Ablation): Target parameters for permanent SP ablation
3. **Red zones** (Unsafe): Avoid - risk of His bundle damage
4. **Strength-Duration**: Shorter pulses require higher voltages but offer more precise control

### Safety Margin Recommendations
- Maintain minimum 5-10 mm distance from His bundle at typical voltages
- Lower voltages provide larger safety margins
- Shorter pulse widths (0.1-10 μs) offer steeper field gradients

## Limitations

1. **2D Simplification**: Actual cardiac anatomy is 3D
2. **Homogeneous Tissue Assumptions**: Real tissue has spatially varying properties
3. **Static Model**: Doesn't account for dynamic tissue changes during ablation
4. **Idealized Geometry**: Simplified representation of complex anatomical structures
5. **Thermal Effects**: Pure electroporation model, no thermal damage consideration

## Future Enhancements

- [ ] 3D mesh implementation
- [ ] Patient-specific anatomy from imaging data
- [ ] Dynamic conductivity changes during ablation
- [ ] Thermal-electric coupled model
- [ ] Biphasic pulse waveform modeling
- [ ] Tissue damage probability models

## References

This simulation framework is designed for research purposes to explore PFA parameter spaces in cardiac ablation. The strength-duration thresholds are based on published cardiac electroporation literature.

## License

Research code - see project documentation for usage terms.

## Contact

For questions or collaborations, please contact the project maintainer.

---

**Last Updated**: November 2025