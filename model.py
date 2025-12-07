import numpy as np
import matplotlib.pyplot as plt
from skfem import *
from skfem.helpers import dot, grad
from skfem.visuals.matplotlib import plot, draw
from skfem.element import ElementTetP1, ElementTetP0

class PFAModel:
    def __init__(self, width=40.0, height=40.0, depth=10.0, resolution=1.0):
        """
        Initialize the PFA FEM Model (3D Anisotropic).
        
        Args:
            width (float): Width of the domain in mm (x-axis).
            height (float): Height of the domain in mm (y-axis).
            depth (float): Depth of the domain in mm (z-axis).
            resolution (float): Mesh resolution.
        """
        self.width = width
        self.height = height
        self.depth = depth
        
        # Create a 3D tetrahedral mesh
        # Centered at (0,0,0)
        self.mesh = MeshTet.init_tensor(
            np.linspace(-width/2, width/2, int(width/resolution) + 1),
            np.linspace(-height/2, height/2, int(height/resolution) + 1),
            np.linspace(-depth/2, depth/2, int(depth/resolution) + 1)
        )
        self.basis = Basis(self.mesh, ElementTetP1())
        
        # Default conductivities (S/m)
        # Longitudinal (along fibers) and Transverse (across fibers)
        self.sigma_myocardium_l = 0.16
        self.sigma_myocardium_t = 0.05  # Anisotropy ratio ~3:1
        
        self.sigma_blood = 0.50
        self.sigma_av_node = 0.05 
        self.sigma_his = 0.15     
        
        # Initialize conductivity tensors for each element
        # Shape: (3, 3, nelems)
        self.conductivity_tensors = np.zeros((3, 3, self.basis.nelems))
        
        # Default fiber direction (e.g., along X-axis)
        self.fiber_direction = np.array([1.0, 0.0, 0.0]) 
        
        # Initialize with myocardium properties
        self._set_conductivity_tensor(
            np.ones(self.basis.nelems, dtype=bool), 
            self.sigma_myocardium_l, 
            self.sigma_myocardium_t,
            self.fiber_direction
        )

    def _set_conductivity_tensor(self, mask, sigma_l, sigma_t, fiber_dir):
        """
        Helper to set conductivity tensor for masked elements.
        Tensor = sigma_t * I + (sigma_l - sigma_t) * (d . d^T)
        where d is the unit fiber direction vector.
        """
        if not np.any(mask):
            return
            
        # Normalize fiber direction
        d = np.array(fiber_dir) / np.linalg.norm(fiber_dir)
        
        # Outer product d . d^T
        ddT = np.outer(d, d) # (3, 3)
        
        # Identity matrix
        I = np.eye(3)
        
        # Calculate tensor
        sigma_tensor = sigma_t * I + (sigma_l - sigma_t) * ddT
        
        # Assign to masked elements
        # We need to broadcast (3,3) to (3,3, N_masked)
        count = np.sum(mask)
        self.conductivity_tensors[:, :, mask] = sigma_tensor[:, :, np.newaxis]

    def define_regions(self, av_node_pos=(5, 5, 0), av_node_radius=2.0,
                       his_bundle_pos=(5, 0, 0), his_bundle_width=1.0, his_bundle_length=5.0,
                       sp_pos=(5, 0, 0), sp_radius=2.0):
        """
        Define tissue regions on the 3D mesh.
        """
        # Get centroids of elements
        centroids = self.mesh.p[:, self.mesh.t].mean(axis=1)
        x, y, z = centroids
        
        # 1. Blood (e.g., top half of the domain for simplicity, or a specific chamber)
        # Let's assume blood is above y = 10
        blood_mask = y > 10
        # Blood is isotropic
        self._set_conductivity_tensor(blood_mask, self.sigma_blood, self.sigma_blood, [1,0,0])
        
        # 2. AV Node (Spherical region)
        av_dist = np.sqrt((x - av_node_pos[0])**2 + (y - av_node_pos[1])**2 + (z - av_node_pos[2])**2)
        av_mask = av_dist < av_node_radius
        # AV Node isotropic for now
        self._set_conductivity_tensor(av_mask, self.sigma_av_node, self.sigma_av_node, [1,0,0])
        
        # 3. His Bundle (Rectangular box extending from AV node)
        # Assuming it has some thickness in Z (e.g., 2mm)
        his_thickness = 2.0
        his_mask = (x > his_bundle_pos[0]) & (x < his_bundle_pos[0] + his_bundle_width) & \
                   (y > his_bundle_pos[1] - his_bundle_length/2) & (y < his_bundle_pos[1] + his_bundle_length/2) & \
                   (z > his_bundle_pos[2] - his_thickness/2) & (z < his_bundle_pos[2] + his_thickness/2)
        # His bundle is conductive, maybe anisotropic along Y (length)? 
        # Let's assume isotropic for simplicity or anisotropic along Y
        self._set_conductivity_tensor(his_mask, self.sigma_his, self.sigma_his, [0,1,0])

        # 4. Slow Pathway (Spherical region inferior to AV Node)
        sp_dist = np.sqrt((x - sp_pos[0])**2 + (y - sp_pos[1])**2 + (z - sp_pos[2])**2)
        sp_mask = sp_dist < sp_radius
        self._set_conductivity_tensor(sp_mask, self.sigma_av_node, self.sigma_av_node, [1,0,0])

    def get_electrode_nodes(self, center, radius, length, axis=0):
        """
        Identify nodes within a cylindrical electrode volume.
        
        Args:
            center (tuple): (x, y, z) center of the electrode.
            radius (float): Radius of the electrode.
            length (float): Length of the electrode.
            axis (int): Axis of orientation (0=x, 1=y, 2=z).
            
        Returns:
            np.array: Indices of nodes within the electrode.
        """
        x, y, z = self.mesh.p
        cx, cy, cz = center
        
        # Transform to local coordinates relative to center
        dx = x - cx
        dy = y - cy
        dz = z - cz
        
        if axis == 0: # Aligned with X
            radial_dist = np.sqrt(dy**2 + dz**2)
            axial_dist = np.abs(dx)
        elif axis == 1: # Aligned with Y
            radial_dist = np.sqrt(dx**2 + dz**2)
            axial_dist = np.abs(dy)
        else: # Aligned with Z
            radial_dist = np.sqrt(dx**2 + dy**2)
            axial_dist = np.abs(dz)
            
        # Check if inside cylinder
        mask = (radial_dist <= radius) & (axial_dist <= length / 2.0)
        nodes = np.where(mask)[0]
        
        if len(nodes) == 0:
            # Fallback: Find closest node
            dist = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
            closest = np.argmin(dist)
            print(f"WARNING: No nodes found within electrode radius. Snapping to closest node {closest} at {self.mesh.p[:, closest]}.")
            return np.array([closest])
            
        print(f"DEBUG: Found {len(nodes)} nodes for electrode at {center} (r={radius}, l={length})")
        return nodes

    def solve(self, voltage=1000.0, 
              tip_pos=(-2.0, 0.0, 0.0), tip_length=4.0, tip_radius=1.0,
              ring_pos=(2.0, 0.0, 0.0), ring_length=2.0, ring_radius=1.0,
              mode='bipolar'):
        """
        Solve the Laplace equation for electric potential in 3D with finite electrodes.
        """
        # Define conductivity function for assembly
        @BilinearForm
        def laplace(u, v, w):
            sig = np.transpose(w['sigma'], (0, 1, 3, 2))
            j_x = sig[0,0]*u.grad[0] + sig[0,1]*u.grad[1] + sig[0,2]*u.grad[2]
            j_y = sig[1,0]*u.grad[0] + sig[1,1]*u.grad[1] + sig[1,2]*u.grad[2]
            j_z = sig[2,0]*u.grad[0] + sig[2,1]*u.grad[1] + sig[2,2]*u.grad[2]
            return v.grad[0]*j_x + v.grad[1]*j_y + v.grad[2]*j_z

        nqp = self.basis.quadrature[0].shape[1]
        sigma_field = np.repeat(self.conductivity_tensors[:, :, np.newaxis, :], nqp, axis=2)
        
        # Assemble stiffness matrix
        A = asm(laplace, self.basis, sigma=sigma_field)
        
        # Boundary Conditions
        
        # 1. Identify Electrode Nodes (Finite Volume)
        tip_nodes = self.get_electrode_nodes(tip_pos, tip_radius, tip_length, axis=0)
        if len(tip_nodes) == 0:
            print("WARNING: No nodes found for Tip Electrode! Mesh resolution might be too coarse.")
        
        if mode == 'bipolar':
            ring_nodes = self.get_electrode_nodes(ring_pos, ring_radius, ring_length, axis=0)
            if len(ring_nodes) == 0:
                print("WARNING: No nodes found for Ring Electrode! Mesh resolution might be too coarse.")
            
            dofs = np.concatenate([tip_nodes, ring_nodes])
            x = np.zeros(self.basis.N)
            x[tip_nodes] = voltage
            x[ring_nodes] = 0.0
            
        elif mode == 'monopolar':
            # Tip = +V
            # Boundary = 0V (Distant patch)
            boundary_dofs = self.mesh.boundary_nodes() # All boundary nodes()
            
            # Exclude tip nodes from boundary dofs
            boundary_dofs = np.setdiff1d(boundary_dofs, tip_nodes)
            
            dofs = np.concatenate([tip_nodes, boundary_dofs])
            x = np.zeros(self.basis.N)
            x[tip_nodes] = voltage
            x[boundary_dofs] = 0.0
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Solve
        self.potential = solve(*condense(A, x=x, D=dofs))
        return self.potential

    def calculate_field(self):
        """
        Calculate Electric Field Magnitude |E| = |-grad(phi)|
        """
        out = self.basis.interpolate(self.potential)
        grad_u = out.grad # Shape: (3, nqp, nelems)
        
        # Compute magnitude: sqrt(Ex^2 + Ey^2 + Ez^2)
        E_mag = np.sqrt(grad_u[0]**2 + grad_u[1]**2 + grad_u[2]**2)
        
        if E_mag.shape[0] == self.basis.nelems:
             self.E_field = np.mean(E_mag, axis=1)
        else:
             self.E_field = np.mean(E_mag, axis=0)
        
        self.E_field_v_cm = self.E_field * 10.0
        return self.E_field_v_cm

    def plot_results(self, title="PFA Simulation"):
        """
        Plot the potential and electric field.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot Potential
        draw(self.mesh, node_data=self.potential, ax=ax[0])
        ax[0].set_title("Electric Potential (V)")
        ax[0].set_aspect('equal')
        
        # Plot Electric Field
        # For P0 data (element-wise), we can use tripcolor
        im = ax[1].tripcolor(self.mesh.p[0], self.mesh.p[1], self.mesh.t.T, 
                             self.E_field_v_cm, shading='flat', cmap='inferno')
        ax[1].set_title("Electric Field (V/cm)")
        ax[1].set_aspect('equal')
        plt.colorbar(im, ax=ax[1])
        
        # Highlight contours for RE and IRE
        # Interpolate P0 to P1 for contour plotting
        # Simple averaging from elements to nodes
        # Or just project P0 to P1
        # For simplicity in this visualization, we rely on the color map
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(f"{title.replace(' ', '_')}.png")
        plt.close()
