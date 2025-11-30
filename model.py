import numpy as np
import matplotlib.pyplot as plt
from skfem import *
from skfem.helpers import dot, grad
from skfem.visuals.matplotlib import plot, draw
from skfem.element import ElementTriP0

class PFAModel:
    def __init__(self, width=40.0, height=40.0, resolution=1.0):
        """
        Initialize the PFA FEM Model.
        
        Args:
            width (float): Width of the domain in mm.
            height (float): Height of the domain in mm.
            resolution (float): Mesh resolution.
        """
        self.width = width
        self.height = height
        # Create a 2D triangular mesh
        # Centered at (0,0)
        self.mesh = MeshTri.init_tensor(
            np.linspace(-width/2, width/2, int(width/resolution) + 1),
            np.linspace(-height/2, height/2, int(height/resolution) + 1)
        )
        self.basis = Basis(self.mesh, ElementTriP1())
        
        # Default conductivities (S/m)
        self.sigma_myocardium = 0.16
        self.sigma_blood = 0.50
        self.sigma_av_node = 0.05 # Estimated low conductivity
        self.sigma_his = 0.15     # Estimated moderate conductivity
        
        self.conductivities = np.zeros(self.basis.nelems) + self.sigma_myocardium

    def define_regions(self, av_node_pos=(5, 5), av_node_radius=2.0,
                       his_bundle_pos=(5, 0), his_bundle_width=1.0, his_bundle_length=5.0,
                       sp_pos=(5, 0), sp_radius=2.0):
        """
        Define tissue regions on the mesh.
        Simple geometric definitions for this iteration.
        """
        # Get centroids of elements
        centroids = self.mesh.p.mean(axis=1) if self.mesh.p.shape[1] == 3 else self.mesh.p[:, self.mesh.t].mean(axis=1)
        x, y = centroids
        
        # 1. Blood (e.g., top half of the domain for simplicity, or a specific chamber)
        # Let's assume blood is above y = 10
        blood_mask = y > 10
        self.conductivities[blood_mask] = self.sigma_blood
        
        # 2. AV Node (Circular region)
        av_dist = np.sqrt((x - av_node_pos[0])**2 + (y - av_node_pos[1])**2)
        av_mask = av_dist < av_node_radius
        self.conductivities[av_mask] = self.sigma_av_node
        
        # 3. His Bundle (Rectangular strip extending from AV node)
        his_mask = (x > his_bundle_pos[0]) & (x < his_bundle_pos[0] + his_bundle_width) & \
                   (y > his_bundle_pos[1] - his_bundle_length/2) & (y < his_bundle_pos[1] + his_bundle_length/2)
        self.conductivities[his_mask] = self.sigma_his

        # 4. Slow Pathway (Circular region inferior to AV Node)
        # Typically located near the CS ostium, inferior/posterior to AVN.
        sp_dist = np.sqrt((x - sp_pos[0])**2 + (y - sp_pos[1])**2)
        sp_mask = sp_dist < sp_radius
        # SP likely has similar conductivity to AVN or Myocardium? Let's assume AVN-like for now or just Myocardium.
        # It's a functional pathway. Let's assign it AVN conductivity for now to distinguish it.
        self.conductivities[sp_mask] = self.sigma_av_node

    def solve(self, voltage=1000.0, electrode1_pos=(-2.0, 0.0), electrode2_pos=(2.0, 0.0)):
        """
        Solve the Laplace equation for electric potential.
        
        Args:
            voltage (float): Applied voltage in Volts.
            electrode1_pos (tuple): (x, y) of Electrode 1 (Anode, +V).
            electrode2_pos (tuple): (x, y) of Electrode 2 (Cathode, 0V).
        """
        # Define conductivity function for assembly
        @BilinearForm
        def laplace(u, v, w):
            return w['sigma'] * dot(grad(u), grad(v))

        # Create a P0 basis for the element-wise conductivity
        basis0 = self.basis.with_element(ElementTriP0())
        sigma_field = basis0.interpolate(self.conductivities)

        # Assemble stiffness matrix
        # We pass the interpolated conductivity
        A = asm(laplace, self.basis, sigma=sigma_field)
        
        # Boundary Conditions
        # Find nodes closest to electrode positions
        e1_x, e1_y = electrode1_pos
        e2_x, e2_y = electrode2_pos
        
        nodes = self.mesh.p
        dist1 = np.sqrt((nodes[0] - e1_x)**2 + (nodes[1] - e1_y)**2)
        dist2 = np.sqrt((nodes[0] - e2_x)**2 + (nodes[1] - e2_y)**2)
        
        node_e1 = np.argmin(dist1)
        node_e2 = np.argmin(dist2)
        
        # For P1 elements, node indices correspond to DOF indices
        dofs = np.array([node_e1, node_e2])
        
        # Enforce Dirichlet BCs
        x = np.zeros(self.basis.N)
        x[node_e1] = voltage
        x[node_e2] = 0.0
        
        # Solve
        self.potential = solve(*condense(A, x=x, D=dofs))
        return self.potential

    def calculate_field(self):
        """
        Calculate Electric Field Magnitude |E| = |-grad(phi)|
        """
        # Interpolate potential to quadrature points of the P1 basis
        out = self.basis.interpolate(self.potential)
        grad_u = out.grad # Shape: (2, nqp, nelems)
        
        # print(f"DEBUG: grad_u shape: {grad_u.shape}")
        
        # Compute magnitude: sqrt(Ex^2 + Ey^2)
        E_mag = np.sqrt(grad_u[0]**2 + grad_u[1]**2) # Shape: (nqp, nelems)
        
        # print(f"DEBUG: E_mag shape: {E_mag.shape}")
        
        # Average over quadrature points to get element-wise value
        # If shape is (nqp, nelems), we mean over axis 0.
        # If shape is (nelems, nqp), we mean over axis 1.
        
        if E_mag.shape[0] == self.basis.nelems:
             # Shape is (nelems, nqp) - unlikely for skfem but possible
             self.E_field = np.mean(E_mag, axis=1)
        else:
             # Shape is (nqp, nelems)
             self.E_field = np.mean(E_mag, axis=0)
             
        # print(f"DEBUG: E_field shape: {self.E_field.shape}")
        
        # Convert to V/cm (model is in mm, so V/mm * 10 = V/cm)
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
