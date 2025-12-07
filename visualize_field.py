import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from model import PFAModel
from scipy.interpolate import LinearNDInterpolator

def visualize_electric_field_detailed(voltage=2000, distance_from_his=15, pulse_width="10 us", 
                                      ire_threshold=1200, re_threshold=600, mode='bipolar'):
    """
    Create a detailed visualization of the electric field for PFA simulation (3D Slice).

    Args:
        voltage: Applied voltage (V)
        distance_from_his: Distance of electrode center from His bundle (mm inferior)
        pulse_width: Pulse width label
        ire_threshold: IRE threshold (V/cm)
        re_threshold: RE threshold (V/cm)
        mode: 'bipolar' or 'monopolar'
    """
    # Geometry
    # Corrected Anatomy: His is Superior to AV Node
    his_pos = (0, 12, 0)
    av_node_pos = (0, 10, 0)
    sp_pos = (0, -5, 0)
    
    # Electrode Dimensions
    tip_length = 4.0
    tip_radius = 1.25 # 2.5mm diameter (7.5 Fr)
    ring_length = 2.0
    ring_radius = 1.25
    spacing = 2.0 # Space between tip and ring
    
    y_center = his_pos[1] - distance_from_his
    
    tip_center_x = -spacing/2 - tip_length/2
    ring_center_x = spacing/2 + ring_length/2
    
    tip_pos = (tip_center_x, y_center, 0)
    ring_pos = (ring_center_x, y_center, 0)

    print(f"Simulating Electric Field (3D Anisotropic) - Mode: {mode}")
    print(f"  Voltage: {voltage} V")
    print(f"  Pulse Width: {pulse_width}")
    print(f"  IRE Threshold: {ire_threshold} V/cm")
    print(f"  RE Threshold: {re_threshold} V/cm")
    print(f"  Electrode Center: {distance_from_his} mm inferior to His")
    print(f"  Tip Position: {tip_pos}")
    if mode == 'bipolar':
        print(f"  Ring Position: {ring_pos}")

    # Initialize model with coarser resolution for 3D performance
    # Resolution 1.5mm is reasonable compromise
    model = PFAModel(width=40, height=40, depth=10, resolution=1.5)

    # Define regions
    model.define_regions(
        av_node_pos=av_node_pos, av_node_radius=2.0,
        his_bundle_pos=his_pos, his_bundle_width=2.0, his_bundle_length=4.0,
        sp_pos=sp_pos, sp_radius=2.0
    )

    # Solve
    print("  Solving FEM (3D)...")
    model.solve(voltage=voltage, 
                tip_pos=tip_pos, tip_length=tip_length, tip_radius=tip_radius,
                ring_pos=ring_pos, ring_length=ring_length, ring_radius=ring_radius,
                mode=mode)

    # Calculate field
    print("  Calculating electric field...")
    E_field = model.calculate_field()

    # Analyze peak fields at anatomical structures
    centroids = model.mesh.p[:, model.mesh.t].mean(axis=1)
    x, y, z = centroids

    # His bundle
    his_dist = np.sqrt((x - his_pos[0])**2 + (y - his_pos[1])**2 + (z - his_pos[2])**2)
    his_mask = his_dist < 1.5
    max_E_his = np.max(E_field[his_mask]) if np.any(his_mask) else 0.0

    # SP
    sp_dist = np.sqrt((x - sp_pos[0])**2 + (y - sp_pos[1])**2 + (z - sp_pos[2])**2)
    sp_mask = sp_dist < 1.5
    max_E_sp = np.max(E_field[sp_mask]) if np.any(sp_mask) else 0.0

    # AV Node
    av_dist = np.sqrt((x - av_node_pos[0])**2 + (y - av_node_pos[1])**2 + (z - av_node_pos[2])**2)
    av_mask = av_dist < 1.5
    max_E_av = np.max(E_field[av_mask]) if np.any(av_mask) else 0.0

    print(f"\n  Peak Electric Fields:")
    print(f"    His Bundle: {max_E_his:.1f} V/cm")
    print(f"    Slow Pathway: {max_E_sp:.1f} V/cm")
    print(f"    AV Node: {max_E_av:.1f} V/cm")

    # Determine clinical outcome
    if max_E_his >= ire_threshold:
        outcome = "UNSAFE (His Ablated)"
        outcome_color = "red"
    elif max_E_sp >= ire_threshold:
        outcome = "Effective Ablation (SP Ablated, His Safe)"
        outcome_color = "dodgerblue"
    elif max_E_sp >= re_threshold and max_E_his < re_threshold:
        outcome = "Safe Mapping (SP Stunned, His Safe)"
        outcome_color = "green"
    elif max_E_his >= re_threshold:
        if max_E_sp >= re_threshold:
            outcome = "Warning (His Stunned, SP Stunned)"
            outcome_color = "orange"
        else:
            outcome = "His Stunned Only"
            outcome_color = "gold"
    else:
        outcome = "Ineffective"
        outcome_color = "gray"

    print(f"    Clinical Outcome: {outcome}")

    # Create comprehensive figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Prepare 2D Slice Interpolation (Z=0)
    print("  Interpolating 2D slice for visualization...")
    grid_x, grid_y = np.mgrid[-20:20:200j, -20:20:200j]
    grid_z = np.zeros_like(grid_x)
    
    # Interpolate Potential (Node-based)
    interp_pot = LinearNDInterpolator(model.mesh.p.T, model.potential)
    pot_slice = interp_pot(grid_x, grid_y, grid_z)
    
    # Interpolate E-field (Element-based)
    interp_field = LinearNDInterpolator(centroids.T, E_field)
    field_slice = interp_field(grid_x, grid_y, grid_z)
    
    # Interpolate Conductivity (Element-based)
    cond_trace = model.conductivity_tensors[0,0,:] + model.conductivity_tensors[1,1,:] + model.conductivity_tensors[2,2,:]
    cond_avg = cond_trace / 3.0
    interp_cond = LinearNDInterpolator(centroids.T, cond_avg)
    cond_slice = interp_cond(grid_x, grid_y, grid_z)

    # 1. Electric Potential
    ax1 = fig.add_subplot(gs[0, 0])
    mesh_plot = ax1.contourf(grid_x, grid_y, pot_slice, levels=50, cmap='RdBu_r')
    plt.colorbar(mesh_plot, ax=ax1, label='Potential (V)')

    # Add anatomical markers
    ax1.add_patch(Circle(his_pos[:2], 2.0, fill=False, edgecolor='red', linewidth=2, label='His Bundle'))
    ax1.add_patch(Circle(sp_pos[:2], 2.0, fill=False, edgecolor='blue', linewidth=2, label='Slow Pathway'))
    ax1.add_patch(Circle(av_node_pos[:2], 2.0, fill=False, edgecolor='purple', linewidth=2, label='AV Node'))
    
    # Add Electrodes (Rectangles for X-aligned cylinders)
    tip_rect = Rectangle((tip_pos[0]-tip_length/2, tip_pos[1]-tip_radius), tip_length, tip_radius*2, 
                         color='red', alpha=0.5, label='Tip (+V)')
    ax1.add_patch(tip_rect)
    
    if mode == 'bipolar':
        ring_rect = Rectangle((ring_pos[0]-ring_length/2, ring_pos[1]-ring_radius), ring_length, ring_radius*2, 
                              color='black', alpha=0.5, label='Ring (0V)')
        ax1.add_patch(ring_rect)

    ax1.set_xlabel('X Position (mm)')
    ax1.set_ylabel('Y Position (mm)')
    ax1.set_title(f'Electric Potential (Z=0 Slice) - {mode.capitalize()}')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Electric Field Magnitude (Zone Map)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Define levels for zones
    # 0 -> RE: Safe (Gray/White)
    # RE -> IRE: Stunned (Yellow/Orange)
    # > IRE: Ablated (Red)
    levels = [0, re_threshold, ire_threshold, 20000]
    colors = ['#f0f0f0', '#ffd700', '#ff4500'] # Light Gray, Gold, OrangeRed
    
    field_plot = ax2.contourf(grid_x, grid_y, field_slice, levels=levels, colors=colors)
    
    # Add anatomical markers
    ax2.add_patch(Circle(his_pos[:2], 2.0, fill=False, edgecolor='black', linewidth=2))
    ax2.add_patch(Circle(sp_pos[:2], 2.0, fill=False, edgecolor='black', linewidth=2))
    ax2.add_patch(Circle(av_node_pos[:2], 2.0, fill=False, edgecolor='black', linewidth=2, linestyle='--'))
    
    # Add Electrodes
    ax2.add_patch(Rectangle((tip_pos[0]-tip_length/2, tip_pos[1]-tip_radius), tip_length, tip_radius*2, 
                            fill=False, edgecolor='black', linewidth=1))
    if mode == 'bipolar':
        ax2.add_patch(Rectangle((ring_pos[0]-ring_length/2, ring_pos[1]-ring_radius), ring_length, ring_radius*2, 
                                fill=False, edgecolor='black', linewidth=1))

    # Add text labels
    ax2.text(his_pos[0]+2.5, his_pos[1], 'His', color='black', fontweight='bold', fontsize=10)
    ax2.text(sp_pos[0]+2.5, sp_pos[1], 'SP', color='black', fontweight='bold', fontsize=10)
    ax2.text(av_node_pos[0]+2.5, av_node_pos[1], 'AVN', color='black', fontweight='bold', fontsize=10)

    ax2.set_xlabel('X Position (mm)')
    ax2.set_ylabel('Y Position (mm)')
    ax2.set_title(f'Ablation Zones (Z=0)\n(Yellow: Stunned, Red: Ablated)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3, color='black', linewidth=0.5)
    
    # Create custom legend for zones
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#f0f0f0', lw=4),
                    Line2D([0], [0], color='#ffd700', lw=4),
                    Line2D([0], [0], color='#ff4500', lw=4)]
    ax2.legend(custom_lines, ['Safe', 'Stunned', 'Ablated'], loc='upper right', fontsize=8)

    # 3. Tissue Conductivity Map
    ax3 = fig.add_subplot(gs[0, 2])
    cond_plot = ax3.contourf(grid_x, grid_y, cond_slice, levels=20, cmap='viridis')
    cbar3 = plt.colorbar(cond_plot, ax=ax3, label='Avg Conductivity (S/m)')

    # Add anatomical markers
    ax3.add_patch(Circle(his_pos[:2], 2.0, fill=False, edgecolor='red', linewidth=2))
    ax3.add_patch(Circle(sp_pos[:2], 2.0, fill=False, edgecolor='blue', linewidth=2))
    ax3.add_patch(Circle(av_node_pos[:2], 2.0, fill=False, edgecolor='purple', linewidth=2))

    ax3.set_xlabel('X Position (mm)')
    ax3.set_ylabel('Y Position (mm)')
    ax3.set_title('Conductivity Distribution (Z=0)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # 4. Field along vertical line through center (x=0)
    ax4 = fig.add_subplot(gs[1, 0])

    # Extract field along x=0
    y_line = np.linspace(-20, 20, 200)
    x_line = np.zeros_like(y_line)
    z_line = np.zeros_like(y_line)

    E_line = interp_field(x_line, y_line, z_line)

    ax4.plot(y_line, E_line, 'b-', linewidth=2)
    ax4.axhline(re_threshold, color='cyan', linestyle='--', linewidth=2, label=f'RE ({re_threshold} V/cm)')
    ax4.axhline(ire_threshold, color='lime', linestyle='-', linewidth=2, label=f'IRE ({ire_threshold} V/cm)')

    # Mark anatomical positions
    ax4.axvline(his_pos[1], color='red', linestyle=':', alpha=0.5, label='His')
    ax4.axvline(sp_pos[1], color='blue', linestyle=':', alpha=0.5, label='SP')
    ax4.axvline(av_node_pos[1], color='purple', linestyle=':', alpha=0.5, label='AV Node')
    ax4.axvline(y_center, color='orange', linestyle=':', alpha=0.7, label='Catheter Center')

    ax4.set_xlabel('Y Position (mm) [Inferior ← → Superior]')
    ax4.set_ylabel('Electric Field (V/cm)')
    ax4.set_title('E-Field Profile Along Midline (x=0, z=0)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-20, 20)

    # 5. Field along horizontal line through electrodes
    ax5 = fig.add_subplot(gs[1, 1])

    x_line_h = np.linspace(-20, 20, 200)
    y_line_h = np.ones_like(x_line_h) * y_center
    z_line_h = np.zeros_like(x_line_h)

    E_line_h = interp_field(x_line_h, y_line_h, z_line_h)

    ax5.plot(x_line_h, E_line_h, 'b-', linewidth=2)
    ax5.axhline(re_threshold, color='cyan', linestyle='--', linewidth=2, label=f'RE ({re_threshold} V/cm)')
    ax5.axhline(ire_threshold, color='lime', linestyle='-', linewidth=2, label=f'IRE ({ire_threshold} V/cm)')

    # Mark electrode positions
    ax5.axvline(tip_pos[0], color='red', linestyle=':', alpha=0.7, label='Tip')
    if mode == 'bipolar':
        ax5.axvline(ring_pos[0], color='black', linestyle=':', alpha=0.7, label='Ring')
    
    ax5.set_xlabel('X Position (mm) [Left ← → Right]')
    ax5.set_ylabel('Electric Field (V/cm)')
    ax5.set_title(f'E-Field Profile Through Electrodes (y={y_center:.1f} mm)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-20, 20)

    # 6. Summary Statistics Table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    # Create summary table
    summary_data = [
        ['Parameter', 'Value'],
        ['─' * 25, '─' * 25],
        ['Mode', mode.capitalize()],
        ['Applied Voltage', f'{voltage} V'],
        ['Pulse Width', pulse_width],
        ['Distance from His', f'{distance_from_his} mm'],
        ['', ''],
        ['Thresholds', ''],
        ['  IRE Threshold', f'{ire_threshold} V/cm'],
        ['  RE Threshold', f'{re_threshold} V/cm'],
        ['', ''],
        ['Peak E-Fields', ''],
        ['  His Bundle', f'{max_E_his:.1f} V/cm'],
        ['  Slow Pathway', f'{max_E_sp:.1f} V/cm'],
        ['  AV Node', f'{max_E_av:.1f} V/cm'],
        ['', ''],
        ['Clinical Outcome', ''],
        ['  ', outcome],
    ]

    table = ax6.table(cellText=summary_data, cellLoc='left', loc='center',
                      colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color code the outcome row
    outcome_row = len(summary_data) - 1
    for col in range(2):
        table[(outcome_row, col)].set_facecolor(outcome_color)
        table[(outcome_row, col)].set_text_props(weight='bold', color='white')

    # Header row
    for col in range(2):
        table[(0, col)].set_facecolor('lightgray')
        table[(0, col)].set_text_props(weight='bold')

    ax6.set_title('Simulation Summary', fontweight='bold', fontsize=12, pad=20)

    # Overall title
    fig.suptitle(f'PFA Electric Field Analysis ({mode.capitalize()}): {voltage}V @ {pulse_width}\n' +
                 f'Electrode Position: {distance_from_his} mm inferior to His Bundle',
                 fontsize=14, fontweight='bold')

    # Save figure
    filename = f'Electric_Field_{mode.capitalize()}_{voltage}V_{pulse_width.replace(" ", "")}_d{distance_from_his}mm.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {filename}")

    plt.close()

    return {
        'max_E_his': max_E_his,
        'max_E_sp': max_E_sp,
        'max_E_av': max_E_av,
        'outcome': outcome
    }

if __name__ == "__main__":
    print("="*60)
    print("Running PFA Simulations (3D Anisotropic)")
    print("="*60)

    # 1. Bipolar Mode
    print("\n--- Bipolar Mode ---")
    visualize_electric_field_detailed(
        voltage=2000,
        distance_from_his=15,
        pulse_width="100 us",
        ire_threshold=750,
        re_threshold=375,
        mode='bipolar'
    )
    
    # 2. Monopolar Mode
    print("\n--- Monopolar Mode ---")
    visualize_electric_field_detailed(
        voltage=2000,
        distance_from_his=15,
        pulse_width="100 us",
        ire_threshold=750,
        re_threshold=375,
        mode='monopolar'
    )

    print("\n" + "="*60)
    print("All Simulations Complete!")
    print("="*60)
