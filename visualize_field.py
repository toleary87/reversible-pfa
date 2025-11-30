import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from model import PFAModel

def visualize_electric_field_detailed(voltage=2000, distance_from_his=15, pulse_width="10 us", ire_threshold=1200, re_threshold=600):
    """
    Create a detailed visualization of the electric field for PFA simulation.

    Args:
        voltage: Applied voltage (V)
        distance_from_his: Distance of electrode center from His bundle (mm inferior)
        pulse_width: Pulse width label
        ire_threshold: IRE threshold (V/cm)
        re_threshold: RE threshold (V/cm)
    """
    # Geometry
    his_pos = (0, 10)
    sp_pos = (0, -5)
    av_node_pos = (0, 12)
    electrode_spacing = 4.0

    # Calculate electrode positions
    y_center = his_pos[1] - distance_from_his
    e1_pos = (-electrode_spacing/2, y_center)
    e2_pos = (electrode_spacing/2, y_center)

    print(f"Simulating Electric Field:")
    print(f"  Voltage: {voltage} V")
    print(f"  Pulse Width: {pulse_width}")
    print(f"  IRE Threshold: {ire_threshold} V/cm")
    print(f"  RE Threshold: {re_threshold} V/cm")
    print(f"  Electrode Center: {distance_from_his} mm inferior to His")
    print(f"  Electrode 1 (Anode): {e1_pos}")
    print(f"  Electrode 2 (Cathode): {e2_pos}")

    # Initialize model with higher resolution for better visualization
    model = PFAModel(width=40, height=40, resolution=0.25)

    # Define regions
    model.define_regions(
        av_node_pos=av_node_pos, av_node_radius=2.0,
        his_bundle_pos=his_pos, his_bundle_width=2.0, his_bundle_length=4.0,
        sp_pos=sp_pos, sp_radius=2.0
    )

    # Solve
    print("  Solving FEM...")
    model.solve(voltage=voltage, electrode1_pos=e1_pos, electrode2_pos=e2_pos)

    # Calculate field
    print("  Calculating electric field...")
    E_field = model.calculate_field()

    # Analyze peak fields at anatomical structures
    centroids = model.mesh.p[:, model.mesh.t].mean(axis=1)
    x, y = centroids

    # His bundle
    his_dist = np.sqrt((x - his_pos[0])**2 + (y - his_pos[1])**2)
    his_mask = his_dist < 1.5
    max_E_his = np.max(E_field[his_mask]) if np.any(his_mask) else 0.0

    # SP
    sp_dist = np.sqrt((x - sp_pos[0])**2 + (y - sp_pos[1])**2)
    sp_mask = sp_dist < 1.5
    max_E_sp = np.max(E_field[sp_mask]) if np.any(sp_mask) else 0.0

    # AV Node
    av_dist = np.sqrt((x - av_node_pos[0])**2 + (y - av_node_pos[1])**2)
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

    # 1. Electric Potential
    ax1 = fig.add_subplot(gs[0, 0])
    mesh_plot = ax1.tripcolor(model.mesh.p[0], model.mesh.p[1], model.mesh.t.T,
                               model.potential, shading='flat', cmap='RdBu_r')
    plt.colorbar(mesh_plot, ax=ax1, label='Potential (V)')

    # Add anatomical markers
    ax1.add_patch(Circle(his_pos, 2.0, fill=False, edgecolor='red', linewidth=2, label='His Bundle'))
    ax1.add_patch(Circle(sp_pos, 2.0, fill=False, edgecolor='blue', linewidth=2, label='Slow Pathway'))
    ax1.add_patch(Circle(av_node_pos, 2.0, fill=False, edgecolor='purple', linewidth=2, label='AV Node'))
    ax1.plot(*e1_pos, 'r*', markersize=15, label=f'Anode (+{voltage}V)')
    ax1.plot(*e2_pos, 'k*', markersize=15, label='Cathode (0V)')

    ax1.set_xlabel('X Position (mm)')
    ax1.set_ylabel('Y Position (mm)')
    ax1.set_title('Electric Potential Distribution')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Electric Field Magnitude
    ax2 = fig.add_subplot(gs[0, 1])
    field_plot = ax2.tripcolor(model.mesh.p[0], model.mesh.p[1], model.mesh.t.T,
                                E_field, shading='flat', cmap='inferno',
                                vmin=0, vmax=min(3000, np.max(E_field)))
    cbar2 = plt.colorbar(field_plot, ax=ax2, label='|E| (V/cm)')

    # Add threshold contours (skip if E_field doesn't reach thresholds)
    # Note: tricontour requires interpolation to nodes for element-wise data
    # For simplicity, we'll skip contours and use manual circles/annotations instead
    # to show threshold regions

    # Add anatomical markers
    ax2.add_patch(Circle(his_pos, 2.0, fill=False, edgecolor='white', linewidth=2))
    ax2.add_patch(Circle(sp_pos, 2.0, fill=False, edgecolor='white', linewidth=2))
    ax2.add_patch(Circle(av_node_pos, 2.0, fill=False, edgecolor='white', linewidth=2, linestyle='--'))
    ax2.plot(*e1_pos, 'r*', markersize=15)
    ax2.plot(*e2_pos, 'w*', markersize=15)

    # Add text labels
    ax2.text(his_pos[0]+2.5, his_pos[1], 'His', color='white', fontweight='bold', fontsize=10)
    ax2.text(sp_pos[0]+2.5, sp_pos[1], 'SP', color='white', fontweight='bold', fontsize=10)

    ax2.set_xlabel('X Position (mm)')
    ax2.set_ylabel('Y Position (mm)')
    ax2.set_title(f'Electric Field Magnitude\n(Cyan: RE={re_threshold}, Green: IRE={ire_threshold} V/cm)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3, color='white', linewidth=0.5)

    # 3. Tissue Conductivity Map
    ax3 = fig.add_subplot(gs[0, 2])
    cond_plot = ax3.tripcolor(model.mesh.p[0], model.mesh.p[1], model.mesh.t.T,
                               model.conductivities, shading='flat', cmap='viridis')
    cbar3 = plt.colorbar(cond_plot, ax=ax3, label='Conductivity (S/m)')

    # Add anatomical markers
    ax3.add_patch(Circle(his_pos, 2.0, fill=False, edgecolor='red', linewidth=2))
    ax3.add_patch(Circle(sp_pos, 2.0, fill=False, edgecolor='blue', linewidth=2))
    ax3.add_patch(Circle(av_node_pos, 2.0, fill=False, edgecolor='purple', linewidth=2))
    ax3.plot(*e1_pos, 'r*', markersize=15)
    ax3.plot(*e2_pos, 'k*', markersize=15)

    ax3.set_xlabel('X Position (mm)')
    ax3.set_ylabel('Y Position (mm)')
    ax3.set_title('Tissue Conductivity Distribution')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # 4. Field along vertical line through center (x=0)
    ax4 = fig.add_subplot(gs[1, 0])

    # Extract field along x=0
    y_line = np.linspace(-20, 20, 200)
    x_line = np.zeros_like(y_line)

    # Interpolate E field to this line
    from scipy.interpolate import LinearNDInterpolator
    interp = LinearNDInterpolator(list(zip(x, y)), E_field)
    E_line = interp(x_line, y_line)

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
    ax4.set_title('E-Field Profile Along Midline (x=0)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-20, 20)

    # 5. Field along horizontal line through electrodes
    ax5 = fig.add_subplot(gs[1, 1])

    x_line_h = np.linspace(-20, 20, 200)
    y_line_h = np.ones_like(x_line_h) * y_center

    E_line_h = interp(x_line_h, y_line_h)

    ax5.plot(x_line_h, E_line_h, 'b-', linewidth=2)
    ax5.axhline(re_threshold, color='cyan', linestyle='--', linewidth=2, label=f'RE ({re_threshold} V/cm)')
    ax5.axhline(ire_threshold, color='lime', linestyle='-', linewidth=2, label=f'IRE ({ire_threshold} V/cm)')

    # Mark electrode positions
    ax5.axvline(e1_pos[0], color='red', linestyle=':', alpha=0.7, label='Anode')
    ax5.axvline(e2_pos[0], color='black', linestyle=':', alpha=0.7, label='Cathode')
    ax5.axvline(0, color='orange', linestyle=':', alpha=0.5, label='Midline')

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
        ['Applied Voltage', f'{voltage} V'],
        ['Pulse Width', pulse_width],
        ['Electrode Spacing', f'{electrode_spacing} mm'],
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
    fig.suptitle(f'PFA Electric Field Analysis: {voltage}V @ {pulse_width}\n' +
                 f'Electrode Position: {distance_from_his} mm inferior to His Bundle',
                 fontsize=14, fontweight='bold')

    # Save figure
    filename = f'Electric_Field_Detail_{voltage}V_{pulse_width.replace(" ", "")}_d{distance_from_his}mm.png'
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
    # Visualize 2000V at 100 microseconds
    print("="*60)
    print("Electric Field Visualization - 2000V @ 100 μs")
    print("="*60)

    # Use 100 μs parameters
    results = visualize_electric_field_detailed(
        voltage=2000,
        distance_from_his=15,  # 15 mm inferior to His (right at SP location)
        pulse_width="100 us",
        ire_threshold=750,
        re_threshold=375
    )

    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)
