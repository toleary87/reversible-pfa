import numpy as np
import matplotlib.pyplot as plt
from model import PFAModel

def run_parameter_sweep():
    # Pulse Width Configurations (Strength-Duration Relationship)
    pulse_configs = [
        {"width": "0.1 us",  "threshold": 3000, "color": "m"},
        {"width": "10 us",   "threshold": 1200, "color": "c"},
        {"width": "100 us",  "threshold": 750,  "color": "g"},
        {"width": "1000 us", "threshold": 500,  "color": "b"}
    ]

    # Sweep from 100V to 2000V in 100V increments
    voltages = np.arange(100, 2100, 100)
    
    # Geometry Definitions
    # His bundle fixed at (0, 10)
    his_pos = (0, 10)
    # Slow Pathway (SP) is 15 mm inferior to His.
    # y = 10 - 15 = -5
    sp_pos = (0, -5)
    
    # Sweep range: 20 mm inferior to 1 mm inferior
    distances = np.arange(20, 0, -1) # 20, 19, ..., 1
    
    electrode_spacing = 4.0 # mm
    
    print("Starting Clinical Objectives Parameter Sweep...")
    print(f"His Bundle Position: {his_pos}")
    print(f"Slow Pathway Position: {sp_pos} (15 mm inferior)")
    print("-" * 40)
    
    # Cache results: results[voltage][distance] = {max_E_his, max_E_sp}
    raw_results = {}
    
    # 1. Run Simulations
    for v in voltages:
        print(f"Simulating Voltage: {v} V")
        raw_results[v] = {}
        
        for d in distances:
            # Calculate dipole center y
            y_center = his_pos[1] - d
            
            # Electrodes at +/- 2 mm from center in x
            e1_pos = (-electrode_spacing/2, y_center)
            e2_pos = (electrode_spacing/2, y_center)
            
            # Initialize model
            model = PFAModel(width=40, height=40, resolution=0.5)
            
            # Define regions
            model.define_regions(av_node_pos=(0, 12), av_node_radius=2.0,
                                 his_bundle_pos=his_pos, his_bundle_width=2.0, his_bundle_length=4.0,
                                 sp_pos=sp_pos, sp_radius=2.0)
            
            # Solve
            model.solve(voltage=v, electrode1_pos=e1_pos, electrode2_pos=e2_pos)
            
            # Calculate field
            E_field = model.calculate_field()
            
            # Analyze results
            centroids = model.mesh.p.mean(axis=1) if model.mesh.p.shape[1] == 3 else model.mesh.p[:, model.mesh.t].mean(axis=1)
            x, y = centroids
            
            # His mask
            his_dist = np.sqrt((x - his_pos[0])**2 + (y - his_pos[1])**2)
            his_mask = his_dist < 1.5
            max_E_his = np.max(E_field[his_mask]) if np.any(his_mask) else 0.0
            
            # SP mask
            sp_dist = np.sqrt((x - sp_pos[0])**2 + (y - sp_pos[1])**2)
            sp_mask = sp_dist < 1.5
            max_E_sp = np.max(E_field[sp_mask]) if np.any(sp_mask) else 0.0
            
            raw_results[v][d] = {"max_E_his": max_E_his, "max_E_sp": max_E_sp}
            
            # Debug print for lowest voltage and relevant distances
            if v == 100 and d in [20, 15, 10, 1]:
                 print(f"DEBUG: V={v}, d={d} | E_sp={max_E_sp:.2f}, E_his={max_E_his:.2f}")

    # 2. Evaluate Clinical Scenarios for each Pulse Width
    print("-" * 40)
    print("Evaluating Clinical Operating Windows...")
    
    for config in pulse_configs:
        width = config['width']
        ire_thresh = config['threshold']
        re_thresh = ire_thresh * 0.5 # RE is 50% of IRE
        
        print(f"Pulse Width: {width} (IRE: {ire_thresh}, RE: {re_thresh} V/cm)")
        
        # Create Outcome Map
        # X-axis: Distance, Y-axis: Voltage
        # Value: 
        # 0 = Ineffective (Gray)
        # 1 = His Stunned (Yellow)
        # 2 = Safe Mapping (Green)
        # 3 = Effective Ablation (Blue)
        # 4 = Unsafe (Red)
        # 1.5 = Warning (Orange) - His Stunned + SP Stunned/Ineffective
        
        outcome_grid = np.zeros((len(voltages), len(distances)))
        
        for i, v in enumerate(voltages):
            for j, d in enumerate(distances):
                res = raw_results[v][d]
                e_his = res['max_E_his']
                e_sp = res['max_E_sp']
                
                # Priority Logic: Safety First? Or Outcome First?
                # Red: His > IRE (Unsafe) - Top Priority
                if e_his >= ire_thresh:
                    outcome_grid[i, j] = 4 # Red
                
                # Blue: SP > IRE (Effective) AND His < IRE
                elif e_sp >= ire_thresh:
                    outcome_grid[i, j] = 3 # Blue
                
                # Green: SP > RE (Mapping) AND His < RE (Safe)
                elif e_sp >= re_thresh and e_his < re_thresh:
                    outcome_grid[i, j] = 2 # Green
                
                # Yellow/Orange: His > RE
                elif e_his >= re_thresh:
                    # If SP > RE too? -> Mapping but His Stunned -> Warning (Orange)
                    if e_sp >= re_thresh:
                        outcome_grid[i, j] = 1.5 # Orange
                    else:
                        outcome_grid[i, j] = 1 # Yellow (His Stunned only)
                
                # Gray: Ineffective
                else:
                    outcome_grid[i, j] = 0 # Gray


        # Plot Operating Window
        plt.figure(figsize=(10, 8))
        # Use pcolormesh
        # X: Distances (need to be sorted for pcolormesh usually, but let's see)
        # Distances are 20, 19... 5. Let's reverse for plotting
        d_grid, v_grid = np.meshgrid(distances, voltages)
        
        # Flip outcome_grid so that columns correspond to increasing distance (1 -> 20)
        # This ensures that when mapped to extent [0.5, 20.5], x=20.5 corresponds to d=20.
        # Combined with invert_xaxis(), this puts d=20 (Far) on the Left.
        outcome_grid_flipped = np.fliplr(outcome_grid)
        
        
        # Define colormap
        from matplotlib.colors import ListedColormap, BoundaryNorm
        # 0=Gray, 1=Yellow, 1.5=Orange, 2=Green, 3=Blue, 4=Red
        # We need to map values to colors.
        # Let's use discrete values: 0, 1, 2, 3, 4, 5 (for 1.5)
        # Remap grid for easier plotting
        plot_grid = np.copy(outcome_grid_flipped)
        plot_grid[plot_grid == 1.5] = 5 # Move Orange to 5
        
        # Colors: 0=Gray, 1=Yellow, 2=Green, 3=Blue, 4=Red, 5=Orange
        # Order in list: 0, 1, 2, 3, 4, 5
        # 0: Gray, 1: Yellow, 2: Green, 3: Blue, 4: Red, 5: Orange
        cmap = ListedColormap(['lightgray', 'gold', 'lightgreen', 'dodgerblue', 'salmon', 'orange'])
        # Bounds must match the values
        # Values: 0, 1, 2, 3, 4, 5
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        norm = BoundaryNorm(bounds, cmap.N)
        
        plt.imshow(plot_grid, aspect='auto', origin='lower', 
                   extent=[distances[-1]-0.5, distances[0]+0.5, voltages[0]-50, voltages[-1]+50],
                   cmap=cmap, norm=norm)
        
        # Custom Legend
        import matplotlib.patches as mpatches
        patches = [
            mpatches.Patch(color='lightgray', label='Ineffective'),
            mpatches.Patch(color='orange', label='His Stunned / SP Ineffective'),
            mpatches.Patch(color='khaki', label='Mapping w/ His Effect'),
            mpatches.Patch(color='lightgreen', label='Safe Mapping (SP Stunned, His Safe)'),
            mpatches.Patch(color='dodgerblue', label='Effective Ablation (SP Ablated, His Safe)'),
            mpatches.Patch(color='salmon', label='Unsafe (His Ablated)')
        ]
        plt.legend(handles=patches, loc='upper right')
        
        plt.xlabel("Distance from His Bundle (mm)")
        plt.ylabel("Applied Voltage (V)")
        plt.title(f"Clinical Operating Window ({width})\nIRE: {ire_thresh} V/cm, RE: {re_thresh} V/cm")
        plt.gca().invert_xaxis() # 20 -> 5
        
        filename = f"Operating_Window_{width.replace(' ', '_')}.png"
        plt.savefig(filename)
        plt.close()
        print(f"  Saved {filename}")

    print("-" * 40)
    print("All Operating Window plots saved.")

    # 3. Summary Plot: Safe Distance vs Voltage (Multi-line)
    # We need to re-calculate safe distances based on the raw_results
    summary_safe_distances = {}
    
    for config in pulse_configs:
        width = config['width']
        threshold = config['threshold']
        safe_dists = []
        
        for v in voltages:
            # Find min safe distance (His < threshold)
            min_safe_dist = 1 # Default (safe at 1 mm)
            
            # Check distances descending (20 -> 5)
            for d in distances:
                if raw_results[v][d]['max_E_his'] > threshold:
                    min_safe_dist = d + 1
                    break
            safe_dists.append(min_safe_dist)
        
        summary_safe_distances[width] = safe_dists
        print(f"  {width}: {safe_dists}")

    plt.figure(figsize=(10, 6))
    
    for config in pulse_configs:
        width = config['width']
        plt.plot(voltages, summary_safe_distances[width], 
                 marker='o', linestyle='-', color=config['color'], 
                 label=f"{width} (Thresh: {config['threshold']} V/cm)")
                 
    plt.xlabel("Applied Voltage (V)")
    plt.ylabel("Minimum Safe Distance (mm)")
    plt.title("Safe Distance from His Bundle vs Voltage\nEffect of Pulse Width (Strength-Duration)")
    plt.legend()
    plt.grid(True)
    plt.xticks(voltages, rotation=45)
    plt.yticks(np.arange(0, 11, 1))
    plt.ylim(0, 10) # Explicitly set Y-axis to 0-10 mm
    plt.tight_layout()
    plt.savefig("Safe_Distance_vs_Voltage_Microseconds.png")
    plt.close()
    
    print("Summary saved to Safe_Distance_vs_Voltage_Microseconds.png")

if __name__ == "__main__":
    run_parameter_sweep()
