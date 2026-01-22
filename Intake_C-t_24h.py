import yaml
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from PDESolver import RiverSolver
import pickle
import os
import pandas as pd

# ===== Load Configuration =====
class Config:
    def __init__(self, path='config.yaml'):
        with open(path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        self.N = 300  # Number of grid points
        self.L = cfg['river_reach']['L_km'] * 1000  # River length (m)
        self.mass = cfg['spill_event']['total_mass_kg']
        self.duration = cfg['spill_event']['release_duration_min']
        self.width = cfg['river_reach']['default_width_m']
        self.depth = cfg['river_reach']['default_depth_m']

# ===== Main Program =====
if __name__ == "__main__":
    # 1. Create configuration object
    config = Config()
    
    # 2. Set fixed parameters
    D_fixed = 50  # m²/s, dispersion coefficient
    
    # 3. Intake location (30 km from source)
    intake_distance_m = 30000  # 30 km
    
    # 4. Different flow velocities to analyze
    u_values = np.arange(0.5, 2.1, 0.3)  # [0.5, 0.8, 1.1, 1.4, 1.7, 2.0] m/s
    
    all_results = []
    
    for u in u_values:
        print(f"\nCalculating u={u:.1f} m/s...")
        
        # Create solver instance
        solver = RiverSolver(config, u=u, D=D_fixed)
        
        # Solve for 24 hours
        sol = solver.solve(T_hours=24)
        
        # Find intake location index
        intake_idx = np.argmin(np.abs(solver.x - intake_distance_m))
        intake_dist_km = solver.x[intake_idx] / 1000
        
        # Get concentration history at intake
        C_intake = sol.y[intake_idx, :]
        
        # Hourly sampling (if sol.t has enough points)
        if hasattr(sol, 'sol'):  # If dense_output=True
            # Create hourly time points for 24 hours
            hourly_times = np.arange(0, 24*3600 + 3600, 3600)
            C_intake_hourly = sol.sol(hourly_times)[intake_idx]
        else:
            # Use existing time points
            hourly_times = sol.t
            C_intake_hourly = C_intake
        
        # Save results
        result = {
            'u': u,
            'intake_distance_km': intake_dist_km,
            'times_h': hourly_times / 3600,
            'times_s': hourly_times,
            'concentration_mg_L': C_intake_hourly,
            'solver': solver,
            'solution': sol
        }
        all_results.append(result)
        
        # Save CSV file (one file per velocity)
        df = pd.DataFrame({
            'Time_h': result['times_h'],
            'Time_s': result['times_s'],
            'Concentration_mg_L': result['concentration_mg_L']
        })
        csv_filename = f'Intake_Concentration_24h_u_{u:.1f}.csv'
        df.to_csv(csv_filename, index=False, encoding='utf-8')
        print(f"  Data saved to: {csv_filename}")
        
        # Print peak information
        peak_idx = np.argmax(C_intake_hourly)
        print(f"  Intake location: {intake_dist_km:.1f} km")
        print(f"  Peak concentration: {C_intake_hourly[peak_idx]:.2e} mg/L")
        print(f"  Peak arrival time: {result['times_h'][peak_idx]:.2f} hours")
    
    # 5. Create comprehensive plots
    plt.figure(figsize=(14, 8))
    
    # Subplot 1: Concentration curves for all velocities
    plt.subplot(2, 2, 1)
    for res in all_results:
        plt.plot(res['times_h'], res['concentration_mg_L'], 
                'o-', linewidth=2, markersize=4, label=f'u={res["u"]:.1f} m/s')
    
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Concentration (mg/L)', fontsize=12)
    plt.title('Intake Concentration vs Time for Different Velocities', fontsize=14)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    
    # Subplot 2: Peak concentration vs velocity
    plt.subplot(2, 2, 2)
    u_list = [res['u'] for res in all_results]
    peak_list = [np.max(res['concentration_mg_L']) for res in all_results]
    
    plt.plot(u_list, peak_list, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Velocity (m/s)', fontsize=12)
    plt.ylabel('Peak Concentration (mg/L)', fontsize=12)
    plt.title('Peak Concentration vs Flow Velocity', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Peak arrival time vs velocity
    plt.subplot(2, 2, 3)
    arrival_times = [res['times_h'][np.argmax(res['concentration_mg_L'])] for res in all_results]
    theoretical_times = [intake_distance_m / (u * 1000) for u in u_list]  # Convert to hours
    
    plt.plot(u_list, arrival_times, 'bo-', linewidth=2, markersize=8, label='Numerical Solution')
    plt.plot(u_list, theoretical_times, 'r--', linewidth=2, label='Theoretical (Pure Advection)')
    plt.xlabel('Velocity (m/s)', fontsize=12)
    plt.ylabel('Peak Arrival Time (hours)', fontsize=12)
    plt.title('Peak Arrival Time vs Flow Velocity', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Spatial distribution at final time
    plt.subplot(2, 2, 4)
    for res in all_results:
        last_idx = -1
        spatial_dist = res['solution'].y[:, last_idx]
        x_km = res['solver'].x / 1000
        plt.plot(x_km, spatial_dist, label=f'u={res["u"]:.1f} m/s', alpha=0.7)
    
    plt.axvline(x=intake_distance_m/1000, color='r', linestyle='--', 
               linewidth=2, label='Intake Location')
    plt.xlabel('Distance (km)', fontsize=12)
    plt.ylabel('Concentration (mg/L)', fontsize=12)
    plt.title('Spatial Distribution after 24 Hours', fontsize=14)
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('Intake_Analysis_Summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Summary table
    print("\n" + "="*80)
    print("SUMMARY RESULTS:")
    print("="*80)
    print(f"{'Velocity(m/s)':<15} {'Peak Conc(mg/L)':<20} {'Arrival Time(h)':<18} {'Theoretical Time(h)':<18}")
    print("-"*80)
    
    for i, res in enumerate(all_results):
        peak_conc = np.max(res['concentration_mg_L'])
        arrival_time = res['times_h'][np.argmax(res['concentration_mg_L'])]
        theoretical_time = intake_distance_m / (res['u'] * 1000)
        
        print(f"{res['u']:<15.1f} {peak_conc:<20.2e} {arrival_time:<18.2f} {theoretical_time:<18.2f}")
    
    # 7. Create comprehensive summary CSV
    summary_data = []
    for res in all_results:
        for t_h, conc in zip(res['times_h'], res['concentration_mg_L']):
            summary_data.append({
                'Velocity_m_s': res['u'],
                'Time_h': t_h,
                'Time_s': t_h * 3600,
                'Concentration_mg_L': conc,
                'Intake_Distance_km': res['intake_distance_km']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('All_Simulations_Summary.csv', index=False)
    print(f"\nSummary data saved to: All_Simulations_Summary.csv")
    
    # 8. Save all results to pickle
    with open('all_simulation_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print(f"All results saved to: all_simulation_results.pkl")
    
    # 9. Additional analysis: Create peak statistics table
    print("\n" + "="*80)
    print("PEAK STATISTICS:")
    print("="*80)
    
    peak_stats = []
    for res in all_results:
        conc_data = res['concentration_mg_L']
        peak_conc = np.max(conc_data)
        mean_conc = np.mean(conc_data)
        std_conc = np.std(conc_data)
        arrival_time = res['times_h'][np.argmax(conc_data)]
        
        peak_stats.append({
            'Velocity': res['u'],
            'Peak_Conc': peak_conc,
            'Mean_Conc': mean_conc,
            'Std_Conc': std_conc,
            'Arrival_Time': arrival_time,
            'Theoretical_Time': intake_distance_m / (res['u'] * 1000),
            'Delay_Ratio': arrival_time / (intake_distance_m / (res['u'] * 1000))
        })
    
    # Create and display peak statistics DataFrame
    peak_df = pd.DataFrame(peak_stats)
    print(peak_df.to_string(index=False))
    
    # Save peak statistics
    peak_df.to_csv('Peak_Statistics.csv', index=False)
    print(f"\nPeak statistics saved to: Peak_Statistics.csv")