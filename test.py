"""
1D Advection-Dispersion Model for River Pollution
Updated version with refined visualizations
"""
import yaml
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os

# ===== Load Configuration =====
class Config:
    def __init__(self, path='config.yaml'):
        with open(path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        self.L = cfg['river_reach']['L_km'] * 1000  # m
        self.mass = cfg['spill_event']['total_mass_kg']
        self.duration = cfg['spill_event']['release_duration_min'] * 60  # seconds
        self.width = cfg['river_reach']['default_width_m']
        self.depth = cfg['river_reach']['default_depth_m']
        self.D = cfg['transport_parameters']['longitudinal_dispersion_D_m2s']['baseline']

cfg = Config()

# ===== Parameters =====
L = cfg.L  # River length (m)
N = 300  # Number of grid points
dx = L / (N - 1)  # Grid spacing (m)
x = np.linspace(0, L, N)  # Spatial coordinates from 0 to L

# Physical parameters
u = 0.5  # Flow velocity (m/s) - typical river velocity
A = cfg.width * cfg.depth  # Cross-sectional area (m²)
D = cfg.D  # Dispersion coefficient (m²/s)

# Source calculation - REDUCED for better visualization
# Reduce source strength by 1000x to get more reasonable concentrations
M_dot = cfg.mass / cfg.duration / 1000  # kg/s (reduced by 1000x)
flux_value = M_dot / A  # kg/(m²·s)

# Stability conditions
CFL = 0.5
dt_max_advection = CFL * dx / abs(u) if u != 0 else 1.0
dt_max_diffusion = 0.25 * dx**2 / (2 * D) if D != 0 else 1.0
dt_max = min(dt_max_advection, dt_max_diffusion)

print("="*60)
print("Model Parameters:")
print(f"  River length: {L/1000:.1f} km")
print(f"  Grid: N={N}, dx={dx:.1f} m")
print(f"  Velocity: u={u} m/s")
print(f"  Dispersion: D={D} m²/s")
print(f"  Cross-section: A={A:.1f} m²")
print(f"  Mass flux: {M_dot:.6f} kg/s (reduced by 1000x for better visualization)")
print(f"  Release duration: {cfg.duration/60:.1f} min")
print("="*60)

# ===== Define PDE System (MOL) with Ghost Cell Method =====
def river_pde(t, C):
    """1D Advection-Dispersion: dC/dt = -u * dC/dx + D * d²C/dx²"""
    dCdt = np.zeros_like(C)
    
    # Create arrays with ghost cells
    C_ext = np.zeros(N + 2)
    C_ext[1:-1] = C
    
    # ===== LEFT BOUNDARY (x=0): Flux boundary =====
    if t <= cfg.duration:
        flux = flux_value
    else:
        flux = 0.0
    
    # Ghost cell method for Robin boundary
    alpha = u * dx / D
    if abs(1 + alpha) > 1e-10:
        C_ext[0] = (2 * flux * dx / D - (alpha - 1) * C_ext[1]) / (1 + alpha)
    else:
        C_ext[0] = C_ext[1]
    
    # ===== RIGHT BOUNDARY (x=L): Zero gradient =====
    C_ext[-1] = C_ext[-2]
    
    # ===== INTERIOR POINTS =====
    for i in range(1, N+1):
        # Advection: Upwind scheme
        if u >= 0:
            adv = -u * (C_ext[i] - C_ext[i-1]) / dx
        else:
            adv = -u * (C_ext[i+1] - C_ext[i]) / dx
        
        # Diffusion: Central difference
        diff = D * (C_ext[i+1] - 2*C_ext[i] + C_ext[i-1]) / (dx**2)
        
        dCdt[i-1] = adv + diff
    
    return dCdt

# ===== Initial Condition =====
C0 = np.zeros(N)

# ===== Simulation for LONGER TIME (96 hours) =====
t_end_3d = 96 * 3600  # 96 hours for 3D plot
t_end_hours = 96 * 3600  # 96 hours for hour-scale plot

print(f"\nStarting simulation (0 to {t_end_hours/3600:.0f} hours)...")
sol = solve_ivp(river_pde, 
                [0, t_end_hours], 
                C0, 
                method='RK45',
                max_step=dt_max,
                rtol=1e-6,
                atol=1e-9)

print(f"Simulation completed: {sol.success}")
print(f"Number of time steps: {len(sol.t)}")
print(f"Concentration range: [{np.min(sol.y):.2e}, {np.max(sol.y):.2e}] kg/m³")

# ===== Interpolation function =====
def get_concentration_at_time(t_seconds):
    """Get concentration at all locations at given time"""
    t_idx = np.searchsorted(sol.t, t_seconds)
    
    if t_idx == 0:
        return sol.y[:, 0] * 1e6  # Convert to μg/L (micrograms per liter)
    elif t_idx >= len(sol.t):
        return sol.y[:, -1] * 1e6
    else:
        # Linear interpolation
        t1 = sol.t[t_idx-1]
        t2 = sol.t[t_idx]
        C1 = sol.y[:, t_idx-1]
        C2 = sol.y[:, t_idx]
        alpha = (t_seconds - t1) / (t2 - t1)
        return (C1 * (1 - alpha) + C2 * alpha) * 1e6  # Convert to μg/L

# ===== Convert units =====
x_km = x / 1000

# Monitoring locations for different plots
monitor_locations_min = list(range(0, 11))  # 0-10 km for minute-scale plot
monitor_locations_hour = [0, 5, 10, 15, 20, 25, 30]  # For hour-scale plot
monitor_indices_min = [np.argmin(np.abs(x_km - loc)) for loc in monitor_locations_min]
monitor_indices_hour = [np.argmin(np.abs(x_km - loc)) for loc in monitor_locations_hour]

# ===== Prepare data for PLOT 1 (minutes scale, 0-10km) =====
print("\nPreparing data for minute-scale plot (0-10 km)...")
t_min_detailed = np.linspace(0, 180, 361)  # 0-180 minutes
t_sec_detailed1 = t_min_detailed * 60
C_monitor_min = np.zeros((len(monitor_locations_min), len(t_min_detailed)))

for i, idx in enumerate(monitor_indices_min):
    for j, t_sec in enumerate(t_sec_detailed1):
        C_all = get_concentration_at_time(t_sec)
        C_monitor_min[i, j] = C_all[idx]

# ===== Prepare data for PLOT 2 (hours scale, 96 hours) =====
print("Preparing data for hour-scale plot (96 hours)...")
t_hour_detailed = np.linspace(0, 96, 577)  # 0-96 hours, 10-minute intervals
t_sec_detailed2 = t_hour_detailed * 3600
C_monitor_hour = np.zeros((len(monitor_locations_hour), len(t_hour_detailed)))

for i, idx in enumerate(monitor_indices_hour):
    for j, t_sec in enumerate(t_sec_detailed2):
        C_all = get_concentration_at_time(t_sec)
        C_monitor_hour[i, j] = C_all[idx]

# ===== Prepare data for PLOT 3 (3D plot) =====
print("Preparing data for 3D plot...")
# Reduce resolution for faster processing
x_3d = np.linspace(0, 30, 60)  # 0-30 km
t_3d = np.linspace(0, 96, 60)  # 0-96 hours
X_3d, T_3d = np.meshgrid(x_3d, t_3d, indexing='ij')
C_3d = np.zeros_like(X_3d)

for i, xi in enumerate(x_3d):
    x_idx = np.argmin(np.abs(x_km - xi))
    for j, tj in enumerate(t_3d):
        t_sec = tj * 3600
        C_all = get_concentration_at_time(t_sec)
        C_3d[i, j] = C_all[x_idx]

print("\n" + "="*60)
print("GENERATING PLOTS")
print("="*60)

# ===== PLOT 1: Minutes scale (0-10 km) =====
plt.figure(figsize=(16, 10))
# Use Set3 colormap for better color distinction
colors = plt.cm.Set3(np.linspace(0, 1, len(monitor_locations_min)))

for i, loc_km in enumerate(monitor_locations_min):
    plt.plot(t_min_detailed, C_monitor_min[i, :],
             linewidth=2.2,
             label=f'{loc_km} km',
             color=colors[i],
             alpha=0.85)

# Mark release end time
plt.axvline(x=cfg.duration/60, color='red', linestyle='--',
            linewidth=2.5, alpha=0.8,
            label=f'Release ends ({cfg.duration/60:.0f} min)')

plt.xlabel('Time (minutes)', fontsize=14, fontweight='bold')
plt.ylabel('Concentration (μg/L)', fontsize=14, fontweight='bold')  # Changed to μg/L
plt.title('Pollutant Concentration During and After Release\nat Locations 0-10 km from Source',
          fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=11, loc='upper right', ncol=2, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--')

# Use linear scale instead of log for better visualization
plt.yscale('linear')
# Set y-limits based on data range
y_max = np.max(C_monitor_min) * 1.1
plt.ylim(0, y_max)
plt.xlim(0, 180)

plt.tight_layout()
plt.savefig('plot1_minutes_scale_0_10km.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Plot 1 saved: plot1_minutes_scale_0_10km.png")

# ===== PLOT 2: Hours scale (96 hours) =====
plt.figure(figsize=(16, 10))
# Use tab20c colormap for hour-scale plot
colors_hour = plt.cm.tab20c(np.linspace(0, 1, len(monitor_locations_hour)))

for i, loc_km in enumerate(monitor_locations_hour):
    plt.plot(t_hour_detailed, C_monitor_hour[i, :],
             linewidth=2.2,
             label=f'{loc_km} km',
             color=colors_hour[i],
             alpha=0.85)

# Mark release end time
plt.axvline(x=cfg.duration/3600, color='red', linestyle='--',
            linewidth=2.5, alpha=0.8,
            label=f'Release ends ({cfg.duration/3600:.2f} h)')

plt.xlabel('Time (hours)', fontsize=14, fontweight='bold')
plt.ylabel('Concentration (μg/L)', fontsize=14, fontweight='bold')  # Changed to μg/L
plt.title('Long-term Evolution of Pollutant Concentration (0-96 hours)\nat Key Locations Along the River',
          fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=11, loc='upper right', ncol=2, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--')

# Use linear scale
plt.yscale('linear')
y_max_hour = np.max(C_monitor_hour) * 1.1
plt.ylim(0, y_max_hour)
plt.xlim(0, 96)

plt.tight_layout()
plt.savefig('plot2_hours_scale_96h.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Plot 2 saved: plot2_hours_scale_96h.png")

# ===== PLOT 3: 3D Surface Plot with improved visualization =====
fig = plt.figure(figsize=(20, 12))
ax = fig.add_subplot(111, projection='3d')

# Use 'YlOrRd' colormap (yellow-orange-red) instead of viridis
# This avoids the purple-heavy visualization
surf = ax.plot_surface(T_3d, X_3d, C_3d,
                      cmap='YlOrRd',  # Changed from viridis to YlOrRd
                      linewidth=0.05,
                      edgecolor='gray',
                      antialiased=True,
                      rstride=1,
                      cstride=1,
                      alpha=0.95,
                      vmin=0,  # Explicitly set minimum
                      vmax=np.max(C_3d) * 0.8)  # Cap at 80% of max for better color range

# Set axes limits and origin
# Set origin to corner by setting all axes to start at 0
ax.set_xlim3d(0, 96)      # Time: 0-96 hours
ax.set_ylim3d(0, 30)      # Distance: 0-30 km
ax.set_zlim3d(0, np.max(C_3d) * 1.1)  # Concentration

# Customize axes labels
ax.set_xlabel('\nTime (hours)', fontsize=13, labelpad=15)
ax.set_ylabel('\nDistance from Source (km)', fontsize=13, labelpad=15)
ax.set_zlabel('\nConcentration (μg/L)', fontsize=13, labelpad=15)  # Changed to μg/L
ax.set_title('3D Spatio-Temporal Evolution of River Pollution (0-96 hours)',
             fontsize=16, fontweight='bold', pad=25)

# Set view angle to better show the corner origin
# elev=25, azim=-120 gives a good corner view
ax.view_init(elev=25, azim=-120)

# Add colorbar with better positioning
cbar = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=25, pad=0.12)
cbar.set_label('Concentration (μg/L)', fontsize=12, rotation=270, labelpad=20)

# Add grid for better depth perception
ax.xaxis._axinfo["grid"].update({"linewidth": 0.5, "color": 'gray', "alpha": 0.3})
ax.yaxis._axinfo["grid"].update({"linewidth": 0.5, "color": 'gray', "alpha": 0.3})
ax.zaxis._axinfo["grid"].update({"linewidth": 0.5, "color": 'gray', "alpha": 0.3})

# Set background color for better contrast
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('lightgray')
ax.yaxis.pane.set_edgecolor('lightgray')
ax.zaxis.pane.set_edgecolor('lightgray')

# Add some key markers
# Mark release end time
release_end_hour = cfg.duration / 3600
ax.plot([release_end_hour, release_end_hour], [0, 30],
        [0, 0],
        color='red', linewidth=3, linestyle='--', alpha=0.9)

# Mark intake location (30 km)
ax.plot(t_3d, np.full_like(t_3d, 30),
        C_3d[-1, :],  # Last row corresponds to 30 km
        color='blue', linewidth=2.5, alpha=0.7)

plt.tight_layout()
plt.savefig('plot3_3d_surface_96h.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Plot 3 saved: plot3_3d_surface_96h.png")

# ===== Analysis =====
print("\n" + "="*60)
print("ANALYSIS RESULTS")
print("="*60)

# Peak analysis for 0-10 km locations
print(f"\nPeak concentrations at 0-10 km locations:")
print(f"{'Location (km)':<12} {'Peak Conc (μg/L)':<18} {'Peak Time (min)':<15}")
print("-" * 50)

for i, loc_km in enumerate(monitor_locations_min):
    peak_conc = np.max(C_monitor_min[i, :])
    peak_time = t_min_detailed[np.argmax(C_monitor_min[i, :])]
    print(f"{loc_km:<12} {peak_conc:<18.1f} {peak_time:<15.1f}")

# Analysis for intake location (30 km)
intake_idx = np.argmin(np.abs(x_km - 30))
intake_data = get_concentration_at_time(t_sec_detailed2[-1])
peak_conc_intake = np.max(intake_data)

print(f"\nIntake location (30 km):")
print(f"  Peak concentration: {peak_conc_intake:.1f} μg/L")
print(f"  Theoretical travel time: {30/(u*3.6):.1f} hours")

# Time for pollution to clear the system
print(f"\nSystem clearance analysis:")
print(f"  Time for pollutant to travel 30 km: {30/(u*3.6):.1f} hours")
print(f"  Simulation duration: {t_end_hours/3600:.0f} hours")
print(f"  Concentration at end of simulation: {C_3d[-1, -1]:.2f} μg/L")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("Three updated plots generated:")
print("1. plot1_minutes_scale_0_10km.png - 0-10 km locations, minutes scale")
print("2. plot2_hours_scale_96h.png - 0-30 km locations, 96 hours scale")  
print("3. plot3_3d_surface_96h.png - 3D visualization, 96 hours, corner origin")
print(f"\nFiles saved in: {os.getcwd()}")