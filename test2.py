"""
1D Advection-Dispersion Model for River Pollution
Corrected version with proper spatial progression
"""
import yaml
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import cm
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
N = 300  # Increased resolution for better accuracy
dx = L / (N - 1)  # Grid spacing (m)
x = np.linspace(0, L, N)  # Spatial coordinates from 0 to L

# Physical parameters
u = 0.5  # Flow velocity (m/s) - typical river velocity
A = cfg.width * cfg.depth  # Cross-sectional area (m²)
D = cfg.D  # Dispersion coefficient (m²/s)

# Source calculation
M_dot = cfg.mass / cfg.duration  # kg/s
flux_value = M_dot / A  # kg/(m²·s)

# Stability conditions - critical for avoiding oscillations
CFL = 0.5  # More conservative Courant number
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
print(f"  Mass flux: {M_dot:.4f} kg/s")
print(f"  Release duration: {cfg.duration/60:.1f} min")
print(f"  Max time step: {dt_max:.2f} s (CFL={CFL})")
print("="*60)

# ===== Define PDE System (MOL) with Ghost Cell Method =====
def river_pde(t, C):
    """
    1D Advection-Dispersion: dC/dt = -u * dC/dx + D * d²C/dx²
    Using ghost cell method for stable boundary conditions
    """
    dCdt = np.zeros_like(C)
    
    # Create arrays with ghost cells
    C_ext = np.zeros(N + 2)  # Extended array with ghost cells
    C_ext[1:-1] = C  # Interior points
    
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
        C_ext[0] = C_ext[1]  # Fallback
    
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

# ===== Simulation =====
t_end = 8 * 3600  # 8 hours
print(f"\nStarting simulation (0 to {t_end/3600:.1f} hours)...")

# Run simulation with dense output enabled
sol = solve_ivp(river_pde, 
                [0, t_end], 
                C0, 
                method='RK45',
                max_step=dt_max,
                rtol=1e-6,
                atol=1e-9)

print(f"Simulation completed: {sol.success}")
print(f"Number of time steps: {len(sol.t)}")
print(f"Concentration range: [{np.min(sol.y):.2e}, {np.max(sol.y):.2e}] kg/m³")

# ===== Convert units and prepare data =====
x_km = x / 1000  # Distance in km
monitor_locations = [0, 5, 10, 15, 20, 25, 30]  # km
monitor_indices = [np.argmin(np.abs(x_km - loc)) for loc in monitor_locations]

# Create interpolation function for concentration
def get_concentration_at_time(t_seconds):
    """Get concentration at all locations at given time"""
    # Find the two nearest time points for linear interpolation
    t_idx = np.searchsorted(sol.t, t_seconds)
    
    if t_idx == 0:
        return sol.y[:, 0] * 1000  # Convert to mg/L
    elif t_idx >= len(sol.t):
        return sol.y[:, -1] * 1000
    else:
        # Linear interpolation
        t1 = sol.t[t_idx-1]
        t2 = sol.t[t_idx]
        C1 = sol.y[:, t_idx-1]
        C2 = sol.y[:, t_idx]
        alpha = (t_seconds - t1) / (t2 - t1)
        return (C1 * (1 - alpha) + C2 * alpha) * 1000

# Create detailed time arrays for plotting
# For Plot 1 (minutes scale): high resolution
t_min_detailed = np.linspace(0, 180, 361)  # 0-180 min, 1 point per 0.5 min
t_sec_detailed1 = t_min_detailed * 60

# For Plot 2 (hours scale)
t_hour_detailed = np.linspace(0, 8, 481)  # 0-8 hours, 1 point per minute
t_sec_detailed2 = t_hour_detailed * 3600

# Prepare concentration data for monitoring locations
C_monitor_mgL_min = np.zeros((len(monitor_locations), len(t_min_detailed)))
C_monitor_mgL_hour = np.zeros((len(monitor_locations), len(t_hour_detailed)))

print("\nInterpolating concentration data...")
for i, idx in enumerate(monitor_indices):
    # For minutes scale
    for j, t_sec in enumerate(t_sec_detailed1):
        C_all = get_concentration_at_time(t_sec)
        C_monitor_mgL_min[i, j] = C_all[idx]
    
    # For hours scale  
    for j, t_sec in enumerate(t_sec_detailed2):
        C_all = get_concentration_at_time(t_sec)
        C_monitor_mgL_hour[i, j] = C_all[idx]

print(f"Data prepared for {len(monitor_locations)} monitoring locations")

# ===== Prepare 3D data =====
print("Preparing 3D data...")
x_3d = np.linspace(0, 30, 80)  # km - reduced resolution for speed
t_3d = np.linspace(0, 8, 80)   # hours
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

# ===== PLOT 1: Minutes scale =====
plt.figure(figsize=(15, 9))
colors = cm.plasma(np.linspace(0.2, 0.9, len(monitor_locations)))

for i, loc_km in enumerate(monitor_locations):
    plt.plot(t_min_detailed, C_monitor_mgL_min[i, :],
             linewidth=2.5,
             label=f'{loc_km} km',
             color=colors[i],
             alpha=0.85)

# Mark release end time
plt.axvline(x=cfg.duration/60, color='red', linestyle='--',
            linewidth=2.5, alpha=0.8,
            label=f'Release ends ({cfg.duration/60:.0f} min)')

# Add thresholds
plt.axhline(y=0.1, color='orange', linestyle=':',
            linewidth=2, alpha=0.7, label='Warning (0.1 mg/L)')
plt.axhline(y=0.3, color='red', linestyle=':',
            linewidth=2, alpha=0.7, label='Shutdown (0.3 mg/L)')

plt.xlabel('Time (minutes)', fontsize=14, fontweight='bold')
plt.ylabel('Concentration (mg/L)', fontsize=14, fontweight='bold')
plt.title('Pollutant Concentration During and Immediately After Release\nat Different Locations Along the River',
          fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12, loc='upper right', ncol=2, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--')
plt.yscale('log')
plt.ylim(1e-4, 50)
plt.xlim(0, 180)
plt.tight_layout()
plt.savefig('plot1_minutes_scale.png', dpi=200, bbox_inches='tight')
plt.close()  # Close figure to free memory
print("✓ Plot 1 saved: plot1_minutes_scale.png")

# ===== PLOT 2: Hours scale =====
plt.figure(figsize=(15, 9))

for i, loc_km in enumerate(monitor_locations):
    plt.plot(t_hour_detailed, C_monitor_mgL_hour[i, :],
             linewidth=2.5,
             label=f'{loc_km} km',
             color=colors[i],
             alpha=0.85)

# Mark release end time
plt.axvline(x=cfg.duration/3600, color='red', linestyle='--',
            linewidth=2.5, alpha=0.8,
            label=f'Release ends ({cfg.duration/3600:.2f} h)')

# Add thresholds
plt.axhline(y=0.1, color='orange', linestyle=':',
            linewidth=2, alpha=0.7, label='Warning (0.1 mg/L)')
plt.axhline(y=0.3, color='red', linestyle=':',
            linewidth=2, alpha=0.7, label='Shutdown (0.3 mg/L)')

plt.xlabel('Time (hours)', fontsize=14, fontweight='bold')
plt.ylabel('Concentration (mg/L)', fontsize=14, fontweight='bold')
plt.title('Long-term Evolution of Pollutant Concentration\nat Different Locations Along the River',
          fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12, loc='upper right', ncol=2, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--')
plt.yscale('log')
plt.ylim(1e-4, 50)
plt.xlim(0, 8)
plt.tight_layout()
plt.savefig('plot2_hours_scale.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Plot 2 saved: plot2_hours_scale.png")

# ===== PLOT 3: 3D Surface Plot =====
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(18, 10))
ax = fig.add_subplot(111, projection='3d')

# Create surface plot
surf = ax.plot_surface(T_3d, X_3d, C_3d,
                      cmap='viridis',
                      linewidth=0.1,
                      edgecolors='k',
                      antialiased=True,
                      rstride=2,
                      cstride=2,
                      alpha=0.95)

# Mark monitoring locations
for i, loc_km in enumerate(monitor_locations):
    loc_idx = np.argmin(np.abs(x_3d - loc_km))
    ax.plot(t_3d, np.full_like(t_3d, loc_km),
            C_3d[loc_idx, :],
            color=colors[i], linewidth=2.5, alpha=0.8,
            label=f'{loc_km} km')

# Mark release end time
release_end_hour = cfg.duration / 3600
ax.plot([release_end_hour, release_end_hour], [0, 30],
        [0, 0],
        color='red', linewidth=3, linestyle='--', alpha=0.9,
        label=f'Release end ({release_end_hour:.2f} h)')

# Customize axes
ax.set_xlabel('\nTime (hours)', fontsize=13, labelpad=20, linespacing=2)
ax.set_ylabel('\nDistance (km)', fontsize=13, labelpad=20, linespacing=2)
ax.set_zlabel('\nConcentration (mg/L)', fontsize=13, labelpad=20, linespacing=2)
ax.set_title('3D Spatio-Temporal Evolution of River Pollution',
             fontsize=16, fontweight='bold', pad=30)

# Set view angle
ax.view_init(elev=25, azim=-135)

# Add colorbar
cbar = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=25, pad=0.1)
cbar.set_label('Concentration (mg/L)', fontsize=12, rotation=270, labelpad=20)

# Add legend
ax.legend(fontsize=11, loc='upper left', bbox_to_anchor=(0.02, 0.98),
          framealpha=0.9)

plt.tight_layout()
plt.savefig('plot3_3d_surface.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Plot 3 saved: plot3_3d_surface.png")

# ===== Detailed Analysis =====
print("\n" + "="*60)
print("DETAILED ANALYSIS")
print("="*60)

# Calculate peak statistics
print(f"\n{'Location (km)':<12} {'Peak Conc (mg/L)':<18} {'Peak Time (min)':<15} {'Travel Delay (min)':<18}")
print("-" * 70)

peak_times = []
peak_concs = []
for i, loc_km in enumerate(monitor_locations):
    conc_data = C_monitor_mgL_min[i, :]
    peak_conc = np.max(conc_data)
    peak_time_idx = np.argmax(conc_data)
    peak_time_min = t_min_detailed[peak_time_idx]
    
    peak_concs.append(peak_conc)
    peak_times.append(peak_time_min)
    
    if i == 0:
        travel_delay = 0
    else:
        travel_delay = peak_time_min - peak_times[0]
    
    print(f"{loc_km:<12} {peak_conc:<18.3f} {peak_time_min:<15.1f} {travel_delay:<18.1f}")

# Intake (30 km) specific analysis
intake_data = C_monitor_mgL_min[-1, :]
peak_conc_intake = peak_concs[-1]
peak_time_intake = peak_times[-1]

print(f"\nIntake (30 km) detailed analysis:")
print(f"  Peak concentration: {peak_conc_intake:.3f} mg/L")
print(f"  Peak arrival time: {peak_time_intake:.1f} min ({peak_time_intake/60:.2f} h)")
print(f"  Theoretical travel time: {30/(u*3.6*60):.1f} min")

# Threshold analysis
thresholds = [0.001, 0.01, 0.1, 0.3]
threshold_names = ['Detection', 'Low', 'Warning', 'Shutdown']

print(f"\nThreshold exceedance times at intake (30 km):")
for thresh, name in zip(thresholds, threshold_names):
    exceed_idx = np.where(intake_data > thresh)[0]
    if len(exceed_idx) > 0:
        first_exceed = t_min_detailed[exceed_idx[0]]
        last_exceed = t_min_detailed[exceed_idx[-1]]
        duration = last_exceed - first_exceed
        print(f"  {name} ({thresh} mg/L): {first_exceed:.1f}-{last_exceed:.1f} min (duration: {duration:.1f} min)")
    else:
        print(f"  {name} ({thresh} mg/L): NOT exceeded")

# Travel wave characteristics
print(f"\nPollution wave characteristics:")
print(f"  Wave speed: {u:.2f} m/s = {u*3.6:.1f} km/h")
print(f"  Time to travel 30 km: {30/(u*3.6):.2f} hours")
print(f"  Dispersion causes peak reduction: {peak_concs[0]/peak_concs[-1]:.1f}x from 0 to 30 km")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("Three high-quality plots saved as PNG files:")
print("1. plot1_minutes_scale.png - Shows pollution wave progression (minutes)")
print("2. plot2_hours_scale.png - Long-term evolution (hours)")
print("3. plot3_3d_surface.png - 3D spatio-temporal visualization")

print("\nFiles created in:", os.getcwd())