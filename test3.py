"""
1D Advection-Dispersion Model for River Pollution
Final version with all requested modifications
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
u = 0.5  # Flow velocity (m/s)
A = cfg.width * cfg.depth  # Cross-sectional area (m²)
D = cfg.D  # Dispersion coefficient (m²/s)

# Source calculation - INCREASE for hour-scale plot (but keep for consistency)
M_dot = cfg.mass / cfg.duration  # kg/s (original value)
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
print(f"  Mass flux: {M_dot:.4f} kg/s")
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

# ===== Simulation for different time scales =====
print(f"\nStarting simulation for hour-scale plot (0-96 hours)...")
t_end_hours = 96 * 3600  # 96 hours

sol_hours = solve_ivp(river_pde, 
                     [0, t_end_hours], 
                     C0, 
                     method='RK45',
                     max_step=dt_max,
                     rtol=1e-6,
                     atol=1e-9)

print(f"Simulation completed: {sol_hours.success}")
print(f"Number of time steps: {len(sol_hours.t)}")
print(f"Concentration range: [{np.min(sol_hours.y):.2e}, {np.max(sol_hours.y):.2e}] kg/m³")

# ===== Interpolation function =====
def get_concentration_at_time(t_seconds, solution):
    """Get concentration at all locations at given time"""
    t_idx = np.searchsorted(solution.t, t_seconds)
    
    if t_idx == 0:
        return solution.y[:, 0]
    elif t_idx >= len(solution.t):
        return solution.y[:, -1]
    else:
        # Linear interpolation
        t1 = solution.t[t_idx-1]
        t2 = solution.t[t_idx]
        C1 = solution.y[:, t_idx-1]
        C2 = solution.y[:, t_idx]
        alpha = (t_seconds - t1) / (t2 - t1)
        return C1 * (1 - alpha) + C2 * alpha

# ===== Convert units =====
x_km = x / 1000

# Monitoring locations
monitor_locations_min = list(range(0, 11))  # 0-10 km for minute-scale plot (KEEP SAME)
monitor_locations_hour = [0, 5, 10, 15, 20, 25, 30]  # For hour-scale plot
monitor_indices_min = [np.argmin(np.abs(x_km - loc)) for loc in monitor_locations_min]
monitor_indices_hour = [np.argmin(np.abs(x_km - loc)) for loc in monitor_locations_hour]

# ===== PLOT 1: Minutes scale (0-10 km) - KEEP EXACTLY AS BEFORE =====
# First simulate for minutes scale (0-3 hours is enough)
print(f"\nStarting simulation for minute-scale plot (0-3 hours)...")
t_end_min = 3 * 3600  # 3 hours

sol_min = solve_ivp(river_pde, 
                   [0, t_end_min], 
                   C0, 
                   method='RK45',
                   max_step=dt_max,
                   rtol=1e-6,
                   atol=1e-9)

# Prepare minute-scale data
t_min_detailed = np.linspace(0, 180, 361)  # 0-180 minutes
t_sec_detailed1 = t_min_detailed * 60
C_monitor_min = np.zeros((len(monitor_locations_min), len(t_min_detailed)))

for i, idx in enumerate(monitor_indices_min):
    for j, t_sec in enumerate(t_sec_detailed1):
        C_all = get_concentration_at_time(t_sec, sol_min)
        C_monitor_min[i, j] = C_all[idx]  # Keep in kg/m³

# Create minute-scale plot (EXACT SAME AS BEFORE)
plt.figure(figsize=(16, 10))
colors = plt.cm.Set3(np.linspace(0, 1, len(monitor_locations_min)))

for i, loc_km in enumerate(monitor_locations_min):
    plt.plot(t_min_detailed, C_monitor_min[i, :] * 1000,  # Convert to g/m³ (×1000)
             linewidth=2.2,
             label=f'{loc_km} km',
             color=colors[i],
             alpha=0.85)

plt.axvline(x=cfg.duration/60, color='red', linestyle='--',
            linewidth=2.5, alpha=0.8,
            label=f'Release ends ({cfg.duration/60:.0f} min)')

plt.xlabel('Time (minutes)', fontsize=14, fontweight='bold')
plt.ylabel('Concentration (g/m³)', fontsize=14, fontweight='bold')
plt.title('Pollutant Concentration During and After Release\nat Locations 0-10 km from Source',
          fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=11, loc='upper right', ncol=2, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--')
plt.yscale('linear')
y_max = np.max(C_monitor_min) * 1000 * 1.1
plt.ylim(0, y_max)
plt.xlim(0, 180)
plt.tight_layout()
plt.savefig('plot1_minutes_scale_0_10km.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Plot 1 saved: plot1_minutes_scale_0_10km.png")

# ===== PLOT 2: Hours scale (96 hours) - ADJUSTED FOR HIGHER PEAKS =====
print("\nPreparing hour-scale plot...")
# Use solution for hours (already simulated)
t_hour_detailed = np.linspace(0, 96, 577)  # 0-96 hours
t_sec_detailed2 = t_hour_detailed * 3600
C_monitor_hour = np.zeros((len(monitor_locations_hour), len(t_hour_detailed)))

for i, idx in enumerate(monitor_indices_hour):
    for j, t_sec in enumerate(t_sec_detailed2):
        C_all = get_concentration_at_time(t_sec, sol_hours)
        C_monitor_hour[i, j] = C_all[idx]

# ADJUSTMENT: Multiply by 10000 to get higher peaks for hour-scale plot
C_monitor_hour_adjusted = C_monitor_hour * 10000  # Scale up for better visualization

plt.figure(figsize=(16, 10))
colors_hour = plt.cm.tab20c(np.linspace(0, 1, len(monitor_locations_hour)))

for i, loc_km in enumerate(monitor_locations_hour):
    plt.plot(t_hour_detailed, C_monitor_hour_adjusted[i, :],  # Use adjusted values
             linewidth=2.2,
             label=f'{loc_km} km',
             color=colors_hour[i],
             alpha=0.85)

plt.axvline(x=cfg.duration/3600, color='red', linestyle='--',
            linewidth=2.5, alpha=0.8,
            label=f'Release ends ({cfg.duration/3600:.2f} h)')

plt.xlabel('Time (hours)', fontsize=14, fontweight='bold')
plt.ylabel('Concentration (10⁻⁴ × kg/m³)', fontsize=14, fontweight='bold')  # Note the scaling
plt.title('Long-term Evolution of Pollutant Concentration (0-96 hours)\nScaled for Better Visualization',
          fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=11, loc='upper right', ncol=2, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--')
plt.yscale('linear')
# Allow y-axis to go negative slightly for better visualization
y_min = np.min(C_monitor_hour_adjusted) * 1.1 if np.min(C_monitor_hour_adjusted) < 0 else -0.1
y_max = np.max(C_monitor_hour_adjusted) * 1.1
plt.ylim(y_min, y_max)
plt.xlim(0, 96)

# Add note about scaling
plt.text(80, y_max * 0.9, 'Values scaled ×10⁴\nfor visualization',
         fontsize=10, ha='right', va='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('plot2_hours_scale_96h.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Plot 2 saved: plot2_hours_scale_96h.png")

# ===== PLOT 3: 3D Surface Plot - REVERT TO PREVIOUS VERSION WITH MODIFICATIONS =====
print("\nPreparing 3D plot...")
# Use a shorter time range for 3D plot (0-12 hours is enough to see the wave)
t_end_3d = 12 * 3600  # 12 hours for 3D plot

# Simulate for 3D plot
sol_3d = solve_ivp(river_pde, 
                   [0, t_end_3d], 
                   C0, 
                   method='RK45',
                   max_step=dt_max,
                   rtol=1e-6,
                   atol=1e-9)

# Prepare 3D data with reduced resolution
x_3d = np.linspace(0, 30, 80)  # 0-30 km
t_3d = np.linspace(0, 12, 80)  # 0-12 hours
X_3d, T_3d = np.meshgrid(x_3d, t_3d, indexing='ij')
C_3d = np.zeros_like(X_3d)

for i, xi in enumerate(x_3d):
    x_idx = np.argmin(np.abs(x_km - xi))
    for j, tj in enumerate(t_3d):
        t_sec = tj * 3600
        C_all = get_concentration_at_time(t_sec, sol_3d)
        C_3d[i, j] = C_all[x_idx]

# Scale down for better visualization (make peaks less extreme)
C_3d_scaled = C_3d * 100  # Scale by 100 for better 3D visualization

# Create 3D plot
fig = plt.figure(figsize=(20, 12))
ax = fig.add_subplot(111, projection='3d')

# Use 'plasma' colormap for more vibrant colors
surf = ax.plot_surface(T_3d, X_3d, C_3d_scaled,
                      cmap='plasma',  # More vibrant than viridis
                      linewidth=0.1,
                      edgecolor='gray',
                      antialiased=True,
                      rstride=1,
                      cstride=1,
                      alpha=0.92)

# REVERSE AXES FOR CORNER ORIGIN
# Instead of changing axis limits, we'll plot the data differently
# We'll flip the X and Y axes and adjust the view

# Option 1: Flip the data itself
# Or we can simply adjust the view angle to see from the corner

# Set axis limits with corner at origin
ax.set_xlim(0, 12)
ax.set_ylim(0, 30)
ax.set_zlim(0, np.max(C_3d_scaled) * 1.1)

# Customize axes
ax.set_xlabel('\nTime (hours)', fontsize=13, labelpad=15)
ax.set_ylabel('\nDistance from Source (km)', fontsize=13, labelpad=15)
ax.set_zlabel('\nConcentration (10⁻² × kg/m³)', fontsize=13, labelpad=15)  # Note scaling

# Set title
ax.set_title('3D Spatio-Temporal Evolution of River Pollution (0-12 hours)\nScaled for 3D Visualization',
             fontsize=16, fontweight='bold', pad=25)

# Set view angle to show corner origin - THIS IS KEY
# elev=30, azim=-135 gives a good corner view
ax.view_init(elev=30, azim=-135)

# Add colorbar
cbar = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=25, pad=0.12)
cbar.set_label('10⁻² × kg/m³', fontsize=12, rotation=270, labelpad=20)

# Improve grid and background
ax.xaxis._axinfo["grid"].update({"linewidth": 0.5, "color": 'gray', "alpha": 0.3})
ax.yaxis._axinfo["grid"].update({"linewidth": 0.5, "color": 'gray', "alpha": 0.3})
ax.zaxis._axinfo["grid"].update({"linewidth": 0.5, "color": 'gray', "alpha": 0.3})

# Set pane properties
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('lightgray')
ax.yaxis.pane.set_edgecolor('lightgray')
ax.zaxis.pane.set_edgecolor('lightgray')
ax.xaxis.pane.set_alpha(0.1)
ax.yaxis.pane.set_alpha(0.1)
ax.zaxis.pane.set_alpha(0.1)

# Add some contour lines on the surface for better depth perception
# We'll add contours at specific levels
levels = np.linspace(0, np.max(C_3d_scaled), 10)
ax.contour(T_3d, X_3d, C_3d_scaled, 
           levels=levels, 
           zdir='z', 
           offset=0,
           colors='black', 
           linewidths=0.5, 
           alpha=0.3)

# Mark release end time
release_end_hour = cfg.duration / 3600
ax.plot([release_end_hour, release_end_hour], [0, 30],
        [0, 0],
        color='red', linewidth=2.5, linestyle='--', alpha=0.8,
        label=f'Release end')

# Mark intake location
ax.plot(t_3d, np.full_like(t_3d, 30),
        C_3d_scaled[-1, :],
        color='blue', linewidth=2.0, alpha=0.7,
        label='Intake (30 km)')

# Add legend
ax.legend(fontsize=11, loc='upper left')

plt.tight_layout()
plt.savefig('plot3_3d_surface_corner.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Plot 3 saved: plot3_3d_surface_corner.png")

# ===== Analysis =====
print("\n" + "="*60)
print("ANALYSIS RESULTS")
print("="*60)

# Analysis for hour-scale plot (real values)
print(f"\nActual peak concentrations (kg/m³) at key locations:")
print(f"{'Location (km)':<12} {'Actual Peak':<15} {'Scaled (×10⁴)':<15}")
print("-" * 45)

for i, loc_km in enumerate(monitor_locations_hour):
    actual_peak = np.max(C_monitor_hour[i, :])
    scaled_peak = np.max(C_monitor_hour_adjusted[i, :])
    print(f"{loc_km:<12} {actual_peak:<15.2e} {scaled_peak:<15.2f}")

# Travel time analysis
print(f"\nTravel time analysis:")
travel_time_30km = 30 / (u * 3.6)  # hours
print(f"  Time for pollutant to reach 30 km: {travel_time_30km:.1f} hours")
print(f"  Wave speed: {u:.2f} m/s = {u*3.6:.1f} km/h")

# Mass balance
total_mass_injected = cfg.mass  # kg
# Calculate approximate mass in system at peak
peak_time_idx = np.argmax(C_monitor_hour[0, :])
peak_time_hour = t_hour_detailed[peak_time_idx]
peak_time_sec = peak_time_hour * 3600
C_peak = get_concentration_at_time(peak_time_sec, sol_hours)
mass_in_system = np.sum(C_peak) * A * dx

print(f"\nMass balance at peak time ({peak_time_hour:.1f} hours):")
print(f"  Mass injected: {total_mass_injected:.1f} kg")
print(f"  Approx. mass in system: {mass_in_system:.1f} kg")
print(f"  Relative error: {abs(total_mass_injected - mass_in_system)/total_mass_injected*100:.1f}%")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("Three plots generated with all requested modifications:")
print("1. plot1_minutes_scale_0_10km.png - Minute scale, 0-10 km (unchanged)")
print("2. plot2_hours_scale_96h.png - Hour scale, 96 hours (peaks scaled ×10⁴)")
print("3. plot3_3d_surface_corner.png - 3D plot, corner origin, plasma colors")
print(f"\nFiles saved in: {os.getcwd()}")