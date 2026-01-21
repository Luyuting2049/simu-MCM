"""
Corrected 1D Advection-Dispersion Model with stable numerical scheme
"""

# 奇怪但后面图像可用
import yaml
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import cm

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
N = 200  # Number of grid points
dx = L / N  # Grid spacing (m)
x = np.linspace(dx/2, L-dx/2, N)  # Cell-centered coordinates

# Physical parameters
u = 0.5  # Flow velocity (m/s)
A = cfg.width * cfg.depth  # Cross-sectional area (m²)
D = cfg.D  # Dispersion coefficient (m²/s)

# Source calculation
M_dot = cfg.mass / cfg.duration  # kg/s
flux_value = M_dot / A  # kg/(m²·s)

# Stability conditions
CFL = 0.8  # Courant number for stability
dt_max_advection = CFL * dx / abs(u) if u != 0 else 1.0
dt_max_diffusion = 0.5 * dx**2 / (2 * D) if D != 0 else 1.0
dt_max = min(dt_max_advection, dt_max_diffusion)

print("="*60)
print("Model Parameters:")
print(f"  River length: {L/1000:.1f} km")
print(f"  Grid: N={N}, dx={dx:.1f} m")
print(f"  Velocity: u={u} m/s")
print(f"  Dispersion: D={D} m²/s")
print(f"  Cross-section: A={A:.1f} m²")
print(f"  Mass flux: {M_dot:.4f} kg/s")
print(f"  Boundary flux: {flux_value:.6f} kg/(m²·s)")
print(f"  Stability limits: dt_adv={dt_max_advection:.2f}s, dt_diff={dt_max_diffusion:.2f}s")
print("="*60)

# ===== Define PDE System (MOL) with Stable Scheme =====
def river_pde(t, C):
    """
    1D Advection-Dispersion: dC/dt = -u * dC/dx + D * d²C/dx²
    Using upwind for advection and central difference for diffusion
    """
    dCdt = np.zeros_like(C)
    
    # Create ghost cells for boundaries
    C_with_ghost = np.zeros(N + 2)
    C_with_ghost[1:-1] = C
    
    # ===== LEFT BOUNDARY (Inflow) =====
    if t <= cfg.duration:
        flux = flux_value
    else:
        flux = 0.0
    
    # Solve for ghost cell using flux boundary condition
    denominator = (D/dx) + (u/2)
    if denominator != 0:
        numerator = flux - (D/dx)*C[0] - (u/2)*C[0]
        C_ghost_left = numerator / denominator
    else:
        C_ghost_left = 0
    
    C_with_ghost[0] = C_ghost_left
    
    # ===== RIGHT BOUNDARY (Outflow) =====
    # Zero gradient: dC/dx = 0 at x=L
    C_ghost_right = C[-1]
    C_with_ghost[-1] = C_ghost_right
    
    # ===== INTERIOR POINTS =====
    for i in range(N):
        idx = i + 1
        
        # Advection: Upwind scheme
        if u >= 0:
            adv = -u * (C_with_ghost[idx] - C_with_ghost[idx-1]) / dx
        else:
            adv = -u * (C_with_ghost[idx+1] - C_with_ghost[idx]) / dx
        
        # Diffusion: Central difference
        diff = D * (C_with_ghost[idx+1] - 2*C_with_ghost[idx] + C_with_ghost[idx-1]) / (dx**2)
        
        dCdt[i] = adv + diff
    
    return dCdt

# ===== Initial Condition =====
C0 = np.zeros(N)

# ===== Time Settings =====
t_end = 18000  # 300 minutes total

print("\nStarting simulation...")
sol = solve_ivp(river_pde, 
                [0, t_end], 
                C0, 
                method='RK45',
                max_step=dt_max,
                rtol=1e-5,
                atol=1e-8)

print(f"Simulation completed: {sol.success}")
print(f"Number of time steps: {len(sol.t)}")
print(f"Concentration range: [{np.min(sol.y):.2e}, {np.max(sol.y):.2e}] kg/m³")

# ===== Convert units =====
t_min = sol.t / 60  # Convert seconds to minutes
x_km = x / 1000  # Convert meters to kilometers
C_kg_m3 = sol.y  # Concentration in kg/m³
C_mg_L = C_kg_m3 * 1000  # Convert to mg/L

# ===== Plot Results =====
print("\nGenerating plots...")

# Figure 1: Concentration time series at different locations
plt.figure(figsize=(14, 8))
positions_km = [0, 5, 10, 15, 20, 25, 30]
colors = cm.viridis(np.linspace(0, 1, len(positions_km)))

for i, pos_km in enumerate(positions_km):
    idx = np.argmin(np.abs(x_km - pos_km))
    plt.plot(t_min, C_mg_L[idx, :], 
             label=f'x = {pos_km} km', 
             linewidth=2.5,
             color=colors[i],
             alpha=0.9)

plt.axvline(x=cfg.duration/60, color='red', linestyle='--', 
            linewidth=2, alpha=0.7, label=f'Release end ({cfg.duration/60:.0f} min)')

# Add warning thresholds
try:
    with open('config.yaml', 'r') as f:
        config_data = yaml.safe_load(f)
    if 'water_quality' in config_data:
        plt.axhline(y=0.1, color='orange', linestyle=':', 
                    linewidth=1.5, alpha=0.7, label='Warning threshold (0.1 mg/L)')
        plt.axhline(y=0.3, color='red', linestyle=':', 
                    linewidth=1.5, alpha=0.7, label='Shutdown threshold (0.3 mg/L)')
except:
    pass

plt.xlabel('Time (minutes)', fontsize=12, fontweight='bold')
plt.ylabel('Concentration (mg/L)', fontsize=12, fontweight='bold')
plt.title('Pollutant Concentration at Different Locations', fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3, linestyle='--')
plt.yscale('log')
plt.ylim(1e-3, 1e2)
plt.tight_layout()
plt.savefig('concentration_time_series.png', dpi=150, bbox_inches='tight')

# Figure 2: Spatial profiles at different times
plt.figure(figsize=(14, 8))
time_points = [0, 30, 60, 120, 180, 240, 300]  # minutes
colors = cm.plasma(np.linspace(0, 1, len(time_points)))

for i, t_val in enumerate(time_points):
    # Find the closest time index
    idx_t = np.argmin(np.abs(t_min - t_val))
    C_at_time = C_mg_L[:, idx_t]
    
    plt.plot(x_km, C_at_time, 
             label=f't = {t_val} min',
             linewidth=2.5,
             color=colors[i],
             alpha=0.9)

plt.xlabel('Distance along river (km)', fontsize=12, fontweight='bold')
plt.ylabel('Concentration (mg/L)', fontsize=12, fontweight='bold')
plt.title('Spatial Distribution of Pollutant Concentration', fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3, linestyle='--')
plt.yscale('log')
plt.ylim(1e-3, 1e2)
plt.tight_layout()
plt.savefig('concentration_spatial_profile.png', dpi=150, bbox_inches='tight')

# Figure 3: Heatmap (using interpolation)
plt.figure(figsize=(15, 6))

# Create regular grid for heatmap
t_heatmap = np.linspace(0, 300, 200)  # 0-300 minutes
x_heatmap = x_km

# Interpolate concentration data onto regular grid
C_heatmap = np.zeros((len(x_heatmap), len(t_heatmap)))

for i, t_val in enumerate(t_heatmap):
    # Find the two nearest time points
    idx_t = np.argmin(np.abs(t_min - t_val))
    
    # Simple nearest neighbor interpolation
    C_heatmap[:, i] = C_mg_L[:, idx_t]

# Create meshgrid for plotting
T_mesh, X_mesh = np.meshgrid(t_heatmap, x_heatmap)

# Create heatmap using pcolormesh
plt.pcolormesh(T_mesh, X_mesh, C_heatmap, 
               shading='auto', 
               cmap='YlOrRd',
               norm=plt.matplotlib.colors.LogNorm(vmin=1e-3, vmax=10))

plt.colorbar(label='Concentration (mg/L)', extend='both')
plt.axvline(x=cfg.duration/60, color='white', linestyle='--', linewidth=2, alpha=0.8)
plt.xlabel('Time (minutes)', fontsize=12, fontweight='bold')
plt.ylabel('Distance along river (km)', fontsize=12, fontweight='bold')
plt.title('Pollutant Transport: Spatio-Temporal Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('concentration_heatmap.png', dpi=150, bbox_inches='tight')

# Figure 4: Contour plot (alternative visualization)
plt.figure(figsize=(15, 6))

# For contour plot, we need to smooth the data a bit
from scipy import ndimage

# Apply mild Gaussian filter to smooth contour lines
C_smooth = ndimage.gaussian_filter(C_heatmap, sigma=1.0)

# Create contour plot
contour = plt.contourf(T_mesh, X_mesh, C_smooth, 
                       levels=50, 
                       cmap='YlOrRd',
                       norm=plt.matplotlib.colors.LogNorm(vmin=1e-3, vmax=10))

plt.colorbar(contour, label='Concentration (mg/L)', extend='both')
plt.axvline(x=cfg.duration/60, color='black', linestyle='--', linewidth=2, alpha=0.8)
plt.xlabel('Time (minutes)', fontsize=12, fontweight='bold')
plt.ylabel('Distance along river (km)', fontsize=12, fontweight='bold')
plt.title('Pollutant Transport: Contour Plot', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('concentration_contour.png', dpi=150, bbox_inches='tight')

# ===== Key Statistics =====
print("\n" + "="*60)
print("Key Statistics:")
print("="*60)

# Find intake location (30 km)
intake_idx = np.argmin(np.abs(x_km - 30))
C_intake = C_mg_L[intake_idx, :]

# Peak concentration at intake
peak_conc = np.max(C_intake)
peak_time = t_min[np.argmax(C_intake)]

print(f"At intake location (30 km):")
print(f"  Peak concentration: {peak_conc:.3f} mg/L")
print(f"  Peak arrival time: {peak_time:.1f} minutes")

# Check thresholds
warning_threshold = 0.1  # mg/L
shutdown_threshold = 0.3  # mg/L

warning_idx = np.where(C_intake > warning_threshold)[0]
shutdown_idx = np.where(C_intake > shutdown_threshold)[0]

if len(warning_idx) > 0:
    warning_time = t_min[warning_idx[0]]
    print(f"  Warning threshold exceeded at: {warning_time:.1f} minutes")
else:
    print(f"  Warning threshold NOT exceeded")

if len(shutdown_idx) > 0:
    shutdown_time = t_min[shutdown_idx[0]]
    print(f"  Shutdown threshold exceeded at: {shutdown_time:.1f} minutes")
else:
    print(f"  Shutdown threshold NOT exceeded")

# Mass conservation check
total_mass_sim = np.sum(C_kg_m3[:, -1]) * A * dx
print(f"\nMass conservation check:")
print(f"  Initial mass injected: {cfg.mass:.1f} kg")
print(f"  Mass in system at end: {total_mass_sim:.1f} kg")
print(f"  Relative error: {abs(total_mass_sim - cfg.mass)/cfg.mass*100:.1f}%")

# Concentration at key times
print(f"\nConcentration at key times (30 km intake):")
key_times = [60, 120, 180, 240, 300]  # minutes
for t_key in key_times:
    idx_t = np.argmin(np.abs(t_min - t_key))
    conc = C_intake[idx_t]
    print(f"  t = {t_key} min: {conc:.4f} mg/L")

print("\n" + "="*60)
print("Plots generated:")
print("1. concentration_time_series.png - Time series at different locations")
print("2. concentration_spatial_profile.png - Spatial profiles at different times")
print("3. concentration_heatmap.png - Spatio-temporal heatmap")
print("4. concentration_contour.png - Contour plot")
print("="*60)

plt.show()