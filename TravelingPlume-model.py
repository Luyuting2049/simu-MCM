"""
1D Advection-Dispersion Model for River Pollutant Transport
Solves the transport equation using finite difference (upwind for advection, central for diffusion)
with ODE solver (RK45)
"""
import yaml
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ===== Load Configuration =====
class Config:
    def __init__(self, path='config.yaml'):
        with open(path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        self.L = cfg['river_reach']['L_km'] * 1000  # River length (m)
        self.mass = cfg['spill_event']['total_mass_kg']
        self.duration = cfg['spill_event']['release_duration_min']
        self.width = cfg['river_reach']['default_width_m']
        self.depth = cfg['river_reach']['default_depth_m']
        self.D = cfg['transport_parameters']['longitudinal_dispersion_D_m2s']['baseline']

cfg = Config()
N, dx = 300, cfg.L / 300  # Grid: 300 cells, each of size dx
u, A, D = 0.5, cfg.width * cfg.depth, cfg.D

print(f"Grid: N={N}, dx={dx:.0f}m | u={u}m/s, A={A}m² | D={D}m²/s | Release: {cfg.mass}kg/{cfg.duration}min")

# ===== Phase 1: Release (0-30 min) =====
print("\n" + "="*60 + "\nPhase 1: Release (0-30 min)")

def ode_phase1(t, C):
    """Advection-dispersion: dC/dt = -u*dC/dx + D*d²C/dx²"""
    dC = np.zeros(N)
    source = cfg.mass / cfg.duration / A  # Source [kg/(m³*min)]
    
    for i in range(N):
        if i == 0:
            # Boundary: source + diffusion + advection out
            dC[i] = source / dx - u * (C[i] - 0) / dx + D * (C[i+1] - C[i]) / dx**2
        elif i == N - 1:
            # Far end: no-flux boundary (backward difference for advection)
            dC[i] = -u * (C[i] - C[i-1]) / dx + D * (0 - C[i]) / dx**2
        else:
            # Interior: standard upwind + central difference
            dC[i] = -u * (C[i] - C[i-1]) / dx + D * (C[i+1] - 2*C[i] + C[i-1]) / dx**2
    
    return dC

t1 = np.linspace(0, cfg.duration, int(cfg.duration * 10) + 1)
sol1 = solve_ivp(ode_phase1, (0, cfg.duration), np.zeros(N), t_eval=t1, 
                 method='RK45', max_step=0.1, dense_output=True)
print(f"Complete: C_max={np.max(sol1.y):.6f} kg/m³, points={len(sol1.t)}")

# ===== Phase 2A: Non-release (30-300 min) =====
print("\nPhase 2A: Non-release (30-300 min)")

def ode_phase2(t, C):
    """No source: dC/dt = -u*dC/dx + D*d²C/dx²"""
    dC = np.zeros(N)
    
    for i in range(N):
        if i == 0:
            # Zero-flux at upstream (no source, no incoming flow)
            dC[i] = -u * (C[i] - 0) / dx + D * (C[i+1] - C[i]) / dx**2
        elif i == N - 1:
            # No-flux at downstream
            dC[i] = -u * (C[i] - C[i-1]) / dx + D * (0 - C[i]) / dx**2
        else:
            # Interior
            dC[i] = -u * (C[i] - C[i-1]) / dx + D * (C[i+1] - 2*C[i] + C[i-1]) / dx**2
    
    return dC

t2a = np.linspace(cfg.duration, 300, int((300 - cfg.duration) * 10) + 1)
sol2a = solve_ivp(ode_phase2, (cfg.duration, 300), sol1.y[:, -1], t_eval=t2a,
                  method='RK45', max_step=0.1, dense_output=True)
print(f"Complete: C_max={np.max(sol2a.y):.6f} kg/m³, points={len(sol2a.t)}")

# ===== Phase 2B: Long-term (5-100 hours) =====
print("\nPhase 2B: Long-term (5-100 hours)")

def ode_phase2b(t, C):
    """Time in hours: same equation, multiply by 60 for unit conversion"""
    dC = np.zeros(N)
    
    for i in range(N):
        if i == 0:
            dC[i] = -u * (C[i] - 0) / dx + D * (C[i+1] - C[i]) / dx**2
        elif i == N - 1:
            dC[i] = -u * (C[i] - C[i-1]) / dx + D * (0 - C[i]) / dx**2
        else:
            dC[i] = -u * (C[i] - C[i-1]) / dx + D * (C[i+1] - 2*C[i] + C[i-1]) / dx**2
    
    return dC * 60

t2b = np.linspace(5, 100, int((100 - 5) * 2) + 1)
sol2b = solve_ivp(ode_phase2b, (5, 100), sol2a.y[:, -1], t_eval=t2b,
                  method='RK45', max_step=0.5, dense_output=True)
print(f"Complete: C_max={np.max(sol2b.y):.6f} kg/m³, points={len(sol2b.t)}")

# ===== Plotting =====
print("\n" + "="*60 + "\nGenerating plots...")

x_km = np.linspace(0, cfg.L/1000, N)
pos_km = [0, 5, 10, 15, 20, 25, 30]
pos_idx = [int(min(x / (cfg.L/1000) * N, N-1)) for x in pos_km]
colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(pos_km)))

# Figure 1: Phase 1 time series
fig, ax = plt.subplots(figsize=(12, 6))
for j, idx in enumerate(pos_idx):
    ax.plot(sol1.t, sol1.y[idx, :], label=f'x={pos_km[j]}km', 
            linewidth=2.5, color=colors[j], marker='o', markersize=3, alpha=0.8)
ax.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
ax.set_ylabel('Concentration (kg/m³)', fontsize=12, fontweight='bold')
ax.set_title('Phase 1: Release Period (0-30 min)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, ncol=2, loc='upper left')
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('phase1_concentration_time.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ phase1_concentration_time.png")

# Figure 2: Phase 2A time series
fig, ax = plt.subplots(figsize=(12, 6))
for j, idx in enumerate(pos_idx):
    ax.plot(sol2a.t, sol2a.y[idx, :], label=f'x={pos_km[j]}km', 
            linewidth=2.5, color=colors[j], marker='s', markersize=3, alpha=0.8)
ax.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
ax.set_ylabel('Concentration (kg/m³)', fontsize=12, fontweight='bold')
ax.set_title('Phase 2: Post-Release Period (30-300 min)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, ncol=2, loc='best')
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('phase2a_concentration_time.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ phase2a_concentration_time.png")

# Figure 3: 3D surface
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')
t_all = np.concatenate([sol1.t, sol2a.t])
C_all = np.concatenate([sol1.y, sol2a.y], axis=1)
T, X = np.meshgrid(t_all, x_km)
surf = ax.plot_surface(T, X, C_all, cmap='YlOrRd', alpha=0.9, edgecolor='none', antialiased=True)
ax.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold', labelpad=10)
ax.set_ylabel('River Distance (km)', fontsize=11, fontweight='bold', labelpad=10)
ax.set_zlabel('Concentration (kg/m³)', fontsize=11, fontweight='bold', labelpad=10)
ax.set_title('3D Concentration Profile C(x,t) - 0-300 min', fontsize=12, fontweight='bold', pad=15)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Concentration (kg/m³)')
ax.view_init(elev=25, azim=45)
plt.tight_layout()
plt.savefig('phase1_phase2a_3d_surface.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ phase1_phase2a_3d_surface.png")

print("\n" + "="*60)
print("All completed! Three plots generated.")
print("="*60)
