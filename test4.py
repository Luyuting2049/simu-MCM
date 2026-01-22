"""
1D Advection-Dispersion Model for River Pollution
Final version with 24-hour limit and reversed 3D axes
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
        self.depth = cfg['river_reach']['default_width_m']
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

# Source calculation - REDUCE source strength for better 3D visualization
M_dot = cfg.mass / cfg.duration / 100  # Reduced by 100x
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
print(f"  Mass flux: {M_dot:.6f} kg/s (reduced 100x)")
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
print(f"\nStarting simulation for hour-scale plot (0-24 hours)...")
t_end_hours = 24 * 3600  # 24 hours only

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
monitor_locations_min = list(range(0, 11))  # 0-10 km for minute-scale plot
monitor_locations_hour = [0, 5, 10, 15, 20, 25, 30]  # For hour-scale plot
monitor_indices_min = [np.argmin(np.abs(x_km - loc)) for loc in monitor_locations_min]
monitor_indices_hour = [np.argmin(np.abs(x_km - loc)) for loc in monitor_locations_hour]


#以下均为画图
# ===== PLOT 1: Minutes scale (0-10 km) - KEEP EXACTLY AS BEFORE =====
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
        C_monitor_min[i, j] = C_all[idx]

# Create minute-scale plot (EXACT SAME AS BEFORE)
plt.figure(figsize=(16, 10))
colors = plt.cm.Set3(np.linspace(0, 1, len(monitor_locations_min)))

one_km_index = monitor_locations_min.index(1)
colors[one_km_index] = (0.5, 0.7, 1.0, 1.0)   # 深蓝色，alpha=0.85

for i, loc_km in enumerate(monitor_locations_min):
    plt.plot(t_min_detailed, C_monitor_min[i, :] * 1000,  # Convert to g/m³
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

# ===== PLOT 2: Hours scale (24 hours only) - IGNORE 0km =====
print("\nPreparing hour-scale plot (0-24 hours, ignoring 0km)...")
t_hour_detailed = np.linspace(0, 24, 289)  # 0-24 hours, 5-minute intervals
t_sec_detailed2 = t_hour_detailed * 3600
C_monitor_hour = np.zeros((len(monitor_locations_hour), len(t_hour_detailed)))

for i, idx in enumerate(monitor_indices_hour):
    for j, t_sec in enumerate(t_sec_detailed2):
        C_all = get_concentration_at_time(t_sec, sol_hours)
        C_monitor_hour[i, j] = C_all[idx]

# Scale and exclude 0km
C_monitor_hour_adjusted = C_monitor_hour * 100  # Scale by 100

plt.figure(figsize=(16, 10))
colors_hour = plt.cm.tab20c(np.linspace(0, 1, len(monitor_locations_hour)))

# Plot all locations EXCEPT 0km (start from i=1)
for i, loc_km in enumerate(monitor_locations_hour):
    if i == 0:  # Skip 0km
        continue
    
    plt.plot(t_hour_detailed, C_monitor_hour_adjusted[i, :],
             linewidth=2.2,
             label=f'{loc_km} km',
             color=colors_hour[i],
             alpha=0.85)

plt.axvline(x=cfg.duration/3600, color='red', linestyle='--',
            linewidth=2.5, alpha=0.8,
            label=f'Release ends ({cfg.duration/3600:.2f} h)')

plt.xlabel('Time (hours)', fontsize=14, fontweight='bold')
plt.ylabel('Concentration (10⁻² × kg/m³)', fontsize=14, fontweight='bold')
plt.title('Pollutant Concentration Evolution (0-24 hours)\n5-30 km Locations (0km excluded for scale)',
          fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=11, loc='upper right', ncol=2, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--')
plt.yscale('linear')

# Calculate y-limits excluding 0km
all_data_except_0km = []
for i in range(1, len(monitor_locations_hour)):
    all_data_except_0km.extend(C_monitor_hour_adjusted[i, :])
    
y_min = min(0, np.min(all_data_except_0km) * 1.1)
y_max = np.max(all_data_except_0km) * 1.1
plt.ylim(y_min, y_max)
plt.xlim(0, 24)

plt.tight_layout()
plt.savefig('plot2_hours_scale_24h.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Plot 2 saved: plot2_hours_scale_24h.png")

# ===== PLOT 3: 3D Surface Plot - FROM 2h to 36h, RAINBOW COLORS =====
print("\nPreparing 3D plot (from 2h to 36h, rainbow colors)...")

# 修改模拟时间为36小时
t_end_3d = 36 * 3600  # 36小时 for 3D plot

# Simulate for 3D plot (36小时)
sol_3d = solve_ivp(river_pde, 
                   [0, t_end_3d], 
                   C0, 
                   method='RK45',
                   max_step=dt_max,
                   rtol=1e-6,
                   atol=1e-9)

print(f"3D plot simulation completed: {sol_3d.success}")
print(f"3D plot time steps: {len(sol_3d.t)}")

# 准备3D数据 - 从2小时开始，到36小时
start_hour = 2.0  # 2小时
end_hour = 24.0   # 36小时
t_3d = np.linspace(start_hour, end_hour, 80)  # 2-36小时
x_3d = np.linspace(0, 30, 80)  # 0-30 km
X_3d, T_3d = np.meshgrid(x_3d, t_3d, indexing='ij')
C_3d = np.zeros_like(X_3d)

for i, xi in enumerate(x_3d):
    x_idx = np.argmin(np.abs(x_km - xi))
    for j, tj in enumerate(t_3d):
        t_sec = tj * 3600
        C_all = get_concentration_at_time(t_sec, sol_3d)
        C_3d[i, j] = C_all[x_idx]

# 移除初始高峰
mask_far = X_3d > 5
max_far = np.max(C_3d[mask_far]) if np.any(mask_far) else np.max(C_3d)
mask_near_source = X_3d <= 1
C_3d[mask_near_source] = np.minimum(C_3d[mask_near_source], max_far * 3)

# 调整浓度坐标比例：放大100倍，突出细节
C_3d_scaled = C_3d * 100  # 放大100倍

# 创建3D图
fig = plt.figure(figsize=(20, 12))
ax = fig.add_subplot(111, projection='3d')

# 使用jet colormap（彩虹色）
surf = ax.plot_surface(T_3d, X_3d, C_3d_scaled,
                      cmap='jet',  # 彩虹色渐变
                      linewidth=0.05,
                      edgecolor='gray',
                      antialiased=True,
                      rstride=1,
                      cstride=1,
                      alpha=0.92,
                      vmin=0,
                      vmax=np.max(C_3d_scaled))

# 设置坐标轴范围和标签
ax.set_xlim(start_hour, end_hour)  # 时间从2小时到36小时
ax.set_ylim(0, 30)  # 距离：0-30km
ax.set_zlim(0, np.max(C_3d_scaled) * 1.1)

# 坐标轴标签
ax.set_xlabel('Time (hours)', fontsize=13, labelpad=15)
ax.set_ylabel('Distance from Source (km)', fontsize=13, labelpad=15)
ax.set_zlabel('Concentration (10⁻² × kg/m³)', fontsize=13, labelpad=15)  # 注明放大100倍

# 设置标题
ax.set_title('3D Spatio-Temporal Evolution (2-36h)\nRainbow Color Gradient',
             fontsize=16, fontweight='bold', pad=25)

# 设置视角 - 从角落看
ax.view_init(elev=30, azim=-125)

# 修改网格线为黑色细虚线
ax.xaxis._axinfo["grid"].update({
    "linewidth": 0.5, 
    "color": 'black', 
    "linestyle": '--',
    "alpha": 0.3
})
ax.yaxis._axinfo["grid"].update({
    "linewidth": 0.5, 
    "color": 'black', 
    "linestyle": '--',
    "alpha": 0.3
})
ax.zaxis._axinfo["grid"].update({
    "linewidth": 0.5, 
    "color": 'black', 
    "linestyle": '--',
    "alpha": 0.3
})

# 设置面板属性
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('lightgray')
ax.yaxis.pane.set_edgecolor('lightgray')
ax.zaxis.pane.set_edgecolor('lightgray')
ax.xaxis.pane.set_alpha(0.1)
ax.yaxis.pane.set_alpha(0.1)
ax.zaxis.pane.set_alpha(0.1)

# 添加黑色虚线等高线
levels = np.linspace(0, np.max(C_3d_scaled), 15)
contours = ax.contour(T_3d, X_3d, C_3d_scaled,
                     levels=levels,
                     zdir='z',
                     offset=0,
                     colors='black',
                     linewidths=0.5,
                     linestyles='--',
                     alpha=0.4)

# 标记取水口位置（30km）
ax.plot(t_3d, np.full_like(t_3d, 30),
        C_3d_scaled[-1, :],  # 最后一行对应30km
        color='white', linewidth=2.5, alpha=0.9,
        label='Intake (30km)')

# 添加时间线显示5km处的浓度变化
idx_5km = np.argmin(np.abs(x_3d - 5))
ax.plot(t_3d, np.full_like(t_3d, 5),
        C_3d_scaled[idx_5km, :],
        color='cyan', linewidth=2.0, alpha=0.8,
        label='5km location')

# 添加时间线显示15km处的浓度变化
idx_15km = np.argmin(np.abs(x_3d - 15))
ax.plot(t_3d, np.full_like(t_3d, 15),
        C_3d_scaled[idx_15km, :],
        color='magenta', linewidth=2.0, alpha=0.8,
        label='15km location')

# 添加颜色条
cbar = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=25, pad=0.12)
cbar.set_label('10⁻² × kg/m³', fontsize=12, rotation=270, labelpad=20)

# 添加图例
ax.legend(fontsize=10, loc='upper left')

# 添加文本说明
ax.text2D(0.02, 0.95, 'Time: 2-36h\nRainbow (jet) colors\nConcentration ×100',
          transform=ax.transAxes, fontsize=10,
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('plot3_3d_2h_36h_rainbow.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Plot 3 saved: plot3_3d_2h_36h_rainbow.png")

# ===== PLOT 4: 3D Surface Plot - 0-3h FULL RANGE =====
print("\nPreparing 3D plot (0-3h, full range with peak)...")
t_end_3d_short = 3 * 3600  # 3小时 for 3D plot

# Simulate for 3D plot (3小时)
sol_3d_short = solve_ivp(river_pde, 
                         [0, t_end_3d_short], 
                         C0, 
                         method='RK45',
                         max_step=dt_max,
                         rtol=1e-6,
                         atol=1e-9)

# 准备3D数据 - 0-3小时，包含初始峰值
start_hour_short = 0.0  # 0小时
end_hour_short = 3.0    # 3小时
t_3d_short = np.linspace(start_hour_short, end_hour_short, 100)  # 0-3小时，更多点
x_3d_short = np.linspace(0, 30, 100)  # 0-30 km，更多点
X_3d_short, T_3d_short = np.meshgrid(x_3d_short, t_3d_short, indexing='ij')
C_3d_short = np.zeros_like(X_3d_short)

for i, xi in enumerate(x_3d_short):
    x_idx = np.argmin(np.abs(x_km - xi))
    for j, tj in enumerate(t_3d_short):
        t_sec = tj * 3600
        C_all = get_concentration_at_time(t_sec, sol_3d_short)
        C_3d_short[i, j] = C_all[x_idx]

# 不移除初始峰值，但为了可视化调整缩放
# 放大倍数适当减小，因为峰值本身已经很高
C_3d_short_scaled = C_3d_short * 50  # 放大50倍（比之前的100倍小）

# 使用jet colormap（彩虹色）
fig = plt.figure(figsize=(20, 12))
ax = fig.add_subplot(111, projection='3d')

# 使用jet colormap（彩虹色渐变）
surf = ax.plot_surface(T_3d_short, X_3d_short, C_3d_short_scaled,
                      cmap='jet',  # 彩虹色渐变
                      linewidth=0.05,
                      edgecolor='gray',
                      antialiased=True,
                      rstride=1,
                      cstride=1,
                      alpha=0.92,
                      vmin=0,
                      vmax=np.max(C_3d_short_scaled))

# 设置坐标轴范围和标签
ax.set_xlim(start_hour_short, end_hour_short)  # 时间从0到3小时
ax.set_ylim(0, 30)  # 距离：0-30km
ax.set_zlim(0, np.max(C_3d_short_scaled) * 1.1)

# 坐标轴标签
ax.set_xlabel('Time (hours)', fontsize=13, labelpad=15)
ax.set_ylabel('Distance from Source (km)', fontsize=13, labelpad=15)
ax.set_zlabel('Concentration (transformed)', fontsize=13, labelpad=15)

# 设置标题
ax.set_title('3D Spatio-Temporal Evolution (0-3h)\nIncluding Initial Peak - Rainbow Colors',
             fontsize=16, fontweight='bold', pad=25)

# 设置视角 - 从角落看
ax.view_init(elev=30, azim=-125)

# 修改网格线为黑色细虚线
ax.xaxis._axinfo["grid"].update({
    "linewidth": 0.5, 
    "color": 'black', 
    "linestyle": '--',
    "alpha": 0.3
})
ax.yaxis._axinfo["grid"].update({
    "linewidth": 0.5, 
    "color": 'black', 
    "linestyle": '--',
    "alpha": 0.3
})
ax.zaxis._axinfo["grid"].update({
    "linewidth": 0.5, 
    "color": 'black', 
    "linestyle": '--',
    "alpha": 0.3
})

# 设置面板属性
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('lightgray')
ax.yaxis.pane.set_edgecolor('lightgray')
ax.zaxis.pane.set_edgecolor('lightgray')
ax.xaxis.pane.set_alpha(0.1)
ax.yaxis.pane.set_alpha(0.1)
ax.zaxis.pane.set_alpha(0.1)

# 添加黑色虚线等高线
levels = np.linspace(0, np.max(C_3d_short_scaled), 15)
contours = ax.contour(T_3d_short, X_3d_short, C_3d_short_scaled,
                     levels=levels,
                     zdir='z',
                     offset=0,
                     colors='black',
                     linewidths=0.5,
                     linestyles='--',
                     alpha=0.4)

# 标记排放结束时间
release_end_hour = cfg.duration / 3600
ax.plot([release_end_hour, release_end_hour], [0, 30],
        [0, 0],
        color='red', linewidth=2.5, linestyle='--', alpha=0.9,
        label=f'Release end ({release_end_hour:.2f}h)')

# 标记取水口位置（30km）
ax.plot(t_3d_short, np.full_like(t_3d_short, 30),
        C_3d_short_scaled[-1, :],  # 最后一行对应30km
        color='white', linewidth=2.5, alpha=0.9,
        label='Intake (30km)')

# 添加时间线显示1km处的浓度变化（初始峰值位置）
idx_1km = np.argmin(np.abs(x_3d_short - 1))
ax.plot(t_3d_short, np.full_like(t_3d_short, 1),
        C_3d_short_scaled[idx_1km, :],
        color='cyan', linewidth=2.0, alpha=0.8,
        label='1km location (peak)')

# 添加时间线显示10km处的浓度变化
idx_10km = np.argmin(np.abs(x_3d_short - 10))
ax.plot(t_3d_short, np.full_like(t_3d_short, 10),
        C_3d_short_scaled[idx_10km, :],
        color='magenta', linewidth=2.0, alpha=0.8,
        label='10km location')

# 添加颜色条
cbar = fig.colorbar(surf, ax=ax, shrink=0.7, aspect=25, pad=0.12)
cbar.set_label('Transformed Concentration', fontsize=12, rotation=270, labelpad=20)

# 添加图例
ax.legend(fontsize=10, loc='upper left')

# 添加文本说明
ax.text2D(0.02, 0.95, 'Time: 0-3h\nIncludes initial peak\nRainbow (jet) colors',
          transform=ax.transAxes, fontsize=10,
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('plot4_3d_0_3h_with_peak.png', dpi=200, bbox_inches='tight')
plt.close()
print("✓ Plot 4 saved: plot4_3d_0_3h_with_peak.png")


# ===== Analysis =====
print("\n" + "="*60)
print("ANALYSIS RESULTS")
print("="*60)

# Peak analysis
print(f"\nPeak concentrations at key locations (0-24 hours):")
print(f"{'Location (km)':<12} {'Peak (kg/m³)':<15} {'Peak Time (h)':<15}")
print("-" * 45)

for i, loc_km in enumerate(monitor_locations_hour):
    actual_peak = np.max(C_monitor_hour[i, :])
    peak_time_idx = np.argmax(C_monitor_hour[i, :])
    peak_time = t_hour_detailed[peak_time_idx]
    
    if i == 0:
        print(f"{loc_km:<12} {actual_peak:<15.2e} {peak_time:<15.1f} (excluded from plot)")
    else:
        print(f"{loc_km:<12} {actual_peak:<15.2e} {peak_time:<15.1f}")

# Travel time to intake
travel_time = 30 / (u * 3.6)  # hours
print(f"\nTravel time to intake (30 km): {travel_time:.1f} hours")

# Mass injected (actual vs reduced)
actual_mass = cfg.mass  # kg
reduced_mass = cfg.mass / 100  # kg (due to 100x reduction)
print(f"\nMass injection:")
print(f"  Actual scenario: {actual_mass:.1f} kg")
print(f"  Reduced for visualization: {reduced_mass:.2f} kg (100x reduction)")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("Three plots generated with final adjustments:")
print("1. plot1_minutes_scale_0_10km.png - Minute scale, 0-10 km")
print("2. plot2_hours_scale_24h.png - Hour scale, 0-24h, 0km excluded")
print("3. plot3_3d_reversed_axes.png - 3D with reversed axes, corner origin")
print(f"\nFiles saved in: {os.getcwd()}")