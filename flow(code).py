import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. 加载数据 ---
df_hourly = pd.read_csv('flow_hourly_baseline_2025_06.csv')
df_usgs = pd.read_csv('flow_monthly_usgs_01491000_raw.csv')

# --- 2. 数据处理与单位转换 ---
df_hourly['datetime'] = pd.to_datetime(df_hourly['datetime'])
avg_2025 = df_hourly['flow_m3s'].mean()

# USGS 历史 6 月数据处理 (cfs -> m3/s)
CFS_TO_M3S = 0.0283168
june_history = df_usgs[df_usgs['month'] == 6].copy()
hist_max = june_history['hmax_cfs'].mean() * CFS_TO_M3S
hist_mean = june_history['hmean_cfs'].mean() * CFS_TO_M3S
hist_min = june_history['hmin_cfs'].mean() * CFS_TO_M3S

# --- 3. 绘图 ---
plt.figure(figsize=(14, 8), dpi=100)

# A. 2025年6月时时流量曲线
plt.plot(df_hourly['datetime'], df_hourly['flow_m3s'], 
         label='June 2025 Baseline Hourly Flow', color='#1f77b4', linewidth=1.5, alpha=0.8)

# B. 2025年6月平均值线 
plt.axhline(y=avg_2025, color='darkblue', linestyle='-', linewidth=2.5, 
            label=f'June 2025 Average ({avg_2025:.2f} m³/s)')

# C. USGS 历史背景参考线
plt.axhline(y=hist_max, color='#d62728', linestyle='--', alpha=0.7, label=f'Hist. June Max (~{hist_max:.2f} m³/s)')
plt.axhline(y=hist_mean, color='#2ca02c', linestyle='--', alpha=0.7, label=f'Hist. June Mean (~{hist_mean:.2f} m³/s)')
plt.axhline(y=hist_min, color='#ff7f0e', linestyle='--', alpha=0.7, label=f'Hist. June Min (~{hist_min:.2f} m³/s)')

# 图表装饰
plt.title('Discharge Context: June 2025 vs. USGS Historical Statistics (01491000)', fontsize=15, pad=20)
plt.ylabel('Discharge (m³/s)', fontsize=12)
plt.xlabel('Time (June 2025)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0)) # 侧边图例防止遮挡

# 高亮标注：2025平均线相对于历史均值的位置
plt.text(df_hourly['datetime'].iloc[0], avg_2025 + 0.1, f'  Current Case: {avg_2025:.2f} m³/s', 
         color='darkblue', fontweight='bold')

plt.tight_layout()
plt.show()

# --- 4. 论文结论支撑 ---
print(f"2025年6月平均流量为: {avg_2025:.2f} m3/s")
print(f"历史6月平均流量为: {hist_mean:.2f} m3/s")
print(f"当前流量占历史均值的比例: {(avg_2025/hist_mean)*100:.1f}%")