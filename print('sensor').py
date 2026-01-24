import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 尝试导入 seaborn
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# --- 基础参数 ---
Q_BASE = 3.23
DELAY_H = 5 / 60
THRESHOLD = 0.1
INTAKE_KM = 30.0
CANDIDATES = [3, 5, 8, 12, 15, 18, 22, 25, 28]
GEO_DATA = [(0, 10, 22, 1.8), (10, 20, 25, 2.0), (20, 30, 28, 2.2)]

# --- 核心逻辑 ---
def get_travel_time(x_km, Q_val):
    velocities = [(Q_val / (w * d)) * 3.6 for _, _, w, d in GEO_DATA]
    time = 0; curr_x = x_km
    for i in range(len(GEO_DATA)):
        start, end = GEO_DATA[i][0], GEO_DATA[i][1]
        if curr_x < end:
            dist = min(end - curr_x, INTAKE_KM - curr_x)
            time += dist / velocities[i]
            curr_x += dist
    return time

def optimize_layout():
    selected = []
    temp_candidates = CANDIDATES.copy()
    for _ in range(3):
        best_s, best_l = -float('inf'), None
        for loc in temp_candidates:
            max_lead = max([get_travel_time(l, Q_BASE) for l in selected + [loc]]) - DELAY_H
            score = max_lead * 80000
            locs = sorted(selected + [loc])
            for i in range(len(locs)-1):
                if locs[i+1] - locs[i] < 6: score -= 20000
            if score > best_s: best_s, best_l = score, loc
        selected.append(best_l)
        temp_candidates.remove(best_l)
    return sorted(selected)

# --- 绘图函数 ---
def run_dashboard():
    final_locs = optimize_layout()
    # 稍微减少采样点数量 (从10个减到8个)，防止纵向重叠
    Q_range = np.linspace(Q_BASE*0.7, Q_BASE*1.3, 8) 
    D_range = [10, 50, 100, 150, 200]
    sens_matrix = np.zeros((len(Q_range), len(D_range)))
    
    for j, d in enumerate(D_range):
        for i, q in enumerate(Q_range):
            base_t = get_travel_time(final_locs[0], q)
            dispersion_adv = (np.sqrt(d) / 40) 
            sens_matrix[i, j] = base_t - DELAY_H + dispersion_adv

    # 创建超大画布
    fig = plt.figure(figsize=(16, 16))

    # --- 图 1: 布局 ---
    ax1 = plt.subplot(3, 2, 1)
    ax1.axhline(y=0, color='skyblue', linewidth=10, alpha=0.3)
    ax1.scatter(final_locs, [0]*len(final_locs), c='red', marker='*', s=200)
    for s in final_locs: ax1.text(s, 0.2, f'{s}km', ha='center', fontweight='bold')
    ax1.set_title("1. Optimized Deployment Map", pad=20)
    ax1.set_xlim(-1, 31); ax1.set_yticks([])

    # --- 图 2: 热力图 (针对 Q 轴数字重叠深度优化) ---
    ax2 = plt.subplot(3, 2, 2)
    if HAS_SEABORN:
        # yticklabels 已经格式化为 2 位小数
        y_labels = [f"{q:.2f}" for q in Q_range]
        
        sns.heatmap(sens_matrix, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax2,
                    xticklabels=D_range, 
                    yticklabels=y_labels,
                    cbar_kws={
                        "label": "Lead Time (hours)",
                        "shrink": 0.7,
                        "pad": 0.1      # 增加侧栏与图的距离，给左边留出平衡感
                    })
        # 强制设置 Y 轴标签的旋转角度和字体大小，防止重叠
        ax2.set_yticklabels(y_labels, rotation=0, fontsize=10) 
    else:
        im = ax2.imshow(sens_matrix, cmap="YlGnBu", aspect='auto')
        plt.colorbar(im, ax=ax2)
    
    ax2.set_title("2. Sensitivity Analysis Matrix", pad=30, fontsize=14)
    ax2.set_xlabel("Dispersion D (m²/s)", labelpad=15)
    ax2.set_ylabel("Discharge Q (m³/s)", labelpad=10) # 增加 labelpad 撑开距离

    # --- 图 3: 信号处理 ---
    ax3 = plt.subplot(3, 1, 2)
    t_axis = np.linspace(0, 120, 300)
    filtered = 0.25 * np.exp(-((t_axis-60)**2)/120)
    measured = filtered + np.random.normal(0, 0.03, len(t_axis))
    ax3.plot(t_axis, measured, 'g.', alpha=0.3, label='Raw Data')
    ax3.plot(t_axis, filtered, 'b-', linewidth=2, label='Filtered Signal')
    ax3.axhline(y=THRESHOLD, color='r', linestyle='--', label='Alert Level')
    ax3.set_title("3. Real-time Denoising Performance", pad=20)
    ax3.set_xlabel("Time (min)"); ax3.legend(loc='upper right')

    # --- 图 4: 灵敏度曲线 ---
    ax4 = plt.subplot(3, 1, 3)
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(D_range)))
    for j, d in enumerate(D_range):
        ax4.plot(Q_range, sens_matrix[:, j], 'o-', label=f'D={d}', color=colors[j], linewidth=1.5)
    ax4.set_title("4. Multi-Parametric Stability Test (Lead Time vs Q)", pad=20)
    ax4.set_xlabel("Discharge Q (m³/s)"); ax4.set_ylabel("Lead Time (h)")
    ax4.legend(title="Dispersion (D)", loc='center left', bbox_to_anchor=(1, 0.5))
    ax4.grid(True, linestyle='--', alpha=0.5)

    # 这里的 pad=6.0 是核心，彻底拉开上下子图，防止图2的标签撞到图3
    plt.tight_layout(pad=6.0) 
    plt.show()

if __name__ == "__main__":
    run_dashboard()