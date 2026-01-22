
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 题目核心参数 (根据上传文件整理)
# ==========================================
Q_AVG = 3.22          # 平均流量 (m3/s)
COST_PER_HOUR = 80000 # 停产损失 (CNY/h)
DELAY_H = 5 / 60      # 5分钟报告延迟 (h)
INTAKE_KM = 30.0      # 取水口位置

# 河道几何分段数据 (来自 river_geometry_segments.csv)
# 每行格式: (开始km, 结束km, 宽度m, 深度m)
GEO_DATA = [
    (0, 10, 22, 1.8),
    (10, 20, 25, 2.0),
    (20, 30, 28, 2.2)
]

# 候选监测点 (来自 candidate_sensor_locations_km.csv)
CANDIDATES = [3, 5, 8, 10, 12, 15, 18, 20, 22, 25, 28]

# ==========================================
# 2. 物理模型函数
# ==========================================
def get_segment_velocity():
    """计算各段流速 (km/h)"""
    velocities = []
    for start, end, w, d in GEO_DATA:
        v_m_s = Q_AVG / (w * d)  # v = Q / (W*H)
        v_km_h = v_m_s * 3.6     # 转换为 km/h
        velocities.append({
            'start': start, 'end': end, 'v': v_km_h
        })
    return velocities

VEL_SEGMENTS = get_segment_velocity()

def get_travel_time(start_x):
    """计算污染物从 start_x 流到 30km 处所需的总时长 (h)"""
    remaining_dist = INTAKE_KM - start_x
    total_time = 0
    curr_x = start_x
    
    for seg in VEL_SEGMENTS:
        if curr_x < seg['end']:
            dist_in_seg = min(seg['end'] - curr_x, remaining_dist)
            if dist_in_seg > 0:
                total_time += dist_in_seg / seg['v']
                curr_x += dist_in_seg
                remaining_dist -= dist_in_seg
    return total_time

# ==========================================
# 3. 贪心算法核心
# ==========================================
def evaluate_network(selected_set):
    """
    评价函数 (Objective Function):
    Score = 预警价值 - 空间冗余惩罚
    """
    if not selected_set: return 0
    
    # 计算这组传感器的最大预警提前量
    # 预警时间 = 水流传导时间 - 报告延迟
    lead_times = [get_travel_time(x) - DELAY_H for x in selected_set]
    max_lead_time = max(lead_times)
    
    # 1. 经济效益 (每一小时价值 80,000)
    benefit = max_lead_time * COST_PER_HOUR
    
    # 2. 空间冗余惩罚 (美赛亮点：防止扎堆)
    # 如果两个传感器距离小于 6km，说明监测范围重叠，扣除其价值
    penalty = 0
    locs = sorted(selected_set)
    for i in range(len(locs) - 1):
        gap = locs[i+1] - locs[i]
        if gap < 6:
            penalty += (6 - gap) * 15000  # 惩罚系数
            
    return benefit - penalty

def run_optimization():
    print("="*50)
    print("MCM 2026: 传感器布置方案优化 (物理-经济耦合模型)")
    print("="*50)
    
    selected = []
    candidates_remaining = CANDIDATES.copy()
    
    for i in range(3): # 放置 3 个传感器
        best_score = -float('inf')
        best_loc = None
        
        for loc in candidates_remaining:
            current_test = selected + [loc]
            score = evaluate_network(current_test)
            
            if score > best_score:
                best_score = score
                best_loc = loc
        
        if best_loc is not None:
            selected.append(best_loc)
            candidates_remaining.remove(best_loc)
            print(f"步骤 {i+1}: 选定 {best_loc:>2} km (当前网络估值: ¥{best_score:,.0f})")

    final_locs = sorted(selected)
    max_warning = get_travel_time(final_locs[0]) - DELAY_H
    
    print("\n" + "-"*30)
    print(f"最终推荐位置: {final_locs} km")
    print(f"最大预警时间: {max_warning:.2f} 小时")
    print(f"预期减损价值: ¥{max_warning * COST_PER_HOUR:,.0f}")
    print("-"*30)
    
    plot_results(final_locs)

# ==========================================
# 4. 可视化
# ==========================================
def plot_results(selected):
    plt.figure(figsize=(12, 5))
    
    # 绘制河道流速背景（颜色越深流速越快）
    colors = ['#e3f2fd', '#90caf9', '#42a5f5']
    for i, seg in enumerate(VEL_SEGMENTS):
        plt.axvspan(seg['start'], seg['end'], color=colors[i], alpha=0.3, 
                    label=f"Seg {i+1} ({seg['v']:.2f} km/h)")
    
    # 画所有候选点
    plt.scatter(CANDIDATES, [0.1]*len(CANDIDATES), c='gray', s=30, label='Candidate Sites', alpha=0.5)
    # 画选中的最优传感器
    plt.scatter(selected, [0.1]*len(selected), c='red', marker='*', s=300, edgecolors='black', label='Final Placements')
    
    for s in selected:
        plt.annotate(f'{s}km', (s, 0.1), xytext=(0, 15), textcoords='offset points', 
                     ha='center', fontweight='bold', color='red')
    
    plt.title("Sensor Placement: Balancing Lead-Time & Spatial Diversity", fontsize=14)
    plt.xlabel("Distance from Spill Source (km)")
    plt.xlim(-1, 31); plt.ylim(-0.5, 1); plt.yticks([]); plt.legend(loc='lower right')
    plt.grid(axis='x', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_optimization()