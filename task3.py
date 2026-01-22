import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

# ====================== 全局配置 ======================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.size'] = 10
plt.rcParams['figure.autolayout'] = False

# ====================== 1. 数据读取（无修改） ======================
def load_task_data(task2_conc_path: str, demand_path: str = None):
    try:
        if not os.path.exists(task2_conc_path):
            raise FileNotFoundError(f"文件不存在：{task2_conc_path}")
        
        task2_df = pd.read_csv(task2_conc_path)
        if len(task2_df) != 24:
            raise ValueError(f"浓度数据需24行，实际{len(task2_df)}行")
        task2_df = task2_df.sort_values("time").reset_index(drop=True)
        if not all(task2_df["time"] == np.arange(24)):
            raise ValueError("time列必须是0-23连续整数")
        
        intake_conc = task2_df["concentration"].values.astype(float)
        intake_conc[np.abs(intake_conc) < 1e-10] = 0.0
        print(f"✅ 读取浓度数据：{task2_conc_path} | 峰值: {np.max(intake_conc):.6f} mg/L")
    except Exception as e:
        raise RuntimeError(f"读取浓度数据失败：{str(e)}")

    try:
        if demand_path and os.path.exists(demand_path):
            demand_df = pd.read_csv(demand_path)
            demand_df = demand_df.sort_values("time").reset_index(drop=True)
            relative_demand = demand_df["relative_demand"].values.astype(float)
            daily_base_demand = 10000
            hourly_demand = daily_base_demand * relative_demand / 24
        else:
            hourly_demand = np.ones(24) * 10000 / 24
    except:
        hourly_demand = np.ones(24) * 10000 / 24

    return intake_conc, hourly_demand

# ====================== 2. 固定参数（无修改） ======================
def get_fixed_params():
    params = {
        "stop_cost_per_hour": 80000,
        "emergency_storage_init": 25000,
        "stop_threshold": 0.3,
        "daily_base_demand": 10000,
        "unit_risk_cost": 50000,
        "supply_loss_coeff": 1e6,
        "alpha": 0.3, "beta": 0.2, "gamma": 0.5,
        "water_bins": [0, 5000, 10000, 15000, 20000, 25000],
        "C_bins": [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0]
    }
    return params

# ====================== 3. 动态规划核心（无修改） ======================
def dp_optimal_strategy(intake_conc: np.ndarray, hourly_demand: np.ndarray, params: dict):
    n_hours = 24
    water_bins = params["water_bins"]
    C_bins = params["C_bins"]
    threshold = params["stop_threshold"]

    def get_c_idx(c):
        for i, bin_val in enumerate(C_bins):
            if c <= bin_val:
                return i
        return len(C_bins) - 1

    dp_table = np.full((n_hours + 1, len(C_bins), 2, len(water_bins)), np.inf)
    dp_table[-1, :, :, :] = 0
    policy_table = np.zeros((n_hours, len(C_bins), 2, len(water_bins)), dtype=int)

    for t in range(n_hours - 1, -1, -1):
        current_c_raw = intake_conc[t]
        c_idx = get_c_idx(current_c_raw)

        for o in [0, 1]:
            for w_idx in range(len(water_bins)):
                current_water = water_bins[w_idx]

                for decision in [0, 1]:
                    stage_cost = 0

                    if decision == 1:
                        stage_cost += params["alpha"] * params["stop_cost_per_hour"]

                    if decision == 0 and current_c_raw > threshold:
                        stage_cost += params["gamma"] * params["unit_risk_cost"]

                    if decision == 1:
                        supply = min(current_water, hourly_demand[t])
                        supply_loss = (hourly_demand[t] - supply) / hourly_demand[t] if hourly_demand[t] > 0 else 0
                        stage_cost += params["beta"] * supply_loss * params["supply_loss_coeff"]

                    next_c_idx = get_c_idx(intake_conc[t+1]) if t+1 < n_hours else 0
                    next_o = decision
                    next_water = max(0, current_water - hourly_demand[t]) if decision == 1 else current_water
                    next_w_idx = min(range(len(water_bins)), key=lambda i: abs(water_bins[i] - next_water))

                    total_cost = stage_cost + dp_table[t+1][next_c_idx][next_o][next_w_idx]
                    if total_cost < dp_table[t][c_idx][o][w_idx]:
                        dp_table[t][c_idx][o][w_idx] = total_cost
                        policy_table[t][c_idx][o][w_idx] = decision

    optimal_policy = np.zeros(n_hours, dtype=int)
    current_c_idx = get_c_idx(0.0)
    current_o = 0
    current_w_idx = water_bins.index(params["emergency_storage_init"])

    for t in range(n_hours):
        optimal_policy[t] = policy_table[t][current_c_idx][current_o][current_w_idx]
        if t+1 < n_hours:
            current_c_idx = get_c_idx(intake_conc[t+1])
        current_o = optimal_policy[t]
        current_water = water_bins[current_w_idx]
        next_water = max(0, current_water - hourly_demand[t]) if optimal_policy[t] == 1 else current_water
        current_w_idx = min(range(len(water_bins)), key=lambda i: abs(water_bins[i] - next_water))

    shut_duration = np.sum(optimal_policy)
    shut_cost = params["stop_cost_per_hour"] * shut_duration

    storage = params["emergency_storage_init"]
    total_supply = 0
    supply_detail = np.zeros(n_hours)
    for t in range(n_hours):
        if optimal_policy[t] == 0:
            supply_t = min(params["daily_base_demand"]/24*2, hourly_demand[t])
        else:
            supply_t = min(storage, hourly_demand[t])
            storage -= supply_t
        supply_detail[t] = supply_t
        total_supply += supply_t
    total_demand = np.sum(hourly_demand)
    supply_loss = 1 - (total_supply / total_demand) if total_demand > 0 else 0
    supply_loss_cost = supply_loss * params["supply_loss_coeff"]

    risk_duration = np.sum((intake_conc > threshold) & (optimal_policy == 0))
    risk_cost = params["unit_risk_cost"] * risk_duration

    total_cost = params["alpha"]*shut_cost + params["beta"]*supply_loss_cost + params["gamma"]*risk_cost

    result = {
        "optimal_policy": optimal_policy,
        "total_cost": total_cost,
        "shut_duration": shut_duration,
        "risk_duration": risk_duration,
        "supply_loss": supply_loss,
        "cost_breakdown": {"shut_cost": shut_cost, "supply_loss_cost": supply_loss_cost, "risk_cost": risk_cost},
        "supply_detail": supply_detail,
        "raw_concentration": intake_conc
    }
    return result

# ====================== 4. 结果输出（无修改） ======================
def print_result_summary(velocity: float, result: dict):
    print("\n" + "="*80)
    print(f"📈 流速 {velocity} m/s - 最优关停策略结果")
    print("="*80)
    print(f"1. 总目标成本：{result['total_cost']:,.2f} 元")
    print(f"2. 关停时长：{result['shut_duration']} 小时")
    print(f"3. 风险时长（超标未关停）：{result['risk_duration']} 小时")
    print(f"4. 供水损失率：{result['supply_loss']:.4f} ({result['supply_loss']*100:.2f}%)")
    print("\n5. 成本分解：")
    total_w = result['total_cost'] if result['total_cost'] != 0 else 1
    print(f"   - 关停成本：{result['cost_breakdown']['shut_cost']:,.2f} 元（占比 {(result['cost_breakdown']['shut_cost']*0.3/total_w)*100:.1f}%）")
    print(f"   - 供水损失成本：{result['cost_breakdown']['supply_loss_cost']:,.2f} 元（占比 {(result['cost_breakdown']['supply_loss_cost']*0.2/total_w)*100:.1f}%）")
    print(f"   - 风险成本：{result['cost_breakdown']['risk_cost']:,.2f} 元（占比 {(result['cost_breakdown']['risk_cost']*0.5/total_w)*100:.1f}%）")
    print("\n6. 最优策略（0=开启，1=关停）：")
    print(f"   {result['optimal_policy']}")
    print("="*80 + "\n")

# ====================== 5. 绘图函数（仅修改供需线偏移） ======================
def plot_single_velocity(velocity: float, intake_conc, hourly_demand, result, params):
    hours = np.arange(24)
    # 1. 策略+浓度图（无修改）
    fig, ax1 = plt.subplots(figsize=(14,6))
    # 绘制关停策略
    policy_line = ax1.step(hours, result["optimal_policy"], color="#2ca02c", linewidth=2.5, where="mid", label="Optimal Policy (1=Shut)")
    ax1.fill_between(hours, 0, result["optimal_policy"], alpha=0.2, color="#2ca02c")
    ax1.set_xlabel("Time (Hour)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Shutdown Policy (0=Open, 1=Shut)", fontsize=14, fontweight="bold")
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xticks(np.arange(0,24,2))
    ax1.grid(alpha=0.3, linestyle="--")

    # 双Y轴：浓度曲线
    ax2 = ax1.twinx()
    conc_line = ax2.plot(hours, intake_conc, color="#1f77b4", linewidth=2, label="Intake Concentration (mg/L)")
    threshold_line = ax2.axhline(y=params["stop_threshold"], color="#d62728", linestyle="--", linewidth=2, label="Stop Threshold (0.3 mg/L)")
    ax2.set_ylabel("Pollutant Concentration (mg/L)", fontsize=14, fontweight="bold", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")

    # 合并所有图例
    all_lines = policy_line + conc_line + [threshold_line]
    all_labels = [l.get_label() for l in all_lines]
    ax1.legend(all_lines, all_labels, loc="upper right", fontsize=12, framealpha=0.9)
    ax1.set_title(f"Velocity = {velocity} m/s - Optimal Strategy vs Concentration", fontsize=16, fontweight="bold", pad=15)
    plt.subplots_adjust(left=0.08, right=0.92, top=0.9, bottom=0.12)
    plt.savefig(f"task3_strategy_{velocity}m_s.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    # 2. 供需对比图（核心修改：需水量+10偏移）
    fig, ax = plt.subplots(figsize=(14,6))
    # 绘制需水量（红色）—— 添加10的偏移，让线分开显示
    demand_line = ax.plot(hours, hourly_demand + 10, color="#d62728", linewidth=2, label="Hourly Water Demand (m³)")
    # 绘制供水量（绿色）—— 无修改
    supply_line = ax.plot(hours, result["supply_detail"], color="#2ca02c", linewidth=2, label="Actual Water Supply (m³)")
    # 绘制供水缺口（粉色填充）
    gap_mask = hourly_demand > result["supply_detail"]
    if np.any(gap_mask):
        gap_fill = ax.fill_between(hours[gap_mask], result["supply_detail"][gap_mask], hourly_demand[gap_mask], 
                                  alpha=0.2, color="#ff9999", label="Supply Gap")
    else:
        gap_fill = []

    # 配置坐标轴和图例
    ax.set_xlabel("Time (Hour)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Water Volume (m³)", fontsize=14, fontweight="bold")
    ax.set_xticks(np.arange(0,24,2))
    # 合并供需图例
    all_supply_lines = demand_line + supply_line
    if gap_fill:
        all_supply_lines += gap_fill
    all_supply_labels = [l.get_label() for l in all_supply_lines]
    ax.legend(all_supply_lines, all_supply_labels, loc="upper right", fontsize=12, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_title(f"Velocity = {velocity} m/s - Hourly Water Demand vs Supply", fontsize=16, fontweight="bold", pad=15)
    plt.subplots_adjust(left=0.08, right=0.92, top=0.9, bottom=0.12)
    plt.savefig(f"task3_supply_{velocity}m_s.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

# ====================== 6. 敏感性分析图（无修改） ======================
def plot_sensitivity(summary_df):
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(16,12))
    # 1. 流速 vs 总目标成本
    ax1.plot(summary_df["velocity"], summary_df["total_cost"], marker='o', linewidth=3, color="#1f77b4", markersize=8)
    ax1.set_xlabel("Velocity (m/s)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Total Cost (Yuan)", fontsize=12, fontweight="bold")
    ax1.set_title("Velocity vs Total Optimal Cost", fontsize=14, fontweight="bold")
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.set_xticks(summary_df["velocity"])
    # 添加数值标注
    for x, y in zip(summary_df["velocity"], summary_df["total_cost"]):
        ax1.text(x, y+5000, f"{y:.0f}", ha='center', va='bottom', fontsize=10)

    # 2. 流速 vs 关停时长
    ax2.plot(summary_df["velocity"], summary_df["shut_duration"], marker='s', linewidth=3, color="#2ca02c", markersize=8)
    ax2.set_xlabel("Velocity (m/s)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Shutdown Duration (Hour)", fontsize=12, fontweight="bold")
    ax2.set_title("Velocity vs Shutdown Duration", fontsize=14, fontweight="bold")
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.set_xticks(summary_df["velocity"])
    # 添加数值标注
    for x, y in zip(summary_df["velocity"], summary_df["shut_duration"]):
        ax2.text(x, y+0.1, f"{y}", ha='center', va='bottom', fontsize=10)

    # 3. 流速 vs 风险时长
    ax3.plot(summary_df["velocity"], summary_df["risk_duration"], marker='^', linewidth=3, color="#d62728", markersize=8)
    ax3.set_xlabel("Velocity (m/s)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Risk Duration (Hour)", fontsize=12, fontweight="bold")
    ax3.set_title("Velocity vs Risk Duration", fontsize=14, fontweight="bold")
    ax3.grid(alpha=0.3, linestyle="--")
    ax3.set_xticks(summary_df["velocity"])
    # 手动调整Y轴范围
    ax3.set_ylim(-0.5, max(summary_df["risk_duration"])+1)
    # 添加数值标注
    for x, y in zip(summary_df["velocity"], summary_df["risk_duration"]):
        ax3.text(x, y+0.1, f"{y}", ha='center', va='bottom', fontsize=10)

    # 4. 流速 vs 供水损失率
    ax4.plot(summary_df["velocity"], summary_df["supply_loss"]*100, marker='d', linewidth=3, color="#9467bd", markersize=8)
    ax4.set_xlabel("Velocity (m/s)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Supply Loss Rate (%)", fontsize=12, fontweight="bold")
    ax4.set_title("Velocity vs Supply Loss Rate", fontsize=14, fontweight="bold")
    ax4.grid(alpha=0.3, linestyle="--")
    ax4.set_xticks(summary_df["velocity"])
    # 手动调整Y轴范围
    ax4.set_ylim(-0.1, max(summary_df["supply_loss"]*100)+0.5)
    # 添加数值标注
    for x, y in zip(summary_df["velocity"], summary_df["supply_loss"]*100):
        ax4.text(x, y+0.05, f"{y:.2f}%", ha='center', va='bottom', fontsize=10)

    plt.suptitle("Sensitivity Analysis: Velocity Impact on Optimal Strategy", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig("task3_sensitivity.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

# ====================== 7. 主函数（无修改） ======================
if __name__ == "__main__":
    DEMAND_CSV = "water_demand_profile.csv"
    VELOCITIES = [0.5, 0.8, 1.1, 1.4, 1.7, 2.0]
    params = get_fixed_params()
    summary_data = []

    try:
        for v in VELOCITIES:
            print(f"\n🔄 处理流速 {v} m/s...")
            conc_csv = f"task2_conc_{v}m_s.csv"
            intake_conc, hourly_demand = load_task_data(conc_csv, DEMAND_CSV)
            result = dp_optimal_strategy(intake_conc, hourly_demand, params)
            print_result_summary(v, result)
            plot_single_velocity(v, intake_conc, hourly_demand, result, params)
            summary_data.append({
                "velocity": v,
                "total_cost": result["total_cost"],
                "shut_duration": result["shut_duration"],
                "risk_duration": result["risk_duration"],
                "supply_loss": result["supply_loss"],
                "shut_cost": result["cost_breakdown"]["shut_cost"],
                "supply_loss_cost": result["cost_breakdown"]["supply_loss_cost"],
                "risk_cost": result["cost_breakdown"]["risk_cost"]
            })
            print(f"✅ 流速 {v} m/s 处理完成！")

        # 保存汇总表
        summary_df = pd.DataFrame(summary_data).sort_values("velocity").reset_index(drop=True)
        summary_df.to_csv("task3_velocity_summary.csv", index=False)
        print(f"\n📊 汇总表保存：task3_velocity_summary.csv")
        # 敏感性分析
        plot_sensitivity(summary_df)
        print(f"✅ 敏感性分析图保存：task3_sensitivity.png")

    except Exception as e:
        print(f"❌ 错误：{str(e)}")
        import traceback
        traceback.print_exc()