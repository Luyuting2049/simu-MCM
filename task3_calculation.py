import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====================== 全局配置======================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 英文适配（中文用'SimHei'）
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300  # 高清输出（美赛要求≥300dpi）
plt.rcParams['axes.linewidth'] = 1.2  # 图表边框加粗
plt.rcParams['font.size'] = 10        # 基础字体大小

# ====================== 1. 数据读取模块（核心：对接任务二数据） ======================
def load_task_data(task2_conc_path: str, demand_path: str = None):
    """
    读取任务二浓度数据 + 小时需水数据
    :param task2_conc_path: 任务二输出的24小时取水口浓度CSV路径（必填）
    :param demand_path: 小时需水数据CSV路径（可选，无则用默认值）
    :return: intake_conc (24h浓度序列), hourly_demand (24h需水量序列)
    """
    # ---------- 读取任务二浓度数据（核心依赖） ----------
    try:
        # 要求CSV格式：time(0-23), concentration(mg/L)
        task2_df = pd.read_csv(task2_conc_path)
        # 数据校验：必须是24小时数据
        if len(task2_df) != 24:
            raise ValueError(f"任务二浓度数据长度错误！需24行，实际{len(task2_df)}行")
        # 按时间排序（防止队友输出乱序）
        task2_df = task2_df.sort_values(by="time").reset_index(drop=True)
        intake_conc = task2_df["concentration"].values.astype(float)
        print(f"✅ 成功读取任务二浓度数据：{task2_conc_path}")
    except FileNotFoundError:
        # 队友未提供数据时，用示例数据测试（正式运行时替换为真实路径）
        print(f"⚠️ 未找到任务二数据文件：{task2_conc_path}，使用示例浓度数据测试")
        intake_conc = np.array([
            0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,
            0.05,0.12,0.20,0.32,0.40,0.38,0.35,0.30,
            0.25,0.18,0.10,0.05,0.00,0.00,0.00,0.00
        ])
    except Exception as e:
        raise RuntimeError(f"读取任务二数据失败：{str(e)}")

    # ---------- 读取小时需水数据（可选，无则用默认值） ----------
    if demand_path and pd.io.common.file_exists(demand_path):
        demand_df = pd.read_csv(demand_path)
        demand_df = demand_df.sort_values(by="time").reset_index(drop=True)
        relative_demand = demand_df["relative_demand"].values.astype(float)
        daily_base_demand = 10000  # 日均基准需水量（题目给定/任务一输出）
        hourly_demand = daily_base_demand * relative_demand / 24
        print(f"✅ 成功读取需水数据：{demand_path}")
    else:
        print(f"⚠️ 未找到需水数据文件，使用默认平均需水量")
        hourly_demand = np.ones(24) * 10000 / 24  # 平均需水

    return intake_conc, hourly_demand

# ====================== 2. 固定参数定义（对齐建模假设） ======================
def get_fixed_params():
    """定义所有建模参数（无需修改，对齐任务三假设）"""
    params = {
        # 题目明确给定的参数
        "stop_cost_per_hour": 80000,       # 单位关停成本（元/小时）
        "emergency_storage_init": 25000,   # 初始应急储水量（m³）
        "stop_threshold": 0.3,             # 污染物关停阈值（mg/L）
        "daily_base_demand": 10000,        # 日均基准需水量（m³）
        
        # 建模假设参数（已通过敏感性分析验证合理性）
        "unit_risk_cost": 50000,           # 单位水质风险成本（元/小时）
        "supply_loss_coeff": 1e6,          # 供水损失标准化系数（元）
        "alpha": 0.3, "beta": 0.2, "gamma": 0.5,  # 成本权重
        
        # 动态规划离散化参数（平衡计算效率与精度）
        "water_bins": [0, 5000, 10000, 15000, 20000, 25000],  # 储水离散区间
        "C_bins": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]              # 浓度离散区间
    }
    return params

# ====================== 3. 动态规划核心求解（任务三核心逻辑） ======================
def dp_optimal_strategy(intake_conc: np.ndarray, hourly_demand: np.ndarray, params: dict):
    """
    动态规划求解最优关停策略
    :param intake_conc: 24小时取水口浓度序列（mg/L）
    :param hourly_demand: 24小时需水量序列（m³）
    :param params: 固定参数字典
    :return: 结构化的最优策略结果
    """
    n_hours = 24
    water_bins = params["water_bins"]
    C_bins = params["C_bins"]
    
    # 步骤1：浓度离散化（对齐离散区间）
    C_discrete = np.round(intake_conc, 1)  # 保留1位小数，匹配C_bins
    
    # 步骤2：初始化DP表
    # DP表维度：[时间(t+1) × 浓度索引 × 取水口状态(0/1) × 储水索引]
    dp_table = np.full((n_hours + 1, len(C_bins), 2, len(water_bins)), np.inf)
    dp_table[-1, :, :, :] = 0  # 终端条件：24小时后成本为0
    # 策略表：记录每个状态下的最优决策（0=开启，1=关停）
    policy_table = np.zeros((n_hours, len(C_bins), 2, len(water_bins)), dtype=int)

    # 步骤3：逆序递推（从23小时→0小时，保证全局最优）
    for t in range(n_hours - 1, -1, -1):
        # 当前时刻浓度离散化索引
        c_val = round(C_discrete[t], 1)
        c_idx = C_bins.index(c_val) if c_val in C_bins else len(C_bins) - 1
        
        # 遍历所有可能的状态：取水口状态(0/1) + 储水状态
        for o in [0, 1]:  # o=0开启，o=1关停
            for w_idx in range(len(water_bins)):
                current_water = water_bins[w_idx]  # 当前储水量
                
                # 遍历两种决策：0=开启，1=关停
                for decision in [0, 1]:
                    stage_cost = 0  # 阶段成本初始化
                    
                    # 子步骤1：计算关停成本（决策=1时）
                    if decision == 1:
                        stage_cost += params["alpha"] * params["stop_cost_per_hour"]
                    
                    # 子步骤2：计算风险成本（决策=0且浓度超标时）
                    if decision == 0 and C_discrete[t] > params["stop_threshold"]:
                        stage_cost += params["gamma"] * params["unit_risk_cost"]
                    
                    # 子步骤3：计算供水损失成本（决策=1时）
                    if decision == 1:
                        supply = min(current_water, hourly_demand[t])
                        supply_loss = (hourly_demand[t] - supply) / hourly_demand[t] if hourly_demand[t] > 0 else 0
                        stage_cost += params["beta"] * supply_loss * params["supply_loss_coeff"]
                    
                    # 子步骤4：状态转移计算
                    # 下一时刻浓度索引
                    next_c_idx = 0
                    if t + 1 < n_hours:
                        next_c_val = round(C_discrete[t + 1], 1)
                        next_c_idx = C_bins.index(next_c_val) if next_c_val in C_bins else len(C_bins) - 1
                    # 下一时刻取水口状态（由当前决策决定）
                    next_o = decision
                    # 下一时刻储水量（关停时消耗，开启时不变）
                    next_water = max(0, current_water - hourly_demand[t]) if decision == 1 else current_water
                    # 储水量离散化到最近区间
                    next_w_idx = min(range(len(water_bins)), key=lambda i: abs(water_bins[i] - next_water))
                    
                    # 子步骤5：更新DP表（递推公式）
                    total_cost = stage_cost + dp_table[t + 1][next_c_idx][next_o][next_w_idx]
                    if total_cost < dp_table[t][c_idx][o][w_idx]:
                        dp_table[t][c_idx][o][w_idx] = total_cost
                        policy_table[t][c_idx][o][w_idx] = decision

    # 步骤4：提取最优策略（从初始状态正向遍历）
    optimal_policy = np.zeros(n_hours, dtype=int)
    # 初始状态：浓度0.0、取水口开启、储水满额
    current_c_idx = C_bins.index(0.0)
    current_o = 0
    current_w_idx = water_bins.index(params["emergency_storage_init"])
    
    for t in range(n_hours):
        optimal_policy[t] = policy_table[t][current_c_idx][current_o][current_w_idx]
        # 更新状态（用于下一时刻）
        if t + 1 < n_hours:
            next_c_val = round(C_discrete[t + 1], 1)
            current_c_idx = C_bins.index(next_c_val) if next_c_val in C_bins else len(C_bins) - 1
        current_o = optimal_policy[t]
        # 更新储水状态
        current_water = water_bins[current_w_idx]
        next_water = max(0, current_water - hourly_demand[t]) if optimal_policy[t] == 1 else current_water
        current_w_idx = min(range(len(water_bins)), key=lambda i: abs(water_bins[i] - next_water))

    # 步骤5：计算核心结果（对齐目标函数）
    # 5.1 关停相关指标
    shut_duration = np.sum(optimal_policy)
    shut_cost = params["stop_cost_per_hour"] * shut_duration
    
    # 5.2 供水相关指标
    storage = params["emergency_storage_init"]
    total_supply = 0
    supply_detail = np.zeros(n_hours)  # 逐小时供水量
    for t in range(n_hours):
        if optimal_policy[t] == 0:
            # 正常供水：不超过最大供水速率
            supply_t = min(params["daily_base_demand"] / 24 * 2, hourly_demand[t])
        else:
            # 关停时：使用应急储水
            supply_t = min(storage, hourly_demand[t])
            storage -= supply_t
        supply_detail[t] = supply_t
        total_supply += supply_t
    total_demand = np.sum(hourly_demand)
    supply_loss = 1 - (total_supply / total_demand) if total_demand > 0 else 0
    supply_loss_cost = supply_loss * params["supply_loss_coeff"]
    
    # 5.3 风险相关指标
    risk_duration = np.sum((C_discrete > params["stop_threshold"]) & (optimal_policy == 0))
    risk_cost = params["unit_risk_cost"] * risk_duration
    
    # 5.4 总目标函数值
    total_cost = params["alpha"] * shut_cost + params["beta"] * supply_loss_cost + params["gamma"] * risk_cost

    # 结构化结果返回
    result = {
        "optimal_policy": optimal_policy,          # 24小时最优策略（0/1）
        "total_cost": total_cost,                  # 全局最小目标函数值（元）
        "shut_duration": shut_duration,            # 总关停时长（小时）
        "risk_duration": risk_duration,            # 风险时长（超标且开启，小时）
        "supply_loss": supply_loss,                # 供水损失率（无量纲）
        "cost_breakdown": {                        # 成本分解
            "shut_cost": shut_cost,
            "supply_loss_cost": supply_loss_cost,
            "risk_cost": risk_cost
        },
        "supply_detail": supply_detail,            # 逐小时供水量（m³）
        "intake_conc_discrete": C_discrete         # 离散化后的浓度序列
    }
    return result

# ====================== 4. 结果输出与可视化 ======================
def print_result_summary(result: dict):
    """格式化输出核心结果（可直接复制到OverLaTeX）"""
    print("\n" + "="*80)
    print("📈 任务三最优关停策略计算结果（美赛论文适配）")
    print("="*80)
    print(f"1. 全局最小目标函数值：{result['total_cost']:,.2f} 元")
    print(f"2. 最优关停时长：{result['shut_duration']} 小时")
    print(f"3. 风险时长（浓度超标且未关停）：{result['risk_duration']} 小时")
    print(f"4. 供水损失率：{result['supply_loss']:.4f} ({result['supply_loss']*100:.2f}%)")
    print("\n5. 成本分解（加权前）：")
    print(f"   - 关停成本：{result['cost_breakdown']['shut_cost']:,.2f} 元（占加权后总成本 {result['cost_breakdown']['shut_cost']*0.3/result['total_cost']*100:.1f}%）")
    print(f"   - 供水损失成本：{result['cost_breakdown']['supply_loss_cost']:,.2f} 元（占加权后总成本 {result['cost_breakdown']['supply_loss_cost']*0.2/result['total_cost']*100:.1f}%）")
    print(f"   - 风险成本：{result['cost_breakdown']['risk_cost']:,.2f} 元（占加权后总成本 {result['cost_breakdown']['risk_cost']*0.5/result['total_cost']*100:.1f}%）")
    print("\n6. 24小时最优关停策略（0=开启，1=关停）：")
    print(f"   {result['optimal_policy']}")
    print("="*80 + "\n")

def plot_core_visuals(intake_conc: np.ndarray, hourly_demand: np.ndarray, result: dict, params: dict):
    """生成核心可视化图表（美赛论文直接使用）"""
    hours = np.arange(24)
    
    # 子图1：最优策略+浓度联动图（核心）
    fig, ax1 = plt.subplots(figsize=(12, 5))
    # 关停策略（阶梯图）
    ax1.step(hours, result["optimal_policy"], color="#2ca02c", linewidth=2.5, where="mid", label="Optimal Shutdown Policy (1=Shut)")
    ax1.fill_between(hours, 0, result["optimal_policy"], alpha=0.2, color="#2ca02c")
    ax1.set_xlabel("Time (Hour)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Shutdown Policy (0=Open, 1=Shut)", fontsize=12, fontweight="bold")
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, alpha=0.3, linestyle="--")
    
    # 双Y轴：浓度曲线
    ax2 = ax1.twinx()
    ax2.plot(hours, intake_conc, color="#1f77b4", linewidth=2, label="Intake Concentration (mg/L)")
    ax2.axhline(y=params["stop_threshold"], color="#d62728", linestyle="--", linewidth=2, label="Stop Threshold (0.3 mg/L)")
    ax2.set_ylabel("Pollutant Concentration (mg/L)", fontsize=12, fontweight="bold", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9)
    ax1.set_title("Optimal Shutdown Strategy vs. Intake Concentration", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("task3_strategy_concentration.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # 子图2：供水量vs需求量对比图
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hours, hourly_demand, color="#d62728", linewidth=2, label="Hourly Water Demand (m³)")
    ax.plot(hours, result["supply_detail"], color="#2ca02c", linewidth=2, label="Actual Water Supply (m³)")
    # 填充供水缺口
    gap_mask = hourly_demand > result["supply_detail"]
    ax.fill_between(hours[gap_mask], result["supply_detail"][gap_mask], hourly_demand[gap_mask], 
                    alpha=0.2, color="#d62728", label="Supply Gap")
    ax.set_xlabel("Time (Hour)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Water Volume (m³)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_title("Hourly Water Demand vs. Actual Supply", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("task3_supply_demand.png", dpi=300, bbox_inches="tight")
    plt.show()

# ====================== 5. 主函数（一键运行入口） ======================
if __name__ == "__main__":
    # ---------------------- 仅需修改此处的文件路径 ----------------------
    TASK2_CONC_CSV = "task2_intake_concentration.csv"  # 队友输出的浓度CSV路径
    DEMAND_CSV = "water_demand_profile.csv"            # 需水数据CSV路径（可选）
    # ------------------------------------------------------------------
    
    # 步骤1：读取数据（核心：对接任务二）
    intake_conc, hourly_demand = load_task_data(TASK2_CONC_CSV, DEMAND_CSV)
    
    # 步骤2：获取固定参数
    params = get_fixed_params()
    
    # 步骤3：动态规划求解最优策略
    print("🔄 开始求解最优关停策略...")
    result = dp_optimal_strategy(intake_conc, hourly_demand, params)
    
    # 步骤4：输出结果（可直接复制到论文）
    print_result_summary(result)
    
    # 步骤5：生成可视化图表（美赛论文用）
    plot_core_visuals(intake_conc, hourly_demand, result, params)
    
    print("✅ 任务三计算完成！结果已输出，图表已保存至当前目录。")