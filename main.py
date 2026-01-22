import yaml
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from PDESolver import RiverSolver
import pickle
import os

# ===== Load Configuration =====
class Config:
    def __init__(self, path='config.yaml'):
        with open(path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        self.N = 300  # 切成多少段
        self.L = cfg['river_reach']['L_km'] * 1000  # River length (m)
        self.mass = cfg['spill_event']['total_mass_kg']
        self.duration = cfg['spill_event']['release_duration_min']
        self.width = cfg['river_reach']['default_width_m']
        self.depth = cfg['river_reach']['default_depth_m']


def main():
    # =============================
    # 基本设置
    # =============================
    D = 50.0  # 固定扩散系数 (m^2/s)
    u_values = np.arange(0.5, 2.01, 0.3)  # 0.5, 0.8, ..., 2.0

    config = Config()  # 你的配置类
    intake_x = 30000.0  # 30 km
    T_hours = 10        # 模拟 10 小时

    plt.figure(figsize=(10, 6))

    # =============================
    # 循环不同流速
    # =============================
    for u in u_values:
        print(f"Solving for u = {u:.2f} m/s")

        solver = RiverSolver(config=config, u=u, D=D)
        sol = solver.solve(T_hours=T_hours)

        # 时间（分钟）
        t_min = sol.t / 60.0

        # 取水口索引
        intake_idx = np.argmin(np.abs(solver.x - intake_x))

        # 取水口浓度
        C_intake = sol.y[intake_idx, :]

        plt.plot(
            t_min,
            C_intake,
            linewidth=2,
            label=f"u = {u:.1f} m/s"
        )

    # =============================
    # 作图设置
    # =============================
    plt.xlabel("Time (minutes)", fontsize=12)
    plt.ylabel("Concentration (mg/L)", fontsize=12)
    plt.title(
        "Concentration–Time Curves at Intake Location (x = 30 km)",
        fontsize=14
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "VaryU",
        dpi=300,
        bbox_inches="tight"
        )

    plt.show()




        # =============================
    # 固定 u = 2，不同 D
    # =============================
    u_fixed = 2.0
    D_values = np.arange(10, 201, 10)

    plt.figure(figsize=(10, 6))

    for D in D_values:
        print(f"Solving for D = {D:.1f} m^2/s")

        solver = RiverSolver(config=config, u=u_fixed, D=D)
        sol = solver.solve(T_hours=T_hours)

        t_min = sol.t / 60.0
        intake_idx = np.argmin(np.abs(solver.x - intake_x))
        C_intake = sol.y[intake_idx, :]

        plt.plot(
            t_min,
            C_intake,
            linewidth=2,
            label=f"D = {D}"
        )

    plt.xlabel("Time (minutes)", fontsize=12)
    plt.ylabel("Concentration (mg/L)", fontsize=12)
    plt.title(
        "Concentration–Time Curves at Intake Location (x = 30 km, U = 2 m/s)",
        fontsize=14
    )
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    plt.savefig(
        "VaryD",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()



if __name__ == "__main__":
    main()

