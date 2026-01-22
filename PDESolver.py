
import numpy as np
from scipy.integrate import solve_ivp

class RiverSolver:
    def __init__(self, config, u, D):
        self.config = config
        self.u = u  # 流速 (m/s)
        self.D = D  # 扩散系数 (m²/s)
        self.N = config.N
        self.L = config.L

        self.dx = self.L/(self.N-1)
        self.x = np.linspace(0, self.L, self.N)
        
        # 🟢 修复：计算源项浓度
        self.Q = self.u * config.width * config.depth  # 流量 (m³/s)
        
        # 总质量释放速率 (kg/s)
        total_mass_kg = config.mass
        release_duration_s = config.duration * 60  # 分钟转秒
        self.mass_rate_kg_per_s = total_mass_kg / release_duration_s
        
        # 边界浓度计算 (mg/L = g/m³)
        # 注意：1 mg/L = 1 g/m³
        if self.Q > 0:
            self.C_boundary = (self.mass_rate_kg_per_s * 1000) / self.Q  # g/m³ = mg/L
        else:
            self.C_boundary = 0.0
        
        print(f"调试信息: u={u} m/s, Q={self.Q:.2f} m³/s, C_boundary={self.C_boundary:.6f} mg/L")
        
        # 初始条件
        self.C0 = np.zeros(self.N)

    def rhs(self, t, C):
        dCdt = np.zeros_like(C)
        
        if t <= 1800:  # 前30分钟（1800秒）
            C_left = self.C_boundary
        else:
            C_left = 0.0
        
        # 左边界处理 - 使用Dirichlet边界条件
        # 对于对流扩散方程，左边界应该直接设为C_left
        # 方法：在每次计算前设置C[0] = C_left
        C[0] = C_left
        
        # 🟢 更简洁的内部点计算（向量化）
        # 对流项：一阶迎风差分（因为u>0）
        # dC/dt = -u * (C_i - C_{i-1})/dx + D * (C_{i+1} - 2C_i + C_{i-1})/dx²
        
        convection = -self.u * (C[1:-1] - C[0:-2]) / self.dx
        diffusion = self.D * (C[2:] - 2*C[1:-1] + C[0:-2]) / (self.dx**2)

        dCdt[1:-1] = convection + diffusion
        
        # 🟢 右边界：零梯度边界条件
        # 方法1：设置 dC/dt[N-1] = 0，并在每一步确保 C[N-1] = C[N-2]
        # 方法2（更稳定）：设置右边界为C[N-2]的镜像点
        
        # 使用镜像方法处理右边界
        i = self.N-1  # 最后一个点
        # 对流项：假设u在边界处为正，使用迎风格式
        convection_last = -self.u * (C[i] - C[i-1]) / self.dx
        
        # 扩散项：使用零梯度条件 C[i+1] = C[i-1]（镜像）
        diffusion_last = self.D * (C[i-1] - 2*C[i] + C[i-1]) / (self.dx**2)
        
        dCdt[i] = convection_last + diffusion_last
        
        # 🟢 左边界：保持Dirichlet条件
        dCdt[0] = 0  # 左边界浓度由边界条件设定，不随时间变化
        
        return dCdt

    def solve(self, T_hours=10):
        T_seconds = T_hours * 3600
        
        # 🟢 添加事件检测：监测取水口浓度
        def peak_event(t, C):
            intake_idx = np.argmin(np.abs(self.x - 30000))  # 30km处
            return C[intake_idx] - 0.001  # 当浓度>0.001时触发
        
        peak_event.terminal = False  # 不终止计算
        peak_event.direction = 1     # 只检测上升沿
        
        sol = solve_ivp(
            fun=self.rhs,
            t_span=(0, T_seconds),
            y0=self.C0,
            method='RK45',
            max_step=0.5*self.dx/abs(self.u + 2*self.D/self.dx),  # CFL条件
            rtol=1e-6,
            atol=1e-9,
            dense_output=True  # 允许插值
        )
        
        # 🟢 确保边界条件在解中正确
        # 重新应用左边界条件
        for i in range(len(sol.t)):
            if sol.t[i] <= 1800:
                sol.y[0, i] = self.C_boundary
            else:
                sol.y[0, i] = 0.0
        
        return sol
