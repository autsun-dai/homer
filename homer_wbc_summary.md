# Homer 全身控制（WBC）技术总结与迁移指南

## 一、问题建模：差分逆运动学（Differential IK）

Homer 的 WBC 基于**速度级别的差分 IK + QP 优化**，核心是在每个控制周期（100 Hz）求解：

$$\min_{\Delta q} \quad \sum_i \| W_i (J_i(q) \Delta q + \alpha_i e_i(q)) \|^2 + \lambda \|\Delta q\|^2$$

$$\text{s.t.} \quad G \Delta q \leq h$$

其中：
- $\Delta q \in \mathbb{R}^{10}$ 是全身关节速度（3 底盘 DoF + 7 机械臂 DoF）
- $J_i(q)$ 是第 $i$ 个任务的雅可比矩阵（从 MuJoCo 计算）
- $e_i(q)$ 是第 $i$ 个任务的误差向量
- $\alpha_i \in [0,1]$ 是任务增益（1.0 = 死拍控制）
- $W_i$ 是对角权重矩阵（cost 向量）
- $\lambda$ 是 Levenberg-Marquardt 阻尼系数

解出 $\Delta q$ 后乘以 $\frac{1}{dt}$ 得到速度，对 $q$ 积分更新构型，迭代至收敛（最多 20 次/步）。

---

## 二、任务约束定义（Task Stack）

Homer 使用 **3 个任务的加权叠加**，构成任务栈：

### 任务 1：末端执行器任务（FrameTask）— 主任务

```python
end_effector_task = mink.FrameTask(
    frame_name="pinch_site",   # MuJoCo 中末端位点名
    frame_type="site",
    position_cost=1.0,         # 位置误差权重
    orientation_cost=1.0,      # 姿态误差权重
    lm_damping=1.0,            # LM 阻尼（目标不可达时防抖）
)
```

**误差定义（6 维）**：

$$e = \log_{SE(3)}(T_{\text{current}}^{-1} \cdot T_{\text{target}})$$

即当前末端姿态到目标姿态的 SE(3) 对数映射（twist 向量），前 3 维为位置误差，后 3 维为旋转误差。

**雅可比**：$J \in \mathbb{R}^{6 \times n_v}$，由 MuJoCo `mj_jacSite` 计算并经坐标系变换获得。

---

### 任务 2：姿态任务（PostureTask）— 防奇异、拉回初始构型

```python
posture_cost = np.zeros(nv)
posture_cost[3:] = 1e-3        # 只约束机械臂（DoF 3~9），不约束底盘

posture_task = mink.PostureTask(model, cost=posture_cost)
posture_task.set_target_from_configuration(retract_config)  # 目标为收缩初始构型
```

**误差定义**：

$$e(q) = q^* \ominus q$$

使用 `mj_differentiatePos` 计算目标关节角与当前关节角之差（支持 SO(3) 关节的正确差值）。  
浮动基关节（底盘）的误差项被清零，保持底盘自由。

**雅可比**：$J = -I_{n_v}$（恒等矩阵取负号）

**作用**：当末端任务有多解时（冗余自由度），引导机械臂保持靠近初始收缩构型，代价极小（1e-3），不干扰主任务。

---

### 任务 3：阻尼任务（DampingTask）— 使底盘偏向静止

```python
immobile_base_cost = np.zeros(nv)
immobile_base_cost[:3] = 1.5   # 只对底盘 3 个 DoF 施加阻尼（x, y, θ）

damping_task = mink.DampingTask(model, immobile_base_cost)
```

等价于 `PostureTask(gain=0, target=qpos0)`，即最小化底盘速度。  
代价 1.5 >> 姿态任务 1e-3，意味着：**能用手臂解决就不移动底盘，手臂达不到时联动底盘**。

**最终任务列表**：

```python
tasks = [end_effector_task, posture_task, damping_task]
```

---

## 三、不等式约束定义

转化为 QP 不等式 $G\Delta q \leq h$：

### 约束 1：关节速度限制（VelocityLimit）

$$-v_{\max} \cdot dt \leq \Delta q \leq v_{\max} \cdot dt$$

Homer 中的具体值（参考）：

| 关节 | 最大速度 |
|------|----------|
| `joint_x`  | 0.5 m/s |
| `joint_y`  | 0.5 m/s |
| `joint_th` | π/2 rad/s |
| 机械臂 1~7 | 80°/s ~ 140°/s |

### 约束 2：关节范围限制（ConfigurationLimit）

$$q_{\min} \ominus q \leq \Delta q \leq q_{\max} \ominus q$$

- 自动从 MuJoCo URDF/XML 读取关节范围
- `gain=0.95`：每步最多移动 95% 的剩余裕量，防止压到极限

### 约束 3：碰撞避免（CollisionAvoidanceLimit，可选）

Homer 已实现但默认关闭（注释状态）：

```python
# self.limits = [velocity_limit, position_limit, collision_avoidance_limit]
self.limits = [velocity_limit, position_limit]  # 实际使用
```

---

## 四、QP 求解过程

所有任务汇聚成一个标准二次规划：

$$\min_{\Delta q} \quad \frac{1}{2} \Delta q^T H \Delta q + c^T \Delta q \qquad \text{s.t.} \quad G\Delta q \leq h$$

其中：

$$H = \sum_i J_i^T W_i^2 J_i + \mu_i I + \lambda I, \qquad c = \sum_i J_i^T W_i^2 (-\alpha_i e_i)$$

- $\mu_i = \lambda_{\text{lm}} \|W_i e_i\|^2$（LM 阻尼，误差大时自动增强正则化）
- Homer 使用 `quadprog` 求解器（活跃集法，适合小规模低维问题）

单步求解流程（`mink.solve_ik`）：

```
1. configuration.update(q)        # 执行前向运动学
2. 对每个 task 调用 compute_qp_objective() → 累加 H, c
3. 对每个 limit 调用 compute_qp_inequalities() → 累加 G, h
4. quadprog.solve_qp(H, c, G, h) → 返回 Δq
```

---

## 五、迁移步骤

### 步骤 1：准备机器人模型

- 创建包含底盘 + 机械臂的统一 MuJoCo XML/URDF 模型
- 底盘建模为 3 个 `slide`/`hinge` 关节（`joint_x`, `joint_y`, `joint_th`），位于根链接
- 在末端执行器处定义一个 `site`（如 `ee_site`）
- 确保各关节的 `range` 和 `damping` 参数正确设置

### 步骤 2：安装依赖

```bash
pip install mink mujoco quadprog
```

### 步骤 3：实例化 IK Solver

```python
import mink
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("your_robot.xml")
nv = model.nv  # 总 DoF 数

# 任务定义
end_effector_task = mink.FrameTask(
    "ee_site", "site",
    position_cost=1.0,
    orientation_cost=1.0,
    lm_damping=1.0,
)

posture_cost = np.zeros(nv)
posture_cost[3:] = 1e-3                           # 只约束机械臂
posture_task = mink.PostureTask(model, cost=posture_cost)

base_damping_cost = np.zeros(nv)
base_damping_cost[:3] = 1.5                        # 底盘阻尼
damping_task = mink.DampingTask(model, base_damping_cost)

tasks = [end_effector_task, posture_task, damping_task]

# 约束定义
velocity_limits = mink.VelocityLimit(model, {
    "joint_x": 0.5, "joint_y": 0.5, "joint_th": np.pi/2,
    "joint_1": np.deg2rad(80), ..., "joint_7": np.deg2rad(140),
})
position_limits = mink.ConfigurationLimit(model)
limits = [velocity_limits, position_limits]
```

### 步骤 4：每个控制周期调用 solve()

```python
def solve_wbc(pos_target, quat_target, curr_qpos, freq=100, max_iters=20):
    configuration = mink.Configuration(model)
    configuration.update(curr_qpos)

    # 设置初始构型目标（收缩姿态）
    posture_task.set_target_from_configuration(configuration)

    # 设置末端目标
    T_target = mink.SE3.from_rotation_and_translation(
        mink.SO3(quat_target), pos_target
    )
    end_effector_task.set_target(T_target)

    dt = 1.0 / freq
    for _ in range(max_iters):
        vel = mink.solve_ik(
            configuration, tasks, dt,
            solver="quadprog",
            damping=1e-3,
            limits=limits,
        )
        configuration.integrate_inplace(vel, dt)

        err = end_effector_task.compute_error(configuration)
        pos_err = np.linalg.norm(err[:3])
        rot_err = np.linalg.norm(err[3:])
        if pos_err < 1e-4 and rot_err < 1e-4:
            break

    return configuration.q  # [base_x, base_y, base_θ, arm_q1, ..., arm_q7]
```

### 步骤 5：坐标系处理（真实机器人关键）

策略输出的末端位置通常是**底盘局部坐标系**，需变换到世界坐标系传给 IK：

```python
import scipy.spatial.transform as transform

def local_to_world(pos_local, base_pose):
    x_b, y_b, theta_b = base_pose
    R = transform.Rotation.from_euler('z', theta_b).as_matrix()
    pos_world = R @ pos_local + np.array([x_b, y_b, base_height])
    return pos_world
```

### 步骤 6：执行器分离

WBC 求解出全身关节角后，按索引分发给底盘和机械臂控制器：

```python
qpos_solution = solve_wbc(...)
base_target = qpos_solution[:3]    # → 底盘控制器（Homer 用 Ruckig OTG）
arm_target  = qpos_solution[3:10]  # → 机械臂控制器（Homer 用 Ruckig + 顺从控制）
```

---

## 六、关键参数调参建议

| 参数 | Homer 值 | 调参方向 |
|------|----------|----------|
| `position_cost` | 1.0 | 增大 → 更重视末端位置精度 |
| `orientation_cost` | 1.0 | 减小 → 允许末端姿态偏差（提高可达性） |
| `posture_cost[arm]` | 1e-3 | 增大 → 机械臂更保守（接近初始构型） |
| `base_damping_cost` | 1.5 | 增大 → 底盘更不爱动（依赖手臂多） |
| `max_iters` | 20 | 增大 → 更精确但延迟更高 |
| `frequency` | 100 Hz | 与实际控制频率对齐 |
| `lm_damping` | 1.0 | 增大 → 目标不可达时更平滑（收敛慢） |
| `config_limit.gain` | 0.95 | 减小 → 关节远离极限但运动保守 |

---

## 七、Homer 文件结构速查

```
homer/
├── mink/
│   ├── configuration.py          # Configuration 类（封装 model/data，前向运动学）
│   ├── solve_ik.py               # solve_ik() 主入口，汇聚 tasks 和 limits 求解 QP
│   ├── tasks/
│   │   ├── task.py               # Task 基类，compute_qp_objective() 模板
│   │   ├── frame_task.py         # FrameTask（末端执行器任务）
│   │   ├── posture_task.py       # PostureTask（姿态回弹任务）
│   │   └── damping_task.py       # DampingTask（阻尼/静止任务）
│   └── limits/
│       ├── limit.py              # Limit 基类，compute_qp_inequalities() 模板
│       ├── velocity_limit.py     # VelocityLimit（速度上界约束）
│       └── configuration_limit.py# ConfigurationLimit（关节范围约束）
└── homer/
    ├── wbc_ik_solver_sim.py      # 仿真 WBC IK Solver（含完整参数）
    └── wbc_ik_solver_real.py     # 真实机器人 WBC IK Solver
```
