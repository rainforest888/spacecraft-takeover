# Spacecraft Takeover — SAC Reinforcement Learning for Non-Cooperative Attitude Control

基于 SAC（Soft Actor-Critic）强化学习的**非合作航天器姿控接管**。服务航天器通过施加力矩消耗目标航天器的燃料，迫使其丧失姿控能力。目标航天器的**质量和剩余燃料对智能体不可知**，必须从动力学响应中隐式推断。

> 参考：师兄的 DDPG+LQR baseline（`ALTITUDE-FIGHT-3D-LQR/`），在此基础上增加了**未知质量估计**和**动态 MJCF 质量递减**。

## 训练结果

| 版本 | 算法 | 成功率 | 关键改动 |
|------|------|--------|----------|
| v1 | SAC | 0% | 6-term reward，alpha 坍塌 |
| v2 | SAC | 46.5% | reward scaling + LQR only |
| v3 | SAC | 76-80%→坍塌 | MAX_TORQUE=5，信号归一化 |
| v4 | SAC | 75.8% | 固定 alpha=0.2，稳定收敛 |
| v5 TD3 | TD3 | 75.6% | 爪型机构 + 效率奖励 |
| **v5 SAC** | **SAC** | **91.0% (eval 93%)** | **爪型机构 + LQR-only + 姿态惩罚 + MAX_TORQUE=7** |

最终：SAC V5 — 1000 集训练，910/1000 成功（91.0%），确定性评估 93/100（93%）。

## 快速开始

### MJCF 模型

V5 使用**爪型机构模型**（`models/mjcf/claw_body.xml`）：
- 服务星（100kg）+ 4 根爪臂形成笼状结构
- 目标星（400-600kg + 100-150kg 燃料）被包裹在爪笼内
- weld 约束保持相对静止状态

### 环境要求

- Windows 10 + MSYS (git-bash)
- Conda 环境 `spacraft`：Python 3.11, MuJoCo 3.6.0, PyTorch 2.12
- GPU：RTX 5060 8GB（CPU 训练也可，但较慢）

### 评估

```bash
# 加载最佳模型，100 集确定性评估
"G:\Conda\envs\spacraft\python.exe" -c "
import sys; sys.path.insert(0,'.')
from envs.spacecraft_env_v5 import SpacecraftTakeoverEnvV5
from algorithms.sac_agent import SACAgent
agent = SACAgent(obs_dim=10, action_dim=3, hidden_dim=256, fixed_alpha=0.1)
agent.load('outputs/checkpoints/best_sac_v5.pt')
agent.actor.eval()
env = SpacecraftTakeoverEnvV5(max_steps=600)
for ep in range(10):
    obs, info = env.reset()
    for _ in range(600):
        obs, r, t, tr, info = env.step(agent.select_action(obs, deterministic=True))
        if t or tr: break
    print(f'Ep {ep}: success={info[\"target_fuel\"]<=0}, steps={info.get(\"step\",\"?\")}')
env.close()
"
```

### 验证环境

```bash
cd "G:\claude code_workspace\spacecraft-takeover"

"G:\Conda\envs\spacraft\python.exe" -c "
import sys; sys.path.insert(0,'.')
from envs.spacecraft_env_v2 import SpacecraftTakeoverEnvV2
env = SpacecraftTakeoverEnvV2(max_steps=600)
obs, info = env.reset(seed=42)
print(f'dry_mass={info[\"dry_mass\"]:.0f}kg fuel={info[\"fuel_mass\"]:.0f}kg')
for i in range(600):
    obs, r, t, tr, info = env.step([0.8, -0.5, 0.3])
    if info.get('phase_switched'):
        print(f'Phase switch at t={i}, est_mass={info[\"est_mass\"]:.0f}kg')
        break
env.close()
"
```

### 训练

```bash
# V5 SAC 训练（1000 集，约 2 小时）
"G:\Conda\envs\spacraft\python.exe" scripts/train_sac_v5.py --episodes 1000

# V5 TD3 训练（备选）
"G:\Conda\envs\spacraft\python.exe" scripts/train_td3.py --episodes 1000
```

## 项目结构

```
spacecraft-takeover/
├── envs/
│   ├── spacecraft_env_v2.py      # 主环境：动态质量 + Phase Switch
│   ├── dynamics.py               # MRP/四元数/重力梯度
│   ├── fuel_predictor.py         # 燃料预测（实验性）
│   ├── mass_estimator.py         # LS 惯量估计（实验性）
│   └── mass_change_detector.py   # 响应比检测（实验性）
├── algorithms/
│   ├── sac_agent.py              # SAC + fixed-alpha 支持
│   ├── target_controllers.py     # LQR/PID/SMC 目标策略
│   ├── td3_agent.py              # TD3 备选
│   └── lqr_controller.py         # LQR 实现
├── scripts/
│   ├── train_sac.py              # 训练入口
│   └── eval_fuel_prediction.py   # 燃料预测评估
├── models/mjcf/
│   └── combo_body.xml            # MuJoCo 场景（双刚体）
├── ALTITUDE-FIGHT-3D-LQR/        # 师兄的 DDPG+LQR 参考
├── outputs/                      # 输出目录
│   ├── checkpoints/              # 模型检查点
│   ├── logs/                     # 训练日志
│   └── videos/                   # 渲染视频
└── tests/                        # 单元测试
```

## 核心设计

### 问题建模

- **服务航天器（我方）**：质量 100kg，3D 力矩控制，自身燃料有限
- **目标航天器（对方）**：质量 400-600kg（干燥）+ 50-200kg（燃料），LQR 姿控，燃料不可观测
- **目标**：消耗对方燃料至 0，同时保持姿态稳定

### 观测空间（10 维）

```
[sigma(3), omega(3), self_fuel(1), omega_dot(1), inertia_response(1), att_err(1)]
```

- `sigma`：MRP 姿态参数
- `omega`：角速度
- `self_fuel`：自身剩余燃料（0-1）
- `omega_dot`：归一化角加速度幅值 — 物理响应强度
- `inertia_response`：`||alpha|| / ||tau_self||` — 质量隐式信号（质量大→响应小）
- `att_err`：姿态误差

**关键创新**：观测中**没有对方燃料或质量的任何直接信息**。Agent 必须从 `omega_dot` 和 `inertia_response` 中隐式推断目标航天器的剩余质量。

### 奖励函数

```
r = 0.01 × (10.0 × ΔFb_target - 1.0 × ΔFs_self - 0.01)   # 主线：燃料消耗
  - 0.01 × 0.5 × ((est_mass - dry_mass) / dry_mass)²       # 辅线：质量估计
  + 成功奖励 / Phase Switch奖励 / 失败惩罚
```

设计对齐 DDPG+LQR baseline 的简洁燃料导向奖励。质量估计作为降权的辅助信号（0.5 vs 原 200）。0.01 倍缩放确保 Q 值在合理范围。

### SAC 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `fixed_alpha` | 0.2 | **固定熵系数，不自动调节** |
| `log_std_min` | -5.0 | 最小标准差 ~0.007 |
| `actor_lr` | 3e-4 | |
| `critic_lr` | 3e-4 | |
| `gamma` | 0.99 | |
| `tau` | 0.005 | |
| `batch_size` | 256 | |
| `buffer_size` | 100000 | |
| `hidden_dim` | 256 | |

**为什么固定 alpha？** 每集随机采样 fuel_mass（50-200kg），等同于每集一个新"任务"。SAC 熵自调节在变环境下会坍塌到 0，导致策略丧失适应能力。固定 alpha=0.2 保持持续探索。

## 关键教训

### 1. 信号归一化必须验证

`inertia_response` 被 `/2000` 归一化后值约为 `1e-6`，对 400kg 和 600kg 航天器完全相同 → 网络学不到质量信息。

**修复**：计算原始值范围，选择分母使不同质量产生 >10% 的信号差异。

### 2. 奖励量级必须计算

W_FUEL=500 但实际燃料消耗仅 0.03/步，而 W_MASS=200 的质量误差惩罚为 -8/步。燃料信号被淹没 267 倍。

**修复**：简化为主线的燃料奖励 + 降权的辅助信号 + 0.01 缩放。

### 3. SAC 自动调节不适合变环境

每集不同的燃料质量 → 需要持续探索 → alpha 自动调节会坍塌。

**修复**：固定 alpha=0.2，关闭自动调节。

## 消融与分析结果（2026-07-04）

### 环境更新

- 燃料范围从 50–200 kg 修正为 **100–150 kg**（符合实际航天器推进剂质量）
- 200 集训练 baseline 在该范围达到 **92.3%**（原 75.8% 使用 50–200 kg 范围）

### Tier A: 失败案例分析

| 维度 | 成功 (75.8%) | 失败 (24.2%) |
|------|-------------|-------------|
| 平均干重 | 510 kg | 508 kg |
| 平均 episode 长度 | 600 步 | 600 步 |
| 相位切换率 | **100%** | **30.6%** |
| 平均奖励 | 575.1 | 13.7 |

**关键发现**：
- **相位切换是最强区分信号**：成功 episode 100% 触发切换，失败仅 30.6%
- 干重对成功率几乎无影响（r=0.02）
- 奖励在成功/失败之间差 40 倍（575 vs 14）

### Tier B: 观测信号消融（200 集 × 3 seeds）

| 变体 | 成功率 | 相对 Baseline |
|------|--------|--------------|
| B0: Full (baseline) | **92.3%** ± 3.4% | — |
| B1: No-ω̇_dot | 94.2% ± 4.2% | **+1.8 pp** ↑ |
| B2: No-IR | 82.3% ± 6.2% | −10.0 pp ↓ |
| B3: No-AttErr | **72.0%** ± 9.2% | **−20.3 pp** ↓↓ |
| B4: No-mass-signals | **94.0%** ± 0.7% | **+1.7 pp** ↑ |

**关键发现**：
- **`att_err`（姿态误差）是最重要的信号**：去掉后成功率暴跌 20.3 pp
- **`inertia_response` 中等重要**：去掉后降 10 pp
- **`omega_dot` 单独无效**：去掉反而略提升 (+1.8 pp)
- **两个质量信号同时去掉反而更好** (+1.7 pp, 标准差 0.7%)：说明智能体主要靠 `att_err` 驱动策略

### Tier C: 质量推断验证

| 实验 | 结果 |
|------|------|
| C1 质量扫描 | 燃料越高成功率越低；500kg+150kg 仅 30% |
| C2 信号扰动 | 随机化 `inertia_response` 后成功率 **升至 92%**（vs baseline 74%）|
| C3 Oracle 对比 | Oracle（直接观测质量）76% vs 盲模型 74% — **仅差 2 pp** |

**关键发现**：
- 智能体**不依赖 `inertia_response` 推断质量**（扰动后反而更好）
- 直接给质量信息（Oracle）几乎不提升性能（+2pp）→ **当前策略不依赖隐式质量推断**
- 策略本质上是"姿态稳定 + 持续输出扭矩消耗对方燃料"，而非基于质量推理

### 核心结论

1. **姿态误差是决定性信号**，不是质量推断
2. 质量相关信号（`omega_dot`, `inertia_response`）在当前策略中**几乎不被使用**
3. 下一步改进方向：**重新设计奖励函数**，让质量推断成为任务必要条件（如缩短 episode 时间限制、增加 self_fuel 消耗惩罚），而非依赖当前的"600步持续输出"策略

## 引用

- Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL", 2018
- 师兄的 DDPG+LQR baseline: `ALTITUDE-FIGHT-3D-LQR/`
- MuJoCo: https://mujoco.org/
