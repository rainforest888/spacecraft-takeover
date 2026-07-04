# Spacecraft Takeover — SAC Reinforcement Learning for Non-Cooperative Attitude Control

基于 SAC（Soft Actor-Critic）强化学习的**非合作航天器姿控接管**。服务航天器通过施加力矩消耗目标航天器的燃料，迫使其丧失姿控能力。目标航天器的**质量和剩余燃料对智能体不可知**，必须从动力学响应中隐式推断。

> 参考：师兄的 DDPG+LQR baseline（`ALTITUDE-FIGHT-3D-LQR/`），在此基础上增加了**未知质量估计**和**动态 MJCF 质量递减**。

## 训练结果

| 版本 | 成功率 | 关键改动 |
|------|--------|----------|
| v1 | 0% | 6-term reward，alpha 坍塌 |
| v2 | 46.5% | reward scaling + LQR only |
| v3 | 76-80%→坍塌 | MAX_TORQUE=5，信号归一化 |
| **v4** | **75.8%** | **固定 alpha=0.2，稳定收敛** |

最终：500 集训练，379/500 成功（75.8%），在 100-300 集区间达到 78-82%。

## 快速开始

### 环境要求

- Windows 10 + MSYS (git-bash)
- Conda 环境 `spacraft`：Python 3.11, MuJoCo 3.6.0, PyTorch 2.12
- GPU：RTX 5060 8GB（CPU 训练也可，但较慢）

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
# 500 集（约 1 小时）
"G:\Conda\envs\spacraft\python.exe" scripts/train_sac.py --episodes 500

# 3000 集完整训练（约 6 小时）
"G:\Conda\envs\spacraft\python.exe" scripts/train_sac.py --episodes 3000 --save-every 500
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

## 引用

- Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL", 2018
- 师兄的 DDPG+LQR baseline: `ALTITUDE-FIGHT-3D-LQR/`
- MuJoCo: https://mujoco.org/
