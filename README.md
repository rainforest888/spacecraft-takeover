# Spacecraft Takeover — SAC Reinforcement Learning for Non-Cooperative Attitude Control

基于 **SAC（Soft Actor-Critic）** 强化学习的**非合作航天器姿控接管**。服务航天器（100kg，带爪型捕获机构）通过施加力矩消耗目标航天器的燃料，迫使其丧失姿控能力。目标航天器的**质量和剩余燃料对智能体不可知**，必须从动力学响应中隐式推断。

> Baseline 参考：师兄的 DDPG+LQR（`ALTITUDE-FIGHT-3D-LQR/`），本项目在此基础上增加了**未知质量估计**、**动态 MJCF 质量递减**、**爪型机构物理模型**和**消融分析框架**。

---

## 训练结果

### 最终结果：SAC V5 — 91.0% 训练 / 93% 评估

| 版本 | 算法 | 成功率 | 关键改动 |
|------|------|--------|----------|
| v1 | SAC | 0% | 6-term reward，alpha 坍塌 |
| v2 | SAC | 46.5% | reward scaling + LQR only |
| v3 | SAC | 76-80%→坍塌 | MAX_TORQUE=5，信号归一化 |
| v4 | SAC | 75.8% | 固定 alpha=0.2，稳定收敛 |
| v5 TD3 | TD3 | 75.6% | 爪型机构 + 效率奖励 |
| **v5 SAC** | **SAC** | **91.0% (eval 93%)** | **爪型机构 + LQR-only + 姿态惩罚 + MAX_TORQUE=7** |

**SAC V5 — 1000 集训练**：910/1000 成功（91.0%），确定性评估 93/100（93%）。

### 超参数扫描结果（100 集快速评估）

| 配置 | 成功率 | 说明 |
|------|--------|------|
| baseline (fb=10, sb=2, att=1) | 88.0% | 默认配置 |
| baseline h512 | 90.0% | 更大隐藏层 |
| att=5, omega=2, h512 | 57.0% | 过度惩罚姿态+角速度 |
| att=5, omega=2, 500步 | 74.0% | 缩短 episode |
| auto alpha, h512, omega | 71.0% | 自动调节 alpha（不稳定） |

**关键发现**：默认 reward 权重 (W_FUEL=10, W_SELF=2, W_ATT=1) 已经是最优配置。过度增加姿态/角速度惩罚会降低性能。

---

## 快速开始

### 环境要求

- Windows 10/11 + Git Bash (MSYS)
- Conda 环境 `spacraft`：Python 3.11, MuJoCo 3.6.0, PyTorch 2.12
- GPU：RTX 5060 8GB（CPU 训练也可用，但较慢）
- 可选：`imageio` + `pillow`（用于视频渲染）

```bash
# 创建环境
conda create -n spacraft python=3.11 -y
conda activate spacraft
pip install mujoco==3.6.0 torch numpy gymnasium imageio pillow
```

### 评估已有模型

```bash
cd "G:\claude code_workspace\spacecraft-takeover"

# 加载最佳模型，10 集确定性评估
python -c "
import sys; sys.path.insert(0,'.')
from envs.spacecraft_env_v5 import SpacecraftTakeoverEnvV5
from algorithms.sac_agent import SACAgent

agent = SACAgent(obs_dim=10, action_dim=3, hidden_dim=256, fixed_alpha=0.1)
agent.load('outputs/checkpoints/best_sac_v5.pt')
agent.actor.eval()

env = SpacecraftTakeoverEnvV5(max_steps=600)
successes = 0
for ep in range(10):
    obs, info = env.reset()
    for _ in range(600):
        obs, r, t, tr, info = env.step(agent.select_action(obs, deterministic=True))
        if t or tr: break
    success = info['target_fuel'] <= 0
    successes += success
    print(f'Ep {ep}: success={success}, steps={info.get(\"step\",\"?\")}, '
          f'att_err={info[\"attitude_error\"]:.2f}')
print(f'Success rate: {successes}/10')
env.close()
"
```

### 渲染演示视频

```bash
# 单角度 1080p 演示（含遥测叠加层）
python scripts/demo_v5.py --checkpoint outputs/checkpoints/best_sac_v5.pt

# 多角度 + 多种子批量渲染
python scripts/demo_v5_multi.py
```

### 训练

```bash
# SAC V5 训练（1000 集，约 2 小时 on RTX 5060）
python scripts/train_sac_v5.py --episodes 1000

# 自定义 reward 权重
python scripts/train_sac_v5.py --episodes 500 \
    --w-fuel-burn 10.0 --w-self-burn 2.0 --w-att 1.0

# 快速超参数扫描（100 集评估）
python scripts/sweep_one.py
```

---

## 项目结构

```
spacecraft-takeover/
├── envs/
│   ├── spacecraft_env_v5.py        # ★ V5 环境：爪型机构 + 动态质量 + Phase Switch
│   ├── spacecraft_env_v2.py        # V2-V4 环境（双刚体模型）
│   ├── spacecraft_env.py           # V1 原始环境
│   ├── dynamics.py                 # MRP/四元数转换 + 重力梯度力矩
│   ├── mass_estimator.py           # 最小二乘惯量估计（实验性）
│   ├── mass_change_detector.py     # 响应比变化检测（实验性）
│   └── fuel_predictor.py           # 燃料消耗预测（实验性）
│
├── algorithms/
│   ├── sac_agent.py                # ★ SAC + fixed-alpha 支持
│   ├── td3_agent.py                # TD3 备选算法
│   ├── target_controllers.py       # LQR / PID / SMC 目标策略
│   ├── lqr_controller.py           # LQR 控制器实现
│   └── switch_manager.py           # 策略切换管理器
│
├── scripts/
│   ├── train_sac_v5.py             # ★ SAC V5 训练入口
│   ├── train_sac.py                # SAC V2-V4 训练
│   ├── train_td3.py                # TD3 训练
│   ├── train.py                    # 通用训练入口
│   ├── demo_v5.py                  # ★ V5 演示渲染（1080p + 遥测叠加）
│   ├── demo_v5_multi.py            # ★ 多角度批量渲染
│   ├── demo_capture.py             # 捕获过程演示
│   ├── demo.py                     # 通用演示
│   ├── sweep_one.py                # 单次超参数扫描
│   ├── sweep_v2.py                 # 批量超参数扫描
│   ├── sweep_maxsteps_reward.py    # max_steps + reward 联合扫描
│   ├── eval.py                     # 模型评估
│   ├── eval_fuel_prediction.py     # 燃料预测评估
│   ├── compare.py                  # 模型对比
│   ├── analyze_failures.py         # Tier A: 失败案例分析
│   ├── run_ablation.py             # Tier B: 观测信号消融
│   ├── validate_mass_inference.py  # Tier C: 质量推断验证
│   ├── quick_diag.py               # 快速诊断脚本
│   └── quick_test_maxsteps.py      # max_steps 快速测试
│
├── models/mjcf/
│   ├── claw_body.xml               # ★ V5 爪型机构场景（服务星+4爪臂+目标星）
│   ├── combo_body.xml              # V2-V4 双刚体场景
│   └── capture_demo.xml            # 捕获演示场景
│
├── outputs/
│   ├── checkpoints/                # 模型检查点（.gitignore 排除）
│   │   ├── best_sac_v5.pt          # ★ 最佳 SAC V5 模型
│   │   └── sweep/                  # 扫描期间检查点
│   ├── logs/                       # 训练日志（CSV + sweep 结果）
│   │   ├── training_log.csv        # 主训练日志
│   │   ├── sweep_baseline.txt      # 超参数扫描结果
│   │   └── sweep/                  # 扫描详细日志
│   └── videos/                     # 渲染视频输出（.gitignore 排除）
│
├── tests/
│   ├── test_env.py                 # 环境单元测试
│   ├── test_td3.py                 # TD3 测试
│   └── test_switch.py              # Phase Switch 测试
│
├── ALTITUDE-FIGHT-3D-LQR/          # 师兄的 DDPG+LQR baseline（参考，不跟踪）
├── .gitignore
└── README.md
```

---

## 核心设计

### 问题建模

```
┌─────────────────────────────────────────────────┐
│                  Claw Mechanism                  │
│  ┌───────────────────────────────────────────┐  │
│  │           Service Satellite               │  │
│  │   mass = 100 kg                           │  │
│  │   ┌───┐ ┌───┐ ┌───┐ ┌───┐               │  │
│  │   │Arm│ │Arm│ │Arm│ │Arm│  ← 4 条爪臂    │  │
│  │   └───┘ └───┘ └───┘ └───┘               │  │
│  │         ┌─────────────┐                   │  │
│  │         │ Target Sat  │ ← 被包裹在爪笼内  │  │
│  │         │ 400-600 kg  │                   │  │
│  │         │ +100-150 kg │   燃料             │  │
│  │         └─────────────┘                   │  │
│  └───────────────────────────────────────────┘  │
│  weld 约束保持相对静止 → 施加力矩消耗对方燃料   │
└─────────────────────────────────────────────────┘
```

- **服务航天器（我方）**：质量 100kg，4 条爪臂形成笼状结构包裹目标星。3D 力矩控制 `[-MAX_TORQUE, +MAX_TORQUE]`，自身燃料有限
- **目标航天器（对方）**：干燥质量 400-600kg（每 100 集重新采样）+ 燃料 100-150kg（每集重新采样）。LQR 姿控策略。燃料对智能体不可观测
- **目标**：消耗对方燃料至 0，同时保持姿态稳定（`att_err < 2.0 rad ≈ 115°`）

### 观测空间（10 维）

```
obs = [sigma(3), omega(3), self_fuel(1), omega_dot(1), inertia_response(1), att_err(1)]
```

| 维度 | 符号 | 范围 | 说明 |
|------|------|------|------|
| 0-2 | `sigma` | ℝ³ | MRP 姿态参数（相对参考姿态） |
| 3-5 | `omega` | ℝ³ | 角速度 (rad/s) |
| 6 | `self_fuel` | [0, 1] | 自身剩余燃料比例 |
| 7 | `omega_dot` | [0, 1] | 归一化角加速度幅值 → 物理响应强度 |
| 8 | `inertia_response` | [0, 1] | `‖α‖ / ‖τ_self‖` → 质量隐式信号（质量大→响应小） |
| 9 | `att_err` | ℝ⁺ | 姿态误差范数 `‖sigma_err‖` |

**关键创新**：观测中**没有对方燃料或质量的任何直接信息**。Agent 必须从 `omega_dot` 和 `inertia_response` 中隐式推断目标航天器的剩余质量。

### 动作空间（3 维）

```
action ∈ [-1, +1]³  →  tau_self = action × MAX_TORQUE (7.0 N·m)
```

### 奖励函数

```
r = 0.01 × (
    W_FUEL_BURN × fuel_burned_kg          ← 消耗对方燃料（主线）
  - W_SELF_BURN × self_burn               ← 自身燃料惩罚
  - W_STEP                                ← 每步时间成本
  - W_ATT × attitude_error                ← 姿态误差惩罚（防翻滚）
  - W_OMEGA × ||omega||                   ← 角速度惩罚（可选，默认关闭）
  - W_MASS × (mass_error / dry_mass)²     ← 质量估计辅助（降权）
)

+ Phase Switch 奖励（检测到干重后）
+ 成功奖励（对方燃料耗尽）
+ 失败惩罚（自身燃料耗尽或翻滚）
```

**设计原则**：
- `0.01` 缩放确保 Q 值在合理范围，避免梯度爆炸
- 主线是简单的"消耗对方燃料"，对齐 DDPG+LQR baseline
- 质量估计作为降权辅助信号（W_MASS=0.5），不作为主驱动
- 姿态惩罚是连续信号（非二元），防止翻滚成为主要失败模式

### SAC 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `fixed_alpha` | 0.1 | **固定熵系数，不自动调节** |
| `log_std_min` | -5.0 | 最小标准差 ~0.007 |
| `actor_lr` | 3e-4 | Actor 学习率 |
| `critic_lr` | 3e-4 | Critic 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `tau` | 0.005 | 目标网络软更新系数 |
| `batch_size` | 256 | 批量大小 |
| `buffer_size` | 100000 | 经验回放缓冲区 |
| `hidden_dim` | 256 | 网络隐藏层维度 |
| `MAX_TORQUE` | 7.0 | 最大控制力矩 (N·m) |
| `FUEL_K_MASS` | 3.0 | 燃料消耗系数 (kg/N·m·s) |

**为什么固定 alpha？** 每集随机采样 fuel_mass（100-150kg），等同于每集一个新"任务"。SAC 熵自调节在变环境下会坍塌到 0，导致策略丧失适应能力。固定 alpha=0.1 保持持续探索，比 alpha=0.2（V4）更偏向 exploitation。

### Phase Switch 机制

智能体在训练中通过两种方式检测目标航天器质量：

1. **质量稳定检测**：当 EMA 质量估计在 10 步内波动 < 3kg 时触发
2. **累计力矩阈值**：当累计目标力矩 > 45 N·m·s 时（≈135kg 燃料消耗），作为后备触发

Phase Switch 后智能体获得奖励（R_DETECT=100），表示已成功识别目标干重。

---

## MJCF 物理模型

### V5：爪型机构（claw_body.xml）

```
服务星 (100kg, 蓝色方块)
  ├── arm_1..4: 4 条爪臂，通过 weld 约束连接
  │   形成笼状结构包裹目标星
  └── 爪臂质量轻（各 5kg），不影响整体动力学

目标星 (400-600kg + 100-150kg 燃料, 红色球体)
  └── 被包裹在爪笼内，weld 约束保持相对静止
```

### V2-V4：双刚体（combo_body.xml）

简化的双刚体模型，两颗卫星并排，通过 weld 约束连接。

---

## 消融与分析结果

### 环境更新（2026-07-04）

- 燃料范围从 50–200 kg 修正为 **100–150 kg**（符合实际航天器推进剂质量）
- 200 集训练 baseline 在该范围达到 **92.3%**

### Tier A：失败案例分析

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

### Tier B：观测信号消融（200 集 × 3 seeds）

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

### Tier C：质量推断验证

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

---

## 关键教训

### 1. 信号归一化必须验证

`inertia_response` 被 `/2000` 归一化后值约为 `1e-6`，对 400kg 和 600kg 航天器完全相同 → 网络学不到质量信息。

**修复**：计算原始值范围，选择分母使不同质量产生 >10% 的信号差异。

### 2. 奖励量级必须计算

W_FUEL=500 但实际燃料消耗仅 0.03/步，而 W_MASS=200 的质量误差惩罚为 -8/步。燃料信号被淹没 267 倍。

**修复**：简化为主线的燃料奖励 + 降权的辅助信号 + 0.01 缩放。

### 3. SAC 自动调节不适合变环境

每集不同的燃料质量 → 需要持续探索 → alpha 自动调节会坍塌。

**修复**：固定 alpha，关闭自动调节。V5 使用 alpha=0.1（比 V4 的 0.2 更偏向 exploitation）。

### 4. 简化优于复杂

V5 最关键的设计决策：
- **LQR-only 对手**（去掉 PID/SMC 的不可预测性）
- **简化奖励**（去掉效率奖励，只保留直接燃料消耗）
- **提高 MAX_TORQUE**（5→7，给智能体更强的控制能力）

从 V4 的 75.8% 提升到 V5 的 91.0%，这些简化带来的提升远大于复杂技巧。

---

## 引用

- Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning", ICML 2018
- Haarnoja et al., "Soft Actor-Critic Algorithms and Applications", 2019
- 师兄的 DDPG+LQR baseline: `ALTITUDE-FIGHT-3D-LQR/`
- MuJoCo Physics Engine: https://mujoco.org/
- Gymnasium: https://gymnasium.farama.org/
