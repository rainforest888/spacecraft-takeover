# Spacecraft Intelligent Attitude Takeover Control

> 空间非合作航天器的智能姿态接管控制研究 — 大学生创新创业训练计划

## Overview

TD3-LQR two-stage attitude takeover control of non-cooperative spacecraft using MuJoCo + PyTorch.

- **Phase 1 (Weakening)**: TD3 deep RL agent learns to deplete target spacecraft's fuel through adversarial torque strategies
- **Phase 2 (Takeover)**: LQR optimal controller stabilizes the combined body after target is exhausted
- **Physics**: MuJoCo 3.0, zero-G, gravity gradient torque in LEO
- **Environment**: Gymnasium interface, 11-dim obs, 3-dim continuous action

## Quick Start

```bash
# Create conda environment
conda create -n spacraft python=3.11 -y
conda activate spacraft
pip install -r requirements.txt

# Train TD3 agent
python scripts/train.py

# Evaluate with rendering + video
python scripts/demo.py
```

## Project Structure

```
spacecraft-takeover/
├── models/mjcf/        # MuJoCo model files
├── models/meshes/      # STL files (SolidWorks export)
├── envs/               # Gym environment
├── algorithms/         # TD3, LQR, switch manager
├── scripts/            # train, eval, demo, compare
├── outputs/            # checkpoints, logs, videos
└── tests/              # unit tests
```

## Requirements

- Python ≥ 3.10
- PyTorch (CUDA recommended)
- MuJoCo ≥ 3.0
- RTX 5060 or equivalent GPU (CPU training works but slow)
