# CSE-PPO for ISAC 联合波束成形

本仓库实现用于 Liu2020 多用户 ISAC/DFRC 联合发射波束成形问题的 PPO 与 CSE-PPO。当前强化学习路径使用通信可行的 ZF-inspired 初始化，策略输出联合波束矩阵 `W=[Wc, Wr]` 的残差更新，并通过逐天线归一化保持功率约束。

ZF、SDR 仍可作为外部 baseline 输出；PPO 训练默认只使用 ZF-inspired 初始化作为可行起点，不调用 SDR。

## 当前强化学习建模

- 系统模型：`x = Wc c + Wr s`，其中 `Wc` 为 `M x K` 通信波束矩阵，`Wr` 为 `M x M` 雷达波束矩阵。
- 状态空间：包含通信信道矩阵 `H`、当前联合波束矩阵 `W`、归一化 SINR、完整雷达损失、各目标主瓣带内 MSE 和旁瓣泄漏。
- 动作空间：策略网络输出复值残差 `delta_W`。vanilla PPO 默认使用完整残差，CSE-PPO 默认使用结构化 null-space residual action。
- 动作执行：环境执行 `W <- row_power_normalize(W + step_scale * action_scale * delta_W)`，保证 `[W W^H]_{m,m}=Pt/M`。
- 初始化：RL episode 默认使用 `--init-mode policy`，即 ZF-inspired 可行初始化；也可显式设置 `--init-mode random` 做纯随机初始化对比。
- 雷达目标：完整 Liu2020 雷达损失 `Lr = Lr,1 + wc*Lr,2`，默认 `wc=1`。
- `alpha`：不进入动作空间。对给定 `W`，代码按闭式最优值计算 `alpha`。
- 奖励函数：

```text
reward = (prev_Lr - new_Lr) / max(abs(prev_Lr), eps)
```

通信约束 cost 为 `mean(max(Gamma - gamma_k, 0) / Gamma)`，进入 primal-dual PPO 的 Lagrangian reward。CSE/vanilla 的 line search 用 `Lr + constraint_score_weight * cost` 选择安全步长。

## ZF-inspired 初始化如何起作用

默认 `--init-mode policy` 下，每个 RL episode 的 `reset()` 不从随机 `W` 开始，而是先根据本 episode 采样到的信道 `H` 构造一个通信可行的初始联合波束矩阵 `W0=[Wc0, Wr0]`。这里的 `policy` 只表示“训练策略从该可行初始点出发”，不是加载预训练策略；`--init-mode zf` 与它使用同一套初始化逻辑。若显式指定 `--init-mode random`，才会回到随机初始化。

初始化流程在 `cse_ppo_isac/math_utils.py::zf_beamformer()` 中实现：

1. 对当前信道 `H` 计算伪逆 `Hplus = H^H (H H^H)^dagger`，并计算 `H` 的零空间基 `N`。
2. 根据 SINR 阈值 `Gamma`、噪声功率 `sigma^2` 和安全系数 `init_comm_safety` 设置通信目标增益：

```text
required_gain = sqrt(Gamma * sigma^2 * init_comm_safety)
Wc0 = Hplus * diag(required_gain, ..., required_gain)
```

这样 `H Wc0` 近似对角化：第 `k` 个用户主要接收自己的通信流，多用户干扰被压低。安全系数会让初始通信增益略高于阈值需求，给后续残差探索留出余量。

3. 雷达列 `Wr0` 优先放在通信信道零空间中。代码把目标方向 steering vector 投影到 `N` 上：

```text
Wr0[:, j] = N * (N^H * a(theta_j))
```

因为 `H N ≈ 0`，这些雷达波束列在理想情况下不会显著增加通信用户干扰，同时仍把雷达能量指向目标角附近。若零空间不可用，代码退化为直接使用目标 steering vector。

4. 代码不会只接受一个固定比例的 `Wc0/Wr0`。它会枚举通信缩放 `comm_scales = geomspace(0.7, 18.0, 18)` 和雷达缩放 `radar_scales = geomspace(0.15, 3.0, 10)`，组合成候选：

```text
W = row_power_normalize([cs * Wc0, rs * Wr0])
```

`row_power_normalize()` 会强制每根天线的发射功率满足 `[W W^H]_{m,m}=Pt/M`。随后代码计算每个候选的最小 SINR 裕量 `min_k(gamma_k-Gamma)` 和完整雷达损失 `Lr`。

5. 候选选择规则是：若存在满足所有用户 SINR 阈值的候选，选择其中 `Lr` 最小的一个；若没有可行候选，则选择 SINR 裕量最大的候选作为兜底。因此默认情况下 RL 从“通信已可行、雷达损失相对较低”的点开始，而不是在高维完整波束空间里盲搜可行解。

ZF-inspired 初始化只发生在 episode reset 阶段。进入 episode 后，策略网络输出的是残差 `delta_W`，环境执行：

```text
W <- row_power_normalize(W + step_scale * action_scale * delta_W)
```

line search 会在多个 `step_scale` 中选择 `Lr + constraint_score_weight * cost` 最小的安全更新。也就是说，ZF 提供可行起点，PPO/CSE-PPO 学的是如何在该起点附近通过残差更新改善雷达波束，同时尽量保持通信 SINR 可行。

这与 `--methods zf` 的含义不同：`--methods zf` 会把 ZF 作为一个独立 baseline 评估并保存；默认 PPO 训练中的 ZF-inspired 初始化只是环境 reset 的起点，不会作为一个额外方法写入结果，除非你显式把 `zf` 加到 `--methods`。

## CSE-PPO

CSE-PPO 在 vanilla PPO 上加入通信安全熵合并：

1. CSE-PPO 默认使用结构化 null-space residual action，减少破坏通信可行性的探索。
2. 计算策略熵的每动作维度值，并按最小 SINR 裕量加权：

```text
H_eff = H_policy / action_dim * sigmoid(beta * min_k(gamma_k-Gamma) / Gamma)
```

3. 严重不可行的早期训练阶段保留一个有效熵下限，避免通信安全熵把探索完全压没。
4. 每个 rollout 后，用 `H_eff` 的分位数自适应更新低熵阈值，再用 EMA 平滑。
5. 连续低有效熵 transition 会被合并为 macro-step，用累计 log-prob ratio 和累计 advantage 进入 PPO clipped objective。

vanilla PPO 与 CSE-PPO 使用同一个残差环境；CSE-PPO 额外启用 null-space residual action、通信可行性加权熵和低有效熵 macro-step 合并。

## 运行环境

推荐使用已有 conda 环境 `YH_RL`：

```bash
conda activate YH_RL
python -c "import torch, numpy; print(torch.__version__)"
```

如果当前 shell 无法 `conda activate`，可以直接使用解释器绝对路径：

```bash
/home/user/anaconda3/envs/YH_RL/bin/python run_experiment.py --help
```

基础依赖见 `requirements.txt`。若需要运行 SDR baseline，还需要 `cvxpy`。

## 快速验证

```bash
python -m py_compile run_experiment.py sweep_experiments.py cse_ppo_isac/*.py
python -m pytest tests -q
```

快速跑通 CSE-PPO：

```bash
python run_experiment.py --methods cse --algorithm cse \
  --K 2 --sinr-db 12 \
  --updates 1 --episodes-per-update 2 --eval-episodes 2 \
  --max-steps 2 --inference-candidates 2 \
  --rollout-backend serial --rollout-workers 1 \
  --device cpu --no-sdr
```

## 推荐训练命令

只训练并保存 CSE-PPO：

```bash
python run_experiment.py --methods cse --algorithm cse \
  --K 2 --sinr-db 12 \
  --updates 40 --episodes-per-update 128 --ppo-epochs 3 \
  --max-steps 8 \
  --eval-batch-size 128 --eval-episodes 256 \
  --inference-candidates 64 \
  --rollout-backend process --rollout-workers 8 \
  --device cuda --no-sdr
```

同时训练 vanilla PPO 与 CSE-PPO，用于纯 RL 对比：

```bash
python run_experiment.py --methods vanilla,cse --algorithm cse \
  --K 2 --sinr-db 12 \
  --updates 80 --episodes-per-update 128 --ppo-epochs 3 \
  --max-steps 8 \
  --eval-batch-size 128 --eval-episodes 256 \
  --inference-candidates 64 \
  --rollout-backend process --rollout-workers 8 \
  --device cuda --no-sdr
```

CPU 或调试环境可使用较小配置：

```bash
python run_experiment.py --methods vanilla,cse --algorithm cse \
  --K 2 --sinr-db 12 \
  --updates 20 --episodes-per-update 32 --ppo-epochs 3 \
  --max-steps 8 \
  --eval-batch-size 32 --eval-episodes 64 \
  --inference-candidates 16 \
  --rollout-backend process --rollout-workers 4 \
  --device cpu --no-sdr
```

如果要把 ZF/SDR 作为外部 baseline 一起输出，可以显式加入：

```bash
python run_experiment.py --methods zf,sdr,vanilla,cse --algorithm cse \
  --K 2 --sinr-db 12 \
  --updates 80 --episodes-per-update 128 --ppo-epochs 3 \
  --max-steps 8 \
  --eval-batch-size 128 --eval-episodes 128 \
  --inference-candidates 64 \
  --rollout-backend process --rollout-workers 8 \
  --device cuda --num-workers 4
```

这条命令中的 ZF/SDR 只用于结果对比和画图，不参与 PPO 训练。

## 批量实验

只 sweep 纯 RL 方法：

```bash
python sweep_experiments.py --algorithms cse,vanilla \
  --K-list 2,4,6 --sinr-db-list 4,8,12,16,20,24 \
  --updates 80 --episodes-per-update 128 \
  --eval-batch-size 128 --eval-episodes 128 \
  --device cuda --no-sdr
```

## 输出文件

每次运行都会创建：

```text
log/YYYYMMDD-HHMMSS/
```

主要文件：

- `config.json`：命令行参数、环境配置、PPO 配置和硬件信息。
- `summary.json`：本次运行摘要。
- `beamformers.npz`：固定信道下各方法保存的复值波束矩阵，包含 `H`。
- `patterns.npz`：角度网格、期望波束图和各方法发射波束图。
- `beamformer_metrics.json`：各方法对应的指标。
- `ppo_beamformers_csv/`：PPO 方法的波束矩阵 CSV，例如 `vanilla_beamformer.csv`、`cse_beamformer.csv`。
- `training_history/`：PPO 训练历史 CSV 与 reward 曲线。

`beamformers.npz` 中的矩阵尺寸为：

```text
W: M x (K+M)
Wc = W[:, :K]
Wr = W[:, K:]
R = W W^H
```

## 绘制波束图

训练完成后，用日志目录绘图：

```bash
python plot_beampatterns.py --input-dir log/YYYYMMDD-HHMMSS
```

只绘制部分方法：

```bash
python plot_beampatterns.py --input-dir log/YYYYMMDD-HHMMSS --methods vanilla,cse
```

输出包括：

- `beampattern.png/pdf`：发射波束图。
- `radar_terms.png/pdf`：`Lr,1`、`wc*Lr,2` 和完整 `Lr` 的损失分解。
- `metrics.json`：绘图对应指标。

波束图主要体现 `Lr,1`，也就是主瓣和旁瓣形状；`Lr,2` 是目标方向间交叉相关项，因此单独通过 `radar_terms.png/pdf` 展示。

## 主要指标

- `radar_loss`：完整 `Lr = Lr,1 + wc*Lr,2`，越小越好。
- `beam_objective`：当前等于 `radar_loss`。
- `beampattern_loss`：`Lr,1`，发射波束图 MSE，越小越好。
- `cross_corr`：`Lr,2`，目标方向交叉相关项，越小越好。
- `weighted_cross_corr`：`wc*Lr,2`。
- `sidelobe_ratio`：旁瓣最大值与目标角平均主瓣中心功率的比值，越小越好。
- `sidelobe_leakage`：旁瓣最大功率相对 Liu2020 最优缩放系数 `alpha` 的比值，越小越好。
- `target_mean`：目标角中心点平均发射功率。
- `target_min_ratio`：最弱目标峰值与平均目标峰值的比值，越接近 1 越均衡。
- `target_band_error_mean`：目标主瓣带内形状误差。
- `min_sinr` / `min_sinr_db`：用户最小 SINR。
- `feasible_rate`：满足全部用户 SINR 阈值的比例。
- `cost`：平均 SINR 违约代价。
- `entropy_threshold`：当前 update 自适应得到的低熵合并阈值。
- `low_entropy_rate`：当前 update 被判定为低有效熵的 transition 比例。

## 文件功能

- `cse_ppo_isac/config.py`：环境参数和 PPO 参数。
- `cse_ppo_isac/math_utils.py`：复数波束、SINR、雷达损失、功率归一化、ZF-inspired 初始化和外部 ZF baseline 工具。
- `cse_ppo_isac/env.py`：残差 RL 环境，负责 ZF-inspired/random reset、动作执行、reward 和指标。
- `cse_ppo_isac/actor_critic.py`：PPO 使用的 Actor-Critic 网络。
- `cse_ppo_isac/trainer.py`：PPO/CSE-PPO 训练器，包含自适应低熵合并。
- `cse_ppo_isac/baselines.py`：ZF/SDR 外部 baseline 评估，不参与 RL 训练。
- `run_experiment.py`：单组训练和保存波束矩阵。
- `sweep_experiments.py`：批量实验。
- `plot_beampatterns.py`：读取日志目录并绘制波束图与雷达损失分解。

## 注意事项

- 当前 RL 默认使用 ZF-inspired 初始化作为每个 episode 的可行起点，但不使用 ZF/SDR 参与 episode 内优化。若 `--methods` 中包含 `zf` 或 `sdr`，它们会作为独立 baseline 额外计算。
- PPO 对随机种子、训练轮数、batch 大小和 SINR 阈值敏感。正式实验建议固定 `--seed` 并多 seed 统计。
- `--rollout-workers` 不会改变样本数，只影响环境并行执行；实际 worker 数不会超过 `episodes_per_update`。
- GPU 主要加速策略网络，环境中的 SINR 和雷达损失计算仍主要在 CPU/NumPy 侧完成。矩阵规模较小时，合理的 CPU rollout 并行通常比单纯增大 GPU 更重要。
