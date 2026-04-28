# HE-PPO for ISAC Beamforming

本仓库实现 **HE-PPO: High-Entropy-step-aware PPO for ISAC Beamforming**。当前代码已经收敛为一个纯 RL 训练框架：环境随机初始化信道和波束矩阵，策略只输出 full residual action，训练阶段不使用传统优化初始化、搜索候选、动作拒绝或可行性投影。

系统设置固定为 Liu2020 多用户 ISAC/DFRC 联合发射模型：

```text
M = 10
K = 2
target angles = [-40, 0, 40] deg
mainlobe width = 10 deg
Pt = 1
noise power = 0.01
angle grid = [-90, 90] deg, step = 0.1 deg
per-antenna power: [W W^H]_{m,m} = Pt / M
```

发射矩阵写作：

```text
W = [W_c, W_r] in C^{M x (K+M)}
```

其中 `W_c` 是通信列，`W_r` 是雷达列。训练目标是在通信 SINR 约束下优化雷达波束图匹配和目标方向交叉相关。

## 方法概览

当前只训练并比较两个方法：

| 方法 | 初始化 | 动作空间 | 更新方式 | 作用 |
| --- | --- | --- | --- | --- |
| `ppo` | 随机信道、随机波束矩阵 | full residual action | 每个 transition 独立进入 PPO clipped update | 标准 actor-critic PPO |
| `heppo` | 与 `ppo` 完全相同 | 与 `ppo` 完全相同 | 高熵 step 独立更新，连续低熵 step 合并为 macro-transition | 本文方法 |

两者公平对比时保持一致：

```text
same random initialization rule
same full residual action
same state
same reward
same actor-critic network
same learning rate
same batch size
same PPO epochs
same fixed evaluation channels
same deterministic inference
```

唯一差异在 PPO update 阶段：

```text
PPO:
  all transitions are optimized independently.

HE-PPO:
  high-entropy transitions are optimized individually;
  consecutive low-entropy transitions are consolidated into macro-transitions.
```

## 代码结构

```text
run_train.py
run_eval.py
run_plot.py

isac_rl/
  config.py
  env.py
  metrics.py
  policy.py
  buffer.py
  ppo.py
  heppo.py
  entropy_macro.py
  trainer.py
  logger.py
  plotting.py
  utils.py

tests/
  test_buffer.py
  test_metrics.py
  test_env.py
  test_policy.py
  test_entropy_macro.py
  test_smoke_train.py
```

训练核心文件只包含纯 RL 逻辑：

```text
isac_rl/env.py
isac_rl/trainer.py
isac_rl/ppo.py
isac_rl/heppo.py
```

仓库内不再保留传统优化对比方法的实现文件。`run_eval.py` 只评估已经训练好的 RL checkpoint。

## 训练加速

当前训练仍是单进程、单 GPU 路线，不在一次实验里同时调用两张 GPU。加速来自三点：

```text
1. MetricCache 缓存固定 angle grid、steering matrix、target steering 和 desired pattern。
2. collect_rollout 在一个 update 内同时创建 episodes_per_update 个环境。
3. 每个 step 把所有 state stack 成 batch，只做一次 policy forward，再用 CPU 线程池并行执行 env.step。
```

因此 `episodes_per_update` 越大，GPU MLP 前向的 batch 越大；同时 `eval_interval` 控制评估频率，避免每个 update 都跑大量 eval channel。

`run_train.py` 提供单 GPU 加速参数：

```text
--eval-interval N       每 N 个 update 评估一次
--torch-threads N       设置 torch CPU 线程数
--allow-tf32            在支持的 CUDA GPU 上打开 TF32
```

如果机器有多张 GPU，推荐分别启动独立实验，并用 `CUDA_VISIBLE_DEVICES` 只暴露一张卡给每个进程；本仓库不需要 DataParallel 或跨卡梯度同步。

## 环境

`ISACBeamformingEnv.reset()` 只做随机初始化：

```text
H ~ CN(0, 1)
W ~ random complex matrix
W <- per_antenna_power_normalize(W, Pt)
```

环境不做以下操作：

```text
pseudo-inverse initialization
convex-relaxation initialization
null-space initialization
line search
candidate selection
action rejection
no-op candidate
component guard
communication feasibility projection
```

`step(action)` 只执行策略动作：

```text
Delta_W = decode_complex_residual_action(action)
W_next = W + action_scale * Delta_W
W_next = per_antenna_power_normalize(W_next, Pt)
```

逐天线功率归一化是物理约束，不是启发式搜索。它保证：

```text
[W_next W_next^H]_{m,m} = Pt / M
```

## 状态空间

状态由六类特征拼接：

```text
s_t = [
  channel feature,
  current beamformer feature,
  SINR feature,
  radar scalar feature,
  coarse beampattern feature,
  progress/time feature
]
```

### Channel Feature

信道按自身平均功率归一化：

```text
H_norm = H / sqrt(mean(abs(H)^2) + eps)
feature = concat(real(H_norm), imag(H_norm))
```

维度为 `2KM`。

### Beamformer Feature

当前波束矩阵按逐天线目标幅度归一化：

```text
W_norm = W / sqrt(Pt / M)
feature = concat(real(W_norm), imag(W_norm))
```

维度为 `2M(K+M)`。

### SINR Feature

每个用户的 SINR 计算包含通信信号、用户间干扰、雷达列干扰和噪声。状态中记录：

```text
gap_db = SINR_db - Gamma_db
violation = softplus((Gamma_db - SINR_db) / tau_sinr)
```

并拼接每用户 gap、每用户 violation、最小 gap、平均 violation 和 feasible flag。

### Radar Scalar Feature

雷达标量特征包含：

```text
log1p(Lr1_cw / Lr1_ref)
log1p(Lr2 / Lr2_ref)
log1p(Lr / Lr_ref)
log1p(C_sinr)
```

其中：

```text
Lr1_cw = center-weighted beampattern matching error
Lr2 = target-direction cross-correlation
Lr  = Lr1_cw + w_cross * Lr2
```

### Coarse Beampattern Feature

只给策略几个 scalar 很难学出清晰波束图，因此状态额外加入粗粒度角度网格误差：

```text
coarse grid = [-90, -88, ..., 90] deg
P_coarse = beampattern(W, coarse grid)
d_coarse = desired pattern(coarse grid)
error = clip(P_coarse / alpha - d_coarse, -5, 5)
```

### Progress Feature

进度特征包含：

```text
prev_reward
objective improvement
SINR-cost improvement
EMA reward
t / T
(T - t) / T
```

## 动作空间

主实验使用 full residual action：

```text
action_dim = 2 * M * (K + M)
```

动作拆成复数 residual：

```text
half = M * (K + M)
A_real = action[:half].reshape(M, K+M)
A_imag = action[half:].reshape(M, K+M)
Delta_W = A_real + j A_imag
```

执行：

```text
W_next = W + action_scale * Delta_W
W_next = per_antenna_power_normalize(W_next, Pt)
```

默认：

```text
action_scale = 0.03
episode_steps = 8
```

## 策略网络

actor 使用 tanh Gaussian：

```text
raw_action ~ Normal(mu, std)
action = tanh(raw_action)
```

训练时 log-prob 使用 tanh correction：

```text
log_prob = Normal.log_prob(raw_action).sum()
log_prob -= log(1 - action^2 + 1e-6).sum()
```

HE-PPO 的 step entropy 和 PPO entropy bonus 使用 tanh 后的有效 entropy，而不是 raw Normal entropy：

```text
effective_entropy =
  Normal.entropy()
  + log(1 - tanh(mu)^2 + 1e-6)
```

这样 entropy 不再只由全局 `log_std` 决定，也会随当前 state 下的 mean action 饱和程度变化。若 `mu` 使 `tanh(mu)` 接近 `-1` 或 `1`，有效 entropy 会下降；若动作均值处于未饱和区域，有效 entropy 更高。

评估时使用 deterministic inference：

```text
action = tanh(mu)
```

critic 输出 `V(s_t)`，用于 GAE 和 value loss。

## 奖励函数

训练 reward 保持简洁，只围绕 Liu2020 雷达目标、通信 SINR 软约束和少量 progress shaping。主瓣对准不作为额外 reward 项，而是通过 center-weighted beampattern MSE 融入原始 beampattern matching loss。

center-weighted beampattern MSE 为：

```text
Lr1_cw =
  sum_l q_l |alpha d(theta_l) - P(theta_l)|^2
  / sum_l q_l

q_l = 1 + center_weight * sum_p exp(
  -0.5 * ((theta_l - theta_p) / center_sigma_deg)^2
)
```

默认中心权重：

```text
center_weight = 4.0
center_sigma_deg = 2.0
```

SINR cost：

```text
gap_db_k = Gamma_db - SINR_db_k
C_sinr = mean(softplus(gap_db_k / 2)^2)
```

训练 objective：

```text
Lr = Lr1_cw + w_cross * Lr2
J = w_radar * log1p(Lr / Lr_ref) + w_sinr * C_sinr
```

每步 reward：

```text
progress = (J_prev - J_current) / (abs(J_prev) + eps)

reward =
  -tanh(J_current / objective_scale)
  + progress_weight * tanh(progress / progress_scale)
```

终止步额外加入：

```text
reward += -terminal_weight * tanh(J_current / objective_scale)
reward = clip(reward, -5, 5)
```

默认：

```text
objective_scale = 10
progress_weight = 1
progress_scale = 0.05
terminal_weight = 0.5
```

`C_side`、`C_band`、`C_balance`、`C_target`、`C_offset` 等仍会记录为诊断指标，但不进入 reward。这样论文主线更清楚：HE-PPO 的收益来自 entropy-aware macro-transition update，而不是复杂 reward engineering。

当前 `compute_gae()` 会让 terminal reward 正确向同一 episode 的前序 step 传播，`done` mask 只阻止跨 episode 泄漏。

## PPO

`ppo` 使用标准 actor-critic PPO：

1. 每个 update 采集 `episodes_per_update` 条完整 episode。
2. 用 GAE 计算 advantage 和 return。
3. 对所有 transition 做 advantage normalization。
4. 使用 PPO clipped surrogate 更新 actor。
5. 使用 return target 更新 critic。
6. 使用固定 entropy coefficient 做熵正则。

PPO actor loss：

```text
ratio_t = exp(log pi_new(a_t|s_t) - log pi_old(a_t|s_t))
L_actor = -mean(min(ratio_t A_t, clip(ratio_t, 1-eps, 1+eps) A_t))
```

## HE-PPO

HE-PPO 保留 PPO 的 actor-critic、GAE、critic loss 和 clipped surrogate。它只改变 update 阶段 transition 的组织方式。

### Step Entropy

每个 step 记录每维 entropy：

```text
h_t = entropy(policy_t) / action_dim
```

动态阈值：

```text
tau_h = EMA(quantile(h_t in batch, q=0.4))
```

分类：

```text
high entropy: h_t >= tau_h
low entropy:  h_t <  tau_h
```

### Macro-Transition

连续低熵步骤：

```text
t, t+1, ..., t+l-1
```

合并成一个 macro-transition：

```text
macro_indices = [t, t+1, ..., t+l-1]
macro_old_log_prob = sum(old_log_prob_i)
macro_new_log_prob = sum(new_log_prob_i)
ratio_macro = exp((macro_new_log_prob - macro_old_log_prob) / macro_len)
```

高熵 step 不合并，等价于长度为 1 的普通 PPO transition。

默认最大合并长度：

```text
max_macro_len = 3
```

macro PPO ratio 使用平均 log-ratio，把连续低熵片段视为一个 consolidated effective step。KL 也使用相同尺度：

```text
approx_kl_macro = abs(macro_old_log_prob - macro_new_log_prob) / macro_len
```

这样避免低熵片段因 log-prob 累加而更容易被 clip，导致 HE-PPO 更新比 PPO 更保守。

### Macro Advantage

先计算标准 GAE：

```text
A_t = GAE(reward_t, value_t, done_t)
```

对 macro segment：

```text
A_gae_macro = sum_j gamma^j * A_{t+j}
R_macro     = sum_j gamma^j * r_{t+j}
```

再对相同 `(start_step, macro_len)` 的 group 做弱归一化修正：

```text
A_group = (R_macro - mean(R_macro_group)) / (std(R_macro_group) + eps)
A_macro = 0.8 * normalize(A_gae_macro) + 0.2 * A_group
```

GAE 仍是主项，group correction 只用于降低连续低熵 refinement step 的 credit noise。

### HE-PPO Loss

```text
loss = actor_loss_macro + value_coef * critic_loss - alpha_he * entropy_active
```

`alpha_he` 根据 high-entropy rate 动态调整：

```text
if high_entropy_rate < target_high_entropy_rate:
    alpha_he *= 1.02
else:
    alpha_he *= 0.98

alpha_he = clip(alpha_he, 1e-5, 5e-2)
```

默认：

```text
alpha_he_init = 1e-3
target_high_entropy_rate = 0.45
```

## 训练命令

快速调试：

```bash
CUDA_VISIBLE_DEVICES=1 python run_train.py \
  --algos ppo,heppo \
  --M 10 \
  --K 2 \
  --target-angles=-40,0,40 \
  --sinr-db 12 \
  --episode-steps 8 \
  --updates 100 \
  --episodes-per-update 128 \
  --ppo-epochs 3 \
  --minibatch-size 256 \
  --lr 1e-4 \
  --action-scale 0.02 \
  --seeds 1 \
  --eval-interval 10 \
  --torch-threads 16 \
  --allow-tf32 \
  --device cuda
```

训练默认显示 `tqdm` 进度条。外层显示当前 `algo/seed`，内层显示 update，并实时显示：

```text
reward
J
feasible rate
entropy
macro_step_rate
eval objective
eval feasible rate
seconds per update
```

如果在后台任务或日志文件中不想显示进度条：

```bash
python run_train.py ... --no-progress
```

注意：目标角包含负数时，推荐写成：

```bash
--target-angles=-40,0,40
```

快速 smoke run：

```bash
python run_train.py \
  --algos ppo,heppo \
  --M 3 \
  --K 1 \
  --target-angles=-40,0,40 \
  --updates 1 \
  --episodes-per-update 2 \
  --ppo-epochs 1 \
  --minibatch-size 4 \
  --hidden-dim 16 \
  --seeds 1 \
  --eval-channels 2 \
  --device cpu
```

正式训练：

```bash
CUDA_VISIBLE_DEVICES=0 python run_train.py \
  --algos ppo,heppo \
  --M 10 \
  --K 2 \
  --target-angles=-40,0,40 \
  --sinr-db 12 \
  --episode-steps 12 \
  --updates 100 \
  --episodes-per-update 128 \
  --ppo-epochs 4 \
  --minibatch-size 2048 \
  --lr 1e-4 \
  --action-scale 0.02 \
  --seeds 1 \
  --eval-interval 5 \
  --torch-threads 16 \
  --allow-tf32 \
  --device cuda
```

如果显存仍然很空，可以再尝试：

```text
--episodes-per-update 2048
--minibatch-size 4096
```

先从 `1024 / 2048` 开始，更容易稳定 PPO/HE-PPO 曲线，也能让单张 GPU 有更大的 batch workload。

## 评估与绘图

评估训练好的 RL checkpoint：

```bash
python run_eval.py \
  --log-dir log/YYYYMMDD-HHMMSS \
  --eval-channels 256 \
  --plot-seed 2026 \
  --save-plots \
  --device cpu
```

`run_eval.py` 会：

1. 读取 `log_dir/checkpoints/*.pt`。
2. 对每个 checkpoint 在 `plot_seed, plot_seed+1, ...` 的固定信道集合上评估。
3. 用 deterministic policy inference。
4. 保存多信道均值/标准差指标到 `metrics.json`。
5. 用 `plot_seed` 这一条代表性信道保存波束矩阵和波束图。

只重新绘图：

```bash
python run_plot.py \
  --log-dir log/YYYYMMDD-HHMMSS \
  --eval-channels 256 \
  --plot-seed 2026 \
  --device cpu
```

`run_plot.py` 会检查 `patterns.npz`。如果不存在，它会自动调用当前无 baseline 的 `run_eval.py` 生成 `beamformers.npz`、`patterns.npz` 和图文件；如果你只想绘制已有文件，可以加 `--no-auto-eval`。

## 输出文件

每次训练保存：

```text
log/YYYYMMDD-HHMMSS/
  config.json
  summary.json
  train_history.csv
  eval_history.csv
  entropy_history.csv
  checkpoints/
    ppo_seed1.pt
    ppo_seed2.pt
    ppo_seed3.pt
    heppo_seed1.pt
    heppo_seed2.pt
    heppo_seed3.pt
```

评估后额外保存：

```text
beamformers.npz
patterns.npz
metrics.json
runtime.json
figures/
  convergence_reward.pdf/png
  convergence_objective.pdf/png
  convergence_radar_loss.pdf/png
  convergence_target_center.pdf/png
  convergence_peak_offset.pdf/png
  beampattern.pdf/png
  entropy_macro_stats.pdf/png
  runtime_bar.pdf/png
```

`train_history.csv` 主要字段：

```text
update
episode
train_reward
train_objective
train_feasible_rate
train_min_sinr_db
actor_loss
critic_loss
entropy
alpha_entropy
approx_kl
clip_fraction
grad_norm
update_time_sec
```

`eval_history.csv` 主要字段：

```text
update
episode
eval_reward
eval_objective
eval_Lr
eval_Lr1
eval_Lr1_plain
eval_Lr2
eval_C_sinr
eval_C_target
eval_C_offset
eval_target_center_error
eval_target_peak_offset_error
eval_min_sinr_db
eval_feasible_rate
eval_peak_sidelobe_ratio
eval_target_band_error
```

`entropy_history.csv` 主要字段：

```text
update
episode
entropy_mean
entropy_threshold
high_entropy_rate
macro_step_rate
average_macro_length
num_macro_segments
```

## Figures and Expected Observations

训练和评估图保存在：

```text
log/YYYYMMDD-HHMMSS/figures/
```

### `convergence_reward.pdf/png`

显示 evaluation reward 随 training episode 的变化。横轴是训练 episode 数，不是 PPO update。它主要用于检查 reward 信号是否稳定改善。

预期：HE-PPO 至少不应明显低于 PPO；如果 reward 不升但 objective 和 radar loss 改善，论文中应以物理指标为主。

### `convergence_objective.pdf/png`

显示 evaluation objective `J` 随 training episode 的变化。`J` 由 center-weighted Liu radar loss 和 soft SINR penalty 构成。

预期：HE-PPO 的 `J` 下降更快或最终更低。若两条曲线都平，优先检查 `action_scale`、学习率和 `eval_feasible_rate`。

### `convergence_radar_loss.pdf/png`

显示 Liu-style radar loss `Lr = Lr1_cw + w_cross Lr2`。它直接反映波束图匹配和目标方向交叉相关。

预期：在通信约束不崩的前提下，HE-PPO 应取得更低 radar loss。

### `convergence_target_center.pdf/png`

显示目标角中心处的归一化 gain error。该指标不参与 reward，只用于诊断主瓣是否真的对准 `-40, 0, 40` 度。

预期：曲线应下降。若它一直很高，说明主瓣虽然可能靠近目标带，但目标角中心增益不足。

### `convergence_peak_offset.pdf/png`

显示每个目标带内局部峰值相对目标角的归一化角度偏移。

预期：曲线应接近 0。若 beampattern 看起来有主瓣但该值高，说明主瓣峰值偏离了预设方向。

### `beampattern.pdf/png`

显示最终归一化波束图。竖向虚线标出 Liu2020 三个目标角 `-40, 0, 40` 度。

预期：HE-PPO 应形成更接近目标角的三个主瓣，并保持可接受旁瓣。若主瓣偏移，先看 `convergence_peak_offset`。

### `entropy_macro_stats.pdf/png`

显示 high-entropy step rate 和 macro-transition rate。

预期：HE-PPO 的 macro-step rate 不应长期接近 0；若接近 0，HE-PPO 退化为 PPO。若过高，credit assignment 可能过粗。

### `runtime_bar.pdf/png`

显示 checkpoint deterministic inference 的平均耗时。

预期：PPO 和 HE-PPO 在线推理时间应同量级；本仓库当前不包含传统优化 baseline，因此该图主要用于比较学习策略之间的推理开销。

## 结果阅读顺序

优先看 `eval_history.csv` 和 `metrics.json`，不要只看训练 reward。

建议判断标准：

```text
1. HE-PPO 的 eval_objective 下降是否快于 PPO。
2. HE-PPO 的最终 eval_Lr / eval_Lr1 是否低于 PPO。
3. HE-PPO 的 eval_target_peak_offset_error 是否接近 0。
4. HE-PPO 的 eval_feasible_rate 是否不低于 PPO。
5. HE-PPO 的 beampattern 是否有更清晰且对准目标角的主瓣。
6. HE-PPO 的 runtime 是否与 PPO 同量级。
```

收敛图按 `algo + episode` 聚合多 seed，画 mean curve 和 std band。波束图使用统一参考功率归一化，不对每条曲线单独归一化到 0 dB。

## 测试

```bash
python -m py_compile run_train.py run_eval.py run_plot.py isac_rl/*.py
python -m pytest -q
```

当前测试覆盖：

- Liu2020 指标和逐天线功率约束。
- 随机 reset 和直接 residual action step。
- tanh Gaussian actor 的 log-prob correction。
- terminal reward 在 GAE 中正确向前传播。
- HE-PPO 低熵 macro-transition 合并。
- rollout seed 随 update 变化，避免每个 update 重复同一批训练环境。
- 多 seed checkpoint 不互相覆盖。
- PPO/HE-PPO smoke training 和日志文件。
- 训练核心模块不包含传统优化对比方法实现。
