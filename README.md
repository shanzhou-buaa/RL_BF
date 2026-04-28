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
log1p(Lr1 / Lr1_ref)
log1p(Lr2 / Lr2_ref)
log1p(Lr / Lr_ref)
log1p(peak_sidelobe_ratio)
log1p(mean_sidelobe_ratio)
target gain balance features
```

其中：

```text
Lr1 = beampattern matching error
Lr2 = target-direction cross-correlation
Lr  = Lr1 + cross_corr_weight * Lr2
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

评估时使用 deterministic inference：

```text
action = tanh(mu)
```

critic 输出 `V(s_t)`，用于 GAE 和 value loss。

## 奖励函数

训练 reward 与最终评估指标对齐。先构造总目标：

```text
J =
  w_bp      * log1p(Lr1 / Lr1_ref)
  + w_cc   * log1p(Lr2 / Lr2_ref)
  + w_sinr * C_sinr
  + w_side * C_side
  + w_band * C_band
  + w_balance * C_balance
```

默认权重：

```text
w_bp = 1.0
w_cc = 0.3
w_sinr = 8.0
w_side = 0.8
w_band = 1.0
w_balance = 0.2
```

SINR cost：

```text
gap_db_k = Gamma_db - SINR_db_k
C_sinr = mean(softplus(gap_db_k / 2)^2)
```

旁瓣 cost：

```text
C_side = log1p(peak_sidelobe / target_mean_gain)
```

目标主瓣形状 cost：

```text
C_band = mean_over_targets(mean((P(theta_band) / alpha - 1)^2))
```

目标方向均衡 cost：

```text
C_balance = max(0, 1 - target_min_gain / target_mean_gain)
```

每步 reward：

```text
progress = (J_prev - J_current) / (abs(J_prev) + eps)

reward =
  -J_current
  + beta_progress * tanh(progress / progress_scale)
  + beta_margin * tanh(min_sinr_gap_db / 5)
  + beta_feasible * feasible_flag
```

终止步额外加入：

```text
reward += -terminal_weight * J_current
reward += feasible_terminal_bonus * feasible_flag
```

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
ratio_macro = exp(macro_new_log_prob - macro_old_log_prob)
```

高熵 step 不合并，等价于长度为 1 的普通 PPO transition。

默认最大合并长度：

```text
max_macro_len = 3
```

macro PPO ratio 仍使用累计 log-prob；日志和 KL early stopping 使用 per-step scale：

```text
approx_kl_macro = abs(macro_old_log_prob - macro_new_log_prob) / macro_len
```

这样避免低熵片段因 log-prob 累加而过早触发 KL early stop。

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

正式训练：

```bash
python run_train.py \
  --algos ppo,heppo \
  --M 10 \
  --K 2 \
  --target-angles=-40,0,40 \
  --sinr-db 12 \
  --episode-steps 8 \
  --updates 100 \
  --episodes-per-update 256 \
  --ppo-epochs 5 \
  --minibatch-size 512 \
  --lr 3e-4 \
  --action-scale 0.03 \
  --seeds 1 \
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
python run_plot.py --log-dir log/YYYYMMDD-HHMMSS
```

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
eval_reward
eval_objective
eval_Lr
eval_Lr1
eval_Lr2
eval_C_sinr
eval_min_sinr_db
eval_feasible_rate
eval_peak_sidelobe_ratio
eval_target_band_error
```

`entropy_history.csv` 主要字段：

```text
update
entropy_mean
entropy_threshold
high_entropy_rate
macro_step_rate
average_macro_length
num_macro_segments
```

## 结果阅读顺序

优先看 `eval_history.csv` 和 `metrics.json`，不要只看训练 reward。

建议判断标准：

```text
1. HE-PPO 的 eval_objective 下降是否快于 PPO。
2. HE-PPO 的最终 eval_Lr / eval_Lr1 是否低于 PPO。
3. HE-PPO 的 eval_feasible_rate 是否不低于 PPO。
4. HE-PPO 的 beampattern 是否有更清晰主瓣和更低旁瓣。
5. HE-PPO 的 runtime 是否与 PPO 同量级。
```

收敛图按 `algo + update` 聚合多 seed，画 mean curve 和 std band。波束图使用统一参考功率归一化，不对每条曲线单独归一化到 0 dB。

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
