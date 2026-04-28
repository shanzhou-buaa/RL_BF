# HE-PPO for ISAC Beamforming

本仓库实现 **HE-PPO: High-Entropy-step-aware PPO for ISAC Beamforming**。问题设置固定为 Liu2020 多用户 ISAC/DFRC 联合发射波束成形：逐天线功率约束、`M=10`、目标角 `[-40, 0, 40]` 度、主瓣宽度 `10` 度、`Pt=1`、噪声功率 `0.01`、角度网格 `[-90, 90]` 且步长 `0.1` 度。

核心目标是在 SINR 约束下最小化 Liu2020 雷达损失：

```text
Lr(R, alpha) = Lr1 beampattern matching + wc * Lr2 target cross-correlation
```

本版本将论文主线收敛到一件事：把 Zhang2026 “高熵步骤驱动有效 RL、低熵步骤需要合并以改善 credit assignment” 的思想改造成适用于 actor-critic PPO 的 entropy-aware macro-transition update。

## 方法定位

训练只比较两个 RL 方法：

| 方法 | 初始化 | 动作空间 | PPO update | 用途 |
| --- | --- | --- | --- | --- |
| `ppo` | 随机 `W` | full residual action | 所有 transition 独立更新 | 原始 PPO 基线 |
| `heppo` | 随机 `W` | full residual action | 高熵 step 独立更新，连续低熵 step 合并成 macro-transition | 主方法 |

公平对比要求两者完全一致：

```text
random initialization
full residual action
same state
same reward
same actor-critic network
same learning rate
same batch size
same PPO epochs
same evaluation channels
same deterministic inference
```

唯一不同是 PPO update 阶段：

```text
PPO:    all transitions are optimized independently.
HE-PPO: consecutive low-entropy transitions are consolidated into macro-transitions.
```

ZF、SDR 只作为 `run_eval.py` 中的离线 baseline，不进入 RL reset、step、rollout 或 trainer。

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
  baselines.py
  logger.py
  plotting.py
  utils.py

tests/
  test_metrics.py
  test_env.py
  test_policy.py
  test_entropy_macro.py
  test_smoke_train.py
```

强制隔离规则：

```text
isac_rl/env.py
isac_rl/trainer.py
isac_rl/ppo.py
isac_rl/heppo.py
```

这些训练核心模块不 import `isac_rl/baselines.py`。`baselines.py` 只被 `run_eval.py` 调用。

## 环境

`ISACBeamformingEnv.reset()` 使用纯随机初始化：

```text
H ~ CN(0, 1)
W ~ random complex matrix
W <- per_antenna_power_normalize(W, Pt)
```

禁止使用：

```text
ZF initialization
SDR initialization
pseudo-inverse initialization
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

逐天线功率归一化是物理约束：

```text
[W W^H]_{m,m} = Pt / M
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

主要设计：

- `H` 按自身平均功率归一化，并拼接实部/虚部。
- `W` 按 `sqrt(Pt/M)` 归一化，并拼接实部/虚部。
- SINR 特征包含每用户 gap、违约 softplus、最小 gap、平均违约和 feasible flag。
- 雷达标量特征包含 `Lr1`、`Lr2`、`Lr`、旁瓣比例、目标均衡等。
- 粗粒度 beampattern 使用 `[-90, 90]` 上每 `2` 度一个采样点的误差。
- progress feature 记录上一奖励、目标改善、SINR cost 改善、EMA reward 和时间进度。

## 动作空间

主实验只使用 full residual action：

```text
action_dim = 2 * M * (K + M)
```

动作前半部分是实部，后半部分是虚部：

```text
Delta_W = A_real + j A_imag
Delta_W in C^{M x (K+M)}
```

actor 使用 tanh Gaussian：

```text
raw_action ~ Normal(mu, std)
action = tanh(raw_action)
```

log-prob 使用 tanh correction：

```text
log_prob = Normal.log_prob(raw_action).sum()
log_prob -= log(1 - action^2 + 1e-6).sum()
```

deterministic inference 使用：

```text
action = tanh(mu)
```

## 奖励函数

训练目标与论文指标对齐，先构造总目标：

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

## HE-PPO

### 基线 PPO 如何训练

`ppo` 使用标准 actor-critic PPO。每个 update 先采集 `episodes_per_update` 条 episode，episode 内逐步执行物理环境：

```text
s0 -> a0 -> s1 -> a1 -> ... -> sT
```

每个 transition 记录：

```text
(s_t, a_t, r_t, done_t, log pi_old(a_t|s_t), H_t, V(s_t))
```

采样结束后用 GAE 计算 advantage：

```text
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t = delta_t + gamma * lambda * A_{t+1}
```

然后每个 transition 独立进入 PPO clipped surrogate：

```text
ratio_t = exp(log pi_new(a_t|s_t) - log pi_old(a_t|s_t))
L_actor = -mean(min(ratio_t A_t, clip(ratio_t, 1-eps, 1+eps) A_t))
```

critic 用 return target 做 MSE，actor 加固定 entropy bonus。这个基线不判断 step 是否高熵，也不合并 transition。

### HE-PPO 的核心思想

HE-PPO 保留 PPO 的 actor-critic、GAE、value loss 和 clipped surrogate。它只改变 update 阶段的样本组织方式：

```text
high-entropy step: 保留为普通 PPO transition
low-entropy run:   连续低熵 transition 合并为一个 macro-transition
```

直觉是：在 beamforming refinement 后期，策略经常输出很确定的小动作，这类低熵步骤之间差异很小，逐步分配 credit 会产生噪声；高熵步骤更可能代表有效探索，应该保留细粒度更新。HE-PPO 因此让高熵步骤单独主导 policy update，把连续低熵片段统一接收一个 macro credit。

### Step entropy 与阈值

每个 step 记录每维策略熵：

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

这里使用每维 entropy，是为了让 entropy 阈值不随动作维度变化。当前主实验 PPO 和 HE-PPO 都使用 full residual action，因此该归一化也让不同配置的日志更可比。

### Macro-transition 构造

连续低熵步骤构造 macro-transition：

```text
macro_indices = [t, t+1, ..., t+l-1]
macro_old_log_prob = sum(old_log_prob_i)
macro_new_log_prob = sum(new_log_prob_i)
ratio_macro = exp(macro_new_log_prob - macro_old_log_prob)
```

最大合并长度为 `max_macro_len=3`。长度不设太大，是为了避免把过长片段压成一个 credit 后重新引入模糊 credit assignment。

高熵 step 不合并，等价于长度为 1 的普通 PPO transition。

### Macro advantage

macro advantage 以 GAE 为主，并加入较弱的 group-normalized correction：

```text
A_macro = 0.8 * normalize(A_gae_macro) + 0.2 * A_group
```

其中：

```text
A_gae_macro = sum_j gamma^j * A_{t+j}
R_macro     = sum_j gamma^j * r_{t+j}
A_group     = normalized R_macro within same (start_step, macro_len) group
```

GAE 仍是主项，因此算法保持 PPO 的 critic-based credit assignment；group correction 只作为弱修正，用来借鉴 E-GRPO 中“同类 consolidated group 内做相对优势归一化”的思想。

### HE-PPO loss

HE-PPO loss 保留 PPO clipped surrogate、critic loss 和 entropy bonus：

```text
loss = actor_loss_macro + value_coef * critic_loss - alpha_he * entropy_active
```

`alpha_he` 根据 high-entropy rate 动态调整，并限制在 `[1e-5, 5e-2]`。

动态规则：

```text
if high_entropy_rate < target_high_entropy_rate:
    alpha_he *= 1.02
else:
    alpha_he *= 0.98
```

默认 `target_high_entropy_rate=0.45`。当高熵样本过少时，提高 entropy coefficient 鼓励探索；当高熵样本足够时，降低 entropy coefficient 让策略更专注于目标优化。

### 与 E-GRPO 的区别

HE-PPO 不是直接照搬 GRPO：

- E-GRPO 面向 GRPO-based flow-model alignment；HE-PPO 面向连续动作 beamforming control。
- E-GRPO 不使用 PPO actor-critic critic；HE-PPO 保留 value function、GAE 和 PPO clipped surrogate。
- E-GRPO 的 consolidated group 直接用于 group-relative update；HE-PPO 只把 group-normalized advantage 作为弱修正，主 advantage 仍来自 GAE。
- HE-PPO 不改变环境交互，不跳步，不做传统优化投影。

因此论文中可以表述为：HE-PPO transfers the high-entropy-step-aware credit assignment idea from E-GRPO into an actor-critic PPO framework for ISAC beamforming.

## 训练

推荐主实验命令：

```bash
python run_train.py \
  --algos ppo,heppo \
  --M 10 \
  --K 2 \
  --target-angles=-40,0,40 \
  --sinr-db 12 \
  --episode-steps 8 \
  --updates 300 \
  --episodes-per-update 256 \
  --ppo-epochs 5 \
  --minibatch-size 512 \
  --lr 3e-4 \
  --action-scale 0.03 \
  --seeds 1,2,3 \
  --device cuda
```

训练时默认显示 `tqdm` 进度条。外层进度显示当前 `algo/seed`，内层进度显示 update，并实时显示：

```text
reward
J
entropy
macro_step_rate
eval objective
feasible rate
seconds per update
```

如果在批处理日志里不想显示进度条，可以加：

```bash
--no-progress
```

注意：如果目标角第一个值为负数，推荐写成 `--target-angles=-40,0,40`。新版 `run_train.py` 也兼容 `--target-angles -40,0,40`，但等号写法更不容易被 shell/argparse 误解。

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
  --device cpu
```

## 评估与绘图

训练完成后评估 ZF/SDR 离线 baseline，并保存波束图数据：

```bash
python run_eval.py \
  --log-dir log/YYYYMMDD-HHMMSS \
  --baselines zf,sdr \
  --eval-channels 256 \
  --save-plots \
  --device cpu
```

只绘图：

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
    ppo.pt
    heppo.pt
```

`run_eval.py` 额外保存：

```text
beamformers.npz
patterns.npz
metrics.json
runtime.json
figures/
  convergence_reward.pdf/png
  convergence_objective.pdf/png
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
actor_loss
critic_loss
entropy
alpha_entropy
approx_kl
clip_fraction
grad_norm
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

## 测试

```bash
python -m py_compile run_train.py run_eval.py run_plot.py isac_rl/*.py
python -m pytest -q
```

当前测试覆盖：

- Liu2020 指标和逐天线功率约束。
- 随机 reset 和直接 residual action step。
- tanh Gaussian actor 的 log-prob correction。
- HE-PPO 低熵 macro-transition 合并。
- PPO/HE-PPO smoke training 和日志文件。
- 训练核心模块不 import baseline。
