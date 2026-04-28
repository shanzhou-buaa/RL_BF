# 状态空间
 - 通信的CSI矩阵$H$，维度M\*K，实部+虚部，共2\*M\*K
 - 各个通信用户的SINR值$\gamma_k$，维度K，实数
 - 总radar loss $L$，维度1，实数
 - 各个目标在其角度的带宽范围内，与理想波束幅值的MSE $L_n$，维度为雷达目标的个数N。这一项是为了防止整体波束只对部分目标效果很好，而对某几个目标优化很差
 - CSE相关的处理

# 动作空间
 - 通信波束成型矩阵$W_c$，维度M\*K，实部+虚部，共2\*M\*K
 - 雷达波束成型矩阵$W_r$，维度M\*M，实部+虚部，共2\*M\*M
 - CSE相关的处理，动作是高斯采样，有熵，可以将若干低熵步骤合并，合并时加入通信SINR的权重，对于SINR距离通信阈值的情况，可以增强探索（确认这部分逻辑，并给出所需的动作空间）

# 奖励空间
$reward=-\alpha_1*L + \Sigma_{i=1}^K R_k*\gamma_k$
radar loss越小越好；$R_k$在$\gamma_k < \Gamma$时是个比较大的数，反之为0，即对不满足通信阈值的进行惩罚。
除此之外在奖励中加入针对各个雷达波束的处理，即不要有太差的雷达目标方向的波束
另外再加入CSE所需的奖励

# 当前代码确认

## 状态空间

当前 `state` 包含：

1. 通信信道矩阵 `H`：代码中形状为 `K x M`，实部+虚部，总维度 `2*K*M`。
2. 当前联合波束矩阵 `W=[Wc, Wr]`：`Wc` 为 `M x K`，`Wr` 为 `M x M`，实部+虚部，总维度 `2*M*(K+M)`。
3. 各通信用户归一化 SINR：`sinr / Gamma`，维度 `K`。
4. 完整雷达损失 `Lr = Lr,1 + wc*Lr,2`，维度 `1`。
5. 各雷达目标主瓣带内 MSE `L_n`，代码中为 `target_band_errors`，维度 `N`。
6. 旁瓣泄漏指标 `sidelobe_leakage`，维度 `1`。

因此当前 state 总维度是：

```text
2*K*M + 2*M*(K+M) + K + 1 + N + 1
```

默认 `M=10, K=2, N=3` 时：

```text
state_dim = 40 + 240 + 2 + 1 + 3 + 1 = 287
```

## 动作空间

策略动作仍是高斯采样：

```text
action ~ Normal(mean(state), std)
```

默认 PPO/CSE-PPO/EC-PPO 的动作都是完整复值残差：

```text
delta_W: M x (K+M)
action_dim = 2*M*(K+M)
```

默认 `M=10, K=2` 时：

```text
action_dim = 2*10*(2+10) = 240
```

可选 `--use-nullspace` 时，CSE-PPO/EC-PPO 使用结构化 null-space residual action：

```text
action_dim = 2 * (K + (M-K)*(K+M))
```

含义是：

- `K` 个复数通信增益残差；
- `(M-K) x K` 个通信 null-space 残差系数；
- `(M-K) x M` 个雷达 null-space 残差系数。

可选 null-space 动作在 `M=10, K=2` 时：

```text
action_dim = 2 * (2 + 8*(2+10)) = 196
```

环境将该动作映射成 `delta_W`，然后执行：

```text
W <- row_power_normalize(W + step_scale * delta_W)
```

`row_power_normalize` 严格满足逐天线功率约束。CSE/EC 默认使用 action line-search，在候选步长中选择 `Lr + constraint_score_weight * SINR_cost` 最小的更新。

## 奖励和约束

当前 reward 只使用完整雷达损失 `Lr` 的相对下降量：

```text
loss_reward = loss_reward_weight * (prev_Lr - new_Lr) / max(abs(prev_Lr), 1e-6)
reward = loss_reward
```

默认：

```text
loss_reward_weight = 1.0
constraint_reward_weight = 0.0
action_penalty = 0.0
cross_corr_weight = 1.0
```

SINR 不再作为额外 reward 项，而是作为约束 cost：

```text
cost = mean(max(Gamma - gamma_k, 0) / Gamma)
```

该 cost 进入 primal-dual PPO，并在 CSE/EC 的安全 line-search 中参与候选动作选择。逐天线功率约束由每步归一化直接满足。

## CSE 相关处理

CSE 不增加额外环境动作维度，主要在 trainer 中处理有效熵：

```text
entropy_per_dim = policy_entropy / action_dim
min_margin = min_k(gamma_k - Gamma)
phi = sigmoid(beta * min_margin / Gamma)
H_eff = entropy_per_dim * phi
```

默认：

```text
beta = 6.0
entropy_threshold = 0.18
max_macro_steps = 4
```

当 `H_eff < entropy_threshold` 时，连续低有效熵 transition 会被合并为 macro-step，用于 PPO 更新。
