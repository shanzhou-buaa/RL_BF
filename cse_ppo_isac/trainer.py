from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from .actor_critic import ActorCritic
from .config import EnvConfig, PPOConfig
from .env import ISACBeamformingEnv
from .selection import is_better_candidate


def _reset_env(env: ISACBeamformingEnv) -> np.ndarray:
    return env.reset()


def _reset_env_with_state(env: ISACBeamformingEnv):
    state = env.reset()
    return env, state


def _step_env(job):
    env, action = job
    return env.step(action)


def _step_env_with_state(job):
    env, action = job
    result = env.step(action)
    return env, result


@dataclass
class Transition:
    state: np.ndarray
    action: np.ndarray
    old_log_prob: float
    entropy: float
    reward: float
    cost: float
    value: float
    done: bool
    h_eff: float
    info: Dict[str, object]


@dataclass
class Segment:
    indices: Sequence[int]
    old_log_prob: float
    advantage: float
    ret: float


class CSEPPOTrainer:
    """PPO trainer with communication-safe entropy consolidation."""

    def __init__(self, env_cfg: EnvConfig, ppo_cfg: PPOConfig):
        self.env_cfg = env_cfg
        self.cfg = ppo_cfg
        torch.manual_seed(ppo_cfg.seed)
        np.random.seed(ppo_cfg.seed)
        self.env = ISACBeamformingEnv(
            self.env_cfg,
            structured_action=ppo_cfg.use_nullspace_action,
            seed=ppo_cfg.seed,
        )
        self.policy = ActorCritic(
            self.env.state_size,
            self.env.action_size,
            ppo_cfg.hidden_dim,
            initial_log_std=ppo_cfg.initial_log_std,
        ).to(ppo_cfg.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=ppo_cfg.learning_rate)
        self.entropy_threshold_current = float(ppo_cfg.entropy_threshold)
        self.dual = 0.0
        self.history: List[Dict[str, float]] = []
        self.last_rollout_timing: Dict[str, float] = {}
        self.rollout_envs = [
                ISACBeamformingEnv(
                    self.env_cfg,
                    structured_action=ppo_cfg.use_nullspace_action,
                    seed=ppo_cfg.seed + 1000 + i,
                )
            for i in range(ppo_cfg.episodes_per_update)
        ]
        requested_workers = int(ppo_cfg.rollout_workers)
        if requested_workers <= 0:
            requested_workers = min(ppo_cfg.episodes_per_update, os.cpu_count() or 1)
        self.rollout_workers = max(1, min(requested_workers, ppo_cfg.episodes_per_update))
        self.rollout_backend = ppo_cfg.rollout_backend
        if self.rollout_backend not in {"serial", "thread", "process"}:
            raise ValueError(
                f"unknown rollout_backend={self.rollout_backend!r}; "
                "expected serial, thread or process"
            )
        if self.rollout_backend == "serial" or self.rollout_workers <= 1:
            self._rollout_pool = None
        elif self.rollout_backend == "thread":
            self._rollout_pool = ThreadPoolExecutor(max_workers=self.rollout_workers)
        else:
            self._rollout_pool = ProcessPoolExecutor(max_workers=self.rollout_workers)

    def close(self) -> None:
        if self._rollout_pool is not None:
            self._rollout_pool.shutdown(wait=True)
            self._rollout_pool = None

    def _reset_envs(self, envs: List[ISACBeamformingEnv]) -> List[np.ndarray]:
        if self._rollout_pool is None or len(envs) <= 1:
            return [env.reset() for env in envs]
        if self.rollout_backend == "process":
            results = list(self._rollout_pool.map(_reset_env_with_state, envs))
            for idx, (env, _) in enumerate(results):
                envs[idx] = env
            return [state for _, state in results]
        return list(self._rollout_pool.map(_reset_env, envs))

    def _step_envs(self, envs: List[ISACBeamformingEnv], actions: np.ndarray):
        if self._rollout_pool is None or len(envs) <= 1:
            return [env.step(action) for env, action in zip(envs, actions)]
        if self.rollout_backend == "process":
            results = list(self._rollout_pool.map(_step_env_with_state, zip(envs, actions)))
            for idx, (env, _) in enumerate(results):
                envs[idx] = env
            return [result for _, result in results]
        return list(self._rollout_pool.map(_step_env, zip(envs, actions)))

    def train(self) -> List[Dict[str, float]]:
        progress = tqdm(
            range(1, self.cfg.updates + 1),
            desc="PPO训练",
            dynamic_ncols=True,
            leave=True,
        )
        for update in progress:
            update_start = perf_counter()
            episodes = self._collect_rollouts()
            flat = [tr for ep in episodes for tr in ep]
            mean_cost = float(np.mean([tr.cost for tr in flat]))
            self._update_dual(mean_cost)
            entropy_threshold = self._update_entropy_threshold(flat)
            segments = []
            offset = 0
            segment_start = perf_counter()
            for ep in episodes:
                segments.extend(
                    self._build_segments(
                        ep, offset=offset, entropy_threshold=entropy_threshold
                    )
                )
                offset += len(ep)
            self._normalize_segment_advantages(segments)
            segment_time = perf_counter() - segment_start
            ppo_start = perf_counter()
            self._ppo_update(flat, segments)
            ppo_update_time = perf_counter() - ppo_start
            update_time = perf_counter() - update_start
            rollout_timing = dict(self.last_rollout_timing)

            low_entropy_rate = float(
                np.mean([tr.h_eff < entropy_threshold for tr in flat])
            )

            row = {
                "update": float(update),
                "reward": float(np.mean([sum(tr.reward for tr in ep) for ep in episodes])),
                "cost": mean_cost,
                "dual": float(self.dual),
                "entropy_threshold": float(entropy_threshold),
                "low_entropy_rate": low_entropy_rate,
                "quality_reward": float(
                    np.mean([tr.info.get("quality_reward", tr.info["loss_reward"]) for tr in flat])
                ),
                "improvement_reward": float(
                    np.mean([tr.info.get("improvement_reward", 0.0) for tr in flat])
                ),
                "relative_loss_improvement": float(
                    np.mean([tr.info.get("relative_loss_improvement", 0.0) for tr in flat])
                ),
                "relative_quality_improvement": float(
                    np.mean(
                        [tr.info.get("relative_quality_improvement", 0.0) for tr in flat]
                    )
                ),
                "radar_loss": float(np.mean([ep[-1].info["radar_loss"] for ep in episodes])),
                "beam_objective": float(
                    np.mean([ep[-1].info["beam_objective"] for ep in episodes])
                ),
                "beampattern_loss": float(
                    np.mean([ep[-1].info.get("beampattern_loss", 0.0) for ep in episodes])
                ),
                "cross_corr": float(
                    np.mean([ep[-1].info.get("cross_corr", 0.0) for ep in episodes])
                ),
                "sidelobe_ratio": float(
                    np.mean([ep[-1].info["sidelobe_ratio"] for ep in episodes])
                ),
                "sidelobe_leakage": float(
                    np.mean([ep[-1].info["sidelobe_leakage"] for ep in episodes])
                ),
                "target_mean": float(np.mean([ep[-1].info["target_mean"] for ep in episodes])),
                "target_min": float(np.mean([ep[-1].info["target_min"] for ep in episodes])),
                "target_min_ratio": float(
                    np.mean([ep[-1].info.get("target_min_ratio", 0.0) for ep in episodes])
                ),
                "target_band_error_mean": float(
                    np.mean(
                        [
                            ep[-1].info.get("target_band_error_mean", 0.0)
                            for ep in episodes
                        ]
                    )
                ),
                "feasible_rate": float(np.mean([ep[-1].info["feasible"] for ep in episodes])),
                "accepted_rate": float(np.mean([tr.info["accepted"] for tr in flat])),
                "mean_step_scale": float(np.mean([tr.info["step_scale"] for tr in flat])),
                "rollout_time_sec": float(rollout_timing.get("rollout_time_sec", 0.0)),
                "reset_time_sec": float(rollout_timing.get("reset_time_sec", 0.0)),
                "policy_time_sec": float(rollout_timing.get("policy_time_sec", 0.0)),
                "env_step_time_sec": float(rollout_timing.get("env_step_time_sec", 0.0)),
                "transition_time_sec": float(
                    rollout_timing.get("transition_time_sec", 0.0)
                ),
                "segment_time_sec": float(segment_time),
                "ppo_update_time_sec": float(ppo_update_time),
                "update_time_sec": float(update_time),
                "transitions_per_sec": float(
                    len(flat) / max(rollout_timing.get("rollout_time_sec", 0.0), 1.0e-9)
                ),
            }
            self.history.append(row)
            progress.set_postfix(
                reward=f"{row['reward']:.4f}",
                cost=f"{row['cost']:.4f}",
                dual=f"{row['dual']:.3f}",
                radar=f"{row['radar_loss']:.4f}",
                feasible=f"{row['feasible_rate']:.2f}",
                rollout=f"{row['rollout_time_sec']:.2f}s",
                ppo=f"{row['ppo_update_time_sec']:.2f}s",
            )
        return self.history

    def _collect_rollouts(self) -> List[List[Transition]]:
        """Collect one synchronous vectorized rollout from many environments."""

        rollout_start = perf_counter()
        reset_time = 0.0
        policy_time = 0.0
        env_step_time = 0.0
        transition_time = 0.0
        episodes: List[List[Transition]] = [[] for _ in self.rollout_envs]
        reset_start = perf_counter()
        states = self._reset_envs(self.rollout_envs)
        reset_time += perf_counter() - reset_start
        active = np.ones(len(self.rollout_envs), dtype=bool)

        for _ in range(self.env_cfg.max_steps):
            active_ids = np.flatnonzero(active)
            if active_ids.size == 0:
                break

            policy_start = perf_counter()
            state_batch = torch.as_tensor(
                np.stack([states[i] for i in active_ids]),
                dtype=torch.float32,
                device=self.cfg.device,
            )
            actions, old_log_probs, entropies, values = self.policy.act_batch(state_batch)
            policy_time += perf_counter() - policy_start
            active_envs = [self.rollout_envs[int(env_id)] for env_id in active_ids]
            step_start = perf_counter()
            step_results = self._step_envs(active_envs, actions)
            env_step_time += perf_counter() - step_start

            transition_start = perf_counter()
            for local_idx, (env_id, result) in enumerate(zip(active_ids, step_results)):
                self.rollout_envs[int(env_id)] = active_envs[local_idx]
                action = actions[local_idx]
                next_state, reward, done, info = result
                h_eff = self._effective_entropy(
                    float(entropies[local_idx]), float(info["min_margin"])
                )
                episodes[int(env_id)].append(
                    Transition(
                        state=states[int(env_id)],
                        action=action.astype(np.float32),
                        old_log_prob=float(old_log_probs[local_idx]),
                        entropy=float(entropies[local_idx]),
                        reward=float(reward),
                        cost=float(info["cost"]),
                        value=float(values[local_idx]),
                        done=bool(done),
                        h_eff=h_eff,
                        info=info,
                    )
                )
                states[int(env_id)] = next_state
                active[int(env_id)] = not done
            transition_time += perf_counter() - transition_start

        self.last_rollout_timing = {
            "rollout_time_sec": float(perf_counter() - rollout_start),
            "reset_time_sec": float(reset_time),
            "policy_time_sec": float(policy_time),
            "env_step_time_sec": float(env_step_time),
            "transition_time_sec": float(transition_time),
        }
        return episodes

    def _effective_entropy(self, entropy: float, min_margin: float) -> float:
        entropy_per_dim = entropy / max(self.env.action_size, 1)
        if not self.cfg.use_feasibility_weighted_entropy:
            return float(entropy_per_dim)
        normalized_margin = min_margin / max(self.env_cfg.sinr_threshold, 1.0e-6)
        logits = np.clip(self.cfg.feasibility_beta * normalized_margin, -60.0, 60.0)
        phi = 1.0 / (1.0 + np.exp(-logits))
        floor = float(np.clip(self.cfg.feasibility_entropy_floor, 0.0, 1.0))
        phi = floor + (1.0 - floor) * phi
        return float(entropy_per_dim * phi)

    def _update_entropy_threshold(self, transitions: Sequence[Transition]) -> float:
        if not transitions:
            return float(self.entropy_threshold_current)
        values = np.asarray([tr.h_eff for tr in transitions], dtype=np.float64)
        quantile = float(np.clip(self.cfg.entropy_threshold_quantile, 0.0, 1.0))
        target = float(np.quantile(values, quantile))
        ema = float(np.clip(self.cfg.entropy_threshold_ema, 0.0, 1.0))
        self.entropy_threshold_current = (
            ema * self.entropy_threshold_current + (1.0 - ema) * target
        )
        return float(self.entropy_threshold_current)

    def _adjusted_rewards(self, episode: Sequence[Transition]) -> np.ndarray:
        rewards = np.asarray([tr.reward for tr in episode], dtype=np.float64)
        if not self.cfg.use_dual_control:
            return rewards
        costs = np.asarray([tr.cost for tr in episode], dtype=np.float64)
        return rewards - self.dual * costs

    def _update_dual(self, mean_cost: float) -> None:
        if not self.cfg.use_dual_control:
            self.dual = 0.0
            return
        self.dual = float(
            max(0.0, self.dual + self.cfg.dual_lr * (mean_cost - self.cfg.cost_limit))
        )

    def _gae(self, episode: Sequence[Transition]) -> np.ndarray:
        rewards = self._adjusted_rewards(episode)
        values = np.asarray([tr.value for tr in episode] + [0.0], dtype=np.float64)
        advantages = np.zeros(len(episode), dtype=np.float64)
        gae = 0.0
        for t in reversed(range(len(episode))):
            non_terminal = 0.0 if episode[t].done else 1.0
            delta = rewards[t] + self.cfg.gamma * values[t + 1] * non_terminal - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * non_terminal * gae
            advantages[t] = gae
        return advantages

    def _build_segments(
        self,
        episode: Sequence[Transition],
        offset: int = 0,
        entropy_threshold: float | None = None,
    ) -> List[Segment]:
        advantages = self._gae(episode)
        segments: List[Segment] = []
        threshold = (
            self.entropy_threshold_current
            if entropy_threshold is None
            else float(entropy_threshold)
        )
        t = 0
        while t < len(episode):
            low_entropy = episode[t].h_eff < threshold
            if not self.cfg.use_macro_consolidation or not low_entropy:
                idx = [t]
                t += 1
            else:
                idx = []
                while (
                    t < len(episode)
                    and episode[t].h_eff < threshold
                    and len(idx) < self.cfg.max_macro_steps
                ):
                    idx.append(t)
                    t += 1
            old_log_prob = float(sum(episode[i].old_log_prob for i in idx))
            adv = float(sum((self.cfg.gamma**j) * advantages[i] for j, i in enumerate(idx)))
            ret = float(episode[idx[0]].value + adv)
            global_idx = [i + offset for i in idx]
            segments.append(
                Segment(indices=global_idx, old_log_prob=old_log_prob, advantage=adv, ret=ret)
            )

        return segments

    @staticmethod
    def _normalize_segment_advantages(segments: Sequence[Segment]) -> None:
        if len(segments) <= 1:
            return
        advantages = np.asarray([seg.advantage for seg in segments], dtype=np.float64)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
        for seg, advantage in zip(segments, advantages):
            seg.advantage = float(advantage)

    def _ppo_update(self, flat: Sequence[Transition], segments: Sequence[Segment]) -> None:
        states_np = np.stack([tr.state for tr in flat])
        actions_np = np.stack([tr.action for tr in flat])
        states = torch.as_tensor(states_np, dtype=torch.float32, device=self.cfg.device)
        actions = torch.as_tensor(actions_np, dtype=torch.float32, device=self.cfg.device)

        for _ in range(self.cfg.ppo_epochs):
            log_probs, entropies, values = self.policy.evaluate_actions(states, actions)
            policy_terms = []
            value_terms = []
            entropy_terms = []
            for seg in segments:
                idx = torch.as_tensor(seg.indices, dtype=torch.long, device=self.cfg.device)
                new_log_prob = log_probs.index_select(0, idx).sum()
                entropy = entropies.index_select(0, idx).mean()
                ratio = torch.exp(new_log_prob - seg.old_log_prob)
                adv = torch.as_tensor(seg.advantage, dtype=torch.float32, device=self.cfg.device)
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio)
                policy_terms.append(torch.min(ratio * adv, clipped * adv))
                value_terms.append((values[idx[0]] - seg.ret) ** 2)
                entropy_terms.append(entropy)

            policy_loss = -torch.stack(policy_terms).mean()
            value_loss = torch.stack(value_terms).mean()
            entropy_bonus = torch.stack(entropy_terms).mean()
            loss = (
                policy_loss
                + self.cfg.value_coef * value_loss
                - self.cfg.entropy_coef * entropy_bonus
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, episodes: int) -> Dict[str, float]:
        losses = []
        beam_objectives = []
        beampattern_losses = []
        cross_corrs = []
        sidelobe_ratios = []
        sidelobe_leakages = []
        target_means = []
        target_mins = []
        target_min_ratios = []
        target_band_error_means = []
        min_sinrs = []
        feasible = []
        costs = []
        batch_size = max(1, min(self.cfg.eval_batch_size, episodes))
        completed = 0
        while completed < episodes:
            cur_batch = min(batch_size, episodes - completed)
            envs = [
                ISACBeamformingEnv(
                    self.env_cfg,
                    structured_action=self.cfg.use_nullspace_action,
                    seed=self.cfg.seed + 10_000 + completed + i,
                )
                for i in range(cur_batch)
            ]
            states = self._reset_envs(envs)
            best = [None for _ in envs]
            active = np.ones(cur_batch, dtype=bool)

            for _ in range(self.env_cfg.max_steps):
                active_ids = np.flatnonzero(active)
                if active_ids.size == 0:
                    break
                state_batch = torch.as_tensor(
                    np.stack([states[i] for i in active_ids]),
                    dtype=torch.float32,
                    device=self.cfg.device,
                )
                actions, _, _, _ = self.policy.act_batch(state_batch, deterministic=True)
                active_envs = [envs[int(env_id)] for env_id in active_ids]
                step_results = self._step_envs(active_envs, actions)
                for local_idx, (env_id, result) in enumerate(zip(active_ids, step_results)):
                    idx = int(env_id)
                    envs[idx] = active_envs[local_idx]
                    states[idx], _, done, info = result
                    if info["feasible"] and (
                        best[idx] is None
                        or is_better_candidate(
                            info,
                            best[idx],
                            rel_tol=self.cfg.selection_rel_tolerance,
                        )
                    ):
                        best[idx] = info
                    active[idx] = not done

            for env, item in zip(envs, best):
                final = item if item is not None else env.current_metrics()
                losses.append(float(final["radar_loss"]))
                beam_objectives.append(float(final.get("beam_objective", final["radar_loss"])))
                beampattern_losses.append(float(final.get("beampattern_loss", 0.0)))
                cross_corrs.append(float(final.get("cross_corr", 0.0)))
                sidelobe_ratios.append(float(final.get("sidelobe_ratio", 0.0)))
                sidelobe_leakages.append(float(final.get("sidelobe_leakage", 0.0)))
                target_means.append(float(final.get("target_mean", 0.0)))
                target_mins.append(float(final.get("target_min", 0.0)))
                target_min_ratios.append(float(final.get("target_min_ratio", 0.0)))
                target_band_error_means.append(
                    float(final.get("target_band_error_mean", 0.0))
                )
                min_sinrs.append(float(final["min_sinr"]))
                feasible.append(float(final["feasible"]))
                costs.append(float(final["cost"]))
            completed += cur_batch

        return {
            "radar_loss": float(np.mean(losses)),
            "beam_objective": float(np.mean(beam_objectives)),
            "beampattern_loss": float(np.mean(beampattern_losses)),
            "cross_corr": float(np.mean(cross_corrs)),
            "sidelobe_ratio": float(np.mean(sidelobe_ratios)),
            "sidelobe_leakage": float(np.mean(sidelobe_leakages)),
            "target_mean": float(np.mean(target_means)),
            "target_min": float(np.mean(target_mins)),
            "target_min_ratio": float(np.mean(target_min_ratios)),
            "target_band_error_mean": float(np.mean(target_band_error_means)),
            "min_sinr": float(np.mean(min_sinrs)),
            "feasible_rate": float(np.mean(feasible)),
            "cost": float(np.mean(costs)),
        }
