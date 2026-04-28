from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Dict

import numpy as np
import torch
from tqdm.auto import tqdm

from .buffer import RolloutBatch
from .config import EnvConfig, PPOConfig, SystemConfig, TrainConfig
from .env import ISACBeamformingEnv
from .heppo import HEPPOAgent
from .logger import append_csv
from .metrics import compute_all_metrics
from .ppo import PPOAgent
from .utils import save_json, set_global_seed


def make_agent(algo: str, state_dim: int, action_dim: int, cfg: PPOConfig, device: str):
    if algo == "ppo":
        return PPOAgent(state_dim, action_dim, cfg, device)
    if algo == "heppo":
        return HEPPOAgent(state_dim, action_dim, cfg, device)
    raise ValueError(f"unknown algorithm: {algo}")


def train_algorithms(
    sys_cfg: SystemConfig,
    env_cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    train_cfg: TrainConfig,
    log_dir: str | Path,
    show_progress: bool = True,
) -> Dict[str, object]:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = log_path / train_cfg.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        log_path / "config.json",
        {
            "system": asdict(sys_cfg),
            "env": asdict(env_cfg),
            "ppo": asdict(ppo_cfg),
            "train": asdict(train_cfg),
        },
    )
    summary = {"algorithms": list(train_cfg.algos), "seeds": list(train_cfg.seeds)}
    run_jobs = [(algo, seed) for algo in train_cfg.algos for seed in train_cfg.seeds]
    run_iter = tqdm(
        run_jobs,
        desc="training runs",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for algo, seed in run_iter:
        run_iter.set_description(f"{algo.upper()} seed={seed}")
        set_global_seed(seed)
        env = ISACBeamformingEnv(sys_cfg, env_cfg, seed=seed)
        env.reset()
        agent = make_agent(algo, env.state_dim, env.action_dim, ppo_cfg, train_cfg.device)
        update_iter = tqdm(
            range(1, ppo_cfg.updates + 1),
            desc=f"{algo.upper()} updates",
            dynamic_ncols=True,
            leave=False,
            disable=not show_progress,
        )
        for update in update_iter:
            start = perf_counter()
            batch, rollout_stats = collect_rollout(
                agent, sys_cfg, env_cfg, seed, ppo_cfg.episodes_per_update, train_cfg.device
            )
            update_stats = agent.update(batch)
            elapsed = perf_counter() - start
            train_row = {
                "algo": algo,
                "seed": seed,
                "update": update,
                "episode": update * ppo_cfg.episodes_per_update,
                "train_reward": rollout_stats["reward"],
                "train_objective": rollout_stats["objective"],
                "actor_loss": update_stats["actor_loss"],
                "critic_loss": update_stats["critic_loss"],
                "entropy": update_stats["entropy"],
                "alpha_entropy": update_stats["alpha_entropy"],
                "approx_kl": update_stats["approx_kl"],
                "clip_fraction": update_stats["clip_fraction"],
                "grad_norm": update_stats["grad_norm"],
                "update_time_sec": elapsed,
            }
            entropy_row = {
                "algo": algo,
                "seed": seed,
                "update": update,
                "entropy_mean": update_stats.get("entropy_mean", update_stats["entropy"]),
                "entropy_threshold": update_stats.get("entropy_threshold", np.nan),
                "high_entropy_rate": update_stats.get("high_entropy_rate", 1.0),
                "macro_step_rate": update_stats.get("macro_step_rate", 0.0),
                "average_macro_length": update_stats.get("average_macro_length", 1.0),
                "num_macro_segments": update_stats.get("num_macro_segments", 0.0),
            }
            append_csv(log_path / "train_history.csv", [train_row])
            append_csv(log_path / "entropy_history.csv", [entropy_row])
            eval_row = None
            if update == 1 or update % train_cfg.eval_interval == 0 or update == ppo_cfg.updates:
                eval_row = evaluate_agent(
                    agent, sys_cfg, env_cfg, train_cfg, algo, seed, update
                )
                append_csv(log_path / "eval_history.csv", [eval_row])
            postfix = {
                "reward": f"{rollout_stats['reward']:.3g}",
                "J": f"{rollout_stats['objective']:.3g}",
                "ent": f"{update_stats['entropy']:.3g}",
                "macro": f"{update_stats.get('macro_step_rate', 0.0):.2f}",
                "sec": f"{elapsed:.1f}",
            }
            if eval_row is not None:
                postfix["evalJ"] = f"{eval_row['eval_objective']:.3g}"
                postfix["feas"] = f"{eval_row['eval_feasible_rate']:.2f}"
            update_iter.set_postfix(postfix)
        torch.save(
            {
                "algo": algo,
                "seed": seed,
                "state_dim": env.state_dim,
                "action_dim": env.action_dim,
                "model": agent.policy.state_dict(),
                "system": asdict(sys_cfg),
                "env": asdict(env_cfg),
                "ppo": asdict(ppo_cfg),
            },
            checkpoint_dir / f"{algo}.pt",
        )
    save_json(log_path / "summary.json", summary)
    return summary


@torch.no_grad()
def collect_rollout(
    agent,
    sys_cfg: SystemConfig,
    env_cfg: EnvConfig,
    seed: int,
    episodes: int,
    device: str,
) -> tuple[RolloutBatch, Dict[str, float]]:
    states = []
    actions = []
    rewards = []
    dones = []
    values = []
    log_probs = []
    entropies = []
    episode_steps = []
    next_values = []
    final_objectives = []
    final_rewards = []
    for ep in range(episodes):
        env = ISACBeamformingEnv(sys_cfg, env_cfg, seed=seed * 100_000 + ep)
        state = env.reset()
        total_reward = 0.0
        for step in range(env_cfg.episode_steps):
            state_t = torch.as_tensor(state[None, :], dtype=torch.float32, device=device)
            action_t, _, log_prob_t, entropy_t, value_t = agent.policy.act(state_t)
            action = action_t.squeeze(0).cpu().numpy().astype(np.float32)
            next_state, reward, done, info = env.step(action)
            next_state_t = torch.as_tensor(next_state[None, :], dtype=torch.float32, device=device)
            next_value = 0.0 if done else float(agent.policy.value(next_state_t).item())
            states.append(state)
            actions.append(action)
            rewards.append(float(reward))
            dones.append(bool(done))
            values.append(float(value_t.item()))
            log_probs.append(float(log_prob_t.item()))
            entropies.append(float(entropy_t.item() / max(env.action_dim, 1)))
            episode_steps.append(step)
            next_values.append(next_value)
            total_reward += float(reward)
            state = next_state
            if done:
                break
        final_objectives.append(float(env.current_info["objective"]))
        final_rewards.append(total_reward)
    batch = RolloutBatch(
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=bool),
        values=np.asarray(values, dtype=np.float32),
        log_probs=np.asarray(log_probs, dtype=np.float32),
        entropies=np.asarray(entropies, dtype=np.float32),
        episode_steps=np.asarray(episode_steps, dtype=np.int32),
        next_values=np.asarray(next_values, dtype=np.float32),
    )
    return batch, {
        "reward": float(np.mean(final_rewards)),
        "objective": float(np.mean(final_objectives)),
    }


@torch.no_grad()
def evaluate_agent(
    agent,
    sys_cfg: SystemConfig,
    env_cfg: EnvConfig,
    train_cfg: TrainConfig,
    algo: str,
    seed: int,
    update: int,
) -> Dict[str, float | int | str]:
    rewards = []
    metrics = []
    for idx in range(train_cfg.eval_channels):
        env = ISACBeamformingEnv(sys_cfg, env_cfg, seed=train_cfg.eval_seed + idx)
        state = env.reset()
        total_reward = 0.0
        for _ in range(env_cfg.episode_steps):
            state_t = torch.as_tensor(state[None, :], dtype=torch.float32, device=train_cfg.device)
            action_t, _, _, _, _ = agent.policy.act(state_t, deterministic=True)
            state, reward, done, _ = env.step(action_t.squeeze(0).cpu().numpy())
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
        metrics.append(compute_all_metrics(env.W, env.H, sys_cfg))
    return {
        "algo": algo,
        "seed": seed,
        "update": update,
        "eval_reward": float(np.mean(rewards)),
        "eval_objective": float(np.mean([m["objective"] for m in metrics])),
        "eval_Lr": float(np.mean([m["Lr"] for m in metrics])),
        "eval_Lr1": float(np.mean([m["Lr1"] for m in metrics])),
        "eval_Lr2": float(np.mean([m["Lr2"] for m in metrics])),
        "eval_C_sinr": float(np.mean([m["C_sinr"] for m in metrics])),
        "eval_min_sinr_db": float(np.mean([m["min_sinr_db"] for m in metrics])),
        "eval_feasible_rate": float(np.mean([float(m["feasible"]) for m in metrics])),
        "eval_peak_sidelobe_ratio": float(np.mean([m["peak_sidelobe_ratio"] for m in metrics])),
        "eval_target_band_error": float(np.mean([m["target_band_error"] for m in metrics])),
    }
