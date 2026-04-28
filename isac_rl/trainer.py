from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
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


def _step_env(job):
    env, action = job
    return env.step(action)


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
    summary: dict[str, object] = {
        "algorithms": list(train_cfg.algos),
        "seeds": list(train_cfg.seeds),
        "checkpoints": {},
    }
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
                agent,
                sys_cfg,
                env_cfg,
                seed,
                update,
                ppo_cfg.episodes_per_update,
                train_cfg.device,
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
                "train_feasible_rate": rollout_stats["feasible_rate"],
                "train_min_sinr_db": rollout_stats["min_sinr_db"],
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
                "feas": f"{rollout_stats['feasible_rate']:.2f}",
                "ent": f"{update_stats['entropy']:.3g}",
                "macro": f"{update_stats.get('macro_step_rate', 0.0):.2f}",
                "sec": f"{elapsed:.1f}",
            }
            if eval_row is not None:
                postfix["evalJ"] = f"{eval_row['eval_objective']:.3g}"
                postfix["evalFeas"] = f"{eval_row['eval_feasible_rate']:.2f}"
            update_iter.set_postfix(postfix)
        checkpoint_name = f"{algo}_seed{seed}.pt"
        checkpoint_path = checkpoint_dir / checkpoint_name
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
            checkpoint_path,
        )
        summary["checkpoints"][f"{algo}_seed{seed}"] = str(checkpoint_path)
    save_json(log_path / "summary.json", summary)
    return summary


@torch.no_grad()
def collect_rollout(
    agent,
    sys_cfg: SystemConfig,
    env_cfg: EnvConfig,
    seed: int,
    update: int,
    episodes: int,
    device: str,
) -> tuple[RolloutBatch, Dict[str, float]]:
    envs = [
        ISACBeamformingEnv(
            sys_cfg,
            env_cfg,
            seed=seed * 1_000_000 + update * episodes + ep,
        )
        for ep in range(episodes)
    ]
    states_np = np.stack([env.reset() for env in envs]).astype(np.float32)
    storage = [
        {
            "states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "values": [],
            "log_probs": [],
            "entropies": [],
            "episode_steps": [],
            "next_values": [],
        }
        for _ in range(episodes)
    ]
    workers = min(max(os.cpu_count() or 1, 1), max(episodes, 1))
    action_dims = [max(env.action_dim, 1) for env in envs]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for step in range(env_cfg.episode_steps):
            states_t = torch.as_tensor(states_np, dtype=torch.float32, device=device)
            action_t, _, log_prob_t, entropy_t, value_t = agent.policy.act(states_t)

            actions_np = action_t.cpu().numpy().astype(np.float32)
            values_np = value_t.cpu().numpy().astype(np.float32)
            log_probs_np = log_prob_t.cpu().numpy().astype(np.float32)
            entropies_np = entropy_t.cpu().numpy().astype(np.float32)

            results = list(pool.map(_step_env, zip(envs, actions_np)))
            next_states_np = np.stack([result[0] for result in results]).astype(np.float32)
            dones_np = np.asarray([bool(result[2]) for result in results], dtype=bool)

            if step == env_cfg.episode_steps - 1:
                next_values_np = np.zeros(episodes, dtype=np.float32)
            else:
                next_states_t = torch.as_tensor(
                    next_states_np,
                    dtype=torch.float32,
                    device=device,
                )
                next_values_np = (
                    agent.policy.value(next_states_t).cpu().numpy().astype(np.float32)
                )

            for env_idx, (_next_state, reward, done, _info) in enumerate(results):
                storage_item = storage[env_idx]
                storage_item["states"].append(states_np[env_idx])
                storage_item["actions"].append(actions_np[env_idx])
                storage_item["rewards"].append(float(reward))
                storage_item["dones"].append(bool(done))
                storage_item["values"].append(float(values_np[env_idx]))
                storage_item["log_probs"].append(float(log_probs_np[env_idx]))
                storage_item["entropies"].append(
                    float(entropies_np[env_idx] / action_dims[env_idx])
                )
                storage_item["episode_steps"].append(step)
                storage_item["next_values"].append(
                    0.0 if dones_np[env_idx] else float(next_values_np[env_idx])
                )

            states_np = next_states_np

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
    final_feasible = []
    final_min_sinr_db = []

    for env_idx, env in enumerate(envs):
        item = storage[env_idx]
        states.extend(item["states"])
        actions.extend(item["actions"])
        rewards.extend(item["rewards"])
        dones.extend(item["dones"])
        values.extend(item["values"])
        log_probs.extend(item["log_probs"])
        entropies.extend(item["entropies"])
        episode_steps.extend(item["episode_steps"])
        next_values.extend(item["next_values"])
        assert env.current_info is not None
        final_objectives.append(float(env.current_info["objective"]))
        final_rewards.append(float(np.sum(item["rewards"])))
        final_feasible.append(float(env.current_info["feasible"]))
        final_min_sinr_db.append(float(env.current_info["min_sinr_db"]))
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
        "feasible_rate": float(np.mean(final_feasible)),
        "min_sinr_db": float(np.mean(final_min_sinr_db)),
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
        assert env.W is not None and env.H is not None
        metrics.append(compute_all_metrics(env.W, env.H, sys_cfg, env.metric_cache))
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
