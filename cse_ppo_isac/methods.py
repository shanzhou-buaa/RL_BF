from __future__ import annotations

from .config import PPOConfig


def method_flags(name: str) -> tuple[bool, bool, bool]:
    if name == "cse":
        return True, True, True
    if name == "structured":
        return True, False, False
    if name == "vanilla":
        return False, False, False
    raise ValueError(f"unknown learning method: {name}")


def build_ppo_config(args, method: str, *, seed: int | None = None) -> PPOConfig:
    use_nullspace, use_macro, use_feas = method_flags(method)
    if getattr(args, "use_nullspace", False):
        use_nullspace = True
    if getattr(args, "no_nullspace", False):
        use_nullspace = False
    if getattr(args, "no_macro", False):
        use_macro = False
    if getattr(args, "no_feasibility_entropy", False):
        use_feas = False

    return PPOConfig(
        updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        rollout_workers=getattr(args, "rollout_workers", 1),
        rollout_backend=getattr(args, "rollout_backend", "process"),
        eval_episodes=args.eval_episodes,
        eval_batch_size=args.eval_batch_size,
        ppo_epochs=getattr(args, "ppo_epochs", 3),
        hidden_dim=getattr(args, "hidden_dim", 128),
        learning_rate=getattr(args, "learning_rate", 6.0e-4),
        entropy_coef=getattr(args, "entropy_coef", 1.0e-3),
        initial_log_std=getattr(args, "initial_log_std", -1.1),
        dual_lr=getattr(args, "dual_lr", 0.05),
        cost_limit=getattr(args, "cost_limit", 0.0),
        selection_rel_tolerance=args.selection_rel_tol,
        seed=args.seed if seed is None else seed,
        device=args.device,
        use_nullspace_action=use_nullspace,
        use_macro_consolidation=use_macro,
        use_feasibility_weighted_entropy=use_feas,
        use_dual_control=not getattr(args, "no_dual", False),
    )
