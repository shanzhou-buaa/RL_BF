import unittest

import numpy as np

from cse_ppo_isac.config import EnvConfig, PPOConfig
from cse_ppo_isac.env import ISACBeamformingEnv
from cse_ppo_isac.math_utils import row_power_normalize
from cse_ppo_isac.methods import method_flags
from cse_ppo_isac.selection import is_better_candidate
from cse_ppo_isac.trainer import CSEPPOTrainer, Transition


class EnvObjectiveAndActionTest(unittest.TestCase):
    def test_state_contains_channel_beamformer_and_diagnostics(self):
        cfg = EnvConfig(num_antennas=4, num_users=2)
        env = ISACBeamformingEnv(cfg, structured_action=True, seed=3)

        state = env.reset()

        expected = (
            2 * cfg.num_users * cfg.num_antennas
            + 2 * cfg.num_antennas * (cfg.num_users + cfg.num_antennas)
            + cfg.num_users
            + 1
            + len(cfg.target_angles_deg)
            + 1
        )
        self.assertEqual(env.state_size, expected)
        self.assertEqual(state.shape, (env.state_size,))

    def test_zero_residual_keeps_initialized_beamformer(self):
        cfg = EnvConfig(num_antennas=3, num_users=1)
        env = ISACBeamformingEnv(cfg, structured_action=True, seed=5)
        env.reset()
        initial_W = env.W.copy()

        env.step(np.zeros(env.action_size))

        np.testing.assert_allclose(env.W, initial_W, atol=1.0e-12)

    def test_policy_reset_uses_feasible_initializer(self):
        cfg = EnvConfig(num_antennas=10, num_users=2, sinr_threshold_db=12.0, init_mode="policy")
        env = ISACBeamformingEnv(cfg, seed=7)

        env.reset()
        metrics = env.current_metrics()

        self.assertTrue(metrics["feasible"])
        self.assertGreaterEqual(metrics["min_sinr"], cfg.sinr_threshold)

    def test_step_reward_uses_relative_improvement_and_configured_penalties(self):
        cfg = EnvConfig(
            num_antennas=3,
            num_users=1,
            sinr_threshold_db=60.0,
            constraint_reward_weight=2.0,
            action_penalty=0.5,
            use_action_line_search=False,
        )
        env = ISACBeamformingEnv(cfg, seed=9)
        env.reset()
        action = np.linspace(-1.0, 1.0, env.action_size)

        _, reward, _, info = env.step(action)

        expected = (
            info["loss_reward"]
            + info["constraint_reward"]
            - cfg.action_penalty * info["action_norm"]
        )
        self.assertAlmostEqual(reward, expected, places=12)
        self.assertEqual(info["sinr_violations"], cfg.num_users)

    def test_sinr_penalty_is_dense_inside_infeasible_region(self):
        cfg = EnvConfig(num_antennas=4, num_users=1, sinr_threshold_db=60.0)
        better_env = ISACBeamformingEnv(cfg, seed=19)
        worse_env = ISACBeamformingEnv(cfg, seed=19)
        better_env.reset()
        worse_env.reset()
        assert better_env.H is not None

        better_W = np.zeros((cfg.num_antennas, cfg.num_users + cfg.num_antennas), dtype=np.complex128)
        better_W[:, 0] = better_env.H[0].conj()

        rng = np.random.default_rng(19)
        worse_W = 1.0e-3 * rng.standard_normal(
            (cfg.num_antennas, cfg.num_users + cfg.num_antennas)
        ).astype(np.complex128)
        worse_W[:, cfg.num_users:] = rng.standard_normal(
            (cfg.num_antennas, cfg.num_antennas)
        ) + 1j * rng.standard_normal((cfg.num_antennas, cfg.num_antennas))

        better_info = better_env._candidate_info(
            row_power_normalize(better_W, cfg.total_power),
            step_scale=1.0,
            action_norm=0.0,
            accepted=True,
        )
        worse_info = worse_env._candidate_info(
            row_power_normalize(worse_W, cfg.total_power),
            step_scale=1.0,
            action_norm=0.0,
            accepted=True,
        )

        self.assertFalse(better_info["feasible"])
        self.assertFalse(worse_info["feasible"])
        self.assertGreater(better_info["min_sinr"], worse_info["min_sinr"])
        self.assertLess(better_info["sinr_penalty"], worse_info["sinr_penalty"])

    def test_entropy_threshold_is_adapted_from_rollout_values(self):
        env_cfg = EnvConfig(num_antennas=3, num_users=1)
        ppo_cfg = PPOConfig(
            episodes_per_update=1,
            rollout_backend="serial",
            entropy_threshold_quantile=0.5,
            entropy_threshold_ema=0.0,
        )
        trainer = CSEPPOTrainer(env_cfg, ppo_cfg)
        transitions = [
            Transition(
                state=np.zeros(env_cfg.num_users * env_cfg.num_antennas * 2, dtype=np.float32),
                action=np.zeros(2 * env_cfg.num_antennas * (env_cfg.num_users + env_cfg.num_antennas)),
                old_log_prob=0.0,
                entropy=0.0,
                reward=0.0,
                cost=0.0,
                value=0.0,
                done=False,
                h_eff=value,
                info={},
            )
            for value in (0.1, 0.2, 0.8, 1.0)
        ]

        threshold = trainer._update_entropy_threshold(transitions)
        trainer.close()

        self.assertAlmostEqual(threshold, 0.5, places=12)
        self.assertNotEqual(threshold, ppo_cfg.entropy_threshold)

    def test_effective_entropy_keeps_exploration_floor_when_infeasible(self):
        env_cfg = EnvConfig(num_antennas=3, num_users=1)
        ppo_cfg = PPOConfig(
            episodes_per_update=1,
            rollout_backend="serial",
            feasibility_entropy_floor=0.2,
        )
        trainer = CSEPPOTrainer(env_cfg, ppo_cfg)
        entropy = 10.0

        h_eff = trainer._effective_entropy(entropy, min_margin=-1.0e6)
        trainer.close()

        self.assertGreaterEqual(
            h_eff,
            ppo_cfg.feasibility_entropy_floor * entropy / trainer.env.action_size,
        )

    def test_beam_objective_is_full_liu_radar_loss(self):
        cfg = EnvConfig(cross_corr_weight=1.0)
        env = ISACBeamformingEnv(cfg, structured_action=False, seed=11)
        env.reset()

        metrics = env.objective_for_W(env.W)

        self.assertAlmostEqual(metrics["beam_objective"], metrics["radar_loss"], places=12)
        self.assertAlmostEqual(
            metrics["radar_loss"],
            metrics["beampattern_loss"] + cfg.cross_corr_weight * metrics["cross_corr"],
            places=12,
        )

    def test_structured_action_space_uses_nullspace_residual_parameters(self):
        cfg = EnvConfig(num_antennas=10, num_users=2)
        structured_env = ISACBeamformingEnv(cfg, structured_action=True, seed=13)
        plain_env = ISACBeamformingEnv(cfg, structured_action=False, seed=13)

        expected_structured = 2 * (
            cfg.num_users
            + (cfg.num_antennas - cfg.num_users)
            * (cfg.num_users + cfg.num_antennas)
        )
        expected_plain = 2 * cfg.num_antennas * (cfg.num_users + cfg.num_antennas)

        self.assertEqual(structured_env.action_size, expected_structured)
        self.assertEqual(plain_env.action_size, expected_plain)

    def test_structured_ppo_baseline_uses_nullspace_without_cse_controls(self):
        self.assertEqual(method_flags("structured"), (True, False, False))

    def test_inference_rejects_beampattern_only_gain_with_worse_cross_corr(self):
        incumbent = {
            "feasible": True,
            "beam_objective": 1.0,
            "radar_loss": 1.0,
            "beampattern_loss": 0.50,
            "cross_corr": 0.50,
            "sidelobe_leakage": 0.80,
            "target_mean": 2.0,
            "cost": 0.0,
        }
        candidate = {
            "feasible": True,
            "beam_objective": 1.01,
            "radar_loss": 1.01,
            "beampattern_loss": 0.40,
            "cross_corr": 0.61,
            "sidelobe_leakage": 0.85,
            "target_mean": 1.8,
            "cost": 0.0,
        }

        self.assertFalse(is_better_candidate(candidate, incumbent, rel_tol=0.0))
        self.assertFalse(is_better_candidate(candidate, incumbent, rel_tol=0.02))

    def test_inference_rejects_cross_corr_only_gain_with_worse_beampattern(self):
        incumbent = {
            "feasible": True,
            "beam_objective": 1.0,
            "radar_loss": 1.0,
            "beampattern_loss": 0.42,
            "cross_corr": 0.58,
            "sidelobe_leakage": 0.88,
            "sidelobe_ratio": 0.80,
            "target_mean": 2.30,
            "target_min_ratio": 0.89,
            "target_band_error_mean": 0.013,
            "cost": 0.0,
        }
        cross_only_candidate = {
            "feasible": True,
            "beam_objective": 0.97,
            "radar_loss": 0.97,
            "beampattern_loss": 0.45,
            "cross_corr": 0.52,
            "sidelobe_leakage": 0.96,
            "sidelobe_ratio": 0.87,
            "target_mean": 2.15,
            "target_min_ratio": 0.86,
            "target_band_error_mean": 0.018,
            "cost": 0.0,
        }

        self.assertFalse(
            is_better_candidate(cross_only_candidate, incumbent, rel_tol=0.0)
        )

    def test_inference_tie_break_prefers_lower_sidelobe_then_stronger_mainlobe(self):
        incumbent = {
            "feasible": True,
            "beam_objective": 1.0,
            "radar_loss": 1.0,
            "beampattern_loss": 0.50,
            "cross_corr": 0.40,
            "sidelobe_leakage": 0.80,
            "target_mean": 2.0,
            "cost": 0.0,
        }
        lower_sidelobe = {
            **incumbent,
            "beam_objective": 0.99,
            "radar_loss": 0.99,
            "cross_corr": 0.39,
            "sidelobe_leakage": 0.70,
            "target_mean": 1.9,
        }
        stronger_mainlobe = {
            **incumbent,
            "beam_objective": 1.0,
            "radar_loss": 1.0,
            "cross_corr": 0.40,
            "target_mean": 2.2,
        }

        self.assertTrue(is_better_candidate(lower_sidelobe, incumbent, rel_tol=0.02))
        self.assertTrue(is_better_candidate(stronger_mainlobe, incumbent, rel_tol=0.02))

    def test_step_reward_reports_new_penalty_terms(self):
        cfg = EnvConfig(num_antennas=3, num_users=1, sinr_violation_penalty=3.0)
        env = ISACBeamformingEnv(cfg, structured_action=False, seed=17)
        env.reset()

        action = np.ones(env.action_size)
        _, reward, _, info = env.step(action)

        self.assertAlmostEqual(reward, info["reward"], places=12)
        self.assertIn("sinr_violations", info)
        self.assertIn("power_error", info)
        self.assertIn("relative_loss_improvement", info)

    def test_line_search_can_reject_worse_residual(self):
        cfg = EnvConfig(num_antennas=3, num_users=1)
        env = ISACBeamformingEnv(cfg, structured_action=True, seed=0)
        env.reset()
        initial_W = env.W.copy()

        rng = np.random.default_rng(0)
        action = 100.0 * rng.standard_normal(env.action_size)
        _, _, _, info = env.step(action)

        self.assertLessEqual(info["step_scale"], 1.0)
        if not info["accepted"]:
            np.testing.assert_allclose(env.W, initial_W, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
