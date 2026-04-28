import torch

import run_train


def test_parse_args_exposes_single_process_acceleration_flags():
    args = run_train.parse_args(
        [
            "--eval-interval",
            "5",
            "--torch-threads",
            "8",
            "--allow-tf32",
        ]
    )

    assert args.eval_interval == 5
    assert args.torch_threads == 8
    assert args.allow_tf32 is True


def test_main_applies_torch_threads_and_tf32(monkeypatch, tmp_path):
    captured = {}

    def fake_train_algorithms(_sys_cfg, _env_cfg, _ppo_cfg, train_cfg, _log_dir, show_progress):
        captured["eval_interval"] = train_cfg.eval_interval
        captured["show_progress"] = show_progress
        return {}

    thread_calls = []
    monkeypatch.setattr(run_train, "train_algorithms", fake_train_algorithms)
    monkeypatch.setattr(torch, "set_num_threads", lambda value: thread_calls.append(value))
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_train.py",
            "--updates",
            "1",
            "--episodes-per-update",
            "1",
            "--eval-interval",
            "7",
            "--torch-threads",
            "3",
            "--allow-tf32",
            "--log-dir",
            str(tmp_path),
            "--no-progress",
        ],
    )

    run_train.main()

    assert thread_calls == [3]
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True
    assert captured == {"eval_interval": 7, "show_progress": False}
