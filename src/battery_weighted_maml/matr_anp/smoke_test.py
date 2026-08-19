"""End-to-end CPU smoke test on synthetic MATR-shaped pickles."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from .config import ExperimentConfig, load_config
from .compare_results import compare_evaluations
from .evaluate import evaluate_run
from .runtime import write_json
from .streaming_demo import streaming_run
from .synthetic import write_synthetic_matr_dataset
from .train import train_run


def smoke_config(
    output_root: str | Path,
    device: str = "cpu",
    base: ExperimentConfig | None = None,
) -> ExperimentConfig:
    config = base or ExperimentConfig()
    config.device = device
    config.paths.output_root = str(Path(output_root))
    config.data.minimum_valid_cycles = 24
    config.data.minimum_discharge_points = 8
    config.data.short_signal_threshold = 12
    config.q_grid.num_points = 32
    config.split.num_folds = 3
    config.split.validation_fraction = 0.25
    config.episode.minimum_current_cycle_position = 10
    config.episode.training_alpha_range = [0.3, 0.7]
    config.episode.evaluation_alphas = [0.3, 0.7]
    config.episode.beta_values = [0.0, 0.5, 1.0]
    config.episode.min_context_points = 4
    config.episode.max_context_points = 16
    config.episode.max_target_points = 16
    config.model.hidden_dim = 16
    config.model.wide_hidden_min = 16
    config.model.wide_hidden_max = 64
    config.model.latent_dim = 8
    config.model.attention_heads = 4
    config.model.mlp_layers = 2
    config.model.iv_channels = [4, 8]
    config.model.iv_embedding_dim = 8
    config.training.learning_rate = 5.0e-4
    config.training.max_steps = 1
    config.training.batch_size = 2
    config.training.kl_warmup_steps = 1
    config.training.validation_interval = 1
    config.training.validation_episodes_per_cell = 1
    config.training.early_stopping_patience = 2
    config.training.checkpoint_interval = 1
    config.training.log_interval = 1
    config.training.use_amp = False
    config.evaluation.mc_samples = 2
    config.evaluation.inference_repeats = 1
    config.evaluation.inference_warmup = 0
    config.validate()
    return config


def run_smoke(
    output_root: str | Path,
    device: str = "cpu",
    base_config: ExperimentConfig | None = None,
) -> Path:
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    data_root = write_synthetic_matr_dataset(
        root / "synthetic_data", num_cells=6, num_cycles=28, signal_points=32
    )
    config = smoke_config(root / "runs", device, base_config)
    run_directories = {}
    for model_name in ("soh_only_anp", "partial_iv_anp"):
        run_directory = train_run(
            config,
            model_name,
            fold=0,
            data_root=data_root,
            max_steps=1,
            output_root=root / "runs",
        )
        checkpoint = run_directory / "checkpoints/best.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(payload["step"]) != 1 or payload["model_spec"]["model_name"] != model_name:
            raise RuntimeError("smoke checkpoint reload verification failed")
        run_directories[model_name] = run_directory

    # Exercise real resume semantics by extending the partial model from step 1 to 2.
    partial_run = run_directories["partial_iv_anp"]
    resumed = train_run(
        config,
        "partial_iv_anp",
        fold=0,
        data_root=data_root,
        resume=partial_run / "checkpoints/last.pt",
        max_steps=2,
    )
    if resumed != partial_run:
        raise RuntimeError("resume unexpectedly created a new run directory")
    resumed_payload = torch.load(
        partial_run / "checkpoints/last.pt", map_location="cpu", weights_only=False
    )
    if int(resumed_payload["step"]) != 2:
        raise RuntimeError("smoke resume did not advance to step 2")

    partial_checkpoint = partial_run / "checkpoints/best.pt"
    evaluation_directory = evaluate_run(
        config,
        partial_checkpoint,
        data_root,
        output_dir=root / "evaluation",
        mc_samples=2,
    )
    soh_evaluation_directory = evaluate_run(
        config,
        run_directories["soh_only_anp"] / "checkpoints/best.pt",
        data_root,
        output_dir=root / "evaluation_soh_only",
        mc_samples=2,
    )
    soh_metrics = pd.read_csv(soh_evaluation_directory / "per_cell_metrics.csv")
    horizontal = soh_metrics[soh_metrics["status"] == "ok"].groupby(
        ["cell_id", "alpha"]
    )["future_rmse"].nunique()
    if horizontal.empty or int(horizontal.max()) != 1:
        raise RuntimeError("SOH-only smoke evaluation is not horizontal across beta")
    comparison_directory = compare_evaluations(
        [soh_evaluation_directory, evaluation_directory], root / "comparison"
    )
    test_cell = torch.load(
        partial_checkpoint, map_location="cpu", weights_only=False
    )["fold_split"]["test_cells"][0]
    streaming_directory = streaming_run(
        config,
        partial_checkpoint,
        data_root,
        alpha=0.5,
        cell_id=test_cell,
        output_dir=root / "streaming",
        mc_samples=2,
    )
    required = [
        evaluation_directory / "aggregate_metrics.csv",
        evaluation_directory / "plots/rmse_vs_beta.png",
        streaming_directory / "latency.csv",
        streaming_directory / "streaming_trajectory.png",
        comparison_directory / "rmse_model_comparison.png",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"smoke run did not create expected artifacts: {missing}")
    write_json(
        root / "smoke_manifest.json",
        {
            "status": "passed",
            "completed_at": datetime.now().isoformat(),
            "device": device,
            "data_root": str(data_root),
            "runs": {key: str(value) for key, value in run_directories.items()},
            "evaluation": str(evaluation_directory),
            "soh_only_evaluation": str(soh_evaluation_directory),
            "comparison": str(comparison_directory),
            "streaming": str(streaming_directory),
        },
    )
    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic MATR ANP smoke test")
    parser.add_argument("--output-root", default="outputs/matr_partial_iv_anp/smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", default="configs/matr_partial_iv_anp.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = run_smoke(
        Path(args.output_root) / timestamp,
        args.device,
        load_config(args.config),
    )
    print(f"Smoke test passed: {destination}")


if __name__ == "__main__":
    main()
