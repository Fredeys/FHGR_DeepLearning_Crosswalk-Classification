#!/usr/bin/env python3
"""Lightweight workflow runner for the existing project scripts.

This file only orchestrates the existing Python files with subprocess. It does
not reimplement preprocessing, training, fine-tuning, evaluation,
visualization, error analysis, or inference logic.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import config


# ---------------------------------------------------------------------------
# Simple workflow switches
# ---------------------------------------------------------------------------

ENABLE_PREPROCESSING = True
ENABLE_BASELINE = True
ENABLE_TRAINING = True
ENABLE_FINE_TUNING = True
ENABLE_EVALUATION = True
ENABLE_VISUALIZATION = True
ENABLE_ERROR_ANALYSIS = True
ENABLE_INFERENCE = False


# ---------------------------------------------------------------------------
# Existing project scripts
# ---------------------------------------------------------------------------

PROJECT_ROOT = config.PROJECT_ROOT

PREPROCESSING_SCRIPT = PROJECT_ROOT / "02_preprocessing.py"
BASELINE_SCRIPT: Path | None = PROJECT_ROOT / "00_baseline.py"
TRAINING_SCRIPT = PROJECT_ROOT / "04_train_phase1.py"
FINE_TUNING_SCRIPT = PROJECT_ROOT / "05_fine_tune.py"
EVALUATION_SCRIPT = PROJECT_ROOT / "06_evaluate_final_model.py"
VISUALIZATION_SCRIPT: Path | None = None
ERROR_ANALYSIS_SCRIPT: Path | None = None
INFERENCE_SCRIPT = PROJECT_ROOT / "07_inference.py"


@dataclass(frozen=True)
class Stage:
    name: str
    enabled: bool
    script: Path | None
    args: tuple[str, ...] = ()
    note: str | None = None


def print_header(stage_name: str) -> None:
    print("\n" + "=" * 80)
    print(f"STAGE: {stage_name}")
    print("=" * 80)


def run_stage(stage: Stage) -> None:
    print_header(stage.name)

    if not stage.enabled:
        print(f"Skipped: {stage.name} is disabled.")
        return

    if stage.script is None:
        print(f"Skipped: no separate script exists for {stage.name}.")
        if stage.note:
            print(stage.note)
        return

    if not stage.script.exists():
        print(f"FAILED: script not found for {stage.name}: {stage.script}")
        raise SystemExit(1)

    command = [sys.executable, str(stage.script), *stage.args]
    print("Command:")
    print("  " + " ".join(command))

    try:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as error:
        print(f"\nFAILED: {stage.name}")
        print(f"Script: {stage.script}")
        print(f"Exit code: {error.returncode}")
        raise SystemExit(error.returncode) from error

    print(f"\nSUCCESS: {stage.name}")


def build_workflow() -> list[Stage]:
    """Return the workflow from training onward, without re-splitting data."""
    stages = [
        Stage(
            name="1. Preprocessing / dataset loading checks",
            enabled=ENABLE_PREPROCESSING,
            script=PREPROCESSING_SCRIPT,
            args=(
                "--dataset-dir",
                str(config.DATASET_DIR),
                "--batch-size",
                str(config.BATCH_SIZE),
                "--seed",
                str(config.RANDOM_SEED),
            ),
        ),
        Stage(
            name="2. Baseline calculation",
            enabled=ENABLE_BASELINE,
            script=BASELINE_SCRIPT,
            args=(
                "--dataset-dir",
                str(config.DATASET_DIR),
                "--output-dir",
                str(config.ARTIFACTS_DIR / "baseline"),
            ),
        ),
        Stage(
            name="3. Training phase 1",
            enabled=ENABLE_TRAINING,
            script=TRAINING_SCRIPT,
            args=(
                "--dataset-dir",
                str(config.DATASET_DIR),
                "--output-dir",
                str(config.PHASE1_OUTPUT_DIR),
                "--batch-size",
                str(config.BATCH_SIZE),
                "--epochs",
                str(config.EPOCHS_PHASE1),
                "--learning-rate",
                str(config.INITIAL_LEARNING_RATE),
                "--seed",
                str(config.RANDOM_SEED),
            ),
        ),
        Stage(
            name="4. Fine-tuning",
            enabled=ENABLE_FINE_TUNING,
            script=FINE_TUNING_SCRIPT,
            args=(
                "--dataset-dir",
                str(config.DATASET_DIR),
                "--phase1-model",
                str(config.PHASE1_OUTPUT_DIR / "best_model.keras"),
                "--output-dir",
                str(config.PHASE2_OUTPUT_DIR),
                "--batch-size",
                str(config.FINE_TUNING_BATCH_SIZE),
                "--epochs",
                str(config.FINE_TUNING_EPOCHS),
                "--learning-rate",
                str(config.FINE_TUNING_LEARNING_RATE),
                "--unfreeze-last",
                str(config.UNFREEZE_TOP_LAYERS),
                "--seed",
                str(config.RANDOM_SEED),
            ),
        ),
        Stage(
            name="5. Evaluation on test set",
            enabled=ENABLE_EVALUATION,
            script=EVALUATION_SCRIPT,
            args=(
                "--dataset-dir",
                str(config.DATASET_DIR),
                "--model-path",
                str(config.PHASE2_OUTPUT_DIR / "best_model.keras"),
                "--output-dir",
                str(config.FINAL_EVALUATION_DIR),
                "--error-analysis-dir",
                str(config.ERROR_ANALYSIS_DIR),
                "--plots-dir",
                str(config.PLOTS_DIR),
                "--logs-dir",
                str(config.LOGS_DIR),
                "--batch-size",
                str(config.BATCH_SIZE),
                "--threshold",
                str(config.EVALUATION_THRESHOLD),
                "--max-error-images",
                str(config.MAX_ERROR_IMAGES),
                "--seed",
                str(config.RANDOM_SEED),
            ),
        ),
        Stage(
            name="6. Visualization export",
            enabled=ENABLE_VISUALIZATION,
            script=VISUALIZATION_SCRIPT,
            note=(
                "Visualization exports are generated by the existing training, "
                "fine-tuning, and evaluation scripts."
            ),
        ),
        Stage(
            name="7. Error analysis",
            enabled=ENABLE_ERROR_ANALYSIS,
            script=ERROR_ANALYSIS_SCRIPT,
            note="Error analysis is generated by 06_evaluate_final_model.py.",
        ),
        Stage(
            name="8. Optional inference on new data",
            enabled=ENABLE_INFERENCE,
            script=INFERENCE_SCRIPT,
            args=(
                str(config.INFERENCE_INPUT_DIR),
                "--model-path",
                str(config.PHASE2_OUTPUT_DIR / "best_model.keras"),
                "--threshold",
                str(config.INFERENCE_THRESHOLD),
                "--batch-size",
                str(config.BATCH_SIZE),
                "--inference-dir",
                str(config.INFERENCE_DIR),
            ),
        ),
    ]
    return stages


def main() -> int:
    print(f"Workflow runner: {config.PROJECT_NAME}")
    print(f"Project root: {PROJECT_ROOT}")
    print("Note: this runner does not execute 01_split_dataset.py.")

    for stage in build_workflow():
        run_stage(stage)

    print("\nWorkflow completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
