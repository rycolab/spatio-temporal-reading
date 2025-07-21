#!/usr/bin/env python3
"""
train.py – High‑level orchestration script for AlternatingRenewalModel
=====================================================================

Run `python train.py -h` for a complete list of CLI flags.
All domain‑specific code (datasets, model, trainer, etc.) is imported;
only experiment management is handled here.
"""
from __future__ import annotations
from dataclasses import asdict, fields, replace

import argparse
import json
import logging
import os
import pdb
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from src.config import RunConfig
from src.visuals.animation import animating_predicted_reading_session


# ────────────────────────────────────────────────────────────────────────────────
# Project‑specific imports
# ────────────────────────────────────────────────────────────────────────────────
from src.dataset.dataset import MecoDataset
import src.dataset.feature_funcs as feature_funcs
from src.dataset.utils import collate_fn
from src.model.durations.log_likelihood import DurationNLL
from src.model.saccades.log_likelihood import set_saccadesNLL
from src.model.neural import MarkedPointProcess

from src.model.saccades.log_likelihood import set_saccadesNLL
from src.trainer import Trainer
import typing
from types import (
    NoneType,
    UnionType as Python10UnionType,
)


# ────────────────────────────────────────────────────────────────────────────────
# 2. Utility helpers
# ────────────────────────────────────────────────────────────────────────────────


def build_run_dir(cfg: RunConfig) -> Path:
    """Create `runs/<model‑spec>_<YYYYmmdd‑HHMMSS>` and return the Path."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if cfg.model_type == "saccade":
        exp_name = (
            f"arrm_"  # model short‑hand
            f"hp_{cfg.saccade_likelihood}_"  # likelihood variant
            f"bsz{cfg.batch_size}_"  # batch size
            f"lr{cfg.learning_rate}_"  # learning rate
            f"s{cfg.seed}_"  # seed
            f"{timestamp}"
        )
    elif cfg.model_type == "duration":
        exp_name = (
            f"arrm_"  # model short‑hand
            f"dur_{cfg.dur_likelihood}_"  # likelihood variant
            f"bsz{cfg.batch_size}_"  # batch size
            f"lr{cfg.learning_rate}_"  # learning rate
            f"s{cfg.seed}_"  # seed
            f"{timestamp}"
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model_type}")

    if cfg.final_testing:
        run_dir = Path(cfg.experiment_dir) / (cfg.directory_name)
    else:
        run_dir = Path(cfg.experiment_dir) / (cfg.directory_name + exp_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def configure_logging(run_dir: Path, loglevel: int = logging.INFO) -> logging.Logger:
    """Send messages to both console and `<run_dir>/train.log`."""
    logger = logging.getLogger()
    logger.setLevel(loglevel)

    # File
    fh = logging.FileHandler(run_dir / "train.log")
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for deterministic behaviour."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(
        seed
    )  # official recommendation  [oai_citation_attribution:PyTorch](https://pytorch.org/docs/stable/generated/torch.manual_seed.html?utm_source=chatgpt.com)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ────────────────────────────────────────────────────────────────────────────────
# 3. Training worker (can be spawned on each GPU)
# ────────────────────────────────────────────────────────────────────────────────
def json_path_serializer(obj: Any) -> str:
    """Custom JSON serializer for Path objects."""
    if isinstance(obj, Path):
        return str(obj)
    # Add other custom serializers here if needed, e.g., for datetime
    # if isinstance(obj, datetime):
    #     return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def main() -> None:
    cfg = RunConfig.from_cli()
    run_dir = build_run_dir(cfg=cfg)

    # ─── Logging & seeding ────────────────────────────────────────────────
    logger = configure_logging(run_dir)
    seed_everything(cfg.seed)
    logger.info(
        "Configuration:\n%s",
        json.dumps(asdict(cfg), indent=4, default=json_path_serializer),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # ─── Dataset ─────────────────────────────────────────────────────────
    dataset_kwargs: Dict[str, Any] = dict(
        splitting_procedure=cfg.splitting_procedure,
        filtering=cfg.dataset_filtering,
        feature_func_stpp=feature_funcs.get_features_func(cfg.saccade_predictors_funcs),
        feature_func_dur=feature_funcs.get_features_func(cfg.duration_predictors_funcs),
        division_factor_space=cfg.division_factor_space,
        division_factor_time=cfg.division_factor_time,
        division_factor_durations=cfg.division_factor_durations,
        past_timesteps_duration_baseline_k=cfg.past_timesteps_duration_baseline_k,
        cfg=cfg,
    )
    train_ds = MecoDataset(mode="train", **dataset_kwargs)
    val_ds = MecoDataset(mode="valid", **dataset_kwargs)

    if cfg.subset:
        train_ds = Subset(train_ds, range(cfg.subset_size))
        val_ds = Subset(val_ds, range(int(cfg.subset_size * 0.2)))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.nworkers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.nworkers,
        pin_memory=True,
    )

    # ─── Model & Trainer ────────────────────────────────────────────────
    model = MarkedPointProcess(
        duration_prediction_func=cfg.duration_predictors_funcs,
        hawkes_predictors_func=cfg.saccade_predictors_funcs,
        model_type=cfg.model_type,
        cfg=cfg,
        logger=logger,
    ).to(device)

    if cfg.load_checkpoint:

        if str(cfg.checkpoint_path) == "None":
            raise ValueError("Checkpoint path must be provided when loading a model.")

        logger.info("Loading model weights from checkpoint...")
        logger.info(
            f"Checkpoint path: {cfg.checkpoint_path / f'best_{cfg.model_type}_model' / 'model_weights.pt'}"
        )

        checkpoint = torch.load(
            f=cfg.checkpoint_path / f"best_{cfg.model_type}_model" / "model_weights.pt",
            map_location=device,
        )
        model.load_state_dict(state_dict=checkpoint, strict=cfg.strict_load)

        model.to(device)

    # 1) Collect conv‐specific params by name and allow them to have a  x10 learning rate  to allow gradients to flow more easily
    #    through the convolutional layers.
    conv_param_names = {"gamma_alpha", "gamma_beta"}
    conv_params = []
    other_params = []

    for name, param in model.named_parameters():
        if name in conv_param_names:
            conv_params.append(param)
        else:
            other_params.append(param)

    # 2) Build optimizer depending on cfg.optimizer
    if cfg.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            [
                {"params": other_params, "lr": cfg.learning_rate},
                {"params": conv_params, "lr": 10 * cfg.learning_rate},
            ],
            weight_decay=cfg.weight_decay,
        )
    else:  # SGDNesterov
        optimizer = torch.optim.SGD(
            [
                {"params": other_params, "lr": cfg.learning_rate},
                {"params": conv_params, "lr": 10 * cfg.learning_rate},
            ],
            weight_decay=cfg.weight_decay,
            momentum=0.9,
            nesterov=True,
        )

    NegativeLogLikelihood = (
        set_saccadesNLL(cfg=cfg)
        if cfg.model_type == "saccade"
        else DurationNLL(distribution=cfg.dur_likelihood)
    )

    trainer = Trainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        criterion=NegativeLogLikelihood,
        run_dir=run_dir,
        logging=logger,
        device=device,
        patience=cfg.patience,
    )

    # ─── Training loop ───────────────────────────────────────────────────
    if cfg.training:
        logger.info("Starting model training...")
        start_time = time.time()  # Record the start time
        trainer.train(train_loader, val_loader=val_loader, epochs=cfg.epochs)
        end_time = time.time()  # Record the end time
        training_duration = end_time - start_time
        logger.info(
            f"Training completed. Total training time: {training_duration:.2f} seconds."
        )
        trainer.save_stats(run_dir)
        if "conv" in cfg.duration_predictors_funcs and cfg.model_type == "duration":
            trainer.plot_gamma_params(run_dir)

    # ─── Final testing ──────────────────────────────────────────────────
    if cfg.final_testing:

        test_model_pth = (
            cfg.test_model_dir if "None" not in str(cfg.test_model_dir) else run_dir
        )
        test_model_pth = Path(test_model_pth)

        test_ds = MecoDataset(mode="test", **dataset_kwargs)
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=cfg.nworkers,
            pin_memory=True,
        )
        if (
            test_model_pth / f"best_{cfg.model_type}_model" / "model_weights.pt"
        ).exists() == False and cfg.load_checkpoint:
            test_model_pth = cfg.checkpoint_path

        checkpoint = torch.load(
            test_model_pth / f"best_{cfg.model_type}_model" / "model_weights.pt",
            map_location=torch.device(device),
        )

        model.load_state_dict(state_dict=checkpoint, strict=cfg.strict_load)
        model.eval()

        # logger.info("Evaluating on test set.")
    # trainer.test_evaluation(
    #    run_dir=run_dir,
    #     model=model,
    #     test_loader=test_loader,
    #     mode="test",
    #     compute_probability=False,
    # )

    # ---- Visualizing model animation on held-out sequence ----

    if cfg.model_type == "saccade" and cfg.saccade_likelihood != "HomogenousPoisson":
        test_ds = MecoDataset(mode="test", **dataset_kwargs)
        plt.style.use("default")
        animating_predicted_reading_session(
            inference_model=model,
            likelihood=NegativeLogLikelihood,
            dataset=test_ds,
            output_dir=run_dir,
            division_factor_space=cfg.division_factor_space,
        )


if __name__ == "__main__":
    main()
