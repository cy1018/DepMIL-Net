#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reference two-stage training and evaluation entry point for DepMIL-Net.

Commands
--------
self-test      Run synthetic shape/numerical contracts without AVEC data.
data-check     Validate an AVEC-style frame root without loading image pixels.
stage1-smoke   Run one real AVEC optimization step before a long training job.
stage1-train   Train Motion32 encoder with BDI MSE + ordinal BCE.
extract        Save one variable-length feature file per participant.
stage2-train   Train our 3-layer Transformer + severity-aware ProtoDual.
stage2-eval    Strictly replay a saved Stage-2 checkpoint.

Protocol
--------
This runner intentionally implements the user-selected ExpADA released-oracle
protocol: Training+Development train the model and Testing selects checkpoints.
Every checkpoint and log marks the result TEST-selected/oracle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # A dependency-free progress bar is provided below.
    _tqdm = None

from datasets.depmil_dataset import (
    RAW_SPLITS,
    WINDOW_MODES,
    AVEC2013Motion32Dataset,
    AVEC2013Motion32FeatureDataset,
    feature_bag_collate,
    make_raw_dataset,
)
from models.depmil_net import (
    FEATURE_DIM,
    Motion32Stage1Model,
    SeverityAwareProtoDual,
    cumulative_ordinal_targets,
)


VERSION = "depmil-reference-1.0"
COMPATIBLE_STAGE1_VERSIONS = {VERSION}
DATASET_NAME = "AVEC2013"
PROTOCOL = "expada_released_oracle"
TRAIN_SPLITS = ("Training", "Development")
SELECTION_SPLIT = "Testing"
# The released ExpADA AVEC2013 configuration saves the lowest Testing RMSE.
# Both stages retain that checkpoint-selection rule for protocol consistency.
STAGE1_SELECTION_METRIC = "rmse"
STAGE2_SELECTION_METRIC = "mae"
ORACLE_WARNING = (
    "Training+Development train the model; Testing selects checkpoints. "
    "Metrics are TEST-selected/oracle, not an independent test estimate."
)


class _ConsoleProgress:
    """Small tqdm-compatible fallback used when tqdm is not installed."""

    def __init__(self, iterable, total: int, desc: str, unit: str = "batch", **_):
        self.iterable = iterable
        self.total = max(1, int(total))
        self.desc = str(desc)
        self.unit = str(unit)
        self.current = 0
        self.started = time.perf_counter()
        self.last_printed = 0.0
        self.postfix: Dict[str, object] = {}

    def set_postfix(self, refresh: bool = True, **values) -> None:
        self.postfix.update(values)
        if refresh:
            self._print(force=True)

    def _print(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and self.current != self.total and now - self.last_printed < 1.0:
            return
        self.last_printed = now
        elapsed = max(now - self.started, 1e-8)
        fraction = min(1.0, self.current / self.total)
        completed = int(round(30 * fraction))
        bar = "=" * completed + "." * (30 - completed)
        rate = self.current / elapsed
        eta = (self.total - self.current) / rate if rate > 0 else float("inf")
        eta_text = "--:--" if not math.isfinite(eta) else time.strftime(
            "%H:%M:%S", time.gmtime(max(0.0, eta))
        )
        details = " ".join(f"{key}={value}" for key, value in self.postfix.items())
        message = (
            f"\r{self.desc}: [{bar}] {100.0 * fraction:6.2f}% "
            f"{self.current}/{self.total} {self.unit} "
            f"{rate:6.2f} {self.unit}/s ETA {eta_text}"
        )
        if details:
            message += f" {details}"
        print(message, end="", file=sys.stderr, flush=True)
        if self.current == self.total:
            print(file=sys.stderr, flush=True)

    def __iter__(self):
        self._print(force=True)
        for item in self.iterable:
            yield item
            self.current += 1
            self._print()
        if self.current < self.total:
            self.total = max(1, self.current)
        self._print(force=True)


def progress_bar(iterable, **kwargs):
    if _tqdm is not None:
        return _tqdm(iterable, **kwargs)
    return _ConsoleProgress(iterable, **kwargs)


def resolved_clip_length(args) -> int:
    """ExpADA's AVEC reader and AVEC2013 config use a 10,000-frame cap."""
    if args.clip_length is not None:
        return int(args.clip_length)
    return 10000


def resolved_stage1_lr(args) -> float:
    """Released ExpADA AVEC2013 Stage-1 learning rate."""
    return 1e-5


def resolve_optional_seed(seed: int | None) -> int:
    """Match ExpADA's seed=None behavior while recording the sampled seed."""
    return (
        int(seed)
        if seed is not None
        else random.SystemRandom().randrange(0, 2**31)
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_from_gpu(gpu: int) -> torch.device:
    if gpu >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def safe_torch_load(path: str | Path, device: torch.device):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(labels: Sequence[float], predictions: Sequence[float]) -> Dict[str, float]:
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(predictions, dtype=np.float64)
    if y.shape != p.shape or y.size < 1:
        raise ValueError("labels and predictions must be non-empty and aligned")
    error = p - y
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    if y.size > 1 and np.std(y) > 0 and np.std(p) > 0:
        pearson = float(np.corrcoef(y, p)[0, 1])
    else:
        pearson = 0.0
    mean_y, mean_p = float(np.mean(y)), float(np.mean(p))
    variance_y, variance_p = float(np.var(y)), float(np.var(p))
    covariance = float(np.mean((y - mean_y) * (p - mean_p)))
    denominator = variance_y + variance_p + (mean_y - mean_p) ** 2
    ccc = float(2.0 * covariance / denominator) if denominator > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "pearson": pearson, "ccc": ccc}


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def save_prediction_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["subj_id", "label", "prediction"]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def amp_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def make_scaler(device: torch.device, enabled: bool):
    active = enabled and device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=active)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=active)


def raw_loader(
    args,
    splits: Sequence[str],
    training: bool,
    stride: int | None = None,
    window_mode: str | None = None,
    workers_override: int | None = None,
) -> DataLoader:
    dataset = make_raw_dataset(
        data_root=args.data_root,
        splits=splits,
        label_file=args.label_file,
        frame_len=32,
        stride=int(stride if stride is not None else args.stride),
        window_mode=window_mode or args.window_mode,
        clip_length=resolved_clip_length(args),
        image_size=args.image_size,
        augment=training,
        hflip_probability=args.hflip_probability,
        color_jitter=args.color_jitter,
        random_erasing_probability=args.random_erasing_probability,
    )
    workers = max(
        0,
        int(args.workers if workers_override is None else workers_override),
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        # Windows uses process spawning. Keeping both training and evaluation
        # worker pools alive can load many concurrent copies of PyTorch/CUDA
        # DLLs and trigger WinError 1455 when the page file is exhausted.
        persistent_workers=training and workers > 0,
        drop_last=False,
    )


def stage1_model_config(args) -> Dict[str, object]:
    temporal_kernels = (3,) if args.temporal_mode == "single3" else (3, 5, 7)
    return {
        "pretrained_backbone": bool(args.pretrained_backbone),
        "hidden_dim": int(args.hidden_dim),
        "feature_dim": FEATURE_DIM,
        "dropout": float(args.dropout),
        "use_motion": bool(args.use_motion),
        "temporal_kernels": temporal_kernels,
        "spatial_pool_mode": str(args.spatial_pool_mode),
    }


def validate_stage1_ablation(args, config: Mapping[str, object]) -> None:
    expected = {
        "full": (True, "sap", (3, 5, 7), True),
        "wo_motion": (False, "sap", (3, 5, 7), True),
        "wo_sap": (True, "gap", (3, 5, 7), True),
        "wo_mstn": (True, "sap", (3,), True),
        "wo_ordinal": (True, "sap", (3, 5, 7), False),
    }[args.ablation_name]
    actual = (
        bool(config["use_motion"]),
        str(config["spatial_pool_mode"]),
        tuple(config["temporal_kernels"]),
        float(args.lambda_ord) > 0.0,
    )
    if actual != expected:
        raise ValueError(
            f"Ablation '{args.ablation_name}' expects {expected}, got {actual}. "
            "Use scripts/run_all_stage1_ablations.ps1 to avoid mismatched flags."
        )


def build_stage1_scheduler(optimizer, args):
    """Reproduce the released AVEC2013 linear-warmup/cosine schedule."""
    requested = args.warmup_epochs
    if requested is None:
        requested = 1
    warmup_epochs = max(0, min(int(requested), max(0, args.epochs - 1)))
    if warmup_epochs == 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
            eta_min=args.min_lr,
        )
        return scheduler, {
            "type": "CosineAnnealingLR",
            "warmup_epochs": 0,
            "eta_min": args.min_lr,
        }

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-2,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - warmup_epochs),
        eta_min=args.min_lr,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )
    return scheduler, {
        "type": "LinearWarmupCosine",
        "warmup_epochs": warmup_epochs,
        "eta_min": args.min_lr,
    }


def build_stage1(config: Mapping[str, object], load_pretrained: bool = True):
    return Motion32Stage1Model(
        pretrained_backbone=(
            bool(config.get("pretrained_backbone", True)) and load_pretrained
        ),
        hidden_dim=int(config.get("hidden_dim", 512)),
        feature_dim=int(config.get("feature_dim", FEATURE_DIM)),
        dropout=float(config.get("dropout", 0.2)),
        use_motion=bool(config.get("use_motion", True)),
        temporal_kernels=tuple(config.get("temporal_kernels", (3, 5, 7))),
        spatial_pool_mode=str(config.get("spatial_pool_mode", "sap")),
    )


def stage1_loss(output: Mapping[str, Tensor], labels: Tensor, lambda_ord: float):
    regression = F.mse_loss(output["regression"], labels)
    ordinal_target = cumulative_ordinal_targets(labels)
    # BCE on sigmoid probabilities is unsafe under CUDA autocast.  Using the
    # mathematically equivalent logits form is both AMP-safe and more stable.
    ordinal = F.binary_cross_entropy_with_logits(
        output["ordinal_logits"], ordinal_target
    )
    return regression + lambda_ord * ordinal, regression, ordinal


def aggregate_stage1_predictions(
    subject_ids: Sequence[str], labels: Sequence[float], predictions: Sequence[float]
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    grouped_predictions: Dict[str, List[float]] = defaultdict(list)
    grouped_labels: Dict[str, List[float]] = defaultdict(list)
    for subject, label, prediction in zip(subject_ids, labels, predictions):
        grouped_predictions[str(subject)].append(float(prediction))
        grouped_labels[str(subject)].append(float(label))
    rows: List[Dict[str, object]] = []
    for subject in sorted(grouped_predictions):
        unique_labels = np.unique(np.asarray(grouped_labels[subject]))
        if unique_labels.size != 1:
            raise ValueError(f"Inconsistent labels for {subject}: {unique_labels.tolist()}")
        rows.append(
            {
                "subj_id": subject,
                "label": float(unique_labels[0]),
                "prediction": float(np.mean(grouped_predictions[subject])),
                "segments": len(grouped_predictions[subject]),
            }
        )
    result = metrics(
        [float(row["label"]) for row in rows],
        [float(row["prediction"]) for row in rows],
    )
    return result, rows


@torch.no_grad()
def evaluate_stage1(
    model,
    loader,
    device,
    lambda_ord: float,
    use_amp: bool,
    description: str = "Stage-1 evaluation",
):
    model.eval()
    all_subjects: List[str] = []
    all_labels: List[float] = []
    all_predictions: List[float] = []
    losses: List[float] = []
    progress = progress_bar(
        loader,
        total=len(loader),
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    )
    for step, batch in enumerate(progress, start=1):
        clips = batch["clip"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).reshape(-1)
        with amp_context(device, use_amp):
            output = model(clips)
            loss, _, _ = stage1_loss(output, labels, lambda_ord)
        losses.append(float(loss.detach().cpu()))
        all_subjects.extend([str(value) for value in batch["subj_id"]])
        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(output["regression"].detach().float().cpu().tolist())
        if step == 1 or step % 20 == 0 or step == len(loader):
            progress.set_postfix(
                loss=f"{sum(losses) / len(losses):.4f}",
                refresh=False,
            )
    result, rows = aggregate_stage1_predictions(
        all_subjects, all_labels, all_predictions
    )
    result["loss"] = float(np.mean(losses))
    return result, rows


def run_stage1_train(args) -> None:
    seed_everything(args.seed)
    device = device_from_gpu(args.gpu)
    output_dir = Path(args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[VERSION] {VERSION}")
    print(f"[PROTOCOL] {PROTOCOL}: {ORACLE_WARNING}")
    print(f"[DATASET] {DATASET_NAME}; root={Path(args.data_root).resolve()}")

    train_loader = raw_loader(args, TRAIN_SPLITS, training=True)
    # Evaluation contains only the Testing participants and does not benefit
    # enough from another worker pool to justify the Windows memory overhead.
    selection_loader = raw_loader(
        args,
        (SELECTION_SPLIT,),
        training=False,
        workers_override=0,
    )
    config = stage1_model_config(args)
    validate_stage1_ablation(args, config)
    print(f"[STAGE-I ABLATION] {args.ablation_name}: {json.dumps(config)}")
    model = build_stage1(
        config, load_pretrained=not bool(args.init_ckpt)
    ).to(device)

    init_checkpoint = None
    if args.init_ckpt:
        init_path = Path(args.init_ckpt).resolve()
        init_checkpoint = safe_torch_load(init_path, device)
        if not isinstance(init_checkpoint, dict):
            raise ValueError(f"Invalid Stage-1 checkpoint: {init_path}")
        if init_checkpoint.get("version") not in COMPATIBLE_STAGE1_VERSIONS:
            raise ValueError(
                f"Checkpoint version {init_checkpoint.get('version')} does not "
                f"match supported Stage-1 versions "
                f"{sorted(COMPATIBLE_STAGE1_VERSIONS)}: {init_path}"
            )
        if init_checkpoint.get("kind") != "stage1":
            raise ValueError(f"Not a Stage-1 checkpoint: {init_path}")
        if dict(init_checkpoint.get("model_config", {})) != config:
            raise ValueError(
                "Fine-tune model configuration differs from the initialization "
                f"checkpoint: {init_path}"
            )
        result = model.load_state_dict(init_checkpoint["model"], strict=True)
        print(
            f"[FINETUNE INIT] {init_path}; "
            f"source_epoch={init_checkpoint.get('epoch')}; "
            f"missing_keys={result.missing_keys}; "
            f"unexpected_keys={result.unexpected_keys}"
        )

    backbone_parameters = list(model.encoder.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    new_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in backbone_ids
    ]
    default_lr = resolved_stage1_lr(args)
    backbone_lr = default_lr if args.backbone_lr is None else args.backbone_lr
    new_module_lr = default_lr if args.new_module_lr is None else args.new_module_lr
    optimizer = torch.optim.Adam(
        [
            {"params": backbone_parameters, "lr": backbone_lr},
            {"params": new_parameters, "lr": new_module_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler, scheduler_config = build_stage1_scheduler(optimizer, args)
    print(f"[OPTIMIZER] Adam backbone_lr={backbone_lr:g} new_module_lr={new_module_lr:g}")
    print(f"[SCHEDULER] {json.dumps(scheduler_config)}")
    print(f"[NUMERICS] amp={args.amp}; grad_clip={args.grad_clip}")
    print(
        f"[BATCHES] train={len(train_loader)} per epoch; "
        f"TEST-selection={len(selection_loader)} per epoch; "
        f"batch_size={args.batch_size}; train_workers={args.workers}; "
        "testing_workers=0",
        flush=True,
    )
    scaler = make_scaler(device, args.amp)
    best_selection = float("inf")
    best_epoch = 0
    history: List[Dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss: List[float] = []
        running_reg: List[float] = []
        running_ord: List[float] = []
        train_progress = progress_bar(
            train_loader,
            total=len(train_loader),
            desc=f"Training {epoch:02d}/{args.epochs:02d}",
            unit="batch",
            dynamic_ncols=True,
            mininterval=1.0,
        )
        for step, batch in enumerate(train_progress, start=1):
            clips = batch["clip"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).reshape(-1)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, args.amp):
                output = model(clips)
                loss, regression, ordinal = stage1_loss(
                    output, labels, args.lambda_ord
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip is not None and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running_loss.append(float(loss.detach().cpu()))
            running_reg.append(float(regression.detach().cpu()))
            running_ord.append(float(ordinal.detach().cpu()))
            if step == 1 or step % 20 == 0 or step == len(train_loader):
                recent = min(50, len(running_loss))
                train_progress.set_postfix(
                    loss=f"{sum(running_loss[-recent:]) / recent:.4f}",
                    mse=f"{sum(running_reg[-recent:]) / recent:.4f}",
                    ordinal=f"{sum(running_ord[-recent:]) / recent:.4f}",
                    refresh=False,
                )

        selection_metrics, selection_rows = evaluate_stage1(
            model,
            selection_loader,
            device,
            args.lambda_ord,
            args.amp,
            description=f"Testing  {epoch:02d}/{args.epochs:02d}",
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(running_loss)),
            "train_mse": float(np.mean(running_reg)),
            "train_ordinal_bce": float(np.mean(running_ord)),
            "selection": selection_metrics,
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "new_module_lr": optimizer.param_groups[1]["lr"],
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d} | train loss={record['train_loss']:.4f} "
            f"| TEST-selection MAE={selection_metrics['mae']:.4f} "
            f"RMSE={selection_metrics['rmse']:.4f} "
            f"PCC={selection_metrics['pearson']:.4f} CCC={selection_metrics['ccc']:.4f}"
        )

        state = {
            "version": VERSION,
            "kind": "stage1",
            "phase": "finetune" if args.init_ckpt else "base",
            "epoch": epoch,
            "protocol": PROTOCOL,
            "test_selected": True,
            "train_splits": list(TRAIN_SPLITS),
            "selection_split": SELECTION_SPLIT,
            "selection_metric": STAGE1_SELECTION_METRIC,
            "dataset_name": DATASET_NAME,
            "data_root": str(Path(args.data_root).resolve()),
            "model_config": config,
            "data_config": {
                "frame_len": 32,
                "stride": args.stride,
                "window_mode": args.window_mode,
                "clip_length": resolved_clip_length(args),
                "image_size": args.image_size,
                "normalization": "RGB; (pixel-127.5)/127.5",
            },
            "loss_config": {"mse_weight": 1.0, "ordinal_bce_weight": args.lambda_ord},
            "scheduler_config": scheduler_config,
            "gradient_clipping": args.grad_clip,
            "early_stopping": False,
            "epochs_configured": args.epochs,
            "selection_metrics": selection_metrics,
            "stage1_ablation": args.ablation_name,
            "init_checkpoint": (
                str(Path(args.init_ckpt).resolve()) if args.init_ckpt else None
            ),
            "init_checkpoint_sha256": (
                sha256(args.init_ckpt) if args.init_ckpt else None
            ),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "seed": args.seed,
        }
        torch.save(state, output_dir / "last_stage1.pt")
        if selection_metrics[STAGE1_SELECTION_METRIC] < best_selection:
            best_selection = selection_metrics[STAGE1_SELECTION_METRIC]
            best_epoch = epoch
            torch.save(state, output_dir / "best_stage1.pt")
            save_prediction_csv(
                output_dir / "best_stage1_testing_predictions.csv", selection_rows
            )
            print(
                f"[BEST ORACLE] epoch={epoch} "
                f"TEST-{STAGE1_SELECTION_METRIC.upper()}={best_selection:.4f}"
            )
        save_json(output_dir / "history_stage1.json", history)
        scheduler.step()

    print(f"[DONE] best_epoch={best_epoch} -> {output_dir / 'best_stage1.pt'}")


def load_stage1_checkpoint(path: str, device: torch.device):
    state = safe_torch_load(path, device)
    if not isinstance(state, dict) or state.get("version") not in COMPATIBLE_STAGE1_VERSIONS:
        raise ValueError(
            f"Unsupported Stage-1 checkpoint version {getattr(state, 'get', lambda *_: None)('version')}: "
            f"{path}; accepted={sorted(COMPATIBLE_STAGE1_VERSIONS)}"
        )
    if state.get("kind") != "stage1":
        raise ValueError(f"Not a Stage-1 checkpoint: {path}")
    model = build_stage1(state["model_config"], load_pretrained=False).to(device)
    result = model.load_state_dict(state["model"], strict=True)
    print(
        f"[PASS] strict=True missing_keys={result.missing_keys} "
        f"unexpected_keys={result.unexpected_keys}"
    )
    return model, state


@torch.no_grad()
def run_extract(args) -> None:
    seed_everything(args.seed)
    device = device_from_gpu(args.gpu)
    model, checkpoint = load_stage1_checkpoint(args.ckpt, device)
    model.eval()
    output_root = Path(args.out_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    all_statistics: Dict[str, object] = {}

    for split in RAW_SPLITS:
        loader = raw_loader(
            args,
            (split,),
            training=False,
            stride=args.extract_stride,
            window_mode=args.extract_window_mode,
        )
        grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for batch in loader:
            clips = batch["clip"].to(device, non_blocking=True)
            with amp_context(device, args.amp):
                output = model(clips)
            features = output["features"].detach().float().cpu()
            labels = batch["label"].detach().cpu().tolist()
            starts = batch["start"].detach().cpu().tolist()
            for index, subject in enumerate(batch["subj_id"]):
                grouped[str(subject)].append(
                    {
                        "feature": features[index],
                        "label": float(labels[index]),
                        "start": int(starts[index]),
                        "kind": str(batch["window_kind"][index]),
                    }
                )

        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        lengths: List[int] = []
        kind_counts: Dict[str, int] = defaultdict(int)
        for subject, records in sorted(grouped.items()):
            # Dense and rate-3 windows remain distinguishable and chronological.
            records.sort(key=lambda item: (str(item["kind"]), int(item["start"])))
            labels = {float(item["label"]) for item in records}
            if len(labels) != 1:
                raise ValueError(f"Inconsistent labels for {subject}: {sorted(labels)}")
            gt_label = labels.pop()
            segment = {
                f"seg_{index}": {
                    "start": int(item["start"]),
                    "kind": str(item["kind"]),
                }
                for index, item in enumerate(records)
            }
            value = {
                "version": VERSION,
                "subj_id": subject,
                "gt_label": gt_label,
                "segment": segment,
                "features": torch.stack([item["feature"] for item in records]),
                "starts": torch.tensor([item["start"] for item in records]),
                "window_kinds": [item["kind"] for item in records],
                "stage1_checkpoint": str(Path(args.ckpt).resolve()),
                "stage1_checkpoint_sha256": sha256(args.ckpt),
                "stage1_checkpoint_version": checkpoint.get("version"),
                "dataset_name": checkpoint.get("dataset_name", DATASET_NAME),
                "extract_stride": args.extract_stride,
                "extract_window_mode": args.extract_window_mode,
            }
            torch.save(value, split_dir / f"{subject}.pkl")
            lengths.append(len(records))
            for item in records:
                kind_counts[str(item["kind"])] += 1
        all_statistics[split] = {
            "participants": len(lengths),
            "segments_min": min(lengths),
            "segments_max": max(lengths),
            "segments_mean": float(np.mean(lengths)),
            "window_kinds": dict(kind_counts),
            "saved_tensors": ["features"],
            "filename_example": next(iter(sorted(grouped))) + ".pkl",
        }
        print(f"[EXTRACT] {split}: {json.dumps(all_statistics[split], ensure_ascii=False)}")

    save_json(output_root / "feature_summary.json", all_statistics)
    print(f"[DONE] participant features -> {output_root}")


def feature_loader(
    feature_root: str,
    splits: Iterable[str],
    training: bool,
    workers: int,
) -> DataLoader:
    dataset = AVEC2013Motion32FeatureDataset(feature_root, splits, FEATURE_DIM)
    worker_count = max(0, int(workers))
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=training,
        num_workers=worker_count,
        collate_fn=feature_bag_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=worker_count > 0,
    )


def stage2_model_config(args) -> Dict[str, object]:
    return {
        "feature_dim": FEATURE_DIM,
        "transformer_dim": args.transformer_dim,
        "transformer_layers": args.transformer_layers,
        "transformer_heads": args.transformer_heads,
        "transformer_ffn": args.transformer_ffn,
        "dropout": args.dropout,
        "key_temperature": args.key_temperature,
        "context_temperature": args.context_temperature,
        "severity_strength": args.severity_strength,
        "use_severity": args.use_severity,
        "fusion_mode": args.fusion_mode,
        "fixed_key_weight": args.fixed_key_weight,
    }


def build_stage2(config: Mapping[str, object]):
    return SeverityAwareProtoDual(**dict(config))


def severity_class_targets(labels: Tensor) -> Tensor:
    labels = labels.reshape(-1)
    classes = torch.zeros_like(labels, dtype=torch.long)
    classes = classes + (labels >= 14).long()
    classes = classes + (labels >= 20).long()
    classes = classes + (labels >= 29).long()
    return classes


def stage2_loss(output: Mapping[str, Tensor], label: Tensor, lambda_severity: float):
    """Bag-level Stage-II objective; no instance-label supervision is used."""
    prediction = output["prediction"].reshape(-1)
    target = label.reshape(-1)
    mse = F.mse_loss(prediction, target)
    smooth_l1 = F.smooth_l1_loss(prediction, target)
    regression = 0.7 * mse + 0.3 * smooth_l1
    class_index = severity_class_targets(target)
    class_target = F.one_hot(
        class_index, num_classes=4
    ).to(dtype=prediction.dtype)
    weak_severity = F.binary_cross_entropy(
        output["video_severity_probs"].clamp(1e-7, 1.0 - 1e-7),
        class_target,
    )
    # AVEC2013 supplies one BDI-II label per participant/bag.  Replicating that
    # label to every temporal instance would introduce an unjustified instance
    # loss, so Stage-II supervises only the bag prediction and weak video-level
    # severity distribution.
    instance_severity = prediction.new_zeros(())
    total = regression + lambda_severity * weak_severity
    return total, mse, smooth_l1, weak_severity, instance_severity


def distribution_effective_n(weights: Tensor) -> Tensor:
    return 1.0 / weights.square().sum(dim=-1).clamp_min(1e-8)


def stage2_diagnostics(output: Mapping[str, Tensor]) -> Dict[str, float]:
    key = output["key_weights"].detach().float()
    base = output["base_key_weights"].detach().float()
    context = output["context_weights"].detach().float()
    entropy = -(key * key.clamp_min(1e-8).log()).sum(dim=-1)
    prototype = output["prototype"].detach().float()
    context_vector = output["context"].detach().float()
    return {
        "key_entropy": float(entropy.mean().cpu()),
        "key_effective_n": float(distribution_effective_n(key).mean().cpu()),
        "key_max_weight": float(key.max(dim=-1).values.mean().cpu()),
        "context_effective_n": float(
            distribution_effective_n(context).mean().cpu()
        ),
        "proto_context_cosine": float(
            F.cosine_similarity(prototype, context_vector, dim=-1).mean().cpu()
        ),
        "severity_key_l1_change": float(torch.abs(key - base).sum(dim=-1).mean().cpu()),
        "severity_consistency_std": float(
            output["severity_consistency"].detach().float().std(unbiased=False).cpu()
        ),
        "fusion_gate_key": float(
            output["fusion_gate_key"].detach().float().mean().cpu()
        ),
        "video_severity_confidence": float(
            output["video_severity_probs"].detach().float().max(dim=-1).values.mean().cpu()
        ),
    }


@torch.no_grad()
def evaluate_stage2(model, loader, device):
    model.eval()
    rows: List[Dict[str, object]] = []
    diagnostics: Dict[str, List[float]] = defaultdict(list)
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        output = model(features)
        prediction = float(output["prediction"].detach().cpu().reshape(-1)[0])
        label = float(batch["label"])
        rows.append(
            {
                "subj_id": batch["subj_id"],
                "label": label,
                "prediction": prediction,
                "segments": int(features.shape[0]),
                "key_prediction": float(output["key_prediction"].detach().cpu()[0]),
                "context_prediction": float(
                    output["context_prediction"].detach().cpu()[0]
                ),
                "fusion_gate_key": float(
                    output["fusion_gate_key"].detach().cpu()[0]
                ),
            }
        )
        for name, value in stage2_diagnostics(output).items():
            diagnostics[name].append(value)
    result = metrics(
        [float(row["label"]) for row in rows],
        [float(row["prediction"]) for row in rows],
    )
    mean_diagnostics = {
        name: float(np.mean(values)) for name, values in diagnostics.items()
    }
    return result, rows, mean_diagnostics


def run_stage2_train(args) -> None:
    args.seed = resolve_optional_seed(args.seed)
    seed_everything(args.seed)
    device = device_from_gpu(args.gpu)
    output_dir = Path(args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = feature_loader(
        args.feature_root, TRAIN_SPLITS, True, args.workers
    )
    selection_loader = feature_loader(
        args.feature_root, (SELECTION_SPLIT,), False, args.workers
    )
    config = stage2_model_config(args)
    model = build_stage2(config).to(device)
    momentum = args.momentum
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=momentum,
        weight_decay=args.weight_decay,
    )
    warmup_epochs = args.warmup_epochs
    warmup_epochs = min(warmup_epochs, max(0, args.epochs - 1))
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.cosine_t_max),
            eta_min=args.min_lr,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, cosine], milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.cosine_t_max), eta_min=args.min_lr
        )
    best_selection = float("inf")
    best_epoch = 0
    history: List[Dict[str, object]] = []
    print(f"[VERSION] {VERSION}")
    print(f"[DATASET] {DATASET_NAME}; seed={args.seed}")
    print(f"[PROTOCOL] {PROTOCOL}: {ORACLE_WARNING}")
    print(
        f"[STAGE 2] transformer_layers={args.transformer_layers}; "
        f"use_severity={args.use_severity}; fusion={args.fusion_mode}; "
        f"input=features_only; instance_supervision=False"
    )
    print(
        f"[OPTIMIZER] SGD lr={args.lr:g} momentum={momentum:g}; "
        f"warmup_epochs={warmup_epochs}; cosine_t_max={args.cosine_t_max}; "
        f"eta_min={args.min_lr:g}; grad_clip={args.grad_clip}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: List[float] = []
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            label = batch["label"].to(device).reshape(1)
            optimizer.zero_grad(set_to_none=True)
            output = model(features)
            loss, _, _, _, _ = stage2_loss(
                output, label, args.lambda_severity
            )
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        selection_metrics, selection_rows, diagnostics = evaluate_stage2(
            model, selection_loader, device
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "selection": selection_metrics,
            "diagnostics": diagnostics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d} | train loss={record['train_loss']:.4f} "
            f"| TEST-selection MAE={selection_metrics['mae']:.4f} "
            f"RMSE={selection_metrics['rmse']:.4f} "
            f"PCC={selection_metrics['pearson']:.4f} CCC={selection_metrics['ccc']:.4f}"
        )
        state = {
            "version": VERSION,
            "kind": "stage2",
            "epoch": epoch,
            "protocol": PROTOCOL,
            "test_selected": True,
            "train_splits": list(TRAIN_SPLITS),
            "selection_split": SELECTION_SPLIT,
            "selection_metric": STAGE2_SELECTION_METRIC,
            "dataset_name": DATASET_NAME,
            "feature_root": str(Path(args.feature_root).resolve()),
            "model_config": config,
            "loss_config": {
                "smooth_l1_weight": 0.3,
                "mse_weight": 0.7,
                "expada_weak_severity_weight": args.lambda_severity,
                "instance_severity_weight": 0.0,
                "instance_supervision": False,
            },
            "optimizer_config": {
                "type": "SGD",
                "lr": args.lr,
                "momentum": momentum,
                "weight_decay": args.weight_decay,
                "warmup_epochs": warmup_epochs,
                "cosine_t_max": args.cosine_t_max,
                "eta_min": args.min_lr,
                "gradient_clipping": args.grad_clip,
            },
            "early_stopping": False,
            "epochs_configured": args.epochs,
            "selection_metrics": selection_metrics,
            "diagnostics": diagnostics,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "seed": args.seed,
        }
        torch.save(state, output_dir / "last_stage2.pt")
        if selection_metrics[STAGE2_SELECTION_METRIC] < best_selection:
            best_selection = selection_metrics[STAGE2_SELECTION_METRIC]
            best_epoch = epoch
            torch.save(state, output_dir / "best_stage2.pt")
            save_prediction_csv(
                output_dir / "best_stage2_testing_predictions.csv", selection_rows
            )
            save_json(output_dir / "best_stage2_diagnostics.json", diagnostics)
            print(
                f"[BEST ORACLE] epoch={epoch} "
                f"TEST-{STAGE2_SELECTION_METRIC.upper()}={best_selection:.4f}"
            )
        save_json(output_dir / "history_stage2.json", history)
        scheduler.step()

    print(f"[DONE] best_epoch={best_epoch} -> {output_dir / 'best_stage2.pt'}")


def load_stage2_checkpoint(path: str, device: torch.device):
    state = safe_torch_load(path, device)
    if not isinstance(state, dict) or state.get("version") != VERSION:
        raise ValueError(f"Not a {VERSION} checkpoint: {path}")
    if state.get("kind") != "stage2":
        raise ValueError(f"Not a Stage-2 checkpoint: {path}")
    model = build_stage2(state["model_config"]).to(device)
    result = model.load_state_dict(state["model"], strict=True)
    print(
        f"[PASS] strict=True missing_keys={result.missing_keys} "
        f"unexpected_keys={result.unexpected_keys}"
    )
    return model, state


def run_stage2_eval(args) -> None:
    seed_everything(args.seed)
    device = device_from_gpu(args.gpu)
    model, state = load_stage2_checkpoint(args.ckpt, device)
    loader = feature_loader(args.feature_root, (args.split,), False, args.workers)
    result, rows, diagnostics = evaluate_stage2(model, loader, device)
    save_prediction_csv(Path(args.out_csv).resolve(), rows)
    print(f"[EVAL] {json.dumps(result)}")
    print(f"[DIAGNOSTICS] {json.dumps(diagnostics)}")
    print(f"[PROTOCOL] checkpoint={state.get('protocol')} test_selected={state.get('test_selected')}")


def run_data_check(args) -> None:
    summaries = {}
    for split in RAW_SPLITS:
        dataset = AVEC2013Motion32Dataset(
            data_root=args.data_root,
            split=split,
            label_file=args.label_file,
            frame_len=32,
            stride=args.stride,
            window_mode=args.window_mode,
            clip_length=resolved_clip_length(args),
            image_size=args.image_size,
            augment=False,
        )
        summaries[split] = dataset.summary()
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(
        f"[DATA CHECK] dataset={DATASET_NAME}; frame_len=32; stride={args.stride}; "
        f"window_mode={args.window_mode}; clip_length={resolved_clip_length(args)}; "
        f"add_resample={args.window_mode != 'dense'}"
    )
    print(f"[PROTOCOL] {PROTOCOL}: {ORACLE_WARNING}")
    print("[PASS] ExpADA AVEC2013 root, labels, participant folders and window grid verified")


def run_stage1_smoke(args) -> None:
    """Exercise image decoding, GPU memory, forward, loss and backward once."""
    seed_everything(args.seed)
    device = device_from_gpu(args.gpu)
    loader = raw_loader(args, ("Training",), training=True)
    batch = next(iter(loader))
    config = stage1_model_config(args)
    model = build_stage1(config, load_pretrained=True).to(device).train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=resolved_stage1_lr(args), weight_decay=1e-4
    )
    scaler = make_scaler(device, args.amp)
    clips = batch["clip"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True).reshape(-1)
    optimizer.zero_grad(set_to_none=True)
    with amp_context(device, args.amp):
        output = model(clips)
        loss, regression, ordinal = stage1_loss(output, labels, args.lambda_ord)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    print(
        f"[REAL BATCH] clip={tuple(clips.shape)} range="
        f"({float(clips.min()):.3f},{float(clips.max()):.3f})"
    )
    print(
        f"[FORWARD] features={tuple(output['features'].shape)} "
        f"ordinal={tuple(output['ordinal_probs'].shape)}"
    )
    print(
        f"[LOSS] total={float(loss.detach()):.4f} "
        f"mse={float(regression.detach()):.4f} "
        f"ordinal_bce={float(ordinal.detach()):.4f} "
        f"grad_norm={float(gradient_norm):.4f}"
    )
    if device.type == "cuda":
        print(
            f"[GPU] peak_allocated_gb="
            f"{torch.cuda.max_memory_allocated(device) / 1024**3:.3f}"
        )
    print("[PASS] one real Motion32 Stage-1 optimization step completed")


@torch.no_grad()
def run_self_test(args) -> None:
    seed_everything(7)
    device = device_from_gpu(args.gpu)
    # 112 is sufficient for a shape contract and uses less memory than 224.
    stage1 = Motion32Stage1Model(pretrained_backbone=False).to(device).eval()
    clips = torch.randn(1, 32, 3, 112, 112, device=device)
    output = stage1(clips, return_details=True)
    assert output["features"].shape == (1, FEATURE_DIM)
    assert output["regression"].shape == (1,)
    assert output["ordinal_logits"].shape == (1, 3)
    assert output["ordinal_cumulative"].shape == (1, 3)
    assert output["ordinal_probs"].shape == (1, 4)
    assert torch.allclose(
        output["ordinal_probs"].sum(dim=1),
        torch.ones(1, device=device),
        atol=1e-5,
    )
    stage2 = SeverityAwareProtoDual(
        transformer_layers=3,
        use_severity=True,
        fusion_mode="fixed",
    ).to(device).eval()
    features = torch.randn(13, FEATURE_DIM, device=device)
    result = stage2(features)
    assert result["prediction"].shape == (1,)
    assert result["key_weights"].shape == (1, 13)
    assert result["context_weights"].shape == (1, 13)
    assert result["instance_severity_probs"].shape == (1, 13, 4)
    assert result["video_severity_probs"].shape == (1, 4)
    assert torch.allclose(result["key_weights"].sum(1), torch.ones(1, device=device), atol=1e-5)
    print("[PASS] Stage1 [1,32,3,112,112] -> [1,1024]")
    print("[PASS] Stage2 [13,1024] -> internal severity + one BDI prediction")
    print(f"[VERSION] {VERSION}")


def add_raw_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--label_file", default="label_dict.json")
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--window_mode", choices=WINDOW_MODES, default="dense")
    parser.add_argument(
        "--clip_length",
        type=int,
        default=None,
        help="Default: 10000, matching ExpADA's AVEC2013 dataset reader",
    )
    parser.add_argument("--image_size", type=int, default=224)


def add_stage1_data_arguments(parser: argparse.ArgumentParser) -> None:
    add_raw_data_arguments(parser)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hflip_probability", type=float, default=0.5)
    parser.add_argument("--color_jitter", type=float, default=0.2)
    parser.add_argument("--random_erasing_probability", type=float, default=0.0)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    self_test = commands.add_parser("self-test")
    self_test.add_argument("--gpu", type=int, default=-1)

    data_check = commands.add_parser("data-check")
    add_raw_data_arguments(data_check)

    smoke = commands.add_parser("stage1-smoke")
    add_stage1_data_arguments(smoke)
    smoke.add_argument("--lambda_ord", type=float, default=1.0)
    smoke.add_argument("--grad_clip", type=float, default=5.0)
    smoke.add_argument("--hidden_dim", type=int, default=512)
    smoke.add_argument("--dropout", type=float, default=0.2)
    smoke.add_argument(
        "--use_motion", action=argparse.BooleanOptionalAction, default=True
    )
    smoke.add_argument(
        "--temporal_mode", choices=("single3", "multiscale"), default="multiscale"
    )
    smoke.add_argument(
        "--spatial_pool_mode", choices=("sap", "gap"), default="sap"
    )
    smoke.add_argument("--gpu", type=int, default=0)
    smoke.add_argument("--seed", type=int, default=952292171)
    smoke.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    smoke.add_argument(
        "--pretrained_backbone", action=argparse.BooleanOptionalAction, default=True
    )

    stage1 = commands.add_parser("stage1-train")
    add_stage1_data_arguments(stage1)
    # Match the released ExpADA AVEC2013 Stage-1 training defaults.  The
    # smoke-test and extraction commands keep their smaller batch defaults.
    stage1.set_defaults(batch_size=16, workers=8)
    stage1.add_argument("--out_dir", required=True)
    stage1.add_argument(
        "--init_ckpt",
        default=None,
        help="Optional best_stage1.pt used to initialize a second fine-tuning run",
    )
    stage1.add_argument("--epochs", type=int, default=11)
    stage1.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Deprecated compatibility option; ignored because early stopping is disabled",
    )
    stage1.add_argument(
        "--backbone_lr", type=float, default=None,
        help="Default: 1e-5, matching ExpADA AVEC2013",
    )
    stage1.add_argument(
        "--new_module_lr", type=float, default=None,
        help="Default: 1e-5, matching ExpADA AVEC2013",
    )
    stage1.add_argument("--min_lr", type=float, default=1e-7)
    stage1.add_argument(
        "--warmup_epochs",
        type=int,
        default=None,
        help="Default: 1 epoch, matching ExpADA AVEC2013",
    )
    stage1.add_argument("--weight_decay", type=float, default=1e-4)
    stage1.add_argument("--lambda_ord", type=float, default=1.0)
    stage1.add_argument(
        "--grad_clip",
        type=float,
        default=None,
        help="Disabled by default, matching ExpADA AVEC2013",
    )
    stage1.add_argument("--hidden_dim", type=int, default=512)
    stage1.add_argument("--dropout", type=float, default=0.2)
    stage1.add_argument(
        "--use_motion", action=argparse.BooleanOptionalAction, default=True
    )
    stage1.add_argument(
        "--temporal_mode", choices=("single3", "multiscale"), default="multiscale"
    )
    stage1.add_argument(
        "--spatial_pool_mode", choices=("sap", "gap"), default="sap"
    )
    stage1.add_argument(
        "--ablation_name",
        choices=("full", "wo_motion", "wo_sap", "wo_mstn", "wo_ordinal"),
        default="full",
    )
    stage1.add_argument("--gpu", type=int, default=0)
    stage1.add_argument("--seed", type=int, default=952292171)
    stage1.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    stage1.add_argument(
        "--pretrained_backbone", action=argparse.BooleanOptionalAction, default=True
    )

    extract = commands.add_parser("extract")
    add_stage1_data_arguments(extract)
    extract.set_defaults(batch_size=1, workers=0)
    extract.add_argument("--ckpt", required=True)
    extract.add_argument("--out_dir", required=True)
    extract.add_argument("--extract_stride", type=int, default=32)
    extract.add_argument(
        "--extract_window_mode", choices=WINDOW_MODES, default="dense"
    )
    extract.add_argument("--gpu", type=int, default=0)
    extract.add_argument("--seed", type=int, default=42)
    extract.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)

    stage2 = commands.add_parser("stage2-train")
    stage2.add_argument("--feature_root", required=True)
    stage2.add_argument("--out_dir", required=True)
    stage2.add_argument(
        "--epochs", type=int, default=80,
        help="ExpADA AVEC2013 Stage-2 default: 80",
    )
    stage2.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Deprecated compatibility option; ignored because early stopping is disabled",
    )
    stage2.add_argument("--workers", type=int, default=2)
    stage2.add_argument("--gpu", type=int, default=0)
    stage2.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Default None follows ExpADA and generates a seed that is saved in the checkpoint",
    )
    stage2.add_argument("--lr", type=float, default=1e-3)
    stage2.add_argument("--min_lr", type=float, default=1e-5)
    stage2.add_argument("--momentum", type=float, default=0.8)
    stage2.add_argument("--warmup_epochs", type=int, default=5)
    stage2.add_argument("--cosine_t_max", type=int, default=20)
    stage2.add_argument("--weight_decay", type=float, default=0.0)
    stage2.add_argument(
        "--grad_clip",
        type=float,
        default=None,
        help="Disabled by default, matching ExpADA AVEC2013",
    )
    stage2.add_argument("--lambda_severity", type=float, default=1.0)
    stage2.add_argument("--transformer_dim", type=int, default=256)
    stage2.add_argument("--transformer_layers", type=int, default=3)
    stage2.add_argument("--transformer_heads", type=int, default=4)
    stage2.add_argument("--transformer_ffn", type=int, default=1024)
    stage2.add_argument("--dropout", type=float, default=0.2)
    stage2.add_argument("--key_temperature", type=float, default=0.1)
    stage2.add_argument("--context_temperature", type=float, default=1.0)
    stage2.add_argument("--severity_strength", type=float, default=1.0)
    stage2.add_argument(
        "--use_severity", action=argparse.BooleanOptionalAction, default=True
    )
    stage2.add_argument("--fusion_mode", choices=("fixed", "adaptive"), default="fixed")
    stage2.add_argument("--fixed_key_weight", type=float, default=0.5)

    evaluate = commands.add_parser("stage2-eval")
    evaluate.add_argument("--ckpt", required=True)
    evaluate.add_argument("--feature_root", required=True)
    evaluate.add_argument("--split", choices=RAW_SPLITS, default="Testing")
    evaluate.add_argument("--out_csv", required=True)
    evaluate.add_argument("--workers", type=int, default=0)
    evaluate.add_argument("--gpu", type=int, default=0)
    evaluate.add_argument("--seed", type=int, default=42)

    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "self-test":
        run_self_test(args)
    elif args.command == "data-check":
        run_data_check(args)
    elif args.command == "stage1-smoke":
        run_stage1_smoke(args)
    elif args.command == "stage1-train":
        run_stage1_train(args)
    elif args.command == "extract":
        run_extract(args)
    elif args.command == "stage2-train":
        run_stage2_train(args)
    elif args.command == "stage2-eval":
        run_stage2_eval(args)
    else:
        raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
