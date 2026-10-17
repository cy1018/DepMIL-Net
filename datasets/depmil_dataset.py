#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frame-folder and feature-bag data adapters for DepMIL-Net.

The included adapter follows the AVEC-style ``Training``, ``Development`` and
``Testing`` frame-folder layout used in our experiments. A Stage-I sample is
one 32-frame segment. Dataset files and labels are not distributed here.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import ConcatDataset, Dataset


RAW_SPLITS = ("Training", "Development", "Testing")
WINDOW_MODES = ("dense", "rate3", "mixed")
# The released AVEC2013 reproduction explicitly uses img_type="jpg".  Keeping
# this strict also prevents accidentally reading both BMP and converted JPG
# copies of the same frame when both formats exist in one participant folder.
IMAGE_SUFFIXES = (".jpg",)


def _torch_load(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _numeric_key(path: Path):
    numbers = re.findall(r"\d+", path.stem)
    return tuple(int(value) for value in numbers) if numbers else (10**18,)


def list_frames(frame_dir: Path) -> List[Path]:
    """Return frames in chronological order.

    ExpADA names are zero padded, but the numeric fallback also handles
    participant folders that were renamed outside the released repository.
    """
    frames = [
        path
        for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not frames:
        raise RuntimeError(f"No image frames found in {frame_dir}")
    lexical = sorted(frames, key=lambda path: path.name)
    if len({path.name for path in lexical}) != len(lexical):
        raise RuntimeError(f"Duplicate frame names in {frame_dir}")
    # Zero-padded ExpADA frames are already correct lexically.  Numeric sort is
    # used only when lexical order disagrees with the extracted integer order.
    numeric = sorted(frames, key=_numeric_key)
    return lexical if [p.name for p in lexical] == [p.name for p in numeric] else numeric


def load_label_dict(root: Path, label_file: str) -> Dict[str, float]:
    path = root / label_file
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    labels: Dict[str, float] = {}
    for key, value in raw.items():
        labels[str(key)] = float(value)
    return labels


def participant_id_from_dir(frame_dir: Path) -> str:
    """Recover the AVEC id (for example ``203_1``) from an aligned folder."""
    match = re.search(r"\d{3}_\d", frame_dir.name)
    if match:
        return match.group(0)
    # This matches the released ExpADA code's ``folder_name[:5]`` fallback.
    return frame_dir.name[:5]


@dataclass(frozen=True)
class WindowRecord:
    participant_index: int
    indices: Tuple[int, ...]
    start: int
    kind: str


def make_window_grid(
    frame_count: int,
    frame_len: int = 32,
    stride: int = 16,
    mode: str = "dense",
    clip_length: int = 10000,
) -> List[Tuple[Tuple[int, ...], int, str]]:
    """Build the released-style dense and/or rate-3 32-frame window grid."""
    if mode not in WINDOW_MODES:
        raise ValueError(f"mode must be one of {WINDOW_MODES}, got {mode}")
    if frame_count < 1 or frame_len < 1 or stride < 1:
        raise ValueError("frame_count, frame_len and stride must be positive")
    records: List[Tuple[Tuple[int, ...], int, str]] = []
    dense_count = min(frame_count, clip_length) if clip_length > 0 else frame_count

    if mode in ("dense", "mixed"):
        for start in range(0, dense_count, stride):
            if start + frame_len <= dense_count:
                records.append(
                    (tuple(range(start, start + frame_len)), start, "dense")
                )

    if mode in ("rate3", "mixed"):
        rate = 3
        span = frame_len * rate
        rate3_stride = span // 2
        # ExpADA's released resampling branch uses the complete sequence.
        for start in range(0, frame_count, rate3_stride):
            if start + span <= frame_count:
                records.append(
                    (
                        tuple(range(start, start + span, rate)),
                        start,
                        "rate3",
                    )
                )

    if not records:
        # Deterministic fallback for abnormally short participant recordings.
        indices = tuple(
            min(frame_count - 1, math.floor(i * frame_count / frame_len))
            for i in range(frame_len)
        )
        records.append((indices, 0, "uniform_fallback"))
    return records


class AVEC2013Motion32Dataset(Dataset):
    """One AVEC2013 split, one 32-frame segment per item."""

    def __init__(
        self,
        data_root: str,
        split: str,
        label_file: str = "label_dict.json",
        frame_len: int = 32,
        stride: int = 16,
        window_mode: str = "dense",
        clip_length: int = 10000,
        image_size: int = 224,
        augment: bool = False,
        hflip_probability: float = 0.5,
        color_jitter: float = 0.2,
        random_erasing_probability: float = 0.0,
    ) -> None:
        if split not in RAW_SPLITS:
            raise ValueError(f"split must be one of {RAW_SPLITS}, got {split}")
        if frame_len != 32:
            raise ValueError("Motion32 requires frame_len=32")
        if image_size < 32:
            raise ValueError("image_size is unexpectedly small")

        self.root = Path(data_root).resolve()
        self.split = split
        self.frame_len = int(frame_len)
        self.stride = int(stride)
        self.window_mode = window_mode
        self.clip_length = int(clip_length)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.hflip_probability = float(hflip_probability)
        self.color_jitter = float(color_jitter)
        self.random_erasing_probability = float(random_erasing_probability)
        self.labels = load_label_dict(self.root, label_file)

        split_dir = self.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(split_dir)
        participant_dirs = sorted(
            path
            for path in split_dir.iterdir()
            if path.is_dir() and "aligned" in path.name.lower()
        )
        if not participant_dirs:
            raise RuntimeError(f"No '*aligned*' participant folders in {split_dir}")

        self.participants: List[Dict[str, object]] = []
        self.records: List[WindowRecord] = []
        for frame_dir in participant_dirs:
            subject_base = participant_id_from_dir(frame_dir)
            if subject_base not in self.labels:
                raise KeyError(
                    f"Missing label for {subject_base} in {self.root / label_file}"
                )
            frames = list_frames(frame_dir)
            participant_index = len(self.participants)
            subject_id = f"{subject_base}_{split}"
            self.participants.append(
                {
                    "subj_id": subject_id,
                    "subject_base": subject_base,
                    "label": self.labels[subject_base],
                    "frame_dir": frame_dir,
                    "frames": frames,
                }
            )
            for indices, start, kind in make_window_grid(
                frame_count=len(frames),
                frame_len=self.frame_len,
                stride=self.stride,
                mode=self.window_mode,
                clip_length=self.clip_length,
            ):
                self.records.append(
                    WindowRecord(participant_index, indices, start, kind)
                )

    def __len__(self) -> int:
        return len(self.records)

    def _augmentation_parameters(self):
        flip = self.augment and random.random() < self.hflip_probability
        strength = self.color_jitter
        if self.augment and strength > 0:
            low, high = 1.0 - strength, 1.0 + strength
            brightness = random.uniform(low, high)
            contrast = random.uniform(low, high)
            saturation = random.uniform(low, high)
        else:
            brightness = contrast = saturation = 1.0
        return flip, brightness, contrast, saturation

    def _load_clip(self, paths: Sequence[Path]) -> torch.Tensor:
        flip, brightness, contrast, saturation = self._augmentation_parameters()
        arrays: List[np.ndarray] = []
        for path in paths:
            with Image.open(path) as opened:
                image = opened.convert("RGB").resize(
                    (self.image_size, self.image_size), Image.Resampling.BILINEAR
                )
                if flip:
                    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if brightness != 1.0:
                    image = ImageEnhance.Brightness(image).enhance(brightness)
                if contrast != 1.0:
                    image = ImageEnhance.Contrast(image).enhance(contrast)
                if saturation != 1.0:
                    image = ImageEnhance.Color(image).enhance(saturation)
                # Keep decoded pixels compact while assembling a clip.  PIL
                # pixels are uint8 already; converting every frame to float32
                # here quadruples temporary CPU memory and can trigger a
                # NumPy ArrayMemoryError on Windows.
                arrays.append(np.asarray(image, dtype=np.uint8).copy())

        # [T,H,W,C] -> [T,C,H,W], then reproduce ExpADA's
        # mean=[127.5]*3, std=[127.5]*3, to_rgb=True preprocessing.
        clip = (
            torch.from_numpy(np.stack(arrays, axis=0))
            .permute(0, 3, 1, 2)
            .to(dtype=torch.float32)
        )
        clip = (clip - 127.5) / 127.5

        if self.augment and random.random() < self.random_erasing_probability:
            height, width = clip.shape[-2:]
            area = height * width
            target = random.uniform(0.02, 1.0 / 16.0) * area
            aspect = random.uniform(0.3, 3.3)
            erase_h = max(1, min(height, int(round(math.sqrt(target * aspect)))))
            erase_w = max(1, min(width, int(round(math.sqrt(target / aspect)))))
            top = random.randint(0, height - erase_h)
            left = random.randint(0, width - erase_w)
            # The same erased region is used in every frame, as in ExpADA's
            # SequentialRandomErasing.
            clip[:, :, top : top + erase_h, left : left + erase_w] = 0.0
        return clip.contiguous()

    def __getitem__(self, index: int):
        record = self.records[index]
        participant = self.participants[record.participant_index]
        all_frames: Sequence[Path] = participant["frames"]  # type: ignore[assignment]
        paths = [all_frames[position] for position in record.indices]
        return {
            "clip": self._load_clip(paths),
            "label": torch.tensor(float(participant["label"]), dtype=torch.float32),
            "subj_id": str(participant["subj_id"]),
            "start": torch.tensor(record.start, dtype=torch.long),
            "window_kind": record.kind,
        }

    def summary(self) -> Dict[str, object]:
        kinds: Dict[str, int] = {}
        for record in self.records:
            kinds[record.kind] = kinds.get(record.kind, 0) + 1
        frame_counts = [len(item["frames"]) for item in self.participants]
        return {
            "root": str(self.root),
            "split": self.split,
            "participants": len(self.participants),
            "windows": len(self.records),
            "window_kinds": kinds,
            "frames_min": min(frame_counts),
            "frames_max": max(frame_counts),
            "frames_mean": float(np.mean(frame_counts)),
            "normalization": "RGB; (pixel-127.5)/127.5 (ExpADA)",
        }


def make_raw_dataset(
    data_root: str,
    splits: Sequence[str],
    **kwargs,
) -> Dataset:
    datasets = [
        AVEC2013Motion32Dataset(data_root=data_root, split=split, **kwargs)
        for split in splits
    ]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


class AVEC2013Motion32FeatureDataset(Dataset):
    """Load AVEC2013 participant bags produced by ``extract``.

    Stage 2 intentionally consumes only the saved 1024-D neck/encoder
    features.  Stage-1 regression scores and ordinal probabilities are not
    part of ExpADA's released extraction interface.
    """

    def __init__(
        self,
        feature_root: str,
        splits: Iterable[str],
        feature_dim: int = 1024,
    ) -> None:
        self.root = Path(feature_root).resolve()
        self.feature_dim = int(feature_dim)
        self.files: List[Path] = []
        for split in splits:
            if split not in RAW_SPLITS:
                raise ValueError(f"Unknown feature split: {split}")
            directory = self.root / split
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            files = sorted(directory.glob("*.pkl"))
            if not files:
                raise RuntimeError(f"No .pkl features in {directory}")
            self.files.extend(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        value = _torch_load(path)
        if not isinstance(value, dict):
            raise ValueError(f"Expected dict in {path}")
        features = torch.as_tensor(value["features"], dtype=torch.float32)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected features (T,{self.feature_dim}) in {path}, got {tuple(features.shape)}"
            )
        if not torch.isfinite(features).all():
            raise ValueError(f"Non-finite tensors in {path}")
        return {
            "features": features,
            "label": torch.tensor(float(value["gt_label"]), dtype=torch.float32),
            "subj_id": str(value.get("subj_id", path.stem)),
            "path": str(path),
        }


def feature_bag_collate(batch):
    if len(batch) != 1:
        raise ValueError("Variable-length Stage-2 bags require batch_size=1")
    return batch[0]


# Backward-friendly aliases are intentionally local to this new AVEC2013
# module.  They let small external utilities reuse the old generic class names
# without importing or modifying any AVEC2014 dataloader.
ExpADAMotion32Dataset = AVEC2013Motion32Dataset
Motion32FeatureDataset = AVEC2013Motion32FeatureDataset
