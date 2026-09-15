from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from sampling import sample_indices


class HARTemporalDataset(Dataset):
    def __init__(self, views, labels):
        self.views = views
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        xs = [torch.tensor(v[idx], dtype=torch.float32) for v in self.views]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return xs, y, idx


def _default_har_path():
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "HAR_dataset" / "UCI HAR Dataset"


def _standardize_view_windows(v):
    n, t, d = v.shape
    flat = v.reshape(n * t, d)

    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True) + 1e-8

    flat = (flat - mean) / std
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    flat = np.clip(flat, -10.0, 10.0)

    return flat.reshape(n, t, d).astype(np.float32)


def _load_signal(data_path, split, signal_name):
    signal_file = data_path / split / "Inertial Signals" / f"{signal_name}_{split}.txt"
    if not signal_file.exists():
        raise FileNotFoundError(f"Missing HAR signal file: {signal_file}")
    return np.loadtxt(signal_file, dtype=np.float32)


def _load_split(data_path, split, view_mode):
    y_file = data_path / split / f"y_{split}.txt"
    if not y_file.exists():
        raise FileNotFoundError(f"Missing HAR label file: {y_file}")

    labels = np.loadtxt(y_file, dtype=np.int64) - 1

    signal_groups = {
        "total_acc": ["total_acc_x", "total_acc_y", "total_acc_z"],
        "body_acc": ["body_acc_x", "body_acc_y", "body_acc_z"],
        "body_gyro": ["body_gyro_x", "body_gyro_y", "body_gyro_z"],
    }

    if view_mode == "groups":
        views = []
        for signal_names in signal_groups.values():
            channels = [_load_signal(data_path, split, name) for name in signal_names]
            views.append(np.stack(channels, axis=2).astype(np.float32))
        return views, labels

    if view_mode == "channels":
        ordered_signals = [
            "total_acc_x",
            "total_acc_y",
            "total_acc_z",
            "body_acc_x",
            "body_acc_y",
            "body_acc_z",
            "body_gyro_x",
            "body_gyro_y",
            "body_gyro_z",
        ]
        views = [
            _load_signal(data_path, split, name)[:, :, None].astype(np.float32)
            for name in ordered_signals
        ]
        return views, labels

    raise ValueError("view_mode must be either 'groups' or 'channels'")


def _select_views(views, selected_view_indices):
    if selected_view_indices is None:
        return views

    selected = []
    for idx in selected_view_indices:
        if idx < 0 or idx >= len(views):
            raise ValueError(f"View index {idx} is out of range for {len(views)} views")
        selected.append(views[idx])
    return selected


def load_har_temporal(
    data_path=None,
    split="train",
    view_mode="groups",
    selected_view_indices=None,
    max_samples=None,
    seed=42,
    sample_strategy="stratified",
):
    data_path = Path(data_path) if data_path is not None else _default_har_path()

    if split == "all":
        splits = ["train", "test"]
    elif split in {"train", "test"}:
        splits = [split]
    else:
        raise ValueError("split must be one of: train, test, all")

    views_by_split = []
    labels_by_split = []

    for split_name in splits:
        split_views, split_labels = _load_split(data_path, split_name, view_mode)
        views_by_split.append(split_views)
        labels_by_split.append(split_labels)

    view_count = len(views_by_split[0])
    views = [
        np.concatenate([split_views[v] for split_views in views_by_split], axis=0)
        for v in range(view_count)
    ]
    labels = np.concatenate(labels_by_split, axis=0).astype(np.int64)

    views = _select_views(views, selected_view_indices)

    if max_samples is not None and len(labels) > max_samples:
        indices = sample_indices(labels, max_samples, seed, sample_strategy)
        views = [v[indices] for v in views]
        labels = labels[indices]

    views = [_standardize_view_windows(v) for v in views]

    print(f"Loaded HAR temporal data from: {data_path}")
    print(f"HAR split: {split}, view_mode: {view_mode}")
    print("Unique labels:", np.unique(labels))
    for i, v in enumerate(views):
        print(f"View {i + 1} shape: {v.shape}")

    dims = [v.shape[2] for v in views]
    view = len(views)
    data_size = len(labels)
    class_num = len(np.unique(labels))

    dataset = HARTemporalDataset(views, labels)
    return dataset, dims, view, data_size, class_num
