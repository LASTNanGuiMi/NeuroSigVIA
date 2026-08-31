import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.utils import save_image

from src.medformer_graph import (
    MedformerGraphRenderer,
    TemporalGranularityGraphBank,
)
from src.neurosigvit import preprocess_graph


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_split(dataset, frac=0.2, random_seed=None):
    if not 0 < frac < 1:
        raise ValueError(f"Validation ratio must be between 0 and 1, got {frac}.")

    split = int(len(dataset) * frac)
    keys = list(range(len(dataset)))
    rng = np.random.default_rng(random_seed)
    rng.shuffle(keys)
    train = keys[split:]
    val = keys[:split]

    return train, val


def write_split_indices(
    result_dir,
    dataset,
    train_indices,
    val_indices,
    random_seed,
    val_ratio,
):
    split_dir = f"{result_dir}/splits"
    os.makedirs(split_dir, exist_ok=True)
    np.savez(
        f"{split_dir}/{dataset}_seed{random_seed}_val{val_ratio}.npz",
        train_indices=np.asarray(train_indices, dtype=np.int64),
        val_indices=np.asarray(val_indices, dtype=np.int64),
        random_seed=random_seed,
        val_ratio=val_ratio,
    )


def get_patch_size(patch_size, T):
    patch_sizes = [None]

    if patch_size == "sqrt":
        patch_sizes = [int(math.sqrt(T))]
    elif patch_size == "linspace":
        patch_sizes = (
            np.linspace(
                1,
                math.ceil(T // 2),
                min(math.ceil(T // 2), 20),
            )
            .astype(int)
            .tolist()
        )

    return patch_sizes


def resize_mantis_input(X):
    X_scaled = F.interpolate(X, size=512, mode="linear", align_corners=False)
    return X_scaled


def pad_timeseries(timeseries: torch.Tensor, seq_len: int):
    T = timeseries.shape[2]
    pad_width = seq_len - T
    padded = F.pad(timeseries, (pad_width, 0), value=0)

    return padded


def downsample_timeseries(timeseries: torch.Tensor, seq_len: int):
    x_down = F.interpolate(
        timeseries,
        size=seq_len,
        mode="linear",
        align_corners=True,
    )

    return x_down


def resize_moment_input(batch_ts):
    T = batch_ts.shape[-1]

    if T < 512:
        batch_ts = pad_timeseries(batch_ts, seq_len=512)
    elif T > 512:
        batch_ts = downsample_timeseries(batch_ts, seq_len=512)

    return batch_ts


def write_result_table(
    result_dir, dataset, val_metrics, test_metrics, patch_size=None, image_mode=None
):
    row = {
        "dataset": dataset,
        "image_mode": image_mode,
        "patch_size": patch_size,
    }
    row.update({f"val_{metric}": value for metric, value in val_metrics.items()})
    row.update({f"test_{metric}": value for metric, value in test_metrics.items()})

    df = pd.DataFrame(
        [
            row
        ]
    )

    filename = f"{result_dir}/train_val.csv"
    file_exists = os.path.isfile(filename)
    df.to_csv(filename, mode="a", index=False, header=not file_exists)

    return


def save_activity_graph_samples(
    result_dir,
    dataset,
    dataloader,
    num_samples,
    image_mode="activity_graph",
    med_activity_patch_lengths=(2, 4, 8),
    med_activity_channel_mix=0.35,
    med_activity_router_temperature=0.2,
    med_activity_router_mix=0.5,
    med_activity_adaptive_granularity=False,
    med_activity_granularity_bank=((1, 2, 4), (2, 4, 8), (4, 8, 16)),
):
    if num_samples <= 0:
        return

    if image_mode == "activity_graph":
        transform = lambda batch: preprocess_graph(
            batch, mode="multicolumn", render="waveform"
        )
    elif image_mode == "med_activity_graph":
        if med_activity_adaptive_granularity:
            transform = TemporalGranularityGraphBank(
                patch_length_bank=med_activity_granularity_bank,
                channel_mix=med_activity_channel_mix,
                router_temperature=med_activity_router_temperature,
                router_mix=med_activity_router_mix,
            )
        else:
            transform = MedformerGraphRenderer(
                patch_lengths=med_activity_patch_lengths,
                channel_mix=med_activity_channel_mix,
                router_temperature=med_activity_router_temperature,
                router_mix=med_activity_router_mix,
            )
    else:
        raise ValueError(f"Unsupported activity graph image mode {image_mode}.")

    sample_dir = os.path.join(result_dir, f"{image_mode}_samples")
    os.makedirs(sample_dir, exist_ok=True)

    saved = 0
    for (batch,) in dataloader:
        graph_images = transform(batch)
        if graph_images.ndim == 5:
            regime_labels = [
                "-".join(str(length) for length in regime)
                for regime in med_activity_granularity_bank
            ]
            for candidate_images in graph_images:
                if saved >= num_samples:
                    return
                for regime_label, image in zip(regime_labels, candidate_images):
                    filename = os.path.join(
                        sample_dir,
                        f"{dataset}_sample{saved}_granularity{regime_label}.png",
                    )
                    save_image(image, filename)
                saved += 1
            continue
        for image in graph_images:
            if saved >= num_samples:
                return
            filename = os.path.join(sample_dir, f"{dataset}_sample{saved}.png")
            save_image(image, filename)
            saved += 1


def render_activity_lineplot(
    signals,
    width=960,
    row_height=120,
    margin_x=48,
    margin_y=24,
    line_width=2,
):
    signals = np.asarray(signals, dtype=np.float32)
    if signals.ndim != 2:
        raise ValueError(f"signals must have shape (n, T), got {signals.shape}.")

    channels, time_steps = signals.shape
    if channels == 0:
        raise ValueError("signals must contain at least one channel.")

    height = margin_y * 2 + row_height * channels
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    plot_width = max(1, width - margin_x * 2)
    for channel_idx, signal in enumerate(signals):
        y_top = margin_y + channel_idx * row_height
        y_bottom = y_top + row_height - 1
        y_center = y_top + row_height // 2

        draw.line(
            [(margin_x, y_center), (width - margin_x, y_center)],
            fill=(225, 225, 225),
            width=1,
        )

        value_min = float(signal.min())
        value_max = float(signal.max())
        value_range = value_max - value_min
        if value_range <= 1e-8:
            y_values = np.full(time_steps, y_center, dtype=np.float32)
        else:
            normalized = (signal - value_min) / (value_range + 1e-8)
            usable_height = max(1, row_height - 20)
            y_values = y_top + 10 + (1.0 - np.clip(normalized, 0.0, 1.0)) * (
                usable_height - 1
            )

        if time_steps == 1:
            points = [(margin_x, float(y_values[0])), (width - margin_x, float(y_values[0]))]
        else:
            x_values = margin_x + np.linspace(0, plot_width, num=time_steps)
            points = list(zip(x_values.tolist(), y_values.tolist()))

        draw.line(points, fill="black", width=line_width, joint="curve")
        draw.rectangle(
            [(margin_x, y_top), (width - margin_x, y_bottom)],
            outline=(210, 210, 210),
            width=1,
        )

    return image


def save_activity_lineplot_samples(result_dir, dataset, dataloader, num_samples):
    if num_samples <= 0:
        return

    sample_dir = os.path.join(result_dir, "activity_lineplot_samples")
    os.makedirs(sample_dir, exist_ok=True)

    saved = 0
    for (batch,) in dataloader:
        for sample in batch:
            if saved >= num_samples:
                return
            filename = os.path.join(sample_dir, f"{dataset}_sample{saved}.png")
            render_activity_lineplot(sample.detach().cpu().numpy()).save(filename)
            saved += 1
