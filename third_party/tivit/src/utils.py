import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_split(dataset, frac=0.2):
    split = int(len(dataset) * frac)
    keys = list(range(len(dataset)))
    np.random.shuffle(keys)
    train = keys[split:]
    val = keys[:split]

    return train, val


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


def write_result_table(result_dir, dataset, acc_val, acc_test, patch_size=None):
    df = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "patch_size": patch_size,
                "val_accuracy": np.mean(acc_val),
                "test_accuracy": acc_test,
            }
        ]
    )

    filename = f"{result_dir}/train_val.csv"
    file_exists = os.path.isfile(filename)
    df.to_csv(filename, mode="a", index=False, header=not file_exists)

    return
