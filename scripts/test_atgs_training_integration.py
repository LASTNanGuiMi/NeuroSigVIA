#!/usr/bin/env python3
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mlp_classifier import train_mlp_classifier  # noqa: E402


class DummyAdaptiveVision(nn.Module):
    image_mode = "med_activity_graph"
    feature_granularity_count = 3
    feature_granularity_base_index = 1
    feature_granularity_labels = ("fine", "base", "coarse")

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward_granularities(self, batch):
        summary = torch.stack(
            (
                batch.mean(dim=(1, 2)),
                batch.std(dim=(1, 2), unbiased=False),
                batch.square().mean(dim=(1, 2)),
                batch.amax(dim=(1, 2)) - batch.amin(dim=(1, 2)),
            ),
            dim=1,
        )
        offsets = torch.tensor(
            (
                (0.15, 0.00, 0.00, 0.05),
                (0.00, 0.15, 0.00, 0.05),
                (0.00, 0.00, 0.15, 0.05),
            ),
            dtype=summary.dtype,
            device=summary.device,
        )
        return (summary.unsqueeze(1) + offsets.unsqueeze(0)) * self.scale


def make_loader(sample_count, seed):
    generator = torch.Generator().manual_seed(seed)
    signals = torch.randn(sample_count, 3, 16, generator=generator)
    labels = np.asarray([index % 2 for index in range(sample_count)])
    signals[torch.as_tensor(labels == 1)] += 0.35
    return (
        DataLoader(
            TensorDataset(signals),
            batch_size=4,
            shuffle=False,
        ),
        labels,
    )


def main():
    train_loader, train_labels = make_loader(20, seed=41)
    test_loader, test_labels = make_loader(8, seed=43)
    model = DummyAdaptiveVision()

    with tempfile.TemporaryDirectory(prefix="atgs_training_") as temp_dir:
        artifact_dir = Path(temp_dir) / "atgs" / "dummy"
        val_metrics, test_metrics, train_indices, val_indices = (
            train_mlp_classifier(
                train_loader=train_loader,
                train_labels=train_labels,
                test_loader=test_loader,
                test_labels=test_labels,
                channels=3,
                device="cpu",
                batch_size=4,
                random_seed=42,
                val_ratio=0.25,
                hidden_dim=8,
                num_layers=2,
                dropout=0.0,
                lr=1e-2,
                weight_decay=0.0,
                epochs=3,
                early_stop_patience=1,
                modal_interaction="concat",
                fusion_dim=8,
                fusion_heads=2,
                cross_attn_query="ts",
                mask_prob=0.0,
                pretrain_epochs=0,
                vision_model_1=model,
                artifact_dir=artifact_dir,
            )
        )
        assert np.isfinite(val_metrics["macro_f1"])
        assert np.isfinite(test_metrics["macro_f1"])
        assert set(train_indices).isdisjoint(val_indices)
        assert sorted([*train_indices, *val_indices]) == list(range(20))

        checkpoint = torch.load(
            artifact_dir / "atgs_checkpoint.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert checkpoint["selection"] == "best_validation_epoch"
        assert 1 <= checkpoint["selected_epoch"] <= 3
        assert "vision_model_1" in checkpoint["selector_state_dicts"]

        with np.load(
            artifact_dir / "atgs_diagnostics_train.npz",
            allow_pickle=False,
        ) as train_diagnostics:
            assert np.array_equal(
                train_diagnostics["sample_index"],
                np.asarray(train_indices, dtype=np.int64),
            )
        with np.load(
            artifact_dir / "atgs_diagnostics_validation.npz",
            allow_pickle=False,
        ) as validation_diagnostics:
            assert np.array_equal(
                validation_diagnostics["sample_index"],
                np.asarray(val_indices, dtype=np.int64),
            )
        assert (artifact_dir / "atgs_diagnostics_test.npz").is_file()
        assert (artifact_dir / "atgs_diagnostics_summary.json").is_file()
        assert (artifact_dir / "atgs_training_history.json").is_file()
        assert not list(artifact_dir.glob(".*.tmp*"))

    assert not model.scale.requires_grad
    assert model.scale.grad is None
    print("ATGS TRAINING INTEGRATION VALIDATION PASSED")


if __name__ == "__main__":
    main()
