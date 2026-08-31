#!/usr/bin/env python3
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.mlp_classifier import (  # noqa: E402
    FusionModule,
    MLPClassifier,
    _atomic_json_dump,
    _evaluate,
    _save_atgs_checkpoint,
    _save_granularity_diagnostics,
)


def main():
    torch.manual_seed(31)
    fusion = FusionModule(
        branch_dims=[12],
        modal_interaction="concat",
        branch_names=["vision/model"],
        branch_granularity_counts=[3],
        branch_granularity_base_indices=[1],
        branch_granularity_labels=[("fine", "base", "coarse")],
        granularity_hidden_dim=8,
        granularity_temperature=0.7,
        granularity_base_prior=0.8,
    )
    mlp = MLPClassifier(
        input_dim=fusion.output_dim,
        hidden_dim=8,
        num_layers=2,
        dropout=0.0,
        num_classes=2,
    )
    features = torch.randn(12, 12)
    labels = torch.tensor([0, 1] * 6, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=5,
        shuffle=False,
    )
    _evaluate(mlp, fusion, loader, np.asarray([0, 1]), "cpu")
    details = fusion.last_evaluation_details
    sample_indices = np.arange(20, 32, dtype=np.int64)

    with tempfile.TemporaryDirectory(prefix="atgs_artifacts_") as temp_dir:
        artifact_dir = Path(temp_dir) / "dataset_a"
        summary = _save_granularity_diagnostics(
            artifact_dir=artifact_dir,
            split_name="validation",
            details=details,
            sample_indices=sample_indices,
            fusion_module=fusion,
        )
        _atomic_json_dump(
            artifact_dir / "atgs_diagnostics_summary.json",
            {"schema_version": 1, "splits": {"validation": summary}},
        )
        _save_atgs_checkpoint(
            artifact_dir=artifact_dir,
            mlp=mlp,
            fusion_module=fusion,
            classes=np.asarray([0, 1]),
            training_history=[{"epoch": 1, "loss": 0.5}],
            selection="final_epoch",
            selected_epoch=1,
            validation_macro_f1=0.5,
            balance_weight=0.01,
            entropy_weight=0.001,
        )

        diagnostic_path = artifact_dir / "atgs_diagnostics_validation.npz"
        with np.load(diagnostic_path, allow_pickle=False) as diagnostic:
            assert all(diagnostic[key].dtype.kind != "O" for key in diagnostic.files)
            assert np.array_equal(diagnostic["sample_index"], sample_indices)
            weight_keys = [key for key in diagnostic.files if key.endswith("_weights")]
            assert len(weight_keys) == 1
            weights = diagnostic[weight_keys[0]]
            assert weights.shape == (12, 3)
            assert np.isfinite(weights).all()
            np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)
            top1_key = weight_keys[0].replace("_weights", "_top1")
            assert np.array_equal(diagnostic[top1_key], weights.argmax(axis=1))

        json_summary = json.loads(
            (artifact_dir / "atgs_diagnostics_summary.json").read_text(
                encoding="utf-8"
            )
        )
        branch_summary = json_summary["splits"]["validation"]["branches"][
            "vision/model"
        ]
        np.testing.assert_allclose(
            branch_summary["mean_weights"], weights.mean(axis=0), atol=1e-6
        )
        assert sum(branch_summary["top1_counts"]) == len(weights)

        checkpoint_path = artifact_dir / "atgs_checkpoint.pt"
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        assert checkpoint["selection"] == "final_epoch"
        assert checkpoint["selected_epoch"] == 1
        assert "vision/model" in checkpoint["selector_state_dicts"]
        assert checkpoint["fusion_configuration"]["input_branch_dims"] == [12]
        selector_config = checkpoint["fusion_configuration"][
            "selector_configurations"
        ]["vision/model"]
        assert selector_config["temperature"] == 0.7
        assert selector_config["base_prior"] == 0.8
        assert checkpoint["mlp_configuration"]["input_dim"] == 4
        assert all(
            tensor.device.type == "cpu"
            for tensor in checkpoint["fusion_state_dict"].values()
        )

        restored_fusion = FusionModule(
            branch_dims=[12],
            modal_interaction="concat",
            branch_names=["vision/model"],
            branch_granularity_counts=[3],
            branch_granularity_base_indices=[1],
            branch_granularity_labels=[("fine", "base", "coarse")],
            granularity_hidden_dim=8,
            granularity_temperature=0.7,
            granularity_base_prior=0.8,
        )
        restored_mlp = MLPClassifier(4, 8, 2, 0.0, 2)
        restored_fusion.load_state_dict(
            checkpoint["fusion_state_dict"], strict=True
        )
        restored_mlp.load_state_dict(checkpoint["mlp_state_dict"], strict=True)
        fusion.eval()
        restored_fusion.eval()
        mlp.eval()
        restored_mlp.eval()
        with torch.no_grad():
            expected_logits = mlp(fusion([features]))
            restored_logits = restored_mlp(restored_fusion([features]))
        torch.testing.assert_close(expected_logits, restored_logits)

        assert not list(artifact_dir.glob(".*.tmp*"))

    print("ATGS ARTIFACT VALIDATION PASSED")


if __name__ == "__main__":
    main()
