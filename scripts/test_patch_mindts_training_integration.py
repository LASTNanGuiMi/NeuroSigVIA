#!/usr/bin/env python3
"""Small deterministic CPU integration test for patch-level MindTS training."""

import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.medformer_graph import TemporalGranularityGraphBank  # noqa: E402
from src.patch_mindts import (  # noqa: E402
    PATCH_MINDTS_ARCHITECTURE,
    PatchMindTSFusionModule,
    train_patch_mindts_classifier,
)


class DummyVision(nn.Module):
    """Cheap image statistics encoder backed by the real graph renderer bank."""

    aggregation = "mean"

    def __init__(self, fail_on_encode=False):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.fail_on_encode = bool(fail_on_encode)
        self.med_activity_granularity_bank = TemporalGranularityGraphBank(
            patch_length_bank=((4,), (8,), (16,)),
            img_size=12,
            channel_mix=0.25,
            router_temperature=0.5,
            router_mix=0.25,
        )

    def forward_vit(self, images):
        if self.fail_on_encode:
            raise AssertionError("hot patch cache unexpectedly called DummyVision")
        flattened = images.float().flatten(start_dim=2)
        channel_mean = flattened.mean(dim=-1)
        channel_std = flattened.std(dim=-1, unbiased=False)
        overall_mean = flattened.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        overall_std = (
            flattened.flatten(start_dim=1)
            .std(dim=-1, unbiased=False)
            .unsqueeze(-1)
        )
        features = torch.cat(
            [channel_mean, channel_std, overall_mean, overall_std],
            dim=-1,
        )
        return features * self.scale

    @staticmethod
    def aggregate_hidden_representations(hidden_states, aggregation):
        if aggregation != "mean":
            raise AssertionError(f"unexpected aggregation: {aggregation}")
        return hidden_states


class DummyMantis(nn.Module):
    def __init__(self, fail_on_encode=False):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.fail_on_encode = bool(fail_on_encode)

    def forward(self, inputs):
        if self.fail_on_encode:
            raise AssertionError("hot patch cache unexpectedly called DummyMantis")
        values = inputs[:, 0].float()
        features = torch.stack(
            [
                values.mean(dim=-1),
                values.std(dim=-1, unbiased=False),
                values.square().mean(dim=-1).sqrt(),
                values.amax(dim=-1),
                values.amin(dim=-1),
                values[:, -1] - values[:, 0],
            ],
            dim=-1,
        )
        return features * self.scale


def make_loader(subject_count, windows_per_subject, seed, split_name):
    sample_count = subject_count * windows_per_subject
    generator = torch.Generator().manual_seed(seed)
    subject_labels = np.asarray(
        [index % 2 for index in range(subject_count)],
        dtype=np.int64,
    )
    labels = np.repeat(subject_labels, windows_per_subject)
    subject_ids = np.repeat(
        np.asarray(
            [f"{split_name}_subject_{index}" for index in range(subject_count)]
        ),
        windows_per_subject,
    )
    signals = 0.25 * torch.randn(
        sample_count,
        3,
        70,
        generator=generator,
    )
    time = torch.linspace(0.0, 2.0 * math.pi, 70)
    class_pattern = torch.sin(time).view(1, 1, -1)
    positive = torch.from_numpy(labels == 1)
    signals[positive] += 0.45 + 0.15 * class_pattern
    lengths = torch.tensor(
        [64 if index % 3 == 0 else 70 for index in range(sample_count)],
        dtype=torch.long,
    )
    dataset = TensorDataset(signals, lengths)
    # Subject IDs deliberately remain CPU-side metadata.  Returning them from
    # __getitem__ would change the raw batch contract used by feature extraction.
    dataset.sample_subject_ids = subject_ids
    return (
        DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0),
        labels,
        subject_ids,
    )


def training_arguments(
    train_loader,
    train_labels,
    validation_loader,
    validation_labels,
    test_loader,
    test_labels,
    vision_model,
    mantis_model,
    cache_dir,
    artifact_dir,
):
    return {
        "train_loader": train_loader,
        "train_labels": train_labels,
        "test_loader": test_loader,
        "test_labels": test_labels,
        "channels": 3,
        "device": "cpu",
        "batch_size": 4,
        "random_seed": 42,
        "val_ratio": 0.25,
        "hidden_dim": 8,
        "num_layers": 2,
        "dropout": 0.0,
        "lr": 1.0e-2,
        "weight_decay": 0.0,
        "epochs": 2,
        "early_stop_patience": 0,
        "fusion_dim": 8,
        "fusion_heads": 2,
        "alignment_dim": 4,
        "alignment_temperature": 0.2,
        "alignment_weight": 0.1,
        "outer_patch_size": 64,
        "outer_patch_stride": 64,
        "visual_encode_batch_size": 16,
        "class_weight": "none",
        "vision_model": vision_model,
        "mantis_model": mantis_model,
        "val_loader": validation_loader,
        "val_labels": validation_labels,
        "feature_cache_dir": cache_dir,
        "feature_cache_signature": "patch-mindts-integration-v4",
        "granularity_temperature": 1.0,
        "granularity_balance_weight": 0.01,
        "granularity_entropy_weight": 0.01,
        "granularity_mix_shrinkage_weight": 0.005,
        "granularity_prior_kl_weight": 0.001,
        "granularity_usage_floor": 0.05,
        "granularity_usage_ema_decay": 0.95,
        "granularity_entropy_floor": 0.55,
        "granularity_entropy_ceiling": 1.0,
        "granularity_router_mode": "adaptive_v4",
        "granularity_local_mix_max": 0.50,
        "granularity_local_mix_init": 0.10,
        "granularity_global_mix_max": 0.75,
        "granularity_global_mix_init": 0.50,
        "granularity_evidence_half_saturation": 0.05,
        "granularity_minimum_weight": 0.0,
        "granularity_score_cap": 1.0,
        "checkpoint_metric": "auto",
        "channel_hidden_dim": 8,
        "artifact_dir": artifact_dir,
    }


def assert_finite_metrics(metrics):
    window_expected = {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "macro_auroc",
        "macro_auprc",
    }
    expected = window_expected | {
        f"subject_{name}" for name in window_expected
    } | {"subject_macro_log_loss"}
    assert set(metrics) == expected
    assert all(np.isfinite(float(metrics[name])) for name in expected)


def assert_diagnostics(
    path,
    expected_samples,
    expected_subject_ids,
    windows_per_subject,
):
    with np.load(path, allow_pickle=False) as diagnostics:
        weights = diagnostics["weights"]
        scores = diagnostics["scores"]
        score_margin = diagnostics["score_margin"]
        graph_pairwise_cosine = diagnostics["graph_pairwise_cosine"]
        patch_mask = diagnostics["patch_mask"].astype(bool)
        top1 = diagnostics["top1"]
        assert int(diagnostics["schema_version"].item()) == 5
        assert str(diagnostics["architecture"].item()) == PATCH_MINDTS_ARCHITECTURE
        assert weights.shape == (expected_samples, 2, 3)
        assert scores.shape == weights.shape
        assert score_margin.shape == (expected_samples, 2)
        assert graph_pairwise_cosine.shape == (expected_samples, 2, 3)
        assert patch_mask.shape == (expected_samples, 2)
        assert top1.shape == (expected_samples, 2)
        assert np.isfinite(weights).all()
        assert np.isfinite(scores).all()
        assert np.isfinite(score_margin).all()
        assert np.isfinite(graph_pairwise_cosine).all()
        np.testing.assert_allclose(
            torch.softmax(torch.from_numpy(scores[patch_mask]), dim=-1).numpy(),
            weights[patch_mask],
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            weights[patch_mask].sum(axis=-1),
            np.ones(int(patch_mask.sum())),
            rtol=1e-6,
            atol=1e-6,
        )
        assert (weights[patch_mask] > 0.0).all()
        np.testing.assert_array_equal(
            top1[patch_mask],
            weights[patch_mask].argmax(axis=-1),
        )
        np.testing.assert_allclose(
            weights[~patch_mask],
            np.zeros_like(weights[~patch_mask]),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            top1[~patch_mask],
            -np.ones(int((~patch_mask).sum()), dtype=top1.dtype),
        )
        assert (~patch_mask).any(), "variable lengths should create invalid patch slots"
        for name in (
            "query_shuffle_weights",
            "query_constant_weights",
        ):
            assert diagnostics[name].shape == weights.shape
            assert np.isfinite(diagnostics[name]).all()
        for name in (
            "query_shuffle_y_score",
            "query_constant_y_score",
            "uniform_y_score",
            "dataset_prior_y_score",
            "sample_global_y_score",
            "patch_local_y_score",
        ):
            assert diagnostics[name].shape == (expected_samples, 2)
            assert np.isfinite(diagnostics[name]).all()
        assert diagnostics["soft_uniform_visual_effect"].shape == (
            expected_samples,
            2,
        )
        assert diagnostics["projected_query_rms"].shape == (expected_samples, 2)
        assert diagnostics["projected_relative_key_rms"].shape == (
            expected_samples,
            2,
        )
        assert diagnostics["projected_key_pairwise_l2"].shape == (
            expected_samples,
            2,
            3,
        )
        for name in (
            "dataset_prior_weights",
            "sample_global_weights",
            "patch_local_weights",
        ):
            assert diagnostics[name].shape == weights.shape
            assert np.isfinite(diagnostics[name]).all()
            np.testing.assert_allclose(
                diagnostics[name][patch_mask].sum(axis=-1),
                np.ones(int(patch_mask.sum())),
                rtol=1e-6,
                atol=1e-6,
            )
        for name in (
            "candidate_evidence",
            "global_mix",
            "local_mix",
            "effective_global_mix",
            "effective_local_mix",
        ):
            assert diagnostics[name].shape == patch_mask.shape
            assert np.isfinite(diagnostics[name]).all()
            assert (diagnostics[name][patch_mask] >= 0.0).all()
            np.testing.assert_allclose(
                diagnostics[name][~patch_mask],
                np.zeros(int((~patch_mask).sum())),
                rtol=0.0,
                atol=0.0,
            )

        unique_subject_ids = np.unique(expected_subject_ids)
        np.testing.assert_array_equal(
            diagnostics["sample_subject_id"],
            np.asarray(expected_subject_ids),
        )
        np.testing.assert_array_equal(
            diagnostics["subject_id"],
            unique_subject_ids,
        )
        assert diagnostics["subject_y_true"].shape == (len(unique_subject_ids),)
        assert diagnostics["subject_y_pred"].shape == (len(unique_subject_ids),)
        assert diagnostics["subject_y_score"].shape == (len(unique_subject_ids), 2)
        np.testing.assert_allclose(
            diagnostics["subject_y_score"].sum(axis=-1),
            np.ones(len(unique_subject_ids)),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            diagnostics["subject_window_count"],
            np.full(len(unique_subject_ids), windows_per_subject),
        )


def assert_artifacts(
    artifact_dir,
    validation_subject_ids,
    test_subject_ids,
    windows_per_subject,
):
    validation_diagnostics = artifact_dir / "patch_atgs_diagnostics_validation.npz"
    test_diagnostics = artifact_dir / "patch_atgs_diagnostics_test.npz"
    summary_path = artifact_dir / "patch_atgs_summary.json"
    history_path = artifact_dir / "patch_mindts_training_history.json"
    checkpoint_path = artifact_dir / "patch_mindts_checkpoint.pt"
    for path in (
        validation_diagnostics,
        test_diagnostics,
        summary_path,
        history_path,
        checkpoint_path,
    ):
        assert path.is_file(), path

    assert_diagnostics(
        validation_diagnostics,
        len(validation_subject_ids),
        validation_subject_ids,
        windows_per_subject,
    )
    assert_diagnostics(
        test_diagnostics,
        len(test_subject_ids),
        test_subject_ids,
        windows_per_subject,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == 5
    assert summary["architecture"] == PATCH_MINDTS_ARCHITECTURE
    assert set(summary["splits"]) == {"validation", "test"}
    for split in summary["splits"].values():
        assert len(split["mean_weights"]) == 3
        assert len(split["top1_counts"]) == 3
        assert sum(split["top1_counts"]) == split["valid_patch_count"]
        assert np.isfinite(split["mean_weights"]).all()
        assert np.isclose(sum(split["top1_time_weighted_fraction"]), 1.0)
        assert len(split["std_weights"]) == 3
        assert len(split["graph_pairwise_cosine_mean"]) == 3
        assert 1.0 <= split["mean_per_patch_effective_granularity_count"] <= 3.0
        assert 1.0 <= split["global_effective_granularity_count"] <= 3.0
        assert set(split["query_counterfactuals"]) >= {
            "shuffle_weighted_total_variation",
            "constant_query_weighted_total_variation",
            "shuffle_prediction_probability_total_variation",
            "shuffle_prediction_probability_total_variation_all_samples",
        }
        assert set(split["router_projected_space"]) == {
            "query_rms_mean",
            "relative_key_rms_mean",
            "relative_key_rms_quantiles",
            "key_pairwise_l2_mean",
        }
        assert set(split["soft_route_effect"]) == {
            "visual_relative_l2_mean",
            "prediction_probability_total_variation",
        }
        assert set(split["router_hierarchy"]) == {
            "mean_dataset_prior_weights",
            "mean_sample_global_weights",
            "mean_patch_local_weights",
            "candidate_evidence",
            "sample_global_score_margin",
            "patch_local_score_margin",
            "sample_global_confidence",
            "patch_local_confidence",
            "global_mix",
            "local_mix",
            "effective_global_mix",
            "effective_local_mix",
            "prediction_probability_total_variation",
        }
        assert split["subject_aggregation"] == "mean_window_softmax_probability"
        assert split["subject_count"] == 2
        assert set(split["subject_metrics"]) == {
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "macro_auroc",
            "macro_auprc",
            "macro_log_loss",
        }
        assert all(
            np.isfinite(float(value))
            for value in split["subject_metrics"].values()
        )

    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert history["schema_version"] == 5
    assert history["architecture"] == PATCH_MINDTS_ARCHITECTURE
    assert history["checkpoint_selection_requested"] == "auto"
    assert history["checkpoint_selection_effective"] == "subject_macro_f1"
    assert len(history["checkpoint_selection_key"]) == 2
    assert np.isfinite(float(history["validation_subject_macro_f1"]))
    assert history["early_stopping"] == {
        "strategy": "raw_selection_key",
        "patience": 0,
        "min_epochs": 0,
        "ema_decay": 0.6,
        "min_delta": 0.0,
        "best_monitor_score": history["early_stopping"]["best_monitor_score"],
        "early_stopped": False,
        "stopped_epoch": None,
        "completed_epochs": 2,
        "checkpoint_selection_uses_raw_metric": True,
    }
    assert np.isfinite(history["early_stopping"]["best_monitor_score"])
    assert len(history["epochs"]) == 2
    for epoch in history["epochs"]:
        for key in (
            "loss",
            "task_loss",
            "alignment_loss",
            "weighted_alignment_loss",
            "granularity_regularization_loss",
            "granularity_balance_loss",
            "granularity_entropy_loss",
            "granularity_entropy_floor_loss",
            "weighted_granularity_balance_loss",
            "weighted_granularity_entropy_loss",
            "granularity_usage_floor_loss",
            "granularity_entropy_ceiling_loss",
            "weighted_granularity_usage_floor_loss",
            "weighted_granularity_entropy_floor_loss",
            "weighted_granularity_entropy_ceiling_loss",
            "granularity_mix_shrinkage_loss",
            "weighted_granularity_mix_shrinkage_loss",
            "granularity_prior_kl_loss",
            "weighted_granularity_prior_kl_loss",
            "mean_granularity_max_weight",
            "mean_granularity_entropy",
            "mean_candidate_evidence",
            "mean_effective_local_mix",
            "mean_effective_global_mix",
            "granularity_local_mix",
            "granularity_global_mix",
            "validation_macro_f1",
            "validation_subject_macro_f1",
            "validation_subject_macro_log_loss",
            "early_stop_raw_score",
            "early_stop_smoothed_score",
            "early_stop_best_score",
            "early_stop_epochs_without_improvement",
        ):
            assert np.isfinite(float(epoch[key])), (key, epoch[key])
        assert isinstance(epoch["checkpoint_improved"], bool)
        assert isinstance(epoch["early_stop_improved"], bool)
        assert isinstance(epoch["early_stop_eligible"], bool)
        assert len(epoch["checkpoint_selection_key"]) == 2
        assert len(epoch["mean_granularity_weights"]) == 3
        assert len(epoch["granularity_top1_fraction"]) == 3

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["schema_version"] == 5
    assert checkpoint["architecture"] == PATCH_MINDTS_ARCHITECTURE
    assert checkpoint["outer_patch_size"] == 64
    assert checkpoint["outer_patch_stride"] == 64
    assert checkpoint["granularity_bank"] == [
        [4],
        [8],
        [16],
    ]
    assert checkpoint["model_configuration"]["num_granularities"] == 3
    assert checkpoint["model_configuration"]["fusion_heads"] == 2
    assert checkpoint["model_configuration"]["granularity_weight_layout"] == (
        "one_shared_distribution_per_patch_BNK"
    )
    assert checkpoint["model_configuration"]["granularity_score"] == (
        "bounded_rbf_dataset_sample_patch_hierarchical_v4"
    )
    assert checkpoint["model_configuration"]["visual_token_composition"] == (
        "graph_value_context_only_no_query_residual"
    )
    assert checkpoint["granularity_balance_weight"] == 0.01
    assert checkpoint["granularity_entropy_weight"] == 0.01
    assert checkpoint["granularity_mix_shrinkage_weight"] == 0.005
    assert checkpoint["granularity_prior_kl_weight"] == 0.001
    assert checkpoint["granularity_regularization_policy"] == (
        "current_marginal_usage_floor_entropy_band_mix_shrinkage_"
        "prior_kl_ema_diagnostics_only_v4"
    )
    assert checkpoint["checkpoint_selection_requested"] == "auto"
    assert checkpoint["checkpoint_selection_effective"] == "subject_macro_f1"
    assert checkpoint["checkpoint_selection_unit"] == "subject"
    assert checkpoint["checkpoint_selection_tie_breaker"] == (
        "subject_macro_log_loss:min,epoch:min"
    )
    assert checkpoint["early_stopping"]["strategy"] == "raw_selection_key"
    assert checkpoint["early_stopping"]["patience"] == 0
    assert checkpoint["early_stopping"]["min_epochs"] == 0
    assert checkpoint["early_stopping"]["early_stopped"] is False
    assert checkpoint["early_stopping"]["stopped_epoch"] is None
    assert checkpoint["early_stopping"][
        "checkpoint_selection_uses_raw_metric"
    ] is True
    assert checkpoint["subject_aggregation"] == "mean_window_softmax_probability"
    assert checkpoint["validation_subject_count"] == 2
    expected_validation_digest = hashlib.sha256(
        "\n".join(str(value) for value in validation_subject_ids).encode("utf-8")
    ).hexdigest()
    assert checkpoint["validation_subject_id_sha256"] == expected_validation_digest
    assert set(checkpoint["selected_validation_window_metrics"]) == {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "macro_auroc",
        "macro_auprc",
    }
    assert set(checkpoint["selected_validation_subject_metrics"]) == {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "macro_auroc",
        "macro_auprc",
        "macro_log_loss",
    }
    assert checkpoint["validation_subject_macro_f1"] == (
        checkpoint["selected_validation_subject_metrics"]["macro_f1"]
    )
    assert checkpoint["model_constructor_configuration"][
        "granularity_router_mode"
    ] == "adaptive_v4"
    assert checkpoint["model_constructor_configuration"][
        "granularity_temperature"
    ] == 1.0
    assert "granularity_attention.line_query_center" in checkpoint["model_state_dict"]
    assert "granularity_attention.dataset_prior_logits" in checkpoint["model_state_dict"]
    assert "granularity_attention.local_mix_logit" in checkpoint["model_state_dict"]
    assert "granularity_attention.global_mix_logit" in checkpoint["model_state_dict"]
    assert "granularity_usage_ema" in checkpoint["model_state_dict"]
    assert "granularity_usage_ema_initialized" in checkpoint["model_state_dict"]
    restored = PatchMindTSFusionModule(
        **checkpoint["model_constructor_configuration"]
    )
    restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
    torch.testing.assert_close(
        restored.granularity_usage_ema,
        checkpoint["model_state_dict"]["granularity_usage_ema"],
    )
    assert bool(restored.granularity_usage_ema_initialized) == bool(
        checkpoint["model_state_dict"]["granularity_usage_ema_initialized"]
    )
    assert len(checkpoint["training_history"]) == 2
    assert not list(artifact_dir.glob(".*.tmp*"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_cold_hot_artifacts_equal(cold_artifacts, hot_artifacts):
    for filename in (
        "patch_atgs_diagnostics_validation.npz",
        "patch_atgs_diagnostics_test.npz",
    ):
        with (
            np.load(cold_artifacts / filename, allow_pickle=False) as cold,
            np.load(hot_artifacts / filename, allow_pickle=False) as hot,
        ):
            assert set(cold.files) == set(hot.files)
            for name in cold.files:
                np.testing.assert_array_equal(hot[name], cold[name])
    for filename in (
        "patch_atgs_summary.json",
        "patch_mindts_training_history.json",
    ):
        assert json.loads((hot_artifacts / filename).read_text(encoding="utf-8")) == (
            json.loads((cold_artifacts / filename).read_text(encoding="utf-8"))
        )
    cold_checkpoint = torch.load(
        cold_artifacts / "patch_mindts_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    hot_checkpoint = torch.load(
        hot_artifacts / "patch_mindts_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert cold_checkpoint.keys() == hot_checkpoint.keys()
    assert (
        cold_checkpoint["model_state_dict"].keys()
        == hot_checkpoint["model_state_dict"].keys()
    )
    for name, cold_tensor in cold_checkpoint["model_state_dict"].items():
        torch.testing.assert_close(
            hot_checkpoint["model_state_dict"][name],
            cold_tensor,
            rtol=0.0,
            atol=0.0,
        )


def assert_uniform_artifacts(artifact_dir):
    for split_name in ("validation", "test"):
        with np.load(
            artifact_dir / f"patch_atgs_diagnostics_{split_name}.npz",
            allow_pickle=False,
        ) as diagnostics:
            valid = diagnostics["patch_mask"].astype(bool)
            expected = np.full_like(diagnostics["weights"][valid], 1.0 / 3.0)
            np.testing.assert_allclose(
                diagnostics["weights"][valid], expected, rtol=0.0, atol=1e-7
            )
            for name in (
                "dataset_prior_weights",
                "sample_global_weights",
                "patch_local_weights",
            ):
                np.testing.assert_allclose(
                    diagnostics[name][valid], expected, rtol=0.0, atol=1e-7
                )
            np.testing.assert_allclose(
                diagnostics["effective_global_mix"], 0.0, rtol=0.0, atol=0.0
            )
            np.testing.assert_allclose(
                diagnostics["effective_local_mix"], 0.0, rtol=0.0, atol=0.0
            )
    checkpoint = torch.load(
        artifact_dir / "patch_mindts_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["model_constructor_configuration"][
        "granularity_router_mode"
    ] == "uniform"
    assert checkpoint["granularity_balance_weight"] == 0.0
    assert checkpoint["granularity_entropy_weight"] == 0.0
    assert checkpoint["granularity_mix_shrinkage_weight"] == 0.0
    assert checkpoint["granularity_prior_kl_weight"] == 0.0
    assert not bool(
        checkpoint["model_state_dict"]["granularity_usage_ema_initialized"]
    )


def main():
    torch.set_num_threads(1)
    windows_per_subject = 2
    train_loader, train_labels, train_subject_ids = make_loader(
        4,
        windows_per_subject,
        seed=31,
        split_name="train",
    )
    validation_loader, validation_labels, validation_subject_ids = make_loader(
        2,
        windows_per_subject,
        seed=37,
        split_name="validation",
    )
    test_loader, test_labels, test_subject_ids = make_loader(
        2,
        windows_per_subject,
        seed=41,
        split_name="test",
    )
    assert not set(train_subject_ids) & set(validation_subject_ids)
    assert not set(train_subject_ids) & set(test_subject_ids)
    assert not set(validation_subject_ids) & set(test_subject_ids)
    for labels, subject_ids in (
        (train_labels, train_subject_ids),
        (validation_labels, validation_subject_ids),
        (test_labels, test_subject_ids),
    ):
        for subject_id in np.unique(subject_ids):
            assert len(np.unique(labels[subject_ids == subject_id])) == 1

    with tempfile.TemporaryDirectory(prefix="patch_mindts_training_") as temp_dir:
        root = Path(temp_dir)
        cache_dir = root / "feature_cache"
        cold_artifacts = root / "cold_artifacts"
        hot_artifacts = root / "hot_artifacts"
        uniform_artifacts = root / "uniform_artifacts"

        cold_vision = DummyVision()
        cold_mantis = DummyMantis()
        cold_result = train_patch_mindts_classifier(
            **training_arguments(
                train_loader,
                train_labels,
                validation_loader,
                validation_labels,
                test_loader,
                test_labels,
                cold_vision,
                cold_mantis,
                cache_dir,
                cold_artifacts,
            )
        )
        cold_validation, cold_test, cold_train_indices, cold_val_indices = cold_result
        assert_finite_metrics(cold_validation)
        assert_finite_metrics(cold_test)
        assert cold_train_indices == list(range(len(train_labels)))
        assert cold_val_indices == list(range(len(validation_labels)))
        assert_artifacts(
            cold_artifacts,
            validation_subject_ids=validation_subject_ids,
            test_subject_ids=test_subject_ids,
            windows_per_subject=windows_per_subject,
        )
        assert not cold_vision.scale.requires_grad
        assert cold_vision.scale.grad is None
        assert not cold_mantis.scale.requires_grad
        assert cold_mantis.scale.grad is None

        cache_paths = {
            split: cache_dir / f"patch_{split}.npz"
            for split in ("train", "vali", "test")
        }
        assert all(path.is_file() for path in cache_paths.values())
        cache_digests = {name: digest(path) for name, path in cache_paths.items()}

        # These encoders deliberately fail if the second run misses the cache.
        hot_result = train_patch_mindts_classifier(
            **training_arguments(
                train_loader,
                train_labels,
                validation_loader,
                validation_labels,
                test_loader,
                test_labels,
                DummyVision(fail_on_encode=True),
                DummyMantis(fail_on_encode=True),
                cache_dir,
                hot_artifacts,
            )
        )
        hot_validation, hot_test, hot_train_indices, hot_val_indices = hot_result
        assert_artifacts(
            hot_artifacts,
            validation_subject_ids=validation_subject_ids,
            test_subject_ids=test_subject_ids,
            windows_per_subject=windows_per_subject,
        )
        assert hot_train_indices == cold_train_indices
        assert hot_val_indices == cold_val_indices
        for cold_metrics, hot_metrics in (
            (cold_validation, hot_validation),
            (cold_test, hot_test),
        ):
            assert set(cold_metrics) == set(hot_metrics)
            for metric in cold_metrics:
                np.testing.assert_allclose(
                    float(hot_metrics[metric]),
                    float(cold_metrics[metric]),
                    rtol=0.0,
                    atol=0.0,
                )
        assert cache_digests == {
            name: digest(path) for name, path in cache_paths.items()
        }
        assert_cold_hot_artifacts_equal(cold_artifacts, hot_artifacts)

        uniform_arguments = training_arguments(
            train_loader,
            train_labels,
            validation_loader,
            validation_labels,
            test_loader,
            test_labels,
            DummyVision(fail_on_encode=True),
            DummyMantis(fail_on_encode=True),
            cache_dir,
            uniform_artifacts,
        )
        uniform_arguments.update(
            {
                "granularity_router_mode": "uniform",
                "granularity_balance_weight": 0.0,
                "granularity_entropy_weight": 0.0,
                "granularity_mix_shrinkage_weight": 0.0,
                "granularity_prior_kl_weight": 0.0,
            }
        )
        uniform_validation, uniform_test, _, _ = train_patch_mindts_classifier(
            **uniform_arguments
        )
        assert_finite_metrics(uniform_validation)
        assert_finite_metrics(uniform_test)
        assert_uniform_artifacts(uniform_artifacts)
        assert cache_digests == {
            name: digest(path) for name, path in cache_paths.items()
        }

    print("PATCH MINDTS TRAINING INTEGRATION VALIDATION PASSED")


if __name__ == "__main__":
    main()
