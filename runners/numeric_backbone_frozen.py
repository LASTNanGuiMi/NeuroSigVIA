"""Train full NeuroSigVIA whose numeric branch is a frozen comparison-method encoder.

The encoder is restored from the comparison-method checkpoint trained on this
dataset, split and seed, frozen, and pooled onto the outer patches by time position
(src/frozen_numeric_tokens.py).  Everything else is the backbone-replacement runner.

Every invocation recomputes temporal patches and line tokens. No cached Mantis,
line-plot, or Activity Graph feature is read or written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_loading.experiment import load_data
from src.adaptive_graph_training import (
    _atomic_json_dump, _atomic_prediction_dump, _balanced_class_weights,
    _build_patch_lr_scheduler, _cpu_state_dict, _evaluate,
    _evaluation_protocol, _run_epoch,
)
from src.datautils import write_eeg_medformer_split_audit, write_wearable_split_audit
from src.frozen_numeric_tokens import (
    ARCHITECTURE, FROZEN_BACKBONES as BACKBONES, FrozenNumericTokenClassifier,
    extract_tokens, restore, verify,
)
from src.patch_fusion import (
    _EarlyStoppingMonitor, _checkpoint_selection_key, _extract_line_tokens,
    make_temporal_patches,
)
from src.utils import set_random_seed


ROOT = Path(__file__).resolve().parents[1]
# 四个数据集共用本 runner；EEG 与 wearable 的受试者划分审计函数不同。
DATASET_KIND = {"apava": "eeg", "tdbrain": "eeg", "shimmer10": "wearable", "pads11": "wearable"}
TRAINING = {
    "batch_size": 8, "epochs": 100, "lr": 3e-4, "weight_decay": 1e-3,
    "early_stop_strategy": "raw_primary", "early_stop_warmup_epochs": 10,
    "early_stop_min_epochs": 0, "early_stop_patience": 12,
    "early_stop_ema_decay": 0.6, "early_stop_min_delta": 0.002,
    "lr_scheduler": "reduce_on_plateau", "lr_scheduler_patience": 4,
    "lr_scheduler_factor": 0.5, "lr_scheduler_min_lr": 1e-6,
    "alignment_weight": 0.1, "selector_balance_weight": 0.001,
    "visual_encode_batch_size": 4,
}


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset", choices=tuple(DATASET_KIND), required=True)
    cli.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    cli.add_argument("--backbone", choices=BACKBONES, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--vision-name", default="../models/CLIP-ViT-H-14-laion2B-s32B-b79K")
    cli.add_argument("--batch-size", type=positive_int, default=TRAINING["batch_size"])
    cli.add_argument("--visual-encode-batch-size", type=positive_int, default=TRAINING["visual_encode_batch_size"])
    cli.add_argument("--smoke", action="store_true")
    cli.add_argument("--d-ff", type=positive_int, default=256)
    cli.add_argument("--e-layers", type=positive_int, default=2)
    cli.add_argument("--n-heads", type=positive_int, default=8)
    cli.add_argument("--backbone-dropout", type=float, default=0.1)
    cli.add_argument("--patch-len-list", default="2,4,8")
    cli.add_argument("--top-k", type=positive_int, default=3)
    cli.add_argument("--num-kernels", type=positive_int, default=6)
    # Fusion/classifier settings of the main method; defaults are the previous fixed values.
    cli.add_argument("--lr", type=float, default=TRAINING["lr"])
    cli.add_argument("--dropout", type=float, default=0.1, help="fusion/classifier dropout (main: --mlp_dropout)")
    cli.add_argument("--alignment-weight", type=float, default=TRAINING["alignment_weight"])
    cli.add_argument("--frozen-run", required=True,
                     help="comparison-method run directory with {seed}, e.g. .../PatchTST/seed{seed}")
    cli.add_argument("--checkpoint-metric", choices=("window_macro_f1", "subject_macro_f1"),
                     default="window_macro_f1", help="main: --patch_checkpoint_metric")
    return cli


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(backbone):
    vendor = ROOT / "third_party" / ("timesnet" if backbone == "TimesNet" else "medformer")
    files = [
        Path(__file__), ROOT / "src/numeric_backbone_replacement.py",
        ROOT / "src/adaptive_graph_training.py", ROOT / "src/multimodal_fusion.py",
        ROOT / "data_loading/experiment.py", ROOT / "src/datautils.py",
    ] + sorted(vendor.rglob("*.py"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in files if p.is_file()}


def build_vision_model(name, device):
    from src.neurosigvia import get_neurosigvia
    model = get_neurosigvia(
        model_name=name, model_layer=14, aggregation="mean", stride=None,
        patch_size=None, image_mode="multiscale_activity_graph",
        activity_graph_patch_lengths=[1], activity_graph_channel_mix=0.35,
        activity_graph_router_temperature=0.2, activity_graph_router_mix=0.5,
        activity_graph_adaptive_granularity=False,
        activity_graph_granularity_bank=[[4], [8], [16]],
    ).to(device)
    model.requires_grad_(False)
    model.eval()
    return model


def balanced_smoke_indices(labels, per_class=2):
    labels = np.asarray(labels)
    return np.concatenate([
        np.flatnonzero(labels == value)[:per_class] for value in np.unique(labels)
    ])


@torch.no_grad()
def fresh_static_split(source, labels, vision_model, device, *, encode_batch_size, smoke=False):
    """Recompute raw patches and line tokens directly from source windows."""
    labels = np.asarray(labels).reshape(-1)
    subject_ids = np.asarray(source.sample_subject_ids)
    if len(labels) != len(source) or subject_ids.shape != labels.shape:
        raise ValueError("source rows, labels and subject IDs are not aligned")
    rows = balanced_smoke_indices(labels) if smoke else np.arange(len(labels))
    raw_source = source.tensors[0][rows]
    loader = DataLoader(TensorDataset(raw_source), batch_size=32, shuffle=False, num_workers=0)
    keys = ("raw_windows", "line_tokens", "patch_mask", "valid_fraction", "valid_lengths")
    collected = {key: [] for key in keys}
    for (windows,) in loader:
        temporal = make_temporal_patches(windows, window_size=64, stride=64, lengths=None)
        batch, patches, channels, length = temporal.patches.shape
        flat = temporal.patches.reshape(batch * patches, channels, length).float()
        flat_lengths = temporal.valid_lengths.reshape(-1)
        indices = temporal.patch_mask.reshape(-1).nonzero(as_tuple=False).flatten()
        valid_line = _extract_line_tokens(
            flat.index_select(0, indices), flat_lengths.index_select(0, indices),
            vision_model, device, encode_batch_size,
        ).float()
        padded = torch.zeros(batch * patches, valid_line.shape[-1], dtype=torch.float32)
        padded.index_copy_(0, indices, valid_line)
        collected["raw_windows"].append(temporal.patches.detach().cpu().float())
        collected["line_tokens"].append(padded.reshape(batch, patches, -1).half())
        collected["patch_mask"].append(temporal.patch_mask.detach().cpu())
        collected["valid_fraction"].append(temporal.valid_fraction.detach().cpu().half())
        collected["valid_lengths"].append(temporal.valid_lengths.detach().cpu())
    result = {key: torch.cat(values, dim=0) for key, values in collected.items()}
    result.update(labels=labels[rows].astype(np.int64), subject_ids=subject_ids[rows], source_rows=rows.tolist(),
                  full_windows=raw_source.float())
    return result


def build_training_loader(split, batch_size, shuffle):
    count = len(split["labels"])
    dataset = TensorDataset(
        split["raw_windows"], split["line_tokens"], split["numeric_tokens"],
        split["patch_mask"], split["valid_fraction"], split["valid_lengths"],
        torch.as_tensor(split["labels"], dtype=torch.long), torch.arange(count),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def run(args):
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0 <= args.backbone_dropout < 1:
        raise ValueError("backbone dropout must be in [0,1)")
    device = torch.device("cuda:0")
    bundle, data_manifest = load_data(args.dataset, smoke=False)
    audit = (write_eeg_medformer_split_audit if DATASET_KIND[args.dataset] == "eeg"
             else write_wearable_split_audit)
    split_audit = audit(bundle, args.output)
    vision_model = build_vision_model(args.vision_name, device)
    splits = {}
    for split in ("train", "vali", "test"):
        source = getattr(bundle, f"{split}_loader").dataset
        labels = getattr(bundle, f"{split}_labels")
        splits[split] = fresh_static_split(
            source, labels, vision_model, device,
            encode_batch_size=args.visual_encode_batch_size,
            smoke=args.smoke and split != "test",
        )
    classes = np.unique(splits["train"]["labels"])
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("classes must be contiguous and zero-based")
    channels = int(splits["train"]["raw_windows"].shape[2])
    visual_dim = int(splits["train"]["line_tokens"].shape[-1])
    # Frozen comparison encoder: same data, split and seed as this run, checked before use.
    # Runs before the training seed is set, so the model initialisation is unaffected.
    frozen_dir = Path(args.frozen_run.format(seed=args.seed))
    frozen, frozen_config, frozen_info = restore(args.backbone, frozen_dir)
    frozen_protocol = frozen_info.pop("protocol")
    if frozen_protocol["data_manifest"] != json.loads(json.dumps(data_manifest)):
        raise ValueError(f"{frozen_dir} was trained on a different data split or content")
    frozen_seed = frozen_protocol.get("random_seed", frozen_protocol.get("training_seed"))
    if frozen_seed != args.seed or frozen_protocol.get("dataset") != args.dataset:
        raise ValueError(f"{frozen_dir} is not the {args.dataset} seed {args.seed} run")
    vali_source = bundle.vali_loader.dataset
    frozen_checks = verify(args.backbone, frozen, frozen_config, frozen_dir, vali_source.tensors[0].float(),
                           bundle.vali_labels, frozen_info["eval_batch_size"], device)
    for split in splits.values():
        split["numeric_tokens"] = extract_tokens(
            args.backbone, frozen, frozen_config, split.pop("full_windows"), frozen_info["eval_batch_size"],
            device, split["raw_windows"].shape[1])
    del frozen
    torch.cuda.empty_cache()
    token_dim = int(splits["train"]["numeric_tokens"].shape[-1])
    provenance = {"name": args.backbone, "frozen": True, "token_dim": token_dim,
                  "representation": "input of the removed classification head, pooled onto 64-point patches by time",
                  "trainable_part": "LayerNorm + Linear(token_dim -> fusion_dim)",
                  **frozen_info, **frozen_checks}
    set_random_seed(args.seed)
    loaders = {
        name: build_training_loader(split, args.batch_size, name == "train")
        for name, split in splits.items()
    }
    model = FrozenNumericTokenClassifier(
        backbone_name=args.backbone, token_dim=token_dim, provenance=provenance,
        visual_dim=visual_dim, temporal_dim=128, num_channels=channels,
        num_classes=len(classes), fusion_dim=128, fusion_heads=2, dropout=args.dropout,
        classifier_hidden_dim=128, classifier_num_layers=2, channel_hidden_dim=64,
        alignment_dim=256, alignment_temperature=0.1, graph_image_size=224,
        graph_token_grid=4, activity_graph_canvas_size=360,
        activity_graph_line_width=1.0, activity_graph_vertical_margin=0.05,
        adaptive_temperature=0.5, freeze_adaptive_gate=False,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr, weight_decay=TRAINING["weight_decay"],
    )
    scheduler = _build_patch_lr_scheduler(
        optimizer, TRAINING["lr_scheduler"], patience=TRAINING["lr_scheduler_patience"],
        factor=TRAINING["lr_scheduler_factor"], min_lr=TRAINING["lr_scheduler_min_lr"],
    )
    weights = _balanced_class_weights(splits["train"]["labels"], len(classes), device)
    criterion = nn.CrossEntropyLoss(weight=weights, reduction="none")
    stopping = _EarlyStoppingMonitor(
        strategy=TRAINING["early_stop_strategy"], patience=TRAINING["early_stop_patience"],
        min_epochs=TRAINING["early_stop_min_epochs"],
        warmup_epochs=TRAINING["early_stop_warmup_epochs"],
        ema_decay=TRAINING["early_stop_ema_decay"], min_delta=TRAINING["early_stop_min_delta"],
    )
    protocol = {
        "architecture": ARCHITECTURE, "dataset": args.dataset, "seed": args.seed,
        "split_seed": 20260917 if args.dataset == "apava" else 42,
        "intervention": f"frozen {args.backbone} (comparison-method checkpoint) replaces frozen Mantis numeric branch",
        "feature_reuse": False, "feature_cache_read": False, "feature_cache_write": False,
        "fresh_features": "raw patches and line tokens recomputed in every invocation",
        "numeric_interface": "whole window -> frozen encoder representation pooled per 64-point patch -> trainable LayerNorm+Linear [B,N,128]",
        "uses_upstream_classification_logits": False, "uses_mantis_model_or_features": False,
        "visual_branch": "original frozen OpenCLIP plus trainable adaptive Activity Graph renderer",
        "alignment": "original within-window patch InfoNCE", "fusion": "original concat_attn",
        "checkpoint_metric_effective": args.checkpoint_metric,
        "test_policy": "test untouched in smoke; formal test exactly once after validation selection",
        "normalization": "fixed EEG loader per-window per-channel StandardScaler ddof=0",
        "data_manifest": data_manifest, "subject_split_audit": str(split_audit.relative_to(args.output)),
        "subject_split_sha256": sha256(split_audit),
        "source_rows": {name: split["source_rows"] for name, split in splits.items()},
        "training": {**TRAINING, "batch_size": args.batch_size, "lr": args.lr, "dropout": args.dropout,
                     "alignment_weight": args.alignment_weight},
        "backbone": model.numeric_backbone.provenance(), "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
            Path(__file__), ROOT / "src/frozen_numeric_tokens.py", ROOT / "src/adaptive_graph_training.py",
            ROOT / "src/multimodal_fusion.py", ROOT / "src/datautils.py")},
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "backbone_trainable_parameters": sum(p.numel() for p in model.numeric_backbone.parameters() if p.requires_grad),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0), "smoke": args.smoke,
    }
    _atomic_json_dump(args.output / "protocol.json", protocol)
    torch.cuda.reset_peak_memory_stats()
    common = {
        "alignment_weight": args.alignment_weight,
        "selector_balance_weight": TRAINING["selector_balance_weight"],
        "visual_encode_batch_size": args.visual_encode_batch_size,
        "graph_spatial_grid_size": 4,
    }
    if args.smoke:
        statistics = _run_epoch(
            model, loaders["train"], vision_model, optimizer, criterion, device,
            vision_gradient_checkpointing=True, **common,
        )
        good_gradient = any(
            p.grad is not None and torch.isfinite(p.grad).all() and bool((p.grad.abs() > 0).any())
            for p in model.numeric_backbone.parameters()
        )
        if not good_gradient:
            raise RuntimeError("numeric backbone received no finite non-zero gradient")
        result = {
            "status": "COMPLETED", "scientific_result": False,
            "dataset": args.dataset, "seed": args.seed, "backbone": args.backbone,
            "numeric_gradient_verified": True, "statistics": statistics,
            "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
            "elapsed_seconds": time.monotonic() - started, "test_evaluation_count": 0,
        }
        _atomic_json_dump(args.output / "smoke.json", result)
        _atomic_json_dump(args.output / "status.json", result)
        print("NUMERIC_BACKBONE_CUDA_SMOKE_PASS " + json.dumps(result, allow_nan=False), flush=True)
        return

    history, best_key, best_epoch, best_state = [], None, None, None
    for epoch in range(1, TRAINING["epochs"] + 1):
        epoch_started = time.monotonic()
        current_lr = float(optimizer.param_groups[0]["lr"])
        statistics = _run_epoch(
            model, loaders["train"], vision_model, optimizer, criterion, device,
            vision_gradient_checkpointing=True, **common,
        )
        validation, _ = _evaluate(
            model, loaders["vali"], classes, vision_model, device, f"Validate {args.backbone}",
            sample_subject_ids=splits["vali"]["subject_ids"],
            visual_encode_batch_size=args.visual_encode_batch_size, graph_spatial_grid_size=4,
        )
        key = _checkpoint_selection_key(validation, args.checkpoint_metric)
        improved = best_key is None or key > best_key
        stop_state = stopping.update(key, epoch)
        if scheduler is not None and epoch > TRAINING["early_stop_warmup_epochs"]:
            scheduler.step(float(key[0]))
        record = {
            "epoch": epoch, "train": statistics, "validation": validation,
            "selection_key": list(key), "checkpoint_improved": improved,
            "learning_rate": current_lr,
            "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "early_stopping": stop_state, "epoch_seconds": time.monotonic() - epoch_started,
        }
        history.append(record)
        if improved:
            best_key, best_epoch, best_state = key, epoch, _cpu_state_dict(model)
        _atomic_json_dump(args.output / "progress.json", {
            "status": "RUNNING", "best_epoch": best_epoch, "history": history,
        })
        print(
            f"{args.backbone} {args.dataset} seed={args.seed} epoch={epoch}/100 "
            f"loss={statistics['loss']:.6f} val_macro_f1={validation['macro_f1']:.6f}", flush=True,
        )
        if stop_state["should_stop"]:
            break
    if best_state is None:
        raise RuntimeError("no validation-selected checkpoint")
    model.load_state_dict(best_state)
    validation, validation_details = _evaluate(
        model, loaders["vali"], classes, vision_model, device, "Best validation",
        sample_subject_ids=splits["vali"]["subject_ids"],
        visual_encode_batch_size=args.visual_encode_batch_size, graph_spatial_grid_size=4,
    )
    test, test_details = _evaluate(
        model, loaders["test"], classes, vision_model, device, "Final test",
        sample_subject_ids=splits["test"]["subject_ids"],
        visual_encode_batch_size=args.visual_encode_batch_size, graph_spatial_grid_size=4,
    )
    checkpoint = {
        "architecture": ARCHITECTURE, "model_state_dict": best_state,
        "model_configuration": model.configuration, "classes": classes.tolist(),
        "best_epoch": best_epoch, "selection_key": list(best_key),
        "checkpoint_metric_effective": args.checkpoint_metric, "protocol": "protocol.json",
    }
    temporary = args.output / ".best_checkpoint.tmp.pt"
    torch.save(checkpoint, temporary)
    temporary.replace(args.output / "best_checkpoint.pt")
    _atomic_prediction_dump(args.output / "validation_predictions.npz", validation_details)
    _atomic_prediction_dump(args.output / "test_predictions.npz", test_details)
    summary = {
        "status": "COMPLETED", "architecture": ARCHITECTURE,
        "dataset": args.dataset, "seed": args.seed, "backbone": args.backbone,
        "best_epoch": best_epoch, "epochs_run": len(history),
        **_evaluation_protocol(args.checkpoint_metric),
        "validation_metrics": validation, "test_metrics": test,
        "test_evaluation_count": 1, "checkpoint": "best_checkpoint.pt",
        "validation_predictions": "validation_predictions.npz",
        "test_predictions": "test_predictions.npz", "history": history,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json_dump(args.output / "summary.json", summary)
    _atomic_json_dump(args.output / "status.json", summary)
    print(json.dumps(summary, allow_nan=False), flush=True)


def main():
    args = parser().parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    _atomic_json_dump(args.output / "args.json", {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    })
    _atomic_json_dump(args.output / "status.json", {"status": "RUNNING", "pid": os.getpid()})
    try:
        run(args)
    except BaseException as error:
        _atomic_json_dump(args.output / "status.json", {
            "status": "FAILED", "error_type": type(error).__name__, "error": str(error),
        })
        (args.output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
