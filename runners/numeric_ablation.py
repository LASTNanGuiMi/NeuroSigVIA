"""Retrain the numeric-only ablation against an immutable paired main run."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.classifier import compute_metrics_from_predictions
from src.numeric_ablation import ARCHITECTURE, NumericOnlyClassifier
from src.numeric_ablation_reference import load_reference
from src.patch_mindts import (
    _EarlyStoppingMonitor, _aggregate_subject_predictions, _balanced_class_weights,
    _build_patch_lr_scheduler, _checkpoint_selection_key, _merge_subject_metrics,
)
from src.utils import set_random_seed


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def loader_for(split, batch_size, shuffle, limit=None):
    n = len(split["labels"])
    count = n if limit is None else min(n, limit)
    dataset = TensorDataset(
        torch.as_tensor(split["mantis_channel_tokens"][:count]),
        torch.as_tensor(split["patch_mask"][:count]),
        torch.as_tensor(split["valid_fraction"][:count]),
        torch.as_tensor(split["labels"][:count], dtype=torch.long),
        torch.arange(count, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def forward_batch(model, batch, device):
    mantis, mask, fractions, labels, indices = batch
    logits = model(mantis.to(device, dtype=torch.float32), mask.to(device), fractions.to(device))
    return logits, labels.to(device), indices


@torch.no_grad()
def evaluate(model, loader, classes, subjects, device):
    model.eval()
    true, scores, indices = [], [], []
    for batch in loader:
        logits, labels, sample_indices = forward_batch(model, batch, device)
        probabilities = logits.float().softmax(dim=-1)
        if not torch.isfinite(probabilities).all():
            raise ValueError("non-finite probabilities")
        true.append(labels.cpu().numpy())
        scores.append(probabilities.cpu().numpy())
        indices.append(sample_indices.numpy())
    details = dict(y_true=np.concatenate(true), y_score=np.concatenate(scores), sample_index=np.concatenate(indices))
    details["y_pred"] = details["y_score"].argmax(axis=-1)
    window = compute_metrics_from_predictions(details["y_true"], details["y_pred"], details["y_score"], np.arange(len(classes)))
    subject, subject_details = _aggregate_subject_predictions(details["y_true"], details["y_score"], details["sample_index"], subjects, classes)
    details.update(subject_details)
    return _merge_subject_metrics(window, subject), details


def state_digest(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def positive_int(value):
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("batch size must be a positive integer") from None
    if result <= 0:
        raise argparse.ArgumentTypeError("batch size must be a positive integer")
    return result


def training_configuration(reference_args, checkpoint_metric, batch_size=None):
    """Override this run's loader size without mutating reference/cache inputs."""
    cfg = dict(reference_args)
    reference_batch_size = positive_int(cfg["batch_size"])
    effective_batch_size = reference_batch_size if batch_size is None else positive_int(batch_size)
    cfg.update(batch_size=effective_batch_size, patch_checkpoint_metric=checkpoint_metric)
    return cfg, dict(
        batch_size_requested=batch_size, batch_size_effective=effective_batch_size,
        reference_batch_size=reference_batch_size,
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", required=True, type=Path)
    parser.add_argument("--dataset", choices=["shimmer10", "pads11", "apava", "tdbrain"], required=True)
    parser.add_argument("--seed", type=int, choices=[42, 43, 44], required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--batch-size", type=positive_int, default=None,
        help="training/evaluation loader batch size; defaults to the paired reference run",
    )
    parser.add_argument(
        "--checkpoint-metric", choices=["window_macro_f1", "subject_macro_f1"],
        default="window_macro_f1",
        help="validation metric for checkpoint selection, early stopping and LR scheduling",
    )
    parser.add_argument("--smoke", action="store_true", help="two train batches, no validation/test evaluation")
    return parser


def checkpoint_selection_protocol(metric):
    if metric not in {"window_macro_f1", "subject_macro_f1"}:
        raise ValueError(f"unsupported checkpoint metric: {metric}")
    level = "window" if metric == "window_macro_f1" else "subject"
    return dict(
        checkpoint_metric_requested=metric, checkpoint_metric_effective=metric,
        metric_level=level, selection_unit=level,
        evaluation_unit="window", supplementary_evaluation_unit="subject",
        selection=(
            "Validation window macro-F1; exact ties retain the earliest checkpoint."
            if level == "window" else
            "Validation subject macro-F1 then negative subject macro-log-loss; exact ties retain the earliest checkpoint."
        ) + " Test evaluated only after checkpoint selection.",
        checkpoint_tie_breaker="earliest_epoch" if level == "window" else "negative_subject_macro_log_loss_then_earliest_epoch",
    )


def validation_selection(metrics, metric, best_key):
    """Use one validation-only key for saving, stopping and scheduling."""
    key = _checkpoint_selection_key(metrics, metric)
    return key, best_key is None or key > best_key


def main():
    args = build_parser().parse_args()
    selection = checkpoint_selection_protocol(args.checkpoint_metric)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    reference = load_reference(args.reference_run, args.seed, args.dataset)
    reference_cfg = reference["args"]
    cfg, batch_protocol = training_configuration(reference_cfg, args.checkpoint_metric, args.batch_size)
    splits, classes = reference["splits"], np.asarray(reference["classes"])
    if not torch.cuda.is_available():
        raise RuntimeError("this invocation requires an assigned CUDA GPU")
    device = torch.device("cuda:0")
    batch_size = int(cfg["batch_size"])
    # As in the main method, reset RNG after extraction/loading; construct
    # loaders before model, and use the original global-RNG shuffle behavior.
    set_random_seed(args.seed)
    loaders = {key: loader_for(value, batch_size, key == "train", 2 * batch_size if args.smoke else None) for key, value in splits.items()}
    model = NumericOnlyClassifier.from_config(reference["model_config"]).to(device)
    initial_sha256 = state_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["mlp_lr"], weight_decay=cfg["mlp_weight_decay"])
    scheduler = _build_patch_lr_scheduler(optimizer, cfg["mlp_lr_scheduler"], patience=cfg["mlp_lr_scheduler_patience"], factor=cfg["mlp_lr_scheduler_factor"], min_lr=cfg["mlp_lr_scheduler_min_lr"])
    weights = _balanced_class_weights(np.asarray(splits["train"]["labels"]), len(classes), device) if cfg["mlp_class_weight"] == "balanced" else None
    criterion = nn.CrossEntropyLoss(weight=weights, reduction="none")
    stopping = _EarlyStoppingMonitor(
        strategy=cfg["mlp_early_stop_strategy"], patience=cfg["mlp_early_stop_patience"],
        min_epochs=cfg["mlp_early_stop_min_epochs"], warmup_epochs=cfg["mlp_early_stop_warmup_epochs"],
        ema_decay=cfg["mlp_early_stop_ema_decay"], min_delta=cfg["mlp_early_stop_min_delta"],
    )
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path("src/numeric_ablation.py"), Path("src/numeric_ablation_reference.py"), Path("src/adaptive_graph_training.py"), Path("src/multimodal_fusion.py"), Path("src/mlp_classifier.py"), Path("src/patch_mindts.py")]}
    protocol = dict(
        architecture=ARCHITECTURE, dataset=args.dataset, training_seed=args.seed, split_seed=42,
        reference=reference["provenance"], model_config=reference["model_config"],
        training_args=cfg,
        reference_checkpoint_metric=reference_cfg.get("patch_checkpoint_metric"),
        initial_state_sha256=initial_sha256,
        initial_channel_pool_sha256=state_digest(model.channel_pool),
        classifier_input_dim=model.classifier_input_dim,
        uses_concat_attn=False, zero_visual_slot=False, source_sha256=source_hashes,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), gpu=torch.cuda.get_device_name(0),
        intervention="No images, zero visual slot or concat_attn. Original channel pool -> valid-duration pooling -> MLP with input width changed from 2*fusion_dim to fusion_dim.",
        removed_modules=["renderer", "visual_cross_attention", "alignment", "external vision encoder", "temporal_visual_fusion"],
        loss="Original balanced CrossEntropyLoss(reduction='none').mean(); no vision InfoNCE or gate balance loss.",
        initialization="Fresh complete main constructor with paired seed; retain channel pool and compatible classifier layers, freshly initialize the narrower classifier input Linear. No trained main weights used.",
        randomness_note="Channel-pool and compatible classifier initialization match the reference; the narrower first Linear and subsequent stochastic trajectories do not match the original full model.",
        **selection,
        **batch_protocol,
        std_ddof=1, smoke=args.smoke,
    )
    write_json(args.output / "protocol.json", protocol)
    best_key, best_epoch, best_state = None, None, None
    history = []
    maximum = 1 if args.smoke else int(cfg["mlp_epochs"])
    progress = tqdm(range(1, maximum + 1), desc=f"{args.dataset} seed{args.seed} epochs", dynamic_ncols=True)
    for epoch in progress:
        epoch_start = time.monotonic()
        model.train()
        losses, count = 0.0, 0
        lr = optimizer.param_groups[0]["lr"]
        for batch in tqdm(loaders["train"], desc=f"train {epoch}", leave=False, dynamic_ncols=True):
            optimizer.zero_grad(set_to_none=True)
            logits, labels, _ = forward_batch(model, batch, device)
            loss = criterion(logits.float(), labels).mean()
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            if not any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.channel_pool.parameters()):
                raise RuntimeError("numeric branch received no gradient")
            optimizer.step()
            losses += float(loss.detach()) * len(labels)
            count += len(labels)
        if args.smoke:
            write_json(args.output / "smoke.json", dict(passed=True, batches=len(loaders["train"]), loss=losses/count, architecture=ARCHITECTURE, classifier_input_dim=model.classifier_input_dim, uses_concat_attn=False, elapsed_seconds=time.monotonic()-started, test_evaluations=0, **batch_protocol))
            print("SMOKE PASSED: finite full-shape training/backward; test untouched", flush=True)
            return
        validation, _ = evaluate(model, loaders["vali"], classes, splits["vali"]["subject_ids"], device)
        key, improved = validation_selection(validation, args.checkpoint_metric, best_key)
        stop_state = stopping.update(key, epoch)
        if scheduler is not None and epoch > cfg["mlp_early_stop_warmup_epochs"]:
            scheduler.step(float(key[0]))
        record = dict(epoch=epoch, train_loss=losses/count, validation=validation, selection_key=list(key), checkpoint_metric_effective=args.checkpoint_metric, metric_level=selection["metric_level"], checkpoint_improved=improved, early_stopping=stop_state, learning_rate=lr, next_learning_rate=optimizer.param_groups[0]["lr"], epoch_seconds=time.monotonic()-epoch_start)
        history.append(record)
        write_json(args.output / "progress.json", dict(state="RUNNING", epoch=epoch, maximum_epochs=maximum, best_epoch=epoch if improved else best_epoch, latest=record, elapsed_seconds=time.monotonic()-started, **selection))
        if improved:
            best_key, best_epoch = key, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(dict(architecture=ARCHITECTURE, model_state_dict=best_state, model_config=reference["model_config"], classes=classes.tolist(), best_epoch=best_epoch, validation_metrics=validation, protocol=protocol, **selection), args.output / "best_checkpoint.pt")
        write_json(args.output / "training_history.json", history)
        progress.set_postfix({f"val_{selection['metric_level']}_f1": f"{key[0]:.4f}", "best": best_epoch, "stale": stop_state["epochs_without_improvement"]})
        print(f"EPOCH {epoch}/{maximum} val_{selection['metric_level']}_f1={key[0]:.6f} loss={losses/count:.6f} best={best_epoch} seconds={record['epoch_seconds']:.2f}", flush=True)
        if stop_state["should_stop"]:
            break
    if best_state is None:
        raise RuntimeError("no valid checkpoint")
    model.load_state_dict(best_state, strict=True)
    validation, val_details = evaluate(model, loaders["vali"], classes, splits["vali"]["subject_ids"], device)
    test, test_details = evaluate(model, loaders["test"], classes, splits["test"]["subject_ids"], device)
    for name, details in [("validation", val_details), ("test", test_details)]:
        np.savez_compressed(args.output / f"{name}_predictions.npz", **details)
    summary = dict(architecture=ARCHITECTURE, state="COMPLETED", dataset=args.dataset, seed=args.seed, best_epoch=best_epoch, epochs=len(history), validation_metrics=validation, test_metrics=test, training_history=history, elapsed_seconds=time.monotonic()-started, protocol="protocol.json", checkpoint="best_checkpoint.pt", reference_summary=reference["provenance"]["reference_summary"], test_evaluations=1, **selection, **batch_protocol)
    write_json(args.output / "numeric_ablation_summary.json", summary)
    write_json(args.output / "progress.json", dict(state="COMPLETED", epochs=len(history), best_epoch=best_epoch, elapsed_seconds=summary["elapsed_seconds"], **selection))
    print("COMPLETED " + json.dumps(test), flush=True)


if __name__ == "__main__":
    main()
