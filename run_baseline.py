"""Retrain unmodified Medformer models on the fixed NeuroSigViT protocol.

Each invocation trains one model/dataset/initialization seed. The data split is
fixed independently of that seed. Smoke runs retain the real sequence length,
use a small train/validation subset, and never evaluate the test set.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
MODELS = ("Medformer", "Crossformer", "FEDformer", "Autoformer", "PatchTST", "Transformer")
DATASETS = ("adftd", "tdbrain", "apava", "shimmer10", "pads11")
DEFAULT_BATCH = dict(adftd=8, tdbrain=8, apava=8, shimmer10=1, pads11=4)
CLASS_NAMES = {"adftd": ["HC", "FTD", "AD"], "tdbrain": ["HC", "PD"],
               "apava": ["HC", "AD"], "shimmer10": ["HC", "PD"], "pads11": ["HC", "PD"]}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics(y_true, probabilities):
    """Match NeuroSigViT's macro one-vs-rest AUROC/AP convention."""
    labels = np.arange(probabilities.shape[1])
    y_pred = probabilities.argmax(axis=1)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }
    for name, function in (("macro_auroc", roc_auc_score), ("macro_auprc", average_precision_score)):
        values = []
        for label in labels:
            binary = (y_true == label).astype(int)
            if binary.min() != binary.max():
                values.append(float(function(binary, probabilities[:, label])))
        result[name] = float(np.mean(values)) if values else None
    nll = -np.log(np.clip(probabilities[np.arange(len(y_true)), y_true], 1e-12, 1.0))
    result["macro_log_loss"] = float(np.mean([nll[y_true == label].mean() for label in np.unique(y_true)]))
    return result


def aggregate_subjects(y_true, probabilities, sample_subject_ids):
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    sample_subject_ids = np.asarray(sample_subject_ids)
    if y_true.ndim != 1 or sample_subject_ids.shape != y_true.shape:
        raise ValueError("Labels and sample subject IDs must be aligned vectors")
    if probabilities.ndim != 2 or len(probabilities) != len(y_true):
        raise ValueError("Probabilities must have shape [sample, class]")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Invalid probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1, rtol=1e-5, atol=1e-6):
        raise ValueError("Probability rows must sum to one")
    subject_ids, inverse = np.unique(sample_subject_ids, return_inverse=True)
    subject_labels, subject_scores, counts = [], [], []
    for index in range(len(subject_ids)):
        members = inverse == index
        labels = np.unique(y_true[members])
        if len(labels) != 1:
            raise ValueError(f"Subject {subject_ids[index]!r} has inconsistent labels")
        subject_labels.append(labels[0])
        subject_scores.append(probabilities[members].mean(axis=0))
        counts.append(int(members.sum()))
    return {
        "subject_id": subject_ids,
        "subject_y_true": np.asarray(subject_labels, dtype=np.int64),
        "subject_y_score": np.asarray(subject_scores, dtype=np.float64),
        "subject_window_count": np.asarray(counts, dtype=np.int64),
    }


def subset_indices(labels, samples_per_class):
    """A deterministic smoke subset; no sequence/channel slicing."""
    return np.concatenate([np.flatnonzero(labels == label)[:samples_per_class] for label in np.unique(labels)])


def build_loaders(bundle, batch_size, seed, smoke=False, smoke_samples_per_class=2):
    loaders, subject_ids, selected_rows = {}, {}, {}
    for split in ("train", "vali", "test"):
        source = getattr(bundle, split + "_loader").dataset
        x = source.tensors[0]
        y = np.asarray(getattr(bundle, split + "_labels"), dtype=np.int64).reshape(-1)
        ids = np.asarray(source.sample_subject_ids)
        rows = subset_indices(y, smoke_samples_per_class) if smoke and split != "test" else np.arange(len(y))
        selected_rows[split] = rows.tolist()
        subject_ids[split] = ids[rows]
        dataset = TensorDataset(x[rows], torch.as_tensor(y[rows], dtype=torch.long))
        if dataset.tensors[0].shape[1:] != x.shape[1:]:
            raise AssertionError("Smoke/data loader changed the sequence length or channel count")
        generator = torch.Generator().manual_seed(seed)
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"),
                                    generator=generator, num_workers=0, drop_last=False)
    return loaders, subject_ids, selected_rows


def model_config(args, sequence_length, channels, num_classes):
    return SimpleNamespace(
        task_name="classification", seq_len=sequence_length, pred_len=0,
        enc_in=channels, dec_in=channels, c_out=channels, num_class=num_classes,
        d_model=args.d_model, d_ff=args.d_ff, e_layers=args.e_layers,
        d_layers=1, n_heads=args.n_heads, dropout=args.dropout, factor=1,
        activation="gelu", embed="timeF", freq="h", output_attention=False,
        label_len=48, moving_avg=25, single_channel=False, no_inter_attn=False,
        patch_len_list=args.patch_len_list, augmentations=args.augmentations,
        patch_len=args.patch_len, stride=args.stride,
    )


def import_model(name, vendor_root):
    vendor_root = Path(vendor_root).resolve()
    model_path = vendor_root / "models" / f"{name}.py"
    if not model_path.is_file():
        raise FileNotFoundError(f"Vendored Medformer model is missing: {model_path}")
    # Original Medformer imports use the top-level names models/layers/utils.
    for package in ("models", "layers", "utils"):
        existing = sys.modules.get(package)
        if existing is not None:
            locations = ([getattr(existing, "__file__", None)] + list(getattr(existing, "__path__", [])))
            if not any(location and Path(location).resolve().is_relative_to(vendor_root) for location in locations):
                raise RuntimeError(f"Conflicting top-level module already loaded: {package}")
    sys.path.insert(0, str(vendor_root))
    module = importlib.import_module(f"models.{name}")
    if Path(module.__file__).resolve() != model_path:
        raise RuntimeError("Imported baseline does not match the vendored source")
    return module.Model


def source_metadata(vendor_root):
    paths = [Path(__file__), ROOT / "experiment_common.py", ROOT / "src/datautils.py", ROOT / "data_loading/datasets.py"]
    paths.extend(sorted(Path(vendor_root).rglob("*.py")))
    paths.extend(p for p in Path(vendor_root).glob("*") if p.is_file() and p.suffix.lower() in (".json", ".md"))
    hashes = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): digest_file(path)
              for path in paths if path.is_file()}
    versions = {}
    for package in ("torch", "numpy", "scikit-learn", "einops", "reformer-pytorch", "scipy", "sympy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"project_revision": revision, "source_sha256": hashes, "python": sys.version,
            "packages": versions, "torch_cuda_build": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cudnn_deterministic": True, "cudnn_benchmark": False,
            "deterministic_algorithms_enforced": False}


def fixed_model_indices(model):
    """FEDformer samples Fourier modes into Python lists outside state_dict."""
    result = {}
    for module_name, module in model.named_modules():
        for name in ("index", "index_q", "index_kv"):
            values = getattr(module, name, None)
            if isinstance(values, (tuple, list)) and all(isinstance(value, (int, np.integer)) for value in values):
                result[f"{module_name}.{name}"] = [int(value) for value in values]
    return result


def forward(model, x, device, num_classes):
    x = x.to(device, dtype=torch.float32).transpose(1, 2).contiguous()
    mask = torch.ones(x.shape[:2], dtype=x.dtype, device=device)
    logits = model(x, mask, None, None)
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.shape != (len(x), num_classes) or not torch.isfinite(logits).all():
        raise ValueError(f"Expected finite [batch,{num_classes}] logits, got {tuple(logits.shape)}")
    return logits


@torch.no_grad()
def evaluate(model, loader, subject_ids, device, num_classes):
    model.eval()
    labels, scores = [], []
    for x, y in loader:
        scores.append(forward(model, x, device, num_classes).softmax(dim=-1).cpu().numpy())
        labels.append(y.numpy())
    y_true, probabilities = np.concatenate(labels), np.concatenate(scores)
    aggregated = aggregate_subjects(y_true, probabilities, subject_ids)
    result = metrics(y_true, probabilities)
    result.update({"subject_" + key: value for key, value in metrics(aggregated["subject_y_true"], aggregated["subject_y_score"]).items()})
    predictions = {"y_true": y_true, "y_pred": probabilities.argmax(axis=1), "y_score": probabilities,
                   "sample_subject_id": subject_ids, **aggregated}
    predictions["subject_y_pred"] = aggregated["subject_y_score"].argmax(axis=1)
    return result, predictions


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--task_name", choices=("classification",), default="classification")
    cli.add_argument("--model", choices=MODELS, required=True)
    cli.add_argument("--dataset", choices=DATASETS, required=True)
    cli.add_argument("--random_seed", type=int, default=42)
    cli.add_argument("--split_seed", type=int, choices=(42,), default=42)
    cli.add_argument("--result_dir", type=Path, required=True)
    cli.add_argument("--vendor_root", type=Path, default=ROOT / "third_party/medformer")
    cli.add_argument("--train_epochs", "--epochs", type=int, default=100)
    cli.add_argument("--patience", type=int, default=12)
    cli.add_argument("--warmup_epochs", type=int, default=10, help="Early-stop/scheduler grace period; no LR ramp")
    cli.add_argument("--min_delta", type=float, default=0.002)
    cli.add_argument("--learning_rate", type=float, default=3e-4)
    cli.add_argument("--weight_decay", type=float, default=1e-3)
    cli.add_argument("--batch_size", type=int)
    cli.add_argument("--d_model", type=int, default=128)
    cli.add_argument("--d_ff", type=int, default=256)
    cli.add_argument("--e_layers", type=int, default=2)
    cli.add_argument("--n_heads", type=int, default=8)
    cli.add_argument("--dropout", type=float, default=0.1)
    cli.add_argument("--patch_len_list", default="2,4,8")
    cli.add_argument("--augmentations", default="none")
    cli.add_argument("--patch_len", type=int, default=16)
    cli.add_argument("--stride", type=int, default=8)
    cli.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    cli.add_argument("--gpu", type=int, default=0, help="Logical CUDA index inside CUDA_VISIBLE_DEVICES")
    cli.add_argument("--smoke", action="store_true")
    cli.add_argument("--smoke_samples_per_class", type=int, default=2)
    return cli


def validate_args(args):
    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH[args.dataset]
    for name in ("batch_size", "train_epochs", "d_model", "d_ff", "e_layers", "n_heads", "patch_len", "stride", "smoke_samples_per_class"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.random_seed < 0 or args.random_seed >= 2 ** 32:
        raise ValueError("random_seed must be in [0, 2**32)")
    if args.patience < 0 or args.warmup_epochs < 0 or args.gpu < 0:
        raise ValueError("patience, warmup_epochs and gpu must be non-negative")
    if args.d_model % args.n_heads:
        raise ValueError("d_model must be divisible by n_heads")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    for name in ("learning_rate", "weight_decay", "min_delta"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if args.learning_rate < 1e-6:
        raise ValueError("learning_rate must be at least the scheduler minimum 1e-6")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu explicitly for a CPU smoke run")


def run(args):
    from experiment_common import load_data
    from src.datautils import write_eeg_medformer_split_audit, write_wearable_split_audit

    started = time.monotonic()
    device = torch.device(f"cuda:{args.gpu}" if args.device == "cuda" else "cpu")
    seed_training(args.random_seed)
    # The shared smoke=True helper truncates wearable inputs. Always load the
    # full audited data and form a row-only smoke subset below instead.
    bundle, data_manifest = load_data(args.dataset, smoke=False)
    if args.dataset in ("adftd", "tdbrain", "apava"):
        split_path = write_eeg_medformer_split_audit(bundle, args.result_dir)
    else:
        split_path = write_wearable_split_audit(bundle, args.result_dir)
    loaders, subject_ids, selected_rows = build_loaders(bundle, args.batch_size, args.random_seed, args.smoke, args.smoke_samples_per_class)
    x_train, y_train = loaders["train"].dataset.tensors
    num_classes = len(np.unique(y_train.numpy()))
    if not np.array_equal(np.unique(y_train.numpy()), np.arange(num_classes)):
        raise ValueError("Class labels must be contiguous and zero based")
    if num_classes != len(CLASS_NAMES[args.dataset]):
        raise ValueError("Training classes differ from the fixed dataset task")
    config = model_config(args, int(x_train.shape[2]), int(x_train.shape[1]), num_classes)
    model = import_model(args.model, args.vendor_root)(config).float().to(device)
    metadata = source_metadata(args.vendor_root)
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        metadata["gpu"] = {"name": properties.name, "total_memory_bytes": properties.total_memory,
                           "logical_index": args.gpu}
    metadata["trainable_parameters"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    metadata["fixed_model_indices_outside_state_dict"] = fixed_model_indices(model)
    write_json(args.result_dir / "source_metadata.json", metadata)
    protocol = {
        "model": args.model, "dataset": args.dataset, "random_seed": args.random_seed,
        "split_seed": 42, "split_seed_note": "Existing fixed subject assignments; training seed never re-splits data",
        "smoke": args.smoke, "scientific_result": not args.smoke,
        "input_layout": "loader [B,C,T] -> model [B,T,C]", "model_config": vars(config),
        "class_names_in_score_column_order": CLASS_NAMES[args.dataset],
        "normalization": ("per_window_per_channel_standard_scaler_ddof0" if args.dataset in ("adftd", "tdbrain", "apava")
                          else "per_channel_zscore_fitted_on_training_records_and_time_only_ddof0"),
        "wearable_label_mode": getattr(bundle, "label_mode", None),
        "wearable_label_mapping": getattr(bundle, "label_mapping", None),
        "data_manifest": data_manifest, "subject_split_file": str(split_path.relative_to(args.result_dir)),
        "subject_split_sha256": digest_file(split_path), "selected_source_rows": selected_rows,
        "checkpoint_metric": "validation_subject_macro_f1",
        "checkpoint_tie_break": "minimum validation subject_macro_log_loss",
        "subject_aggregation": "arithmetic mean of window probabilities",
        "loss": "mean of sample-wise balanced weighted cross entropy",
        "class_weight_basis": "training window/record frequencies only",
        "optimizer": "AdamW", "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "train_epochs": 1 if args.smoke else args.train_epochs, "batch_size": args.batch_size,
        "early_stopping": {"strategy": "raw_primary", "patience": args.patience, "warmup_epochs": args.warmup_epochs, "min_delta": args.min_delta},
        "scheduler": {"name": "ReduceLROnPlateau", "mode": "max", "factor": 0.5, "patience": 4, "min_lr": 1e-6},
        "warmup_meaning": "Grace period for early stopping and scheduler, not linear LR warmup",
        "test_policy": "No test evaluation in smoke; otherwise exactly once after restoring the best validation checkpoint",
        "model_scope": "Unmodified Medformer source retrained with this study's fixed protocol; not original-paper scores",
    }
    write_json(args.result_dir / "protocol.json", protocol)
    counts = np.bincount(y_train.numpy(), minlength=num_classes)
    weights = torch.as_tensor(len(y_train) / (num_classes * counts), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6)
    best_key, best_epoch, stop_best, stale_epochs = None, None, None, 0
    history = []
    for epoch in range(1, (1 if args.smoke else args.train_epochs) + 1):
        model.train()
        loss_sum, observations = 0.0, 0
        learning_rate = float(optimizer.param_groups[0]["lr"])
        for x, y in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(forward(model, x, device, num_classes), y.to(device)).mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            observations += len(y)
        validation, _ = evaluate(model, loaders["vali"], subject_ids["vali"], device, num_classes)
        key = (validation["subject_macro_f1"], -validation["subject_macro_log_loss"])
        improved = best_key is None or key > best_key
        if improved:
            best_key, best_epoch = key, epoch
            checkpoint = {"model_state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
                          "epoch": epoch, "selection_key": key, "config": vars(config),
                          "random_seed": args.random_seed, "split_seed": 42, "smoke": args.smoke,
                          "fixed_model_indices_outside_state_dict": fixed_model_indices(model)}
            temporary = args.result_dir / "best_checkpoint.tmp.pt"
            torch.save(checkpoint, temporary)
            os.replace(temporary, args.result_dir / "best_checkpoint.pt")
        if epoch > args.warmup_epochs:
            if stop_best is None or key[0] > stop_best + args.min_delta:
                stop_best, stale_epochs = key[0], 0
            else:
                stale_epochs += 1
            scheduler.step(key[0])
        record = {"epoch": epoch, "train_loss": loss_sum / observations, "validation": validation,
                  "learning_rate": learning_rate, "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
                  "checkpoint_improved": improved, "early_stop_epochs_without_improvement": stale_epochs}
        history.append(record)
        write_json(args.result_dir / "history.json", history)
        print(f"{args.model} {args.dataset} seed={args.random_seed} epoch={epoch} loss={record['train_loss']:.6f} val_subject_f1={key[0]:.6f} lr={learning_rate:.3g}", flush=True)
        if args.patience > 0 and epoch > args.warmup_epochs and stale_epochs >= args.patience:
            break
    checkpoint = torch.load(args.result_dir / "best_checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_validation, validation_predictions = evaluate(model, loaders["vali"], subject_ids["vali"], device, num_classes)
    np.savez_compressed(args.result_dir / "validation_predictions.npz", **validation_predictions)
    final_test = None
    if not args.smoke:
        final_test, test_predictions = evaluate(model, loaders["test"], subject_ids["test"], device, num_classes)
        np.savez_compressed(args.result_dir / "test_predictions.npz", **test_predictions)
    final = {"status": "COMPLETED", "model": args.model, "dataset": args.dataset, "random_seed": args.random_seed,
             "split_seed": 42, "smoke": args.smoke, "scientific_result": not args.smoke,
             "best_epoch": best_epoch, "epochs_run": len(history), "validation": final_validation,
             "test": final_test, "test_evaluation_count": 0 if args.smoke else 1,
             "elapsed_seconds": time.monotonic() - started}
    write_json(args.result_dir / "metrics.json", final)
    write_json(args.result_dir / "status.json", final)
    print(json.dumps(final, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    args = parser().parse_args()
    validate_args(args)
    args.result_dir = args.result_dir.expanduser().resolve()
    args.vendor_root = args.vendor_root.expanduser().resolve()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if any(args.result_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty result directory: {args.result_dir}")
    # Exclusive claim also prevents two jobs racing for the same empty directory.
    with (args.result_dir / ".run_claim").open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    write_json(args.result_dir / "args.json", {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    write_json(args.result_dir / "status.json", {"status": "RUNNING", "pid": os.getpid(), "smoke": args.smoke})
    try:
        run(args)
    except BaseException as error:
        write_json(args.result_dir / "status.json", {"status": "FAILED", "error_type": type(error).__name__, "error": str(error), "smoke": args.smoke})
        (args.result_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
