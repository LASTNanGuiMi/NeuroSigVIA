"""Train the visual-only ablation with the paired main-run protocol.

The existing subject splits and frozen line features are inputs, never inferred
or reshuffled here. The online Activity Graph keeps its image-to-gate gradient
through the frozen visual encoder. No numeric features or trained main-model
weights enter this trainer.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.adaptive_graph_training import (
    _atomic_json_dump,
    _atomic_prediction_dump,
    _cpu_state_dict,
    _encoder_contract,
    _evaluation_protocol,
)
from src.classifier import compute_metrics_from_predictions
from src.patch_fusion import (
    _EarlyStoppingMonitor,
    _aggregate_subject_predictions,
    _balanced_class_weights,
    _build_patch_lr_scheduler,
    _checkpoint_selection_key,
    _merge_subject_metrics,
    _subject_ids_digest,
)
from src.utils import set_random_seed
from src.vision_ablation import VISION_ONLY_ARCHITECTURE, VisionOnlyClassifier


VISUAL_KEYS = (
    "raw_windows", "line_tokens", "patch_mask", "valid_fraction", "valid_lengths",
)
CHECKPOINT_METRIC = "window_macro_f1"


def _array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _state_sha256(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _prepare_splits(splits):
    """Validate the supplied fixed partition without consuming numeric tokens."""
    if set(splits) != {"train", "vali", "test"}:
        raise ValueError("fixed train, vali and test splits are required")
    prepared = {}
    for name in ("train", "vali", "test"):
        split = splits[name]
        tensors = {key: torch.as_tensor(split[key]) for key in VISUAL_KEYS}
        raw, line = tensors["raw_windows"], tensors["line_tokens"]
        labels, subjects = _array(split["labels"]), _array(split["subject_ids"])
        if raw.ndim != 4 or line.ndim != 3 or raw.shape[0] == 0:
            raise ValueError(f"{name}: expected nonempty raw [M,N,C,L], line [M,N,Dv]")
        if tuple(line.shape[:2]) != tuple(raw.shape[:2]):
            raise ValueError(f"{name}: raw/line sample and patch axes disagree")
        if labels.shape != (len(raw),) or subjects.shape != labels.shape:
            raise ValueError(f"{name}: labels and subject IDs must align with windows")
        if labels.dtype.kind not in "iu" or subjects.dtype.hasobject:
            raise TypeError(f"{name}: integer labels and non-object subject IDs required")
        if not raw.is_floating_point() or not line.is_floating_point():
            raise TypeError(f"{name}: raw and line features must be floating point")
        for key in VISUAL_KEYS:
            tensor = tensors[key]
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"{name}: non-finite {key}")
        for key in ("patch_mask", "valid_fraction", "valid_lengths"):
            if tuple(tensors[key].shape) != tuple(raw.shape[:2]):
                raise ValueError(f"{name}: {key} must have shape [M,N]")
        mask, lengths = tensors["patch_mask"], tensors["valid_lengths"]
        if not torch.equal(mask, mask.bool()) or not torch.equal(lengths, lengths.long()):
            raise ValueError(f"{name}: nonbinary mask or fractional valid length")
        lengths = lengths.long()
        if (lengths < 0).any() or (lengths > raw.shape[-1]).any():
            raise ValueError(f"{name}: invalid valid_lengths")
        if not torch.equal(mask.bool(), lengths > 0) or (mask.bool().sum(1) == 0).any():
            raise ValueError(f"{name}: every sample needs valid, consistently masked patches")
        if not torch.allclose(tensors["valid_fraction"].float(), lengths.float() / raw.shape[-1], rtol=1e-3, atol=1e-3):
            raise ValueError(f"{name}: fractions do not match valid_lengths / patch_length")
        for subject in np.unique(subjects):
            if len(np.unique(labels[subjects == subject])) != 1:
                raise ValueError(f"{name}: a subject has inconsistent labels")
        prepared[name] = {**tensors, "labels": labels.astype(np.int64), "subject_ids": subjects}
    classes = np.unique(prepared["train"]["labels"])
    if len(classes) < 2 or not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("training labels must include contiguous class indices 0..K-1")
    for name, split in prepared.items():
        if np.setdiff1d(split["labels"], classes).size:
            raise ValueError(f"{name}: unseen class index")
        if split["raw_windows"].shape[1:] != prepared["train"]["raw_windows"].shape[1:]:
            raise ValueError(f"{name}: patch/channel/length dimensions changed")
        if split["line_tokens"].shape[1:] != prepared["train"]["line_tokens"].shape[1:]:
            raise ValueError(f"{name}: visual feature dimensions changed")
    for left, right in (("train", "vali"), ("train", "test"), ("vali", "test")):
        if np.intersect1d(prepared[left]["subject_ids"], prepared[right]["subject_ids"]).size:
            raise ValueError(f"subject overlap between {left} and {right}")
    return prepared, classes


def _loader(split, batch_size, shuffle):
    # Classification units are original windows; N internal patches share one label.
    dataset = TensorDataset(
        *(split[key] for key in VISUAL_KEYS),
        torch.as_tensor(split["labels"], dtype=torch.long),
        torch.arange(len(split["labels"]), dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _forward(model, batch, vision_model, device, encode_batch_size, training):
    raw, line, mask, fractions, lengths, labels, indices = batch
    logits, details = model(
        raw.to(device=device, dtype=torch.float32),
        line.to(device=device, dtype=torch.float32),
        mask.to(device=device, dtype=torch.bool),
        fractions.to(device=device, dtype=torch.float32),
        lengths.to(device=device, dtype=torch.long),
        vision_model,
        encode_batch_size=encode_batch_size,
        vision_gradient_checkpointing=training,
    )
    return logits, details, labels.to(device=device, dtype=torch.long), indices


def _verify_gradients(model):
    # This repeats the paired main trainer's total-loss gate checks. With nonzero
    # balance regularization it alone does not isolate the CE-to-gate derivative.
    gate = [p for p in model.renderer.gate.region_cls.parameters() if p.requires_grad]
    if gate and not all(p.grad is not None for p in gate):
        raise RuntimeError("adaptive gate did not receive gradients")
    if gate and not any(bool(p.grad.detach().abs().gt(0).any()) for p in gate):
        raise RuntimeError("adaptive gate gradients are all zero")
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise ValueError("visual-only training produced non-finite gradients")


def _run_epoch(model, loader, vision_model, optimizer, criterion, device, balance_weight, encode_batch_size):
    model.train()
    vision_model.eval()
    totals = {"loss": 0.0, "task_loss": 0.0, "selector_balance_loss": 0.0}
    count = 0
    for batch in tqdm(loader, desc="Train vision-only Activity Graph", leave=False):
        optimizer.zero_grad(set_to_none=True)
        logits, details, labels, _ = _forward(model, batch, vision_model, device, encode_batch_size, True)
        task_loss = criterion(logits, labels).mean()
        balance_loss = details["selector_balance_loss"]
        loss = task_loss + balance_weight * balance_loss
        if not torch.isfinite(loss):
            raise ValueError("visual-only training produced non-finite loss")
        loss.backward()
        _verify_gradients(model)
        # The paired adaptive_graph_training._run_epoch has no gradient clipping.
        optimizer.step()
        size = len(labels)
        count += size
        for name, value in (("loss", loss), ("task_loss", task_loss), ("selector_balance_loss", balance_loss)):
            totals[name] += float(value.detach()) * size
        del logits, details, loss, task_loss, balance_loss
    if not count:
        raise ValueError("cannot train an empty split")
    return {name: value / count for name, value in totals.items()}


@torch.no_grad()
def evaluate(model, loader, classes, subjects, vision_model, device, encode_batch_size):
    model.eval()
    vision_model.eval()
    true, scores, indices = [], [], []
    for batch in loader:
        logits, _, labels, sample_index = _forward(model, batch, vision_model, device, encode_batch_size, False)
        probabilities = logits.float().softmax(-1)
        if not torch.isfinite(probabilities).all():
            raise ValueError("visual-only classifier returned non-finite probabilities")
        true.append(labels.cpu().numpy())
        scores.append(probabilities.cpu().numpy())
        indices.append(sample_index.cpu().numpy())
    if not true:
        raise ValueError("cannot evaluate an empty split")
    details = dict(y_true=np.concatenate(true), y_score=np.concatenate(scores), sample_index=np.concatenate(indices))
    details["y_pred"] = details["y_score"].argmax(-1)
    window = compute_metrics_from_predictions(details["y_true"], details["y_pred"], details["y_score"], np.arange(len(classes)))
    subject, subject_details = _aggregate_subject_predictions(
        details["y_true"], details["y_score"], details["sample_index"], subjects, classes,
    )
    details.update(subject_details)
    return _merge_subject_metrics(window, subject), details


def _save_checkpoint(path, payload):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def train_vision_only(splits, vision_model, config, output, seed, device):
    """Fresh visual-only training; return the completed, auditable summary dict."""
    started = time.monotonic()
    output, device = Path(output), torch.device(device)
    cfg = dict(config)
    cfg["random_seed"] = int(seed)
    if cfg.get("patch_checkpoint_metric", CHECKPOINT_METRIC) != CHECKPOINT_METRIC:
        raise ValueError("this ablation requires validation window macro-F1 selection")
    if cfg.get("mlp_class_weight", "balanced") != "balanced":
        raise ValueError("paired protocol requires balanced train-only class weights")
    for name in ("summary.json", "protocol.json", "best_checkpoint.pt", "training_history.json", "validation_predictions.npz", "test_predictions.npz"):
        if (output / name).exists():
            raise FileExistsError(f"refusing to overwrite {output / name}")
    batch_size = int(cfg.get("batch_size", 8))
    epochs = int(cfg.get("mlp_epochs", 100))
    encode_batch_size = int(cfg.get("visual_encode_batch_size", 4))
    lr, decay = float(cfg.get("mlp_lr", 3e-4)), float(cfg.get("mlp_weight_decay", 1e-3))
    balance_weight = float(cfg.get("granularity_balance_weight", 0.001))
    if min(batch_size, epochs, encode_batch_size) <= 0:
        raise ValueError("batch sizes and maximum epochs must be positive")
    if not all(math.isfinite(x) for x in (lr, decay, balance_weight)) or lr <= 0 or min(decay, balance_weight) < 0:
        raise ValueError("invalid optimizer or balance-loss configuration")
    prepared, classes = _prepare_splits(splits)
    class_names = cfg.get("class_names", ["HC", "FTD", "AD"] if len(classes)==3 else [str(x) for x in classes])
    if len(class_names) != len(classes) or len(set(class_names)) != len(classes):
        raise ValueError("class_names must uniquely match the probability columns")
    output.mkdir(parents=True, exist_ok=True)
    # Reset after feature extraction/loading, then create loaders before the model
    # exactly as in the paired main/numeric-only runs. Splits are never resampled.
    set_random_seed(int(seed))
    loaders = {name: _loader(split, batch_size, name == "train") for name, split in prepared.items()}
    constructor = dict(
        visual_dim=int(prepared["train"]["line_tokens"].shape[-1]),
        num_channels=int(prepared["train"]["raw_windows"].shape[2]), num_classes=len(classes),
        fusion_dim=int(cfg.get("fusion_dim", 128)), fusion_heads=int(cfg.get("fusion_heads", 2)),
        classifier_hidden_dim=int(cfg.get("mlp_hidden_dim", 128)), classifier_num_layers=int(cfg.get("mlp_num_layers", 2)),
        dropout=float(cfg.get("mlp_dropout", 0.1)), graph_image_size=224,
        graph_token_grid=int(cfg.get("granularity_graph_token_grid", 4)),
        activity_graph_canvas_size=int(cfg.get("activity_graph_canvas_size", 360)),
        activity_graph_line_width=float(cfg.get("activity_graph_line_width", 1.0)),
        activity_graph_vertical_margin=float(cfg.get("activity_graph_vertical_margin", 0.05)),
        adaptive_temperature=float(cfg.get("granularity_gate_temperature", 0.5)),
        freeze_adaptive_gate=False, cross_attention_ffn_hidden_dim=None,
        cross_attention_bias=True, adaptive_gate_checkpoint=None, strict_gate_checkpoint=True,
    )
    model = VisionOnlyClassifier(**constructor).to(device)
    initial_digest = _state_sha256(model)
    vision_model.to(device).eval()
    for parameter in vision_model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    vision_contract = _encoder_contract(vision_model, "vision", source_identity=cfg.get("vision_source_identity"))
    if hasattr(model, "bind_vision_encoder_contract"):
        model.bind_vision_encoder_contract(vision_contract, validated_encoder=vision_model)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=decay)
    scheduler_cfg = dict(
        scheduler_type=cfg.get("mlp_lr_scheduler", "reduce_on_plateau"),
        patience=int(cfg.get("mlp_lr_scheduler_patience", 4)),
        factor=float(cfg.get("mlp_lr_scheduler_factor", 0.5)), min_lr=float(cfg.get("mlp_lr_scheduler_min_lr", 1e-6)),
    )
    scheduler = _build_patch_lr_scheduler(optimizer, **scheduler_cfg)
    stopping_cfg = dict(
        strategy=cfg.get("mlp_early_stop_strategy", "raw_primary"),
        patience=int(cfg.get("mlp_early_stop_patience", 12)), min_epochs=int(cfg.get("mlp_early_stop_min_epochs", 0)),
        warmup_epochs=int(cfg.get("mlp_early_stop_warmup_epochs", 10)),
        ema_decay=float(cfg.get("mlp_early_stop_ema_decay", 0.6)), min_delta=float(cfg.get("mlp_early_stop_min_delta", 0.002)),
    )
    stopping = _EarlyStoppingMonitor(**stopping_cfg)
    weights = _balanced_class_weights(prepared["train"]["labels"], len(classes), device)
    criterion = nn.CrossEntropyLoss(weight=weights, reduction="none")
    protocol = dict(
        architecture=VISION_ONLY_ARCHITECTURE, dataset=cfg.get("dataset", "ADFTD"), training_seed=int(seed), split_seed=42,
        class_names=class_names, external_encoder_contracts={"vision": vision_contract},
        **_evaluation_protocol(CHECKPOINT_METRIC), checkpoint_metric_requested=CHECKPOINT_METRIC,
        training_args=cfg, model_constructor_configuration=constructor,
        initial_state_sha256=initial_digest, trained_main_weights_loaded=False,
        initialization="Fresh visual-only constructor after seed reset; no trained main weights or gate checkpoint.",
        randomness_note="The branch-only constructor consumes RNG differently from the full paired main constructor.",
        classifier_input_dim=constructor["fusion_dim"], uses_mantis=False, uses_alignment=False,
        uses_concat_attn=False, zero_numeric_slot=False, vision_encoder_parameters_frozen=True,
        vision_encoder_parameters_included=False, online_graph_input_gradients_enabled=True,
        features_used=list(VISUAL_KEYS), removed_modules=["Mantis", "channel_pool", "alignment", "concat_attn"],
        loss="Balanced CrossEntropyLoss(reduction='none').mean() + selector_balance_weight * selector_balance_loss",
        selector_balance_weight=balance_weight, alignment_weight_effective=0.0,
        gradient_clip_max_norm=None, class_weights=weights.detach().cpu().tolist(),
        gradient_check="Finite nonzero gate gradients from total loss; does not separately prove the CE-only path.",
        optimizer=dict(type="AdamW", learning_rate=lr, weight_decay=decay),
        early_stopping=stopping_cfg, lr_scheduler=scheduler_cfg, batch_size_effective=batch_size,
        visual_encode_batch_size=encode_batch_size, maximum_epochs=epochs,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), device=str(device),
        split_metadata={name: dict(windows=len(split["labels"]), subject_ids=np.unique(split["subject_ids"]).tolist(), ordered_subject_ids_sha256=_subject_ids_digest(split["subject_ids"])) for name, split in prepared.items()},
        std_ddof=1,
    )
    _atomic_json_dump(output / "protocol.json", protocol)
    best_key, best_epoch, best_state = None, None, None
    history = []
    for epoch in range(1, epochs + 1):
        epoch_started = time.monotonic()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        stats = _run_epoch(model, loaders["train"], vision_model, optimizer, criterion, device, balance_weight, encode_batch_size)
        validation, _ = evaluate(model, loaders["vali"], classes, prepared["vali"]["subject_ids"], vision_model, device, encode_batch_size)
        key = _checkpoint_selection_key(validation, CHECKPOINT_METRIC)
        improved = best_key is None or key > best_key
        stop_state = stopping.update(key, epoch)
        scheduler_stepped = scheduler is not None and epoch > stopping_cfg["warmup_epochs"]
        if scheduler_stepped:
            scheduler.step(float(key[0]))
        record = dict(epoch=epoch, **stats, validation=validation, validation_macro_f1=float(validation["macro_f1"]),
                      checkpoint_metric_effective=CHECKPOINT_METRIC, checkpoint_selection_key=list(key), checkpoint_improved=improved,
                      early_stopping=stop_state, learning_rate=learning_rate, next_learning_rate=float(optimizer.param_groups[0]["lr"]),
                      lr_scheduler_stepped=scheduler_stepped, epoch_seconds=time.monotonic()-epoch_started)
        history.append(record)
        if improved:
            best_key, best_epoch, best_state = key, epoch, _cpu_state_dict(model)
            _save_checkpoint(output / "best_checkpoint.pt", dict(
                architecture=VISION_ONLY_ARCHITECTURE, model_state_dict=best_state,
                model_constructor_configuration=model.get_config(), classes=classes.tolist(),
                class_names=class_names, external_encoder_contracts={"vision": vision_contract},
                best_epoch=best_epoch, validation_metrics=validation, protocol=protocol,
                checkpoint_metric_effective=CHECKPOINT_METRIC, evaluation_unit="window", test_evaluations=0,
            ))
        _atomic_json_dump(output / "training_history.json", history)
        _atomic_json_dump(output / "progress.json", dict(state="RUNNING", epoch=epoch, maximum_epochs=epochs,
                          best_epoch=best_epoch, latest=record, test_evaluations=0, elapsed_seconds=time.monotonic()-started))
        print(f"VISION_ONLY EPOCH {epoch}/{epochs} loss={stats['loss']:.6f} val_window_f1={key[0]:.6f} best={best_epoch} seconds={record['epoch_seconds']:.2f}", flush=True)
        if stop_state["should_stop"]:
            break
    if best_state is None:
        raise RuntimeError("no valid validation-selected checkpoint")
    model.load_state_dict(best_state, strict=True)
    validation, val_details = evaluate(model, loaders["vali"], classes, prepared["vali"]["subject_ids"], vision_model, device, encode_batch_size)
    # The sole test forward occurs here, after best checkpoint restoration.
    test, test_details = evaluate(model, loaders["test"], classes, prepared["test"]["subject_ids"], vision_model, device, encode_batch_size)
    _atomic_prediction_dump(output / "validation_predictions.npz", val_details)
    _atomic_prediction_dump(output / "test_predictions.npz", test_details)
    summary = dict(
        architecture=VISION_ONLY_ARCHITECTURE, state="COMPLETED", dataset=cfg.get("dataset", "ADFTD"), seed=int(seed),
        class_names=class_names,
        best_epoch=best_epoch, epochs=len(history), validation_metrics=validation, test_metrics=test,
        test_evaluations=1, **_evaluation_protocol(CHECKPOINT_METRIC), checkpoint_metric_requested=CHECKPOINT_METRIC,
        checkpoint="best_checkpoint.pt", protocol="protocol.json", training_history="training_history.json",
        validation_predictions="validation_predictions.npz", test_predictions="test_predictions.npz",
        classifier_input_dim=constructor["fusion_dim"], uses_mantis=False, uses_alignment=False, uses_concat_attn=False,
        training_args=cfg, elapsed_seconds=time.monotonic()-started,
    )
    _atomic_json_dump(output / "summary.json", summary)
    _atomic_json_dump(output / "progress.json", dict(state="COMPLETED", epochs=len(history), best_epoch=best_epoch,
                      test_evaluations=1, elapsed_seconds=summary["elapsed_seconds"]))
    print(f"VISION_ONLY COMPLETED seed={seed} test_window_f1={test['macro_f1']:.6f}", flush=True)
    return summary
