"""Run the upstream TeCh classifier on frozen NeuroSigVIA data splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, TensorDataset

PROJECT_SOURCE = Path(__file__).resolve().parents[1]
TECH_SOURCE = PROJECT_SOURCE / "third_party" / "tech"
sys.path.insert(0, str(TECH_SOURCE))
sys.path.insert(1, str(PROJECT_SOURCE))

from data_loading.experiment import load_data  # noqa: E402
from models.TeCh import Model  # noqa: E402

CONFIG = {
    "apava": dict(batch_size=128, lr=1e-4, t_layer=6, v_layer=6, d_model=256,
                  patch_len=1, epochs=40, patience=20,
                  augmentations="flip0.2,frequency0.2,jitter0.,mask0.,channel0.,drop0.4"),
    "tdbrain": dict(batch_size=128, lr=1e-4, t_layer=6, v_layer=0, d_model=128,
                    patch_len=6, epochs=60, patience=60,
                    augmentations="flip0.,frequency0.2,jitter0.,mask0.,channel0.,drop0.4"),
    "shimmer10": dict(batch_size=32, lr=1e-4, t_layer=6, v_layer=6, d_model=256,
                      patch_len=16, epochs=60, patience=20,
                      augmentations="flip0.,frequency0.2,jitter0.,mask0.,channel0.,drop0.4"),
    "pads11": dict(batch_size=64, lr=1e-4, t_layer=6, v_layer=6, d_model=256,
                   patch_len=16, epochs=60, patience=20,
                   augmentations="flip0.,frequency0.2,jitter0.,mask0.,channel0.,drop0.4"),
}


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_loader(bundle, split, batch_size, seed):
    source = getattr(bundle, split + "_loader").dataset
    x = source.tensors[0].float().transpose(1, 2).contiguous()
    y = torch.as_tensor(getattr(bundle, split + "_labels"), dtype=torch.long)
    ids = np.asarray(source.sample_subject_ids)
    if x.shape[0] != len(y) or len(y) != len(ids):
        raise ValueError(f"misaligned {split} data")
    if not torch.isfinite(x).all():
        raise ValueError(f"nonfinite {split} data")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size,
                        shuffle=(split == "train"), generator=generator,
                        num_workers=0, drop_last=False)
    return loader, ids


def metrics(y, logits, num_class):
    probs = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
    pred = probs.argmax(1)
    y_onehot = np.eye(num_class, dtype=np.int64)[y]
    values = {
        "accuracy": accuracy_score(y, pred),
        "macro_precision": precision_score(y, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "macro_auroc": roc_auc_score(y_onehot, probs, average="macro"),
        "macro_auprc": average_precision_score(y_onehot, probs, average="macro"),
    }
    return values, probs, pred


@torch.no_grad()
def evaluate(model, loader, device, num_class):
    model.eval()
    all_y, all_logits = [], []
    for x, y in loader:
        logits = model(x.to(device)).float().cpu()
        all_y.append(y.numpy())
        all_logits.append(logits.numpy())
    y = np.concatenate(all_y)
    logits = np.concatenate(all_logits)
    values, probs, pred = metrics(y, logits, num_class)
    return values, y, probs, pred


def train(dataset, seed, output, smoke=False):
    cfg = CONFIG[dataset]
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(8)
    bundle, manifest = load_data(dataset)
    loaders, subject_ids = {}, {}
    for split in ("train", "vali", "test"):
        loaders[split], subject_ids[split] = make_loader(bundle, split, cfg["batch_size"], seed)
    classes = sorted(set(map(int, np.asarray(bundle.train_labels))))
    if classes != list(range(len(classes))):
        raise ValueError(f"unsupported class IDs: {classes}")
    x0 = loaders["train"].dataset.tensors[0]
    args = SimpleNamespace(
        seq_len=x0.shape[1], enc_in=x0.shape[2], num_class=len(classes),
        d_model=cfg["d_model"], dropout=0.0, t_layer=cfg["t_layer"],
        v_layer=cfg["v_layer"], patch_len=cfg["patch_len"],
        augmentations=cfg["augmentations"],
    )
    device = torch.device("cuda:0")
    model = Model(args).to(device)
    split_ids = {k: sorted(set(map(str, ids.tolist()))) for k, ids in subject_ids.items()}
    protocol = {
        "method": "upstream_TeCh_CoTAR", "upstream_commit": "9a378cc546a5d97c871eff282148175b3c7cd75b",
        "dataset": dataset, "training_seed": seed,
        "data_split": "APAVA_resplit_20260917" if dataset == "apava" else "project_fixed_subject_split",
        "data_manifest": manifest, "split_subject_ids": split_ids,
        "config": cfg, "model_config": vars(args),
        "data_input": "project normalized windows [N,C,T] transposed to TeCh [N,T,C]",
        "selection": "best validation window macro-F1, earliest epoch on tie",
        "test_policy": "evaluate once after checkpoint selection",
        # Keys keep the layout that src/frozen_numeric_tokens.check_source reads.
        "source_sha256": {
            **{f"tech_source/{name}": sha256(TECH_SOURCE / name) for name in
               ("models/TeCh.py", "layers/Transformer_EncDec.py", "layers/Augmentation.py")},
            **{f"project_source/{name}": sha256(PROJECT_SOURCE / name) for name in
               ("src/datautils.py", "data_loading/experiment.py")},
        },
    }
    atomic_json(output / "protocol.json", protocol)
    if smoke:
        model.train()
        x, y = next(iter(loaders["train"]))
        x, y = x.to(device), y.to(device)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        grad = sum(float(p.grad.abs().sum().item()) for p in model.parameters() if p.grad is not None)
        if not np.isfinite(grad) or grad <= 0:
            raise RuntimeError("no finite nonzero gradient")
        val_metrics, _, _, _ = evaluate(model, loaders["vali"], device, len(classes))
        print(json.dumps({"smoke": "PASS", "dataset": dataset, "shape": list(x.shape),
                          "loss": float(loss.item()), "grad_l1": grad,
                          "validation_metrics": val_metrics,
                          "manifest": manifest}, allow_nan=False), flush=True)
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    best_f1, best_epoch, stale = -1.0, 0, 0
    history = []
    start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        losses = []
        for x, y in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device))
            loss = torch.nn.functional.cross_entropy(logits, y.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            losses.append(float(loss.item()))
        val_metrics, _, _, _ = evaluate(model, loaders["vali"], device, len(classes))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "validation": val_metrics, "elapsed_seconds": round(time.time() - start, 1)}
        history.append(row)
        atomic_json(output / "progress.json", {"best_epoch": best_epoch, "history": history})
        print(json.dumps(row, allow_nan=False), flush=True)
        f1 = val_metrics["macro_f1"]
        if f1 > best_f1 + 1e-5:
            best_f1, best_epoch, stale = f1, epoch, 0
            torch.save(model.state_dict(), output / "best_checkpoint.pt")
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    model.load_state_dict(torch.load(output / "best_checkpoint.pt", map_location=device, weights_only=True))
    val_metrics, val_y, val_prob, val_pred = evaluate(model, loaders["vali"], device, len(classes))
    test_metrics, test_y, test_prob, test_pred = evaluate(model, loaders["test"], device, len(classes))
    np.savez_compressed(output / "validation_predictions.npz", y_true=val_y, y_pred=val_pred,
                        probabilities=val_prob, subject_ids=subject_ids["vali"])
    np.savez_compressed(output / "test_predictions.npz", y_true=test_y, y_pred=test_pred,
                        probabilities=test_prob, subject_ids=subject_ids["test"])
    summary = {"status": "COMPLETED", "dataset": dataset, "seed": seed,
               "best_epoch": best_epoch, "validation": val_metrics, "test": test_metrics,
               "elapsed_seconds": round(time.time() - start, 1)}
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(CONFIG), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    a = parser.parse_args()
    train(a.dataset, a.seed, a.output, a.smoke)
