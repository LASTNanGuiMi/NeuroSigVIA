"""Frozen numeric branch (Medformer / TimesNet / PatchTST / TeCh encoder) for TiViT late fusion.

TiViT fuses frozen embeddings of a pretrained time-series model (Mantis /
MOMENT). Here that branch is the feature-extraction part of a Medformer /
TimesNet / PatchTST model restored from an existing NeuroSigVIA baseline
checkpoint (runners/baselines.py), or of a TeCh model restored from a TeCh
baseline run (runners/tech.py), frozen without further
training. The model's classification layer is replaced by an identity, so the
branch outputs exactly the representation that layer consumed. TeCh's in-model
augmentations are all gated on training mode, so its eval output is deterministic.

TimesNet picks its top-k periods from the batch-averaged spectrum, so its
output depends on batch composition. Callers therefore pass the batch size the
baseline was evaluated with and keep the loader order.
"""
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

NUMERIC_MODELS = ("Medformer", "TimesNet", "PatchTST", "TeCh")
VENDOR_ROOT = Path(__file__).resolve().parents[1] / "third_party"
TECH_SOURCE_PREFIX = "tech_source/"


def vendor_root_for(model_name):
    return VENDOR_ROOT / {"TimesNet": "timesnet", "TeCh": "tech"}.get(model_name, "medformer")


def import_model(name, vendor_root):
    vendor_root = Path(vendor_root).resolve()
    model_path = vendor_root / "models" / f"{name}.py"
    if not model_path.is_file():
        raise FileNotFoundError(f"Vendored baseline model is missing: {model_path}")
    # Both vendored upstreams use the top-level names models/layers/utils.
    for package in ("models", "layers", "utils"):
        existing = sys.modules.get(package)
        if existing is not None:
            locations = [getattr(existing, "__file__", None)] + list(getattr(existing, "__path__", []))
            if not any(location and Path(location).resolve().is_relative_to(vendor_root) for location in locations):
                raise RuntimeError(f"Conflicting top-level module already loaded: {package}")
    sys.path.insert(0, str(vendor_root))
    module = importlib.import_module(f"models.{name}")
    if Path(module.__file__).resolve() != model_path:
        raise RuntimeError("Imported baseline does not match the vendored source")
    return module.Model


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_vendor_source(run_dir, model_name):
    """The copied model code must be byte-identical to the code that trained the checkpoint."""
    vendor_root = vendor_root_for(model_name)
    if model_name == "TeCh":
        # The TeCh batch records its source hashes in protocol.json under tech_source/.
        recorded = json.loads((Path(run_dir) / "protocol.json").read_text())["source_sha256"]
        prefix = TECH_SOURCE_PREFIX
    else:
        recorded = json.loads((Path(run_dir) / "source_metadata.json").read_text())["source_sha256"]
        prefix = f"third_party/{vendor_root.name}/"
    checked = {}
    for key, digest in recorded.items():
        relative = key.split(prefix, 1)[-1] if prefix in key else None
        if relative is None or not relative.startswith(("models/", "layers/")):
            continue
        if not relative.startswith("layers/") and relative != f"models/{model_name}.py":
            continue
        local = vendor_root / relative
        if sha(local) != digest:
            raise ValueError(f"{local} differs from the source that trained {run_dir}")
        checked[relative] = digest
    if f"models/{model_name}.py" not in checked:
        raise ValueError(f"No recorded hash for models/{model_name}.py in {run_dir}")
    return checked


def _logits(model, x, device):
    # Loader layout [B, C, T]; the upstream models take [B, T, C].
    x = x.to(device, dtype=torch.float32).transpose(1, 2).contiguous()
    if getattr(model, "takes_input_only", False):  # TeCh: forward(x_enc)
        output = model(x)
    else:
        mask = torch.ones(x.shape[:2], dtype=x.dtype, device=device)
        output = model(x, mask, None, None)
    if isinstance(output, tuple):
        output = output[0]
    if not torch.isfinite(output).all():
        raise ValueError("Non-finite model output")
    return output


@torch.no_grad()
def _forward_all(model, x_all, batch_size, device):
    loader = DataLoader(TensorDataset(x_all), batch_size=batch_size, shuffle=False, num_workers=0)
    return np.concatenate([_logits(model, x, device).float().cpu().numpy() for (x,) in loader])


def load_frozen_encoder(run_dir, model_name, x_vali, batch_size, device):
    """Restore the trained baseline, check it reproduces its stored validation scores, drop its head."""
    run_dir = Path(run_dir)
    checkpoint = torch.load(run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=True)
    if model_name == "TeCh":
        # The TeCh runner saves the bare state_dict and records the model config in protocol.json.
        protocol = json.loads((run_dir / "protocol.json").read_text())
        config_values, state, epoch = protocol["model_config"], checkpoint, None
        head, score_key = "projector", "probabilities"
    else:
        config_values, state, epoch = checkpoint["config"], checkpoint["model_state_dict"], checkpoint.get("epoch")
        head, score_key = "projection", "y_score"
    model = import_model(model_name, vendor_root_for(model_name))(SimpleNamespace(**config_values)).float()
    model.load_state_dict(state, strict=True)
    model.takes_input_only = model_name == "TeCh"
    model.requires_grad_(False)
    model.eval().to(device)

    stored = np.load(run_dir / "validation_predictions.npz", allow_pickle=True)
    scores = torch.from_numpy(_forward_all(model, x_vali, batch_size, device)).softmax(dim=-1).numpy()
    max_abs_diff = float(np.abs(scores - stored[score_key]).max())
    if max_abs_diff > 1e-4:
        raise ValueError(f"Restored {model_name} does not reproduce its validation scores (max diff {max_abs_diff})")

    feature_dim = int(getattr(model, head).in_features)
    setattr(model, head, nn.Identity())
    return model, dict(checkpoint_epoch=epoch, feature_dim=feature_dim,
                       validation_score_max_abs_diff=max_abs_diff, config=config_values)


def extract_features(encoder, x_all, batch_size, device):
    features = _forward_all(encoder, x_all, batch_size, device)
    # Same per-sample L2 normalization TiViT applies to every embedding branch.
    features /= np.linalg.norm(features, axis=-1, keepdims=True)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite numeric features")
    return features
