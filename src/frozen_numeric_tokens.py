"""Frozen comparison-method encoders as the numeric branch of the full NeuroSigVIA model.

The numeric encoder is restored from the comparison-method checkpoint trained on
this dataset and split (TeCh: runners/tech.py runs; PatchTST / Medformer /
TimesNet: the runners/baselines.py runs behind the paper's baseline table) and
frozen, as NeuroSigVIA freezes Mantis.  Those checkpoints were trained on whole
windows, so each window is encoded once as a whole and the representation that the
removed classification head consumed is pooled onto NeuroSigVIA's 64-point outer
patches by time position:

  TimesNet   GELU(hidden) per time step                       -> steps inside the patch
  PatchTST   encoder output per sub-patch (len 16, stride 8),
             averaged over channels                            -> sub-patches centred in the patch
  Medformer  GELU(encoder output), one token list per granularity
             (patch length = stride)                           -> per granularity, then averaged
  TeCh       temporal-encoder tokens                           -> tokens centred in the patch,
             plus the channel-encoder mean (window-global, as the model sums the two)

Tokens are computed once per split with the checkpoint's own evaluation batch size
and loader order (TimesNet's period selection depends on the batch), and a
trainable LayerNorm + Linear maps them to the fusion width.  Restoring is checked
against the checkpoint's stored validation scores, and the extracted representation
is checked to reproduce the full model's logits through the original head.
"""
from __future__ import annotations

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

from runners.baselines import import_model
from src.adaptive_graph_training import NeuroSigVIAClassifier
from src.numeric_backbone_replacement import DirectTemporalFusionModule

ROOT = Path(__file__).resolve().parents[1]
FROZEN_BACKBONES = ("TeCh", "PatchTST", "Medformer", "TimesNet")
ARCHITECTURE = "neurosigvia_frozen_comparison_encoder_v1"
PATCH = 64
TECH_FILES = ("models/TeCh.py", "layers/Transformer_EncDec.py", "layers/Augmentation.py")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vendor_root(name):
    return ROOT / "third_party" / {"TeCh": "tech", "TimesNet": "timesnet"}.get(name, "medformer")


def check_source(name, run_dir):
    """Vendored model code must be byte-identical to the code that trained the checkpoint."""
    root = vendor_root(name)
    if name == "TeCh":
        recorded = json.loads((run_dir / "protocol.json").read_text())["source_sha256"]
        wanted = {f"tech_source/{f}": f for f in TECH_FILES}
    else:
        recorded = json.loads((run_dir / "source_metadata.json").read_text())["source_sha256"]
        prefix = f"third_party/{root.name}/"
        # Recorded layer files of upstream models this repository no longer ships are skipped;
        # a missing layer the model does import fails when the model is built.
        wanted = {k: k.split(prefix, 1)[1] for k in recorded if prefix in k
                  and ((k.split(prefix, 1)[1].startswith("layers/") and (root / k.split(prefix, 1)[1]).is_file())
                       or k.endswith(f"models/{name}.py"))}
    checked = {}
    for key, relative in wanted.items():
        if sha256(root / relative) != recorded[key]:
            raise ValueError(f"{root / relative} differs from the source that trained {run_dir}")
        checked[relative] = recorded[key]
    if not any(r.startswith("models/") for r in checked):
        raise ValueError(f"no recorded model source hash in {run_dir}")
    return checked


def restore(name, run_dir):
    """Frozen comparison model, its config, and the evaluation batch size of its run."""
    run_dir = Path(run_dir)
    source = check_source(name, run_dir)
    checkpoint = torch.load(run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=True)
    protocol = json.loads((run_dir / "protocol.json").read_text())
    if name == "TeCh":
        config, state = protocol["model_config"], checkpoint
        batch_size = int(protocol["config"]["batch_size"])
        sys.path.insert(0, str(vendor_root(name)))
        module = importlib.import_module("models.TeCh")
        if Path(module.__file__).resolve() != (vendor_root(name) / "models/TeCh.py").resolve():
            raise RuntimeError("imported TeCh does not match the vendored source")
        model_type = module.Model
    else:
        config, state = checkpoint["config"], checkpoint["model_state_dict"]
        batch_size = int(json.loads((run_dir / "args.json").read_text())["batch_size"])
        model_type = import_model(name, vendor_root(name))
    model = model_type(SimpleNamespace(**config)).float()
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval()
    return model, SimpleNamespace(**config), dict(
        run_dir=str(run_dir), checkpoint_sha256=sha256(run_dir / "best_checkpoint.pt"),
        eval_batch_size=batch_size, source_sha256=source, protocol=protocol)


def full_logits(name, model, x):
    """x [B,T,C]; the unmodified model with its classification head."""
    if name == "TeCh":
        return model(x)
    mask = torch.ones(x.shape[:2], dtype=x.dtype, device=x.device)
    out = model(x, mask, None, None)
    return out[0] if isinstance(out, tuple) else out


def representation(name, model, config, x):
    """What the classification head consumes, as time-indexed tokens.

    Returns tokens [B,K,D], spans [K,2] (start, end) in time steps, groups [K]
    (granularity index; 0 otherwise), global [B,D] or None, and head(x)->logits
    reproduced from these pieces for the consistency check.
    """
    batch, length, _ = x.shape
    if name == "TimesNet":
        hidden = model.enc_embedding(x, None)
        for index in range(model.layer):
            hidden = model.layer_norm(model.model[index](hidden))
        tokens = model.act(hidden)
        spans = torch.stack([torch.arange(length), torch.arange(length) + 1], 1)
        groups = torch.zeros(length, dtype=torch.long)
        logits = model.projection(tokens.reshape(batch, -1))
        return tokens, spans, groups, None, logits
    if name == "PatchTST":
        means = x.mean(1, keepdim=True).detach()
        centred = x - means
        stdev = torch.sqrt(torch.var(centred, dim=1, keepdim=True, unbiased=False) + 1e-5)
        hidden, variables = model.patch_embedding((centred / stdev).permute(0, 2, 1))
        hidden, _ = model.encoder(hidden)
        hidden = hidden.reshape(-1, variables, hidden.shape[-2], hidden.shape[-1])  # [B,C,P,D]
        logits = model.projection(model.flatten(hidden.permute(0, 1, 3, 2)).reshape(batch, -1))
        count = hidden.shape[2]
        start = torch.arange(count) * config.stride
        spans = torch.stack([start, start + config.patch_len], 1)
        return hidden.mean(1), spans, torch.zeros(count, dtype=torch.long), None, logits
    if name == "Medformer":
        hidden, _ = model.encoder(model.enc_embedding(x), attn_mask=None)
        tokens = model.act(hidden)
        logits = model.projection(tokens.reshape(batch, -1))
        spans, groups = [], []
        for group, patch_len in enumerate(map(int, config.patch_len_list.split(","))):
            count = int((config.seq_len - patch_len) / patch_len + 2)
            start = torch.arange(count) * patch_len
            spans.append(torch.stack([start, start + patch_len], 1))
            groups.append(torch.full((count,), group, dtype=torch.long))
        spans, groups = torch.cat(spans), torch.cat(groups)
        if len(spans) != tokens.shape[1]:
            raise ValueError("Medformer token count does not match its patch layout")
        return tokens, spans, groups, None, logits
    if name == "TeCh":
        channel = model.channel_encoder(x).mean(1) if model.v_layer > 0 else None
        temporal = model.temporal_encoder(x)  # [B,P,D]
        count = temporal.shape[1]
        step = int(config.patch_len)
        if step == 1 and count != length or step > 1 and count != length // step + 1:
            raise ValueError("TeCh temporal token count does not match its patch layout")
        start = torch.arange(count) * step
        spans = torch.stack([start, start + step], 1)
        pooled = temporal.mean(1) + (channel if channel is not None else 0)
        return temporal, spans, torch.zeros(count, dtype=torch.long), channel, model.projector(pooled)
    raise ValueError(f"unsupported frozen backbone {name}")


def pooling_matrix(spans, groups, length, patches):
    """[N,K] weights: mean of tokens centred in each 64-point patch per group, then mean over groups."""
    centre = ((spans[:, 0] + spans[:, 1] - 1).float() / 2).clamp(max=length - 1)
    patch = (centre // PATCH).long().clamp(max=patches - 1)
    weights = torch.zeros(patches, len(spans))
    for n in range(patches):
        present = [g for g in groups.unique().tolist() if bool(((patch == n) & (groups == g)).any())]
        if not present:
            raise ValueError(f"outer patch {n} receives no encoder token")
        for g in present:
            members = (patch == n) & (groups == g)
            weights[n, members] = 1.0 / (members.sum() * len(present))
    return weights


@torch.no_grad()
def extract_tokens(name, model, config, windows, batch_size, device, patches):
    """windows [n,C,T] in loader order -> patch tokens [n,N,D] (float32, CPU)."""
    model.to(device)
    loader = DataLoader(TensorDataset(windows), batch_size=batch_size, shuffle=False, num_workers=0)
    out, weights = [], None
    for (block,) in loader:
        x = block.to(device, dtype=torch.float32).transpose(1, 2).contiguous()
        tokens, spans, groups, glob, _ = representation(name, model, config, x)
        if weights is None:
            weights = pooling_matrix(spans, groups, x.shape[1], patches).to(device)
        pooled = torch.einsum("nk,bkd->bnd", weights, tokens)
        if glob is not None:
            pooled = pooled + glob.unsqueeze(1)
        if not torch.isfinite(pooled).all():
            raise ValueError("non-finite frozen numeric tokens")
        out.append(pooled.float().cpu())
    return torch.cat(out)


@torch.no_grad()
def verify(name, model, config, run_dir, windows, labels, batch_size, device):
    """Stored validation scores reproduced; extracted representation reproduces the head's logits."""
    model.to(device)
    stored = np.load(Path(run_dir) / "validation_predictions.npz", allow_pickle=True)
    stored_scores = stored["probabilities" if name == "TeCh" else "y_score"]
    if not np.array_equal(np.asarray(stored["y_true"]).reshape(-1), np.asarray(labels).reshape(-1)):
        raise ValueError(f"{run_dir} validation rows are not in this order")
    loader = DataLoader(TensorDataset(windows), batch_size=batch_size, shuffle=False, num_workers=0)
    scores, head_diff = [], 0.0
    for (block,) in loader:
        x = block.to(device, dtype=torch.float32).transpose(1, 2).contiguous()
        logits = full_logits(name, model, x)
        rebuilt = representation(name, model, config, x)[-1]
        head_diff = max(head_diff, float((logits - rebuilt).abs().max()))
        scores.append(logits.float().softmax(-1).cpu().numpy())
    score_diff = float(np.abs(np.concatenate(scores) - stored_scores).max())
    if score_diff > 1e-4 or head_diff > 1e-4:
        raise ValueError(f"frozen {name} check failed: score diff {score_diff}, head diff {head_diff}")
    return dict(validation_score_max_abs_diff=score_diff, representation_to_head_max_abs_diff=head_diff)


class FrozenTokenProjector(nn.Module):
    """Trainable map from frozen patch tokens [B,N,D] to the fusion width; invalid patches are zeroed."""

    def __init__(self, input_dim, output_dim, provenance):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)
        self._provenance = provenance

    def forward(self, tokens, patch_mask):
        projected = self.linear(self.norm(tokens.float()))
        return projected * patch_mask.to(projected.dtype).unsqueeze(-1)

    def provenance(self):
        return dict(self._provenance)


class FrozenNumericTokenClassifier(NeuroSigVIAClassifier):
    """Full NeuroSigVIA whose numeric branch is a frozen comparison-method encoder."""

    def __init__(self, *, backbone_name, token_dim, provenance, **kwargs):
        super().__init__(**kwargs)
        fusion_dim = int(kwargs.get("fusion_dim", 512))
        self.numeric_backbone = FrozenTokenProjector(token_dim, fusion_dim, provenance)
        fusion_args = {
            key: self.constructor_configuration[key]
            for key in (
                "visual_dim", "temporal_dim", "num_channels", "num_classes",
                "fusion_dim", "fusion_heads", "dropout", "classifier_hidden_dim",
                "classifier_num_layers", "channel_hidden_dim", "alignment_dim",
                "alignment_temperature", "cross_attention_ffn_hidden_dim",
                "cross_attention_bias",
            )
            if key in self.constructor_configuration
        }
        fusion_args["temporal_dim"] = fusion_dim
        self.fusion = DirectTemporalFusionModule(**fusion_args)
        self.backbone_name = backbone_name
        self.constructor_configuration.update(architecture=ARCHITECTURE, backbone_name=backbone_name,
                                              backbone=provenance)
        self.configuration.update(architecture=ARCHITECTURE, numeric_backbone=provenance,
                                  fusion=self.fusion.configuration, mantis_features_used=False,
                                  numeric_backbone_trainable=False)

    def forward(self, raw_windows, line_tokens, frozen_numeric_tokens, patch_mask, valid_fraction,
                valid_lengths, vision_model, *, visual_encode_batch_size=16, graph_spatial_grid_size=None,
                vision_gradient_checkpointing=True, return_attention_weights=False):
        self._validate_bound_vision_encoder(vision_model)
        grid = self.graph_token_grid if graph_spatial_grid_size is None else graph_spatial_grid_size
        if grid != self.graph_token_grid:
            raise ValueError("graph_spatial_grid_size conflicts with the model contract")
        numeric_tokens = self.numeric_backbone(frozen_numeric_tokens, patch_mask)
        graph_tokens, selector_details = self._encode_adaptive_graphs(
            raw_windows, valid_lengths, patch_mask, vision_model,
            encode_batch_size=visual_encode_batch_size, spatial_grid_size=grid,
            use_gradient_checkpointing=vision_gradient_checkpointing,
        )
        logits, details = self.fusion(line_tokens, graph_tokens, numeric_tokens, patch_mask, valid_fraction,
                                      return_attention_weights=return_attention_weights)
        details.update(selector_details)
        details["numeric_tokens"] = numeric_tokens
        return logits, details
