"""Fixed-granularity ablation: precompute the (parameter-free) Activity Graph tokens once.

With a fixed block length the rendered graph no longer depends on any trainable
parameter and the OpenCLIP encoder is frozen, so the graph tokens of every window
are constants.  They are computed with the very same ``_encode_adaptive_graphs`` call
the online path uses, kept in float32, cached next to the static line/Mantis cache,
and fed to the classifier instead of re-rendering at every training step.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

FIXED_GRAPH_CACHE_VERSION = "fixed_graph_tokens_v1"


def valid_decisions_per_window(valid_lengths, patch_mask, channels, region_length=16):
    """Number of gate decisions (channel x 16-point region) per window, as in the online gate."""
    lengths = valid_lengths.to(torch.long) * patch_mask.to(torch.long)
    regions = torch.div(lengths + region_length - 1, region_length, rounding_mode="floor")
    return (regions.sum(dim=1) * int(channels)).to(torch.float32)


def _raw_digest(raw_windows, valid_lengths):
    h = hashlib.sha256()
    h.update(raw_windows.detach().cpu().contiguous().numpy().tobytes())
    h.update(valid_lengths.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def attach_fixed_graph_tokens(model, bundle, vision_model, device, *, split_name, cache_dir,
                              encode_batch_size, graph_token_grid, q, chunk_windows=32):
    raw = bundle["raw_windows"]
    lengths = bundle["valid_lengths"]
    mask = bundle["patch_mask"]
    digest = _raw_digest(raw, lengths)
    path = None
    if cache_dir:
        path = Path(cache_dir) / f"{FIXED_GRAPH_CACHE_VERSION}_q{int(q)}_grid{int(graph_token_grid)}_{split_name}.pt"
        if path.exists():
            payload = torch.load(path, map_location="cpu")
            if payload.get("raw_digest") == digest and int(payload.get("q")) == int(q):
                bundle["fixed_graph_tokens"] = payload["tokens"]
                bundle["fixed_valid_decisions"] = payload["valid_decisions"]
                print(f"Loaded fixed-granularity graph tokens: {path}", flush=True)
                return
            raise ValueError(f"fixed graph cache {path} does not match this split; refusing to overwrite")
    was_training = model.training
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(raw), chunk_windows):
            stop = min(start + chunk_windows, len(raw))
            tokens, _ = model._encode_adaptive_graphs(
                raw[start:stop].to(device=device, dtype=torch.float32),
                lengths[start:stop].to(device=device, dtype=torch.long),
                mask[start:stop].to(device=device, dtype=torch.bool),
                vision_model,
                encode_batch_size=encode_batch_size,
                spatial_grid_size=graph_token_grid,
                use_gradient_checkpointing=False,
            )
            chunks.append(tokens.detach().float().cpu())
            if start == 0 or (start // chunk_windows) % 20 == 0:
                print(f"Fixed q={q} graph tokens {split_name}: {stop}/{len(raw)}", flush=True)
    model.train(was_training)
    tokens = torch.cat(chunks, dim=0)
    decisions = valid_decisions_per_window(lengths, mask, raw.shape[2],
                                           region_length=model.renderer.gate.region_length)
    bundle["fixed_graph_tokens"] = tokens
    bundle["fixed_valid_decisions"] = decisions
    if path is not None:
        tmp = path.with_suffix(".tmp")
        torch.save({"version": FIXED_GRAPH_CACHE_VERSION, "q": int(q), "raw_digest": digest,
                    "tokens": tokens, "valid_decisions": decisions}, tmp)
        tmp.replace(path)
        print(f"Saved fixed-granularity graph tokens: {path}", flush=True)
