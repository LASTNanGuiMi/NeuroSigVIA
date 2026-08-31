"""Medformer-inspired graph rendering for multichannel time series.

The canonical renderer is :class:`MedformerGraphRenderer`.  Historical class
names remain available as aliases.  The transform is deterministic and
contains no learned parameters.  It transfers three ideas from Medformer into
image generation:

1. non-overlapping patches at one or three temporal scales;
2. correlation-weighted cross-channel activity propagation at each scale;
3. optional router-style information exchange between three legacy scales.

The resulting fine, medium, and coarse activity maps are encoded as the RGB
planes consumed by the existing vision backbone.  ``TemporalGranularityGraphBank``
adds a bank of such RGB renderers so a downstream, trainable selector can pick
the most useful temporal-granularity regime per temporal window.  These are image
transforms inspired by Medformer, not implementations of the Medformer
classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pair_covering_order(num_channels: int) -> list[int]:
    """Return a deterministic minimum-length walk covering all channel pairs.

    For odd ``num_channels``, the complete graph is Eulerian.  For even
    ``num_channels``, duplicating a matching on vertices ``1..n-2`` leaves
    exactly vertices ``0`` and ``n-1`` odd and therefore yields a minimum
    open Euler trail.  Unique edge IDs keep the parallel copies distinct.
    """
    if num_channels < 1:
        raise ValueError("A Med activity graph requires at least one channel.")
    if num_channels == 1:
        return [0]

    adjacency: list[list[tuple[int, int]]] = [
        [] for _ in range(num_channels)
    ]
    edge_count = 0

    def add_edge(left: int, right: int) -> None:
        nonlocal edge_count
        adjacency[left].append((right, edge_count))
        adjacency[right].append((left, edge_count))
        edge_count += 1

    for left in range(num_channels):
        for right in range(left + 1, num_channels):
            add_edge(left, right)

    if num_channels % 2 == 0:
        for left in range(1, num_channels - 1, 2):
            add_edge(left, left + 1)

    for edges in adjacency:
        edges.sort(key=lambda item: (item[0], item[1]))

    used = [False] * edge_count
    cursors = [0] * num_channels
    stack = [0]
    reverse_order: list[int] = []

    while stack:
        current = stack[-1]
        edges = adjacency[current]
        cursor = cursors[current]
        while cursor < len(edges) and used[edges[cursor][1]]:
            cursor += 1
        cursors[current] = cursor

        if cursor == len(edges):
            reverse_order.append(stack.pop())
            continue

        next_channel, edge_id = edges[cursor]
        cursors[current] += 1
        used[edge_id] = True
        stack.append(next_channel)

    if not all(used):
        raise RuntimeError("Failed to construct a pair-covering Euler walk.")
    return list(reversed(reverse_order))


def _resize_rows(values: torch.Tensor, width: int) -> torch.Tensor:
    """Linearly resize ``(batch, channels, patches)`` along patch time."""
    if values.shape[-1] == width:
        return values
    if values.shape[-1] == 1:
        return values.expand(*values.shape[:-1], width)
    return F.interpolate(
        values,
        size=width,
        mode="linear",
        align_corners=True,
    )


def _stable_softmax(values: torch.Tensor, temperature: float) -> torch.Tensor:
    """Compute temperature-scaled softmax without leaving the tensor device."""
    return F.softmax(values / temperature, dim=-1)


def _inverse_occurrence_row_weights(
    order: torch.Tensor,
    num_channels: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Give every source channel equal total weight after row repetition."""
    occurrences = torch.bincount(order, minlength=num_channels)
    weights = occurrences.index_select(0, order).to(dtype=dtype).reciprocal()
    return weights / weights.mean()


class MedformerGraphRenderer(nn.Module):
    """Create an RGB multi-scale activity graph from ``(B, channels, time)``.

    In legacy three-scale mode, RGB has a fixed semantic meaning: red is
    fine-scale activity, green is medium-scale activity, and blue is
    coarse-scale activity.  In single-scale expert mode the same activity map
    is repeated over RGB, so colour cannot leak the expert identity to CLIP.
    """

    def __init__(
        self,
        patch_lengths=(2, 4, 8),
        img_size=224,
        channel_mix=0.35,
        router_temperature=0.2,
        router_mix=0.5,
    ):
        super().__init__()
        if isinstance(patch_lengths, int):
            patch_lengths = (int(patch_lengths),)
        else:
            patch_lengths = tuple(int(length) for length in patch_lengths)
        if len(patch_lengths) not in {1, 3}:
            raise ValueError(
                "MedformerGraphRenderer requires either one single-scale "
                "expert length or exactly three legacy RGB encoding lengths; "
                f"got {patch_lengths}."
            )
        if any(length <= 0 for length in patch_lengths):
            raise ValueError(f"Patch lengths must be positive, got {patch_lengths}.")
        if len(patch_lengths) == 3 and tuple(sorted(set(patch_lengths))) != patch_lengths:
            raise ValueError(
                "Patch lengths must be three unique increasing integers, got "
                f"{patch_lengths}."
            )
        if img_size <= 0:
            raise ValueError(f"img_size must be positive, got {img_size}.")
        if not 0.0 <= channel_mix <= 1.0:
            raise ValueError(f"channel_mix must be in [0, 1], got {channel_mix}.")
        if router_temperature <= 0.0:
            raise ValueError(
                "router_temperature must be positive, got "
                f"{router_temperature}."
            )
        if not 0.0 <= router_mix <= 1.0:
            raise ValueError(f"router_mix must be in [0, 1], got {router_mix}.")

        self.patch_lengths = patch_lengths
        self.img_size = int(img_size)
        self.channel_mix = float(channel_mix)
        self.router_temperature = float(router_temperature)
        self.router_mix = float(router_mix)

    @staticmethod
    def _normalize_channels(signals: torch.Tensor) -> torch.Tensor:
        signals = torch.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
        center = torch.quantile(signals, 0.5, dim=-1, keepdim=True)
        deviations = signals - center
        mad = (
            torch.quantile(deviations.abs(), 0.5, dim=-1, keepdim=True)
            * 1.4826
        )
        std = signals.std(dim=-1, keepdim=True, unbiased=False)
        scale = torch.where(mad > 1e-6, mad, std.clamp_min(1e-6))
        normalized = deviations / scale
        return torch.nan_to_num(
            normalized,
            nan=0.0,
            posinf=8.0,
            neginf=-8.0,
        ).clamp(-8.0, 8.0)

    def _patch_activity(
        self,
        signals: torch.Tensor,
        patch_length: int,
    ) -> torch.Tensor:
        _, channels, time_steps = signals.shape
        patch_length = min(patch_length, time_steps)
        padding = (-time_steps) % patch_length
        if padding:
            signals = F.pad(signals, (0, padding), mode="replicate")

        patches = signals.reshape(
            signals.shape[0],
            channels,
            -1,
            patch_length,
        )
        amplitude = torch.sqrt(patches.square().mean(dim=-1) + 1e-8)
        if patch_length > 1:
            variation = patches.diff(dim=-1).abs().mean(dim=-1)
        else:
            variation = torch.zeros_like(amplitude)
        activity = amplitude + variation

        centered = activity - activity.mean(dim=-1, keepdim=True)
        norms = torch.linalg.vector_norm(centered, dim=-1, keepdim=True)
        normalized = centered / norms.clamp_min(1e-8)
        affinity = torch.bmm(normalized, normalized.transpose(1, 2)).abs()
        affinity = torch.nan_to_num(affinity, nan=0.0, posinf=0.0, neginf=0.0)
        affinity = affinity + torch.eye(
            channels,
            dtype=affinity.dtype,
            device=affinity.device,
        ).unsqueeze(0)
        affinity = affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        propagated = torch.bmm(affinity, activity)
        return (
            (1.0 - self.channel_mix) * activity
            + self.channel_mix * propagated
        )

    def _render_batch(self, signals: torch.Tensor) -> torch.Tensor:
        signals = self._normalize_channels(signals)
        scale_maps = torch.stack(
            [
                _resize_rows(
                    self._patch_activity(signals, patch_length),
                    self.img_size,
                )
                for patch_length in self.patch_lengths
            ],
            dim=1,
        )

        if len(self.patch_lengths) == 1:
            # A graph expert must encode exactly one temporal granularity.
            # Repeating the neutral map over RGB retains the vision encoder's
            # expected input shape without introducing a scale-specific colour.
            scale_maps = scale_maps.expand(-1, 3, -1, -1)
        else:
            routers = scale_maps.mean(dim=2)
            routers = routers - routers.mean(dim=-1, keepdim=True)
            router_norms = torch.linalg.vector_norm(routers, dim=-1, keepdim=True)
            normalized_routers = routers / router_norms.clamp_min(1e-8)
            router_similarity = torch.bmm(
                normalized_routers,
                normalized_routers.transpose(1, 2),
            )
            router_weights = _stable_softmax(
                router_similarity,
                temperature=self.router_temperature,
            )
            exchanged = torch.einsum(
                "bst,btcw->bscw",
                router_weights,
                scale_maps,
            )
            scale_maps = (
                (1.0 - self.router_mix) * scale_maps
                + self.router_mix * exchanged
            )

        flattened = scale_maps.flatten(start_dim=1)
        lower = torch.quantile(flattened, 0.01, dim=1).view(-1, 1, 1, 1)
        upper = torch.quantile(flattened, 0.99, dim=1).view(-1, 1, 1, 1)
        span = upper - lower
        normalized_maps = ((scale_maps - lower) / span.clamp_min(1e-8)).clamp(
            0.0,
            1.0,
        )
        degenerate = span <= 1e-8
        scale_maps = torch.where(
            degenerate,
            torch.full_like(normalized_maps, 0.5),
            normalized_maps,
        )

        num_channels = signals.shape[1]
        order = torch.tensor(
            _pair_covering_order(num_channels),
            dtype=torch.long,
            device=signals.device,
        )
        row_weights = _inverse_occurrence_row_weights(
            order,
            num_channels,
            scale_maps.dtype,
        )
        ordered_maps = scale_maps.index_select(2, order)
        weighted_maps = ordered_maps * row_weights.view(1, 1, -1, 1)
        ordered_maps = torch.where(degenerate, ordered_maps, weighted_maps)

        image = F.interpolate(
            ordered_maps,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        return image.clamp(0.0, 1.0)

    def forward(self, signals):
        """Render tensors or NumPy-compatible arrays without a host round trip."""
        if torch.is_tensor(signals):
            output_dtype = (
                signals.dtype if signals.is_floating_point() else torch.float32
            )
            working_dtype = (
                torch.float64 if signals.dtype == torch.float64 else torch.float32
            )
            signals_tensor = signals.to(dtype=working_dtype)
        else:
            output_dtype = torch.float32
            signals_tensor = torch.as_tensor(signals, dtype=torch.float32)

        if signals_tensor.ndim == 2:
            signals_tensor = signals_tensor.unsqueeze(0)
        elif signals_tensor.ndim != 3:
            raise ValueError(
                "signals must have shape (channels, time) or "
                f"(batch, channels, time), got {tuple(signals_tensor.shape)}."
            )
        if signals_tensor.shape[0] < 1:
            raise ValueError("signals batch dimension must be non-empty.")
        if signals_tensor.shape[1] < 1 or signals_tensor.shape[2] < 1:
            raise ValueError(
                "Signal dimensions must be non-empty, got "
                f"{tuple(signals_tensor.shape)}."
            )

        return self._render_batch(signals_tensor).to(dtype=output_dtype)


class TemporalGranularityGraphBank(nn.Module):
    """Render a fixed bank of temporal-granularity graph experts.

    The output layout is ``(batch, regimes, RGB, height, width)``.  Each regime
    is an ordinary :class:`MedformerGraphRenderer`; keeping the existing renderer
    intact makes a bank entry that uses ``(2, 4, 8)`` exactly equal to the
    legacy graph.  A one-element entry such as ``(8,)`` is a true single-scale
    expert rendered as neutral RGB.  The bank intentionally has no trainable
    parameters.  The selector is trained after frozen vision-feature
    extraction, where it is compatible with the project's feature-cache
    workflow.
    """

    def __init__(
        self,
        patch_length_bank=((1, 2, 4), (2, 4, 8), (4, 8, 16)),
        img_size=224,
        channel_mix=0.35,
        router_temperature=0.2,
        router_mix=0.5,
    ):
        super().__init__()
        patch_length_bank = tuple(
            tuple(int(length) for length in patch_lengths)
            for patch_lengths in patch_length_bank
        )
        if not patch_length_bank:
            raise ValueError("A granularity bank requires at least one regime.")
        if len(set(patch_length_bank)) != len(patch_length_bank):
            raise ValueError(
                "Granularity-bank regimes must be unique, got "
                f"{patch_length_bank}."
            )

        # MedformerGraphRenderer performs per-regime length/range validation.
        self.graphs = nn.ModuleList(
            [
                MedformerGraphRenderer(
                    patch_lengths=patch_lengths,
                    img_size=img_size,
                    channel_mix=channel_mix,
                    router_temperature=router_temperature,
                    router_mix=router_mix,
                )
                for patch_lengths in patch_length_bank
            ]
        )
        self.patch_length_bank = patch_length_bank
        self.img_size = int(img_size)

    def forward(self, signals):
        return torch.stack([graph(signals) for graph in self.graphs], dim=1)


# Backward-compatible public names.
MedActitivy_graph = MedformerGraphRenderer
MedActivityGraph = MedformerGraphRenderer
MedActivityGranularityBank = TemporalGranularityGraphBank
MultiGranularityGraphBank = TemporalGranularityGraphBank


__all__ = [
    "MedformerGraphRenderer",
    "TemporalGranularityGraphBank",
    "MultiGranularityGraphBank",
    "MedActivityGraph",
    "MedActitivy_graph",
    "MedActivityGranularityBank",
]
