"""Paper-faithful multi-column Activity Graph rendering.

The Activity Graph of Yang et al. is a waveform image. Algorithm 1 extends a
deterministic signal-ID sequence until every unordered pair of signals appears
next to each other. Algorithm 3 then draws, for every sequence position, the
previous, current, and next signal in three columns. It does not construct a
weighted adjacency matrix and it does not propagate values between channels.

Reference:
    P. Yang, C. Yang, V. Lanfranchi, and F. Ciravegna, "Activity Graph Based
    Convolutional Neural Network for Human Activity Recognition Using
    Acceleration and Gyroscope Data," IEEE TII 18(10), 2022.
    https://doi.org/10.1109/TII.2022.3142315
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


PAPER_ACTIVITY_GRAPH_DOI = "10.1109/TII.2022.3142315"
PAPER_ACTIVITY_GRAPH_LAYOUT = "yang2022_algorithm1_algorithm3_multicolumn"
PAPER_ACTIVITY_GRAPH_CANVAS_SIZE = 360


def _edge(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


@lru_cache(maxsize=None)
def paper_signal_order(num_channels: int) -> tuple[int, ...]:
    """Return the exact deterministic order described by paper Algorithm 1.

    IDs in the paper are one-based; this implementation uses zero-based tensor
    indices. The initial sequence is ``0, 1, ..., C-1`` and the paper's scan
    rule appends IDs until every unordered pair occurs in adjacent positions.
    For the four-signal example this returns ``(0,1,2,3,0,2,3,1)``.
    """

    if isinstance(num_channels, bool) or not isinstance(num_channels, int):
        raise TypeError("num_channels must be an integer")
    if num_channels < 1:
        raise ValueError("an Activity Graph requires at least one signal")
    if num_channels == 1:
        return (0,)

    order = list(range(num_channels))
    covered = {
        _edge(left, right)
        for left, right in zip(order, order[1:])
        if left != right
    }
    target_count = num_channels * (num_channels - 1) // 2

    def pair_is_covered(left: int, right: int) -> bool:
        return left == right or _edge(left, right) in covered

    def append(signal_id: int) -> None:
        if not 0 <= signal_id < num_channels:
            raise RuntimeError(
                "paper Algorithm 1 produced an out-of-range signal index"
            )
        previous = order[-1]
        order.append(signal_id)
        if previous != signal_id:
            covered.add(_edge(previous, signal_id))

    # Literal zero-based transcription of Algorithm 1. Keeping this scan rule,
    # rather than substituting another Euler walk, is necessary to reproduce
    # the published four-signal sequence and its three-column layout.
    current = num_channels - 1
    candidate = 0
    while len(covered) < target_count:
        if current == candidate:
            candidate += 1
            if candidate >= num_channels:
                candidate -= 2
                append(candidate)
                current = candidate
                candidate = 0

        if pair_is_covered(current, candidate):
            candidate += 1
            if candidate >= num_channels:
                current += 1
                candidate = 0
                while pair_is_covered(current, candidate):
                    candidate += 1
                    if current == candidate:
                        candidate += 1
                    if candidate >= num_channels:
                        current += 1
                        break
                candidate = 0
                append(current)
        else:
            append(candidate)
            current = candidate
            candidate = 0

    return tuple(order)


def paper_multicolumn_indices(num_channels: int) -> tuple[tuple[int, int, int], ...]:
    """Return paper Algorithm 3 rows as ``(previous, current, next)`` IDs."""

    order = paper_signal_order(num_channels)
    return tuple(
        (order[row - 1], signal_id, order[(row + 1) % len(order)])
        for row, signal_id in enumerate(order)
    )


# Historical internal name retained for imports outside this repository. Its
# behavior now follows the cited paper instead of the former custom Euler walk.
def _pair_covering_order(num_channels: int) -> list[int]:
    return list(paper_signal_order(num_channels))


def _coerce_valid_lengths(
    valid_lengths: Optional[torch.Tensor],
    batch_size: int,
    time_steps: int,
    device: torch.device,
) -> torch.Tensor:
    if valid_lengths is None:
        return torch.full(
            (batch_size,), time_steps, dtype=torch.long, device=device
        )
    lengths = torch.as_tensor(valid_lengths, device=device)
    if lengths.ndim != 1 or lengths.shape[0] != batch_size:
        raise ValueError(f"valid_lengths must have shape [{batch_size}]")
    if lengths.dtype == torch.bool or lengths.is_floating_point():
        raise ValueError("valid_lengths must use an integer dtype")
    lengths = lengths.to(dtype=torch.long)
    if bool(((lengths < 1) | (lengths > time_steps)).any()):
        raise ValueError(f"valid_lengths entries must lie in [1, {time_steps}]")
    return lengths


def block_average_waveform(
    signals: torch.Tensor,
    valid_lengths: torch.Tensor,
    block_length: int,
) -> torch.Tensor:
    """Return a piecewise-mean waveform used by optional granularity experts."""

    if block_length <= 1:
        return signals
    batch_size, channels, time_steps = signals.shape
    sample_positions = torch.arange(time_steps, device=signals.device)
    valid = sample_positions[None, :] < valid_lengths[:, None]
    block_ids = sample_positions.div(block_length, rounding_mode="floor")
    block_count = int(block_ids[-1].item()) + 1
    expanded_ids = block_ids.view(1, 1, -1).expand(batch_size, channels, -1)
    sums = signals.new_zeros((batch_size, channels, block_count))
    sums.scatter_add_(2, expanded_ids, signals * valid[:, None, :])
    counts = signals.new_zeros((batch_size, 1, block_count))
    counts.scatter_add_(
        2,
        block_ids.view(1, 1, -1).expand(batch_size, 1, -1),
        valid[:, None, :].to(dtype=signals.dtype),
    )
    means = sums / counts.clamp_min(1.0)
    reconstructed = means.gather(2, expanded_ids)
    return reconstructed.masked_fill(~valid[:, None, :], 0.0)


def _normalize_waveform_lanes(
    signals: torch.Tensor,
    valid_lengths: torch.Tensor,
    vertical_margin: float,
) -> torch.Tensor:
    """Map valid signal lanes to plot bounds while ignoring padded tails.

    The paper does not report plotting bounds. Per-signal min/max scaling with
    a small plotting margin is the explicit deterministic choice used here to
    emulate independent waveform axes; it is not presented as a paper formula.
    """

    time_steps = signals.shape[-1]
    sample_positions = torch.arange(time_steps, device=signals.device)
    valid = sample_positions[None, None, :] < valid_lengths[:, None, None]
    finite = torch.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
    minimum = finite.masked_fill(~valid, float("inf")).amin(dim=-1, keepdim=True)
    maximum = finite.masked_fill(~valid, float("-inf")).amax(dim=-1, keepdim=True)
    span = maximum - minimum
    unit = (finite - minimum) / span.clamp_min(1e-8)
    unit = torch.where(span <= 1e-8, torch.full_like(unit, 0.5), unit)
    unit = unit.clamp(0.0, 1.0)
    scaled = vertical_margin + (1.0 - 2.0 * vertical_margin) * unit
    return scaled.masked_fill(~valid, 0.5)


def render_paper_activity_graph(
    signals: torch.Tensor,
    *,
    valid_lengths: Optional[torch.Tensor] = None,
    output_size: int = 224,
    canvas_size: int = PAPER_ACTIVITY_GRAPH_CANVAS_SIZE,
    line_width: float = 1.0,
    vertical_margin: float = 0.05,
) -> torch.Tensor:
    """Differentiably draw paper Algorithm 1+3 waveform graphs.

    ``canvas_size=360`` reproduces the paper's reported square raster. The
    result is resized only at the visual-backbone boundary when ``output_size``
    differs from 360.
    """

    if not torch.is_tensor(signals) or signals.ndim != 3:
        raise ValueError("signals must be a tensor with shape [B,C,T]")
    if min(signals.shape) < 1:
        raise ValueError("signals must have non-empty batch/channel/time axes")
    if not signals.is_floating_point():
        raise ValueError("signals must be floating point")
    for name, value, minimum in (
        ("output_size", output_size, 1),
        ("canvas_size", canvas_size, 3),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    if canvas_size % 3 != 0:
        raise ValueError("canvas_size must be divisible by three for Algorithm 3")
    if not math.isfinite(line_width) or line_width <= 0.0:
        raise ValueError("line_width must be positive and finite")
    if (
        not math.isfinite(vertical_margin)
        or vertical_margin < 0.0
        or vertical_margin >= 0.5
    ):
        raise ValueError("vertical_margin must lie in [0, 0.5)")

    output_dtype = signals.dtype
    working_dtype = torch.float64 if signals.dtype == torch.float64 else torch.float32
    working = signals.to(dtype=working_dtype)
    batch_size, channels, time_steps = working.shape
    lengths = _coerce_valid_lengths(
        valid_lengths, batch_size, time_steps, working.device
    )
    normalized = _normalize_waveform_lanes(
        working, lengths, float(vertical_margin)
    )

    layout = torch.tensor(
        paper_multicolumn_indices(channels),
        dtype=torch.long,
        device=working.device,
    )
    row_count = layout.shape[0]
    # Rasterize every original sample-to-sample segment. Drawing only a
    # resampled value at each image column can erase narrow spikes and leaves
    # steep waveforms disconnected. At least eight vertical samples per row
    # retain all Algorithm 1 rows before reducing the reference canvas.
    panel_width = canvas_size // 3
    lane_height = max(8, math.ceil(canvas_size / row_count))
    x_grid = torch.arange(panel_width, device=working.device, dtype=working.dtype)
    y_grid = torch.arange(lane_height, device=working.device, dtype=working.dtype)
    grid_x = x_grid.view(1, 1, 1, panel_width, 1)
    grid_y = y_grid.view(1, 1, lane_height, 1, 1)
    samples_x = torch.arange(time_steps, device=working.device, dtype=working.dtype)
    samples_x = samples_x[None, :] * (panel_width - 1) / (lengths[:, None] - 1).clamp_min(1)
    samples_y = (1.0 - normalized) * (lane_height - 1)
    # A one-sample record is a horizontal constant trace across the lane.
    if time_steps == 1:
        samples_x = torch.stack((samples_x[:, 0], samples_x[:, 0] + panel_width - 1), -1)
        samples_y = samples_y.expand(-1, -1, 2)
    else:
        singleton = lengths == 1
        samples_x = samples_x.clone()
        samples_x[:, 1] = torch.where(singleton, panel_width - 1, samples_x[:, 1])
        samples_y = samples_y.clone()
        samples_y[:, :, 1] = torch.where(singleton[:, None], samples_y[:, :, 0], samples_y[:, :, 1])
    distance_squared = None
    segment_count = samples_x.shape[-1] - 1
    for start in range(0, segment_count, 32):
        stop = min(start + 32, segment_count)
        ax = samples_x[:, start:stop].view(batch_size, 1, 1, 1, -1)
        ay = samples_y[:, :, start:stop].unsqueeze(2).unsqueeze(2)
        dx = (samples_x[:, start + 1:stop + 1] - samples_x[:, start:stop]).view(batch_size, 1, 1, 1, -1)
        dy = (samples_y[:, :, start + 1:stop + 1] - samples_y[:, :, start:stop]).unsqueeze(2).unsqueeze(2)
        projection = ((grid_x - ax) * dx + (grid_y - ay) * dy) / (dx.square() + dy.square()).clamp_min(1e-12)
        projection = projection.clamp(0.0, 1.0)
        distances = (grid_x - ax - projection * dx).square() + (grid_y - ay - projection * dy).square()
        segment_ids = torch.arange(start, stop, device=working.device)
        valid_segments = segment_ids[None, :] < (lengths - 1).clamp_min(1)[:, None]
        distances = distances.masked_fill(~valid_segments[:, None, None, None, :], float("inf"))
        nearest = distances.amin(dim=-1)
        distance_squared = nearest if distance_squared is None else torch.minimum(distance_squared, nearest)
    sigma = max(float(line_width) / 2.354820045, 0.25)
    lanes = 1.0 - torch.exp(-0.5 * distance_squared / sigma**2)
    panels = lanes.index_select(1, layout.reshape(-1)).reshape(
        batch_size, row_count, 3, lane_height, panel_width
    )
    gray = panels.permute(0, 1, 3, 2, 4).reshape(
        batch_size, 1, row_count * lane_height, canvas_size
    )
    if gray.shape[-2] != canvas_size:
        gray = F.interpolate(gray, size=(canvas_size, canvas_size), mode="area")
    image = gray.expand(-1, 3, -1, -1)
    if output_size != canvas_size:
        image = F.interpolate(
            image, size=(output_size, output_size), mode="bilinear", align_corners=False
        )
    return image.clamp(0.0, 1.0).to(dtype=output_dtype)


class ActivityGraphRenderer(nn.Module):
    """Render the paper's deterministic three-column waveform Activity Graph."""

    def __init__(
        self,
        patch_lengths=None,
        img_size: int = 224,
        channel_mix: float | None = None,
        router_temperature: float | None = None,
        router_mix: float | None = None,
        *,
        canvas_size: int = PAPER_ACTIVITY_GRAPH_CANVAS_SIZE,
        line_width: float = 1.0,
        vertical_margin: float = 0.05,
    ) -> None:
        super().__init__()
        # Historical arguments remain accepted so old entry points construct
        # cleanly. They no longer alter graph topology or mix channels because
        # the cited method contains neither operation.
        if patch_lengths is None:
            normalized_lengths: tuple[int, ...] = ()
        elif isinstance(patch_lengths, int):
            normalized_lengths = (int(patch_lengths),)
        else:
            normalized_lengths = tuple(int(value) for value in patch_lengths)
        if any(value <= 0 for value in normalized_lengths):
            raise ValueError("patch lengths must be positive")
        if len(normalized_lengths) > 1:
            raise ValueError("The paper waveform renderer accepts one smoothing scale; legacy RGB scale triplets are unsupported")
        self.patch_lengths = normalized_lengths
        self.granularity = (
            normalized_lengths[0] if len(normalized_lengths) == 1 else None
        )
        self.img_size = int(img_size)
        self.canvas_size = int(canvas_size)
        self.line_width = float(line_width)
        self.vertical_margin = float(vertical_margin)

    def provenance(self) -> dict[str, object]:
        return {
            "reference_doi": PAPER_ACTIVITY_GRAPH_DOI,
            "layout": PAPER_ACTIVITY_GRAPH_LAYOUT,
            "signal_order": "paper_algorithm_1_zero_based_transcription",
            "columns": "paper_algorithm_3_previous_current_next_cyclic",
            "render": "black_waveforms_on_white_rgb",
            "reference_canvas_size": self.canvas_size,
            "output_size": self.img_size,
            "line_width": self.line_width,
            "line_width_units": "intermediate_lane_pixels",
            "stroke": "minimum_distance_to_all_valid_sample_segments",
            "minimum_lane_height": 8,
            "reference_canvas_reduction": "area",
            "plot_bounds": "per_signal_minmax_with_explicit_vertical_margin",
            "vertical_margin": self.vertical_margin,
            "legacy_granularity": self.granularity,
            "channel_propagation": False,
        }

    def forward(
        self,
        signals,
        valid_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if torch.is_tensor(signals):
            tensor = signals
        else:
            tensor = torch.as_tensor(signals, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError(
                "signals must have shape [C,T] or [B,C,T], got "
                f"{tuple(tensor.shape)}"
            )
        if not tensor.is_floating_point():
            tensor = tensor.float()
        lengths = _coerce_valid_lengths(
            valid_lengths, tensor.shape[0], tensor.shape[2], tensor.device
        )
        if self.granularity is not None:
            tensor = block_average_waveform(tensor, lengths, self.granularity)
        return render_paper_activity_graph(
            tensor,
            valid_lengths=lengths,
            output_size=self.img_size,
            canvas_size=self.canvas_size,
            line_width=self.line_width,
            vertical_margin=self.vertical_margin,
        )


class TemporalGranularityGraphBank(nn.Module):
    """Render paper-layout waveform graphs for fixed smoothing granularities."""

    def __init__(
        self,
        patch_length_bank=((4,), (8,), (16,)),
        img_size=224,
        channel_mix=None,
        router_temperature=None,
        router_mix=None,
        *,
        canvas_size=PAPER_ACTIVITY_GRAPH_CANVAS_SIZE,
        line_width=1.0,
        vertical_margin=0.05,
    ) -> None:
        super().__init__()
        bank = tuple(tuple(int(value) for value in regime) for regime in patch_length_bank)
        if not bank or len(set(bank)) != len(bank):
            raise ValueError("granularity-bank regimes must be non-empty and unique")
        self.graphs = nn.ModuleList(
            [
                ActivityGraphRenderer(
                    patch_lengths=regime,
                    img_size=img_size,
                    canvas_size=canvas_size,
                    line_width=line_width,
                    vertical_margin=vertical_margin,
                )
                for regime in bank
            ]
        )
        self.patch_length_bank = bank
        self.img_size = int(img_size)

    def forward(self, signals):
        return torch.stack([graph(signals) for graph in self.graphs], dim=1)


# Backward-compatible public names.
MedActitivy_graph = ActivityGraphRenderer
MedActivityGraph = ActivityGraphRenderer
MedActivityGranularityBank = TemporalGranularityGraphBank
MultiGranularityGraphBank = TemporalGranularityGraphBank


__all__ = [
    "ActivityGraphRenderer",
    "MedActivityGraph",
    "MedActivityGranularityBank",
    "MedActitivy_graph",
    "MultiGranularityGraphBank",
    "PAPER_ACTIVITY_GRAPH_CANVAS_SIZE",
    "PAPER_ACTIVITY_GRAPH_DOI",
    "PAPER_ACTIVITY_GRAPH_LAYOUT",
    "TemporalGranularityGraphBank",
    "block_average_waveform",
    "paper_multicolumn_indices",
    "paper_signal_order",
    "render_paper_activity_graph",
]
