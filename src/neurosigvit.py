from abc import ABC, abstractmethod
import os

import einops
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import Resize
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    ViTMAEForPreTraining,
)

from src.medformer_graph import (
    MedformerGraphRenderer,
    TemporalGranularityGraphBank,
)


OPENCLIP_LAION_MODELS = {
    "clip-vit-b-32-laion2b-s34b-b79k": ("ViT-B-32", "laion2b_s34b_b79k"),
    "clip-vit-b-16-laion2b-s34b-b88k": ("ViT-B-16", "laion2b_s34b_b88k"),
    "clip-vit-l-14-laion2b-s32b-b82k": ("ViT-L-14", "laion2b_s32b_b82k"),
    "clip-vit-h-14-laion2b-s32b-b79k": ("ViT-H-14", "laion2b_s32b_b79k"),
}


def _adaptive_granularity_metadata(base_patch_lengths, granularity_bank):
    """Resolve legacy-base metadata without rejecting single-scale banks."""
    if len(granularity_bank) < 2:
        raise ValueError(
            "Adaptive granularity requires at least two candidate regimes."
        )
    if all(len(regime) == 1 for regime in granularity_bank):
        return -1, "single_scale_granularity_bank_scale_major_flat_v3"
    if granularity_bank.count(base_patch_lengths) != 1:
        raise ValueError(
            "A legacy/mixed adaptive granularity bank must contain the base "
            f"regime {base_patch_lengths} exactly once, got {granularity_bank}."
        )
    return (
        granularity_bank.index(base_patch_lengths),
        "granularity_bank_scale_major_flat_v1",
    )


def get_optimal_order(n):
    if n < 0:
        raise ValueError(f"Number of signal channels must be non-negative, got {n}.")
    if n <= 1:
        return list(range(n))

    total_pairs = n * (n - 1) // 2
    visited_pairs = set()
    id_list = [0]

    while len(visited_pairs) < total_pairs:
        current = id_list[-1]
        next_id = None

        for candidate in range(n):
            if candidate == current:
                continue
            pair = tuple(sorted((current, candidate)))
            if pair not in visited_pairs:
                next_id = candidate
                break

        if next_id is None:
            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) not in visited_pairs:
                        next_id = i if current != i else j
                        break
                if next_id is not None:
                    break

        pair = tuple(sorted((current, next_id)))
        visited_pairs.add(pair)
        id_list.append(next_id)

    return id_list


def build_single_column_graph(signals, id_list):
    signals = np.asarray(signals)
    if signals.ndim != 2:
        raise ValueError(f"signals must have shape (n, T), got {signals.shape}.")

    return signals[np.asarray(id_list, dtype=np.int64)]


def build_multicolumn_graph(signals, id_list):
    signals = np.asarray(signals)
    if signals.ndim != 2:
        raise ValueError(f"signals must have shape (n, T), got {signals.shape}.")

    rows = []
    for k, signal_id in enumerate(id_list):
        left = signals[id_list[k - 1]]
        center = signals[signal_id]
        right = signals[id_list[(k + 1) % len(id_list)]]
        rows.append(np.concatenate([left, center, right], axis=0))

    return np.stack(rows, axis=0)


def _draw_waveform(canvas, signal, x0, y0, width, height, value_min, value_max, line_width):
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 1:
        raise ValueError(f"signal must be one-dimensional, got {signal.shape}.")
    if width <= 0 or height <= 0:
        return

    if signal.size == 1:
        values = np.full(width, signal[0], dtype=np.float32)
    else:
        source_x = np.linspace(0.0, 1.0, num=signal.size, dtype=np.float32)
        target_x = np.linspace(0.0, 1.0, num=width, dtype=np.float32)
        values = np.interp(target_x, source_x, signal).astype(np.float32)

    value_range = value_max - value_min
    if value_range <= 1e-8:
        y = np.full(width, y0 + (height - 1) / 2.0, dtype=np.float32)
    else:
        normalized = (values - value_min) / (value_range + 1e-8)
        y = y0 + (1.0 - np.clip(normalized, 0.0, 1.0)) * (height - 1)

    radius = max(0, int(np.ceil(line_width)))
    for x in range(width):
        y_center = int(round(y[x]))
        top = max(y0, y_center - radius)
        bottom = min(y0 + height - 1, y_center + radius)
        canvas[top : bottom + 1, x0 + x] = 0.0

        if x == 0:
            continue

        prev_y = y[x - 1]
        cur_y = y[x]
        y_start = int(np.floor(min(prev_y, cur_y))) - radius
        y_end = int(np.ceil(max(prev_y, cur_y))) + radius
        y_start = max(y0, y_start)
        y_end = min(y0 + height - 1, y_end)
        if y_end >= y_start:
            canvas[y_start : y_end + 1, x0 + x] = 0.0


def render_activity_waveform_graph(
    signals,
    mode="multicolumn",
    id_list=None,
    img_size=224,
    min_strip_height=8,
    line_width=1.0,
):
    signals = np.asarray(signals, dtype=np.float32)
    if signals.ndim != 2:
        raise ValueError(f"signals must have shape (n, T), got {signals.shape}.")

    if id_list is None:
        id_list = get_optimal_order(signals.shape[0])

    if mode == "single_column":
        rows = [[signal_id] for signal_id in id_list]
    elif mode == "multicolumn":
        rows = [
            [id_list[k - 1], signal_id, id_list[(k + 1) % len(id_list)]]
            for k, signal_id in enumerate(id_list)
        ]
    else:
        raise ValueError(f"Unsupported activity graph mode {mode}.")

    columns = len(rows[0]) if rows else 1
    row_height = max(min_strip_height, int(np.ceil(img_size / max(len(rows), 1))))
    canvas_height = max(img_size, row_height * max(len(rows), 1))
    canvas_width = img_size
    canvas = np.ones((canvas_height, canvas_width), dtype=np.float32)

    value_min = float(signals.min())
    value_max = float(signals.max())
    column_edges = np.linspace(0, canvas_width, columns + 1, dtype=np.int64)

    for row_idx, row_signal_ids in enumerate(rows):
        y0 = row_idx * row_height
        for col_idx, signal_id in enumerate(row_signal_ids):
            x0 = int(column_edges[col_idx])
            x1 = int(column_edges[col_idx + 1])
            _draw_waveform(
                canvas=canvas,
                signal=signals[signal_id],
                x0=x0,
                y0=y0,
                width=x1 - x0,
                height=row_height,
                value_min=value_min,
                value_max=value_max,
                line_width=line_width,
            )

    return canvas


def generate_activity_graph(signals, mode="multicolumn", id_list=None):
    signals = np.asarray(signals)
    if signals.ndim != 2:
        raise ValueError(f"signals must have shape (n, T), got {signals.shape}.")

    if id_list is None:
        id_list = get_optimal_order(signals.shape[0])

    if mode == "single_column":
        return build_single_column_graph(signals, id_list)
    if mode == "multicolumn":
        return build_multicolumn_graph(signals, id_list)

    raise ValueError(f"Unsupported activity graph mode {mode}.")


def preprocess_graph(signals, mode="multicolumn", img_size=224, render="waveform"):
    """
    signals: np.ndarray, shape (n, T), or torch.Tensor, shape (B, n, T)
    returns: torch.Tensor, shape (B, 3, img_size, img_size)
    """
    device = signals.device if torch.is_tensor(signals) else None
    dtype = signals.dtype if torch.is_tensor(signals) else torch.float32

    if torch.is_tensor(signals):
        signals_np = signals.detach().cpu().numpy()
    else:
        signals_np = np.asarray(signals)

    if signals_np.ndim == 2:
        signals_np = signals_np[None, ...]
    elif signals_np.ndim != 3:
        raise ValueError(f"signals must have shape (n, T) or (B, n, T), got {signals_np.shape}.")

    graphs = []
    order_cache = {}
    for sample in signals_np:
        n = sample.shape[0]
        id_list = order_cache.setdefault(n, get_optimal_order(n))
        if render == "waveform":
            graph = render_activity_waveform_graph(
                sample,
                mode=mode,
                id_list=id_list,
                img_size=img_size,
            )
        elif render == "matrix":
            graph = generate_activity_graph(sample, mode=mode, id_list=id_list).astype(
                np.float32, copy=False
            )
            graph_min = graph.min()
            graph_max = graph.max()
            graph = (graph - graph_min) / (graph_max - graph_min + 1e-8)
        else:
            raise ValueError(f"Unsupported activity graph render {render}.")
        graphs.append(graph)

    graph_tensor = torch.from_numpy(np.stack(graphs, axis=0)).unsqueeze(1)
    graph_tensor = F.interpolate(
        graph_tensor,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )
    graph_tensor = graph_tensor.repeat(1, 3, 1, 1)

    if device is not None:
        graph_tensor = graph_tensor.to(device=device, dtype=dtype)

    return graph_tensor


LINE_PLOT_COLORS = np.asarray(
    [
        (0.1216, 0.4667, 0.7059),
        (1.0000, 0.4980, 0.0549),
        (0.1725, 0.6275, 0.1725),
        (0.8392, 0.1529, 0.1569),
        (0.5804, 0.4039, 0.7412),
        (0.5490, 0.3373, 0.2941),
        (0.8902, 0.4667, 0.7608),
        (0.4980, 0.4980, 0.4980),
        (0.7373, 0.7412, 0.1333),
        (0.0902, 0.7451, 0.8118),
    ],
    dtype=np.float32,
)


def render_multichannel_lineplot(signals, img_size=224, line_width=1.0):
    """Overlay every channel as a colored line on one white RGB canvas."""
    signals = np.asarray(signals, dtype=np.float32)
    if signals.ndim != 2:
        raise ValueError(f"signals must have shape (n, T), got {signals.shape}.")
    if signals.shape[0] == 0:
        raise ValueError("signals must contain at least one channel.")

    canvas = np.ones((3, img_size, img_size), dtype=np.float32)
    value_min = float(signals.min())
    value_max = float(signals.max())
    for channel_idx, signal in enumerate(signals):
        mask = np.ones((img_size, img_size), dtype=np.float32)
        _draw_waveform(
            canvas=mask,
            signal=signal,
            x0=0,
            y0=0,
            width=img_size,
            height=img_size,
            value_min=value_min,
            value_max=value_max,
            line_width=line_width,
        )
        line_pixels = mask < 0.5
        color = LINE_PLOT_COLORS[channel_idx % len(LINE_PLOT_COLORS)]
        canvas[:, line_pixels] = color[:, None]
    return canvas


def render_stacked_multichannel_lineplot(
    signals,
    img_size=224,
    line_width=1.0,
):
    """Render every channel in its own fixed vertical lane on one RGB canvas.

    Each channel is scaled independently using only its finite values.  Invalid
    values inside a channel are linearly interpolated from the finite samples;
    an entirely invalid or constant channel is rendered through the centre of
    its lane.  Channel identity is therefore carried by the lane position and
    does not depend on a repeating colour palette.
    """
    if not isinstance(img_size, (int, np.integer)) or img_size <= 0:
        raise ValueError(f"img_size must be a positive integer, got {img_size}.")
    if not np.isfinite(line_width) or line_width <= 0.0:
        raise ValueError(f"line_width must be finite and positive, got {line_width}.")

    if torch.is_tensor(signals):
        signals = signals.detach().to(device="cpu", dtype=torch.float32).numpy()
    signals = np.asarray(signals, dtype=np.float32)
    if signals.ndim != 2:
        raise ValueError(f"signals must have shape (n, T), got {signals.shape}.")

    num_channels, time_steps = signals.shape
    if num_channels == 0 or time_steps == 0:
        raise ValueError(
            "signals must contain at least one channel and one time step, "
            f"got {signals.shape}."
        )
    if num_channels > img_size:
        raise ValueError(
            "A fixed vertical lane requires at least one image row per channel; "
            f"got {num_channels} channels for img_size={img_size}."
        )

    canvas = np.ones((3, img_size, img_size), dtype=np.float32)
    lane_edges = np.linspace(
        0,
        img_size,
        num=num_channels + 1,
        dtype=np.int64,
    )
    sample_positions = np.arange(time_steps, dtype=np.float32)

    for channel_idx, signal in enumerate(signals):
        finite = np.isfinite(signal)
        if not finite.any():
            safe_signal = np.zeros(time_steps, dtype=np.float32)
            value_min = value_max = 0.0
        else:
            valid_positions = sample_positions[finite]
            valid_values = signal[finite]
            value_min = float(valid_values.min())
            value_max = float(valid_values.max())
            if finite.all():
                safe_signal = signal
            elif valid_values.size == 1:
                safe_signal = np.full(
                    time_steps,
                    valid_values[0],
                    dtype=np.float32,
                )
            else:
                safe_signal = np.interp(
                    sample_positions,
                    valid_positions,
                    valid_values,
                ).astype(np.float32)

        lane_start = int(lane_edges[channel_idx])
        lane_end = int(lane_edges[channel_idx + 1])
        lane_height = lane_end - lane_start
        lane_mask = np.ones((img_size, img_size), dtype=np.float32)
        _draw_waveform(
            canvas=lane_mask,
            signal=safe_signal,
            x0=0,
            y0=lane_start,
            width=img_size,
            height=lane_height,
            value_min=value_min,
            value_max=value_max,
            line_width=line_width,
        )
        line_pixels = lane_mask < 0.5
        canvas[:, line_pixels] = 0.0

    return np.nan_to_num(
        canvas,
        nan=1.0,
        posinf=1.0,
        neginf=0.0,
    ).clip(0.0, 1.0)


def preprocess_multichannel_lineplot(signals, img_size=224):
    """Render a batch of multichannel samples as ordinary RGB line plots."""
    device = signals.device if torch.is_tensor(signals) else None
    dtype = signals.dtype if torch.is_tensor(signals) else torch.float32

    if torch.is_tensor(signals):
        signals_np = signals.detach().cpu().numpy()
    else:
        signals_np = np.asarray(signals)

    if signals_np.ndim == 2:
        signals_np = signals_np[None, ...]
    elif signals_np.ndim != 3:
        raise ValueError(
            "signals must have shape (n, T) or (B, n, T), "
            f"got {signals_np.shape}."
        )

    images = []
    for sample in signals_np:
        images.append(render_multichannel_lineplot(sample, img_size=img_size))

    image_tensor = torch.from_numpy(np.stack(images, axis=0))
    image_tensor = F.interpolate(
        image_tensor,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )
    if device is not None:
        image_tensor = image_tensor.to(device=device, dtype=dtype)

    return image_tensor


def preprocess_stacked_multichannel_lineplot(signals, img_size=224):
    """Render ``(channels, time)`` or ``(batch, channels, time)`` as lanes."""
    device = signals.device if torch.is_tensor(signals) else None
    if torch.is_tensor(signals) and signals.is_floating_point():
        output_dtype = signals.dtype
    else:
        output_dtype = torch.float32

    if torch.is_tensor(signals):
        signals_np = (
            signals.detach().to(device="cpu", dtype=torch.float32).numpy()
        )
    else:
        signals_np = np.asarray(signals)

    if signals_np.ndim == 2:
        signals_np = signals_np[None, ...]
    elif signals_np.ndim != 3:
        raise ValueError(
            "signals must have shape (n, T) or (B, n, T), "
            f"got {signals_np.shape}."
        )
    if signals_np.shape[0] == 0:
        raise ValueError("signals batch dimension must be non-empty.")

    images = [
        render_stacked_multichannel_lineplot(sample, img_size=img_size)
        for sample in signals_np
    ]
    image_tensor = torch.from_numpy(np.stack(images, axis=0))
    image_tensor = image_tensor.to(dtype=output_dtype)
    if device is not None:
        image_tensor = image_tensor.to(device=device)

    return image_tensor.clamp(0.0, 1.0)


def get_openclip_config(model_name):
    model_key = model_name.lower()
    model_key = os.path.basename(os.path.normpath(model_key))

    if model_key in OPENCLIP_LAION_MODELS:
        return OPENCLIP_LAION_MODELS[model_key]

    raise ValueError(f"Unsupported OpenCLIP model {model_name}.")


def find_openclip_checkpoint(model_dir):
    preferred_names = [
        "open_clip_pytorch_model.bin",
        "open_clip_model.bin",
        "pytorch_model.bin",
        "model.safetensors",
        "model.bin",
        "model.pt",
        "model.pth",
    ]

    for name in preferred_names:
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            return path

    for name in os.listdir(model_dir):
        if name.endswith((".safetensors", ".bin", ".pt", ".pth")):
            return os.path.join(model_dir, name)

    raise FileNotFoundError(
        f"No OpenCLIP checkpoint file found in {model_dir}. "
        "Expected a .bin, .pt, .pth, or .safetensors file."
    )


def get_processor_vit(model_name):
    model_key = model_name.lower()

    if "clip" in model_key:
        if "laion" in model_key or os.path.isdir(model_name):
            openclip_model_name, openclip_pretrained = get_openclip_config(model_name)
            if os.path.isdir(model_name):
                openclip_pretrained = find_openclip_checkpoint(model_name)

            model, _, processor = open_clip.create_model_and_transforms(
                model_name=openclip_model_name,
                pretrained=openclip_pretrained,
            )
            vit = model.visual
        else:
            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name)
            vit = model.vision_model
    elif "dinov2" in model_key:
        processor = AutoImageProcessor.from_pretrained(model_name)
        vit = AutoModel.from_pretrained(model_name)
    elif "siglip" in model_key:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        vit = model.vision_model
    elif "mae" in model_key:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = ViTMAEForPreTraining.from_pretrained(model_name)
        vit = model.vit
    else:
        raise ValueError(f"Unsupported model {model_name}.")

    return processor, vit


def get_neurosigvit(
    model_name,
    model_layer,
    aggregation,
    stride,
    patch_size,
    image_mode="line_plot",
    med_activity_patch_lengths=(2, 4, 8),
    med_activity_channel_mix=0.35,
    med_activity_router_temperature=0.2,
    med_activity_router_mix=0.5,
    med_activity_adaptive_granularity=False,
    med_activity_granularity_bank=((1, 2, 4), (2, 4, 8), (4, 8, 16)),
):
    processor, vit = get_processor_vit(model_name)

    if hasattr(vit, "transformer") and hasattr(vit.transformer, "resblocks"):
        NeuroSigViTClass = NeuroSigViT_OpenCLIP
    elif hasattr(vit, "encoder") and (
        hasattr(vit.encoder, "layers") or hasattr(vit.encoder, "layer")
    ):
        NeuroSigViTClass = NeuroSigViT_HF
    else:
        raise ValueError("Unsupported model structure.")

    neurosigvit = NeuroSigViTClass(
        processor=processor,
        vit=vit,
        layer_idx=model_layer,
        aggregation=aggregation,
        patch_size=patch_size,
        stride=stride,
        image_mode=image_mode,
    )
    neurosigvit.med_activity_graph = MedformerGraphRenderer(
        patch_lengths=med_activity_patch_lengths,
        channel_mix=med_activity_channel_mix,
        router_temperature=med_activity_router_temperature,
        router_mix=med_activity_router_mix,
    )

    base_patch_lengths = tuple(int(length) for length in med_activity_patch_lengths)
    granularity_bank = tuple(
        tuple(int(length) for length in regime)
        for regime in med_activity_granularity_bank
    )
    neurosigvit.adaptive_granularity_enabled = bool(
        med_activity_adaptive_granularity
    )
    neurosigvit.feature_granularity_count = 1
    neurosigvit.feature_granularity_labels = (
        "-".join(str(length) for length in base_patch_lengths),
    )
    neurosigvit.feature_granularity_base_index = 0
    neurosigvit.feature_layout = "single_embedding_v1"
    neurosigvit.med_activity_granularity_bank = None

    if neurosigvit.adaptive_granularity_enabled:
        if image_mode != "med_activity_graph":
            raise ValueError(
                "Adaptive granularity is only available with "
                "image_mode='med_activity_graph'."
            )
        base_index, feature_layout = _adaptive_granularity_metadata(
            base_patch_lengths,
            granularity_bank,
        )
        neurosigvit.med_activity_granularity_bank = (
            TemporalGranularityGraphBank(
                patch_length_bank=granularity_bank,
                channel_mix=med_activity_channel_mix,
                router_temperature=med_activity_router_temperature,
                router_mix=med_activity_router_mix,
            )
        )
        neurosigvit.feature_granularity_count = len(granularity_bank)
        neurosigvit.feature_granularity_labels = tuple(
            "-".join(str(length) for length in regime)
            for regime in granularity_bank
        )
        neurosigvit.feature_granularity_base_index = base_index
        neurosigvit.feature_layout = feature_layout

    return neurosigvit


class BaseNeuroSigViT(nn.Module, ABC):
    def __init__(
        self, processor, vit, layer_idx, aggregation, patch_size, stride, image_mode
    ):
        super().__init__()
        self.processor = processor
        self.vit = vit
        self.layer_idx = layer_idx
        self.aggregation = aggregation
        self.patch_size = patch_size
        self.stride = stride
        self.image_mode = image_mode
        self.processor = processor
        self.truncate_layers()

    @abstractmethod
    def truncate_layers(self):
        """Truncate transformer layers"""
        pass

    @abstractmethod
    def forward_vit(self, inputs):
        """Forward pass through ViT to extract hidden representations"""
        pass

    def forward(self, inputs):
        if self.image_mode == "activity_graph":
            inputs = preprocess_graph(inputs, mode="multicolumn", render="waveform")
        elif self.image_mode == "med_activity_graph":
            inputs = self.med_activity_graph(inputs)
        elif self.image_mode == "multichannel_line_plot":
            inputs = preprocess_multichannel_lineplot(inputs)
        elif self.image_mode == "activity_matrix":
            inputs = preprocess_graph(inputs, mode="multicolumn", render="matrix")
        elif self.image_mode == "line_plot":
            inputs = self.ts2line_plot_transformation(inputs)
        elif self.image_mode == "segment":
            inputs = self.ts2image_transformation(
                inputs, patch_size=self.patch_size, stride=self.stride
            )
        else:
            raise ValueError(f"Unsupported image mode {self.image_mode}")

        hidden = self.forward_vit(inputs)

        return self.aggregate_hidden_representations(
            hidden, aggregation=self.aggregation
        )

    def forward_granularities(self, inputs):
        """Extract one frozen vision embedding per granularity-bank regime.

        Candidates are encoded sequentially rather than as a ``batch * K``
        tensor so the peak memory of large ViTs stays close to the legacy
        single-graph path.  The returned layout is ``(batch, regimes, dim)``.
        """
        if self.image_mode != "med_activity_graph":
            raise ValueError(
                "Granularity-bank extraction requires med_activity_graph mode."
            )
        if not getattr(self, "adaptive_granularity_enabled", False):
            raise ValueError("Adaptive granularity is not enabled for this model.")
        bank = getattr(self, "med_activity_granularity_bank", None)
        if bank is None:
            raise RuntimeError("Adaptive granularity bank is not initialized.")

        graph_candidates = bank(inputs)
        candidate_embeddings = []
        for candidate_index in range(graph_candidates.shape[1]):
            hidden = self.forward_vit(graph_candidates[:, candidate_index])
            embedding = self.aggregate_hidden_representations(
                hidden,
                aggregation=self.aggregation,
            )
            if embedding.ndim != 2:
                raise ValueError(
                    "Adaptive granularity expects one vector per sample and "
                    f"regime, got {tuple(embedding.shape)}."
                )
            candidate_embeddings.append(embedding)

        return torch.stack(candidate_embeddings, dim=1)

    def aggregate_hidden_representations(self, hidden_states, aggregation):
        if aggregation == "mean":
            if hidden_states.ndim != 3 or hidden_states.shape[1] < 2:
                raise ValueError(
                    "Mean aggregation expects [batch, cls+patches, hidden] with "
                    f"at least one patch token, got {tuple(hidden_states.shape)}."
                )
            # NeuroSigViT mean-pools spatial patch tokens only.  The leading
            # OpenCLIP/HuggingFace token is the CLS token and is deliberately
            # excluded from the activity-graph representation.
            pooled = hidden_states[:, 1:, :].mean(dim=1)
        elif aggregation == "cls_token":
            pooled = hidden_states[:, 0, :]
        else:
            raise ValueError(f"Unsupported aggregation {aggregation}")

        return self.project_pooled_representation(pooled)

    def project_pooled_representation(self, pooled):
        """Map a pooled backbone state to the model's public embedding space."""
        return pooled

    def robust_scale(self, x):
        median = x.median(1, keepdim=True)[0]
        q_tensor = torch.tensor([0.75, 0.25], device=x.device, dtype=x.dtype)
        q75, q25 = torch.quantile(x, q_tensor, dim=1, keepdim=True)
        x = x - median
        iqr = q75 - q25
        return x / (iqr + 1e-5)

    def ts2line_plot_transformation(self, x, image_size=224, line_width=1.5):
        # x: B x T x D. Each channel is rendered as a separate white-background
        # black-line image so the embedding code can concatenate channel features.
        x = self.robust_scale(x)
        x = einops.rearrange(x, "b t d -> (b d) t")

        min_vals = x.min(dim=-1, keepdim=True)[0]
        max_vals = x.max(dim=-1, keepdim=True)[0]
        value_range = max_vals - min_vals
        y = (x - min_vals) / (value_range + 1e-5)
        y = torch.where(value_range <= 1e-5, torch.full_like(y, 0.5), y)

        if y.shape[-1] == 1:
            y = y.expand(-1, image_size)
        else:
            y = F.interpolate(
                y.unsqueeze(1),
                size=image_size,
                mode="linear",
                align_corners=True,
            ).squeeze(1)

        y = (1.0 - y.clamp(0.0, 1.0)) * (image_size - 1)

        y0 = y[:, :-1].unsqueeze(-1)
        y1 = y[:, 1:].unsqueeze(-1)
        y_min = torch.minimum(y0, y1) - line_width
        y_max = torch.maximum(y0, y1) + line_width
        rows = torch.arange(image_size, device=x.device, dtype=x.dtype).view(1, 1, -1)

        above = y_min - rows
        below = rows - y_max
        distance = torch.maximum(torch.maximum(above, below), torch.zeros_like(rows))
        alpha = (1.0 - distance / line_width).clamp(0.0, 1.0)
        alpha = alpha.permute(0, 2, 1)

        last_col_distance = (rows.squeeze(1) - y[:, -1:].abs()).abs()
        last_col_alpha = (1.0 - last_col_distance / line_width).clamp(0.0, 1.0)
        last_col_alpha = last_col_alpha.unsqueeze(-1)
        alpha = torch.cat([alpha, last_col_alpha], dim=-1)

        image_input = 1.0 - alpha.unsqueeze(1)
        image_input = einops.repeat(image_input, "b 1 h w -> b c h w", c=3)

        return image_input

    def ts2image_transformation(
        self,
        x,
        patch_size,
        stride,
        image_size=224,
    ):
        if patch_size is None:
            raise ValueError("patch_size must be set when image_mode='segment'.")

        # x: B x T x D
        # Normalization using robust scaling
        x = self.robust_scale(x)

        x = einops.rearrange(x, "b t d -> b d t")
        T = x.shape[-1]

        if stride == 1:  # No overlapping patches
            pad_left = 0
            if T % patch_size != 0:
                pad_left = patch_size - T % patch_size
            x_pad = F.pad(x, (pad_left, 0), mode="replicate")
            x_2d = einops.rearrange(x_pad, "b d (p f) -> (b d) 1 f p", f=patch_size)
        elif stride > 0 and stride < 1:  # Overlapping patches
            pad_left = 0
            if int(patch_size * stride) == 0:
                stride_len = 1
            else:
                stride_len = int(patch_size * stride)
            remainder = (T - patch_size) % stride_len
            if remainder != 0:
                pad_left = stride_len - remainder
            x_pad = F.pad(x, (pad_left, 0), mode="replicate")
            x_2d = x_pad.unfold(dimension=2, size=patch_size, step=stride_len)
        else:
            raise ValueError(
                f"Stride is set to {stride}, but should be a fraction of the patch size, and thus lie between 0 and 1."
            )

        # Adjust contrast
        min_vals = x_2d.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_vals = x_2d.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        x_2d = (x_2d - min_vals) / (max_vals - min_vals + 1e-5)
        x_2d = torch.pow(x_2d, 0.8)

        # Resize to ViT input resolution
        x_resized = Resize((image_size, image_size), interpolation=0, antialias=False)(
            x_2d
        )

        # Generate grayscale images
        image_input = einops.repeat(x_resized, "b 1 h w -> b c h w", c=3)

        return image_input


class NeuroSigViT_HF(BaseNeuroSigViT):
    def __init__(
        self, processor, vit, layer_idx, aggregation, patch_size, stride, image_mode
    ):
        super().__init__(
            processor, vit, layer_idx, aggregation, patch_size, stride, image_mode
        )
        self.to_pil = T.ToPILImage()

    def truncate_layers(self):
        if self.layer_idx and self.layer_idx != -1:
            if hasattr(self.vit.encoder, "layers"):
                self.vit.encoder.layers = self.vit.encoder.layers[: self.layer_idx]
            elif hasattr(self.vit.encoder, "layer"):
                self.vit.encoder.layer = self.vit.encoder.layer[: self.layer_idx]
            else:
                raise ValueError("Unknown model architecture cannot be truncated.")

    def forward_vit(self, inputs):
        device = inputs.device
        # ToPILImage cannot convert CUDA tensors through NumPy.  Renderers may
        # operate on the model device, so make the host transfer explicit and
        # move only the processor output back to the encoder device.
        inputs = [self.to_pil(im.detach().cpu()) for im in inputs]
        inputs = self.processor(images=inputs, return_tensors="pt").to(device)
        outputs = self.vit(
            **inputs,
            output_hidden_states=(self.layer_idx is None),
        )

        if self.layer_idx:
            return outputs.last_hidden_state
        else:
            return torch.stack(outputs.hidden_states, dim=-1)


class NeuroSigViT_OpenCLIP(BaseNeuroSigViT):
    def __init__(
        self, processor, vit, layer_idx, aggregation, patch_size, stride, image_mode
    ):
        self.hidden_representations = {}
        super().__init__(
            processor, vit, layer_idx, aggregation, patch_size, stride, image_mode
        )
        self.processor.transforms = [self.processor.transforms[-1]]

    def truncate_layers(self):
        if self.layer_idx is not None and self.layer_idx != -1:
            self.vit.transformer.resblocks = self.vit.transformer.resblocks[
                : self.layer_idx
            ]

    def forward_vit(self, inputs):
        hidden_states = []

        inputs = self.processor(inputs)

        x = self.vit._embeds(inputs)
        hidden_states.append(x)

        for blk in self.vit.transformer.resblocks:
            x = blk(x)
            hidden_states.append(x)

        if self.layer_idx is not None:
            return x
        else:
            return torch.stack(hidden_states, dim=-1)

    def project_pooled_representation(self, pooled):
        # OpenCLIP ViT-H/14 has transformer width 1280 and a learned 1024-D
        # output projection.  Applying the frozen post norm/projection makes
        # the implementation match NeuroSigViT Eq. (5), while keeping the
        # selected intermediate transformer depth fixed.
        if hasattr(self.vit, "ln_post"):
            pooled = self.vit.ln_post(pooled)
        projection = getattr(self.vit, "proj", None)
        if projection is not None:
            pooled = pooled @ projection
        return pooled
