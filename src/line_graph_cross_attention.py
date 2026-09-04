"""Cross-attention from line-plot tokens to Activity Graph spatial tokens.

The module keeps the roles of the two visual views explicit: one pooled
line-plot token is the query for each temporal window, while the spatial
Activity Graph tokens are the keys and values.  Keeping at least two graph
tokens is required; attention over a single key would collapse to a constant
weight and would not depend on the query.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class LineGraphCrossAttention(nn.Module):
    """Fuse line-plot and Activity Graph features with cross-attention.

    Parameters are deliberately limited to JSON-serializable scalar values.
    The same values are exposed through :attr:`constructor_configuration` and
    :meth:`get_config`, so checkpoints can reconstruct the module exactly.

    Args:
        line_dim: Last dimension of the pooled line-plot tokens.
        graph_dim: Last dimension of the Activity Graph spatial tokens.
        fusion_dim: Shared dimension used by query, keys, values, and output.
        num_heads: Number of cross-attention heads.
        ffn_hidden_dim: Hidden dimension of the post-attention feed-forward
            network.  Defaults to ``4 * fusion_dim``.
        dropout: Dropout probability used by attention and the feed-forward
            network.
        bias: Whether linear projections and attention use additive biases.

    Input shapes:
        ``line_tokens``: ``[B, N, line_dim]``.
        ``graph_spatial_tokens``: ``[B, N, P, graph_dim]``, with ``P >= 2``.
        ``patch_mask``: optional ``[B, N]`` mask where nonzero entries denote
            valid temporal windows.

    Output shape:
        ``[B, N, fusion_dim]``. Invalid windows are exactly zero.  When
        ``return_attention_weights=True``, the method additionally returns
        per-head weights with shape ``[B, N, num_heads, 1, P]``; invalid
        windows have zero weights.
    """

    def __init__(
        self,
        line_dim: int,
        graph_dim: int,
        fusion_dim: int = 128,
        num_heads: int = 4,
        ffn_hidden_dim: int | None = None,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        line_dim = self._positive_int("line_dim", line_dim)
        graph_dim = self._positive_int("graph_dim", graph_dim)
        fusion_dim = self._positive_int("fusion_dim", fusion_dim)
        num_heads = self._positive_int("num_heads", num_heads)
        if fusion_dim % num_heads != 0:
            raise ValueError(
                "fusion_dim must be divisible by num_heads, got "
                f"fusion_dim={fusion_dim} and num_heads={num_heads}"
            )
        if ffn_hidden_dim is None:
            ffn_hidden_dim = 4 * fusion_dim
        ffn_hidden_dim = self._positive_int(
            "ffn_hidden_dim", ffn_hidden_dim
        )
        dropout = float(dropout)
        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"dropout must be in [0, 1), got {dropout}"
            )
        if not isinstance(bias, bool):
            raise TypeError(f"bias must be bool, got {type(bias).__name__}")

        self.line_dim = line_dim
        self.graph_dim = graph_dim
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads

        self.query_projection = nn.Linear(line_dim, fusion_dim, bias=bias)
        self.key_projection = nn.Linear(graph_dim, fusion_dim, bias=bias)
        self.value_projection = nn.Linear(graph_dim, fusion_dim, bias=bias)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(fusion_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(fusion_dim, ffn_hidden_dim, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, fusion_dim, bias=bias),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(fusion_dim)

        self.constructor_configuration: dict[str, Any] = {
            "line_dim": line_dim,
            "graph_dim": graph_dim,
            "fusion_dim": fusion_dim,
            "num_heads": num_heads,
            "ffn_hidden_dim": ffn_hidden_dim,
            "dropout": dropout,
            "bias": bias,
        }
        self.configuration: dict[str, Any] = {
            **self.constructor_configuration,
            "query_source": "pooled_line_plot_token",
            "key_value_source": "activity_graph_spatial_tokens",
            "minimum_key_value_tokens": 2,
            "attention_residual": "projected_line_query",
            "output_layout": "B_N_fusion_dim",
        }

    @staticmethod
    def _positive_int(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{name} must be an integer, got {type(value).__name__}"
            )
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return int(value)

    def get_config(self) -> dict[str, Any]:
        """Return an independent JSON-serializable constructor dictionary."""

        return dict(self.constructor_configuration)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LineGraphCrossAttention":
        """Reconstruct a module from :meth:`get_config` output."""

        return cls(**dict(config))

    def _validate_inputs(
        self,
        line_tokens: torch.Tensor,
        graph_spatial_tokens: torch.Tensor,
        patch_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if not torch.is_tensor(line_tokens):
            raise TypeError("line_tokens must be a torch.Tensor")
        if not torch.is_tensor(graph_spatial_tokens):
            raise TypeError("graph_spatial_tokens must be a torch.Tensor")
        if line_tokens.ndim != 3:
            raise ValueError(
                "line_tokens must have shape [B, N, D_line], got "
                f"{tuple(line_tokens.shape)}"
            )
        if graph_spatial_tokens.ndim != 4:
            raise ValueError(
                "graph_spatial_tokens must have shape [B, N, P, D_graph], got "
                f"{tuple(graph_spatial_tokens.shape)}"
            )

        batch_size, num_patches, line_dim = line_tokens.shape
        graph_batch, graph_patches, graph_token_count, graph_dim = (
            graph_spatial_tokens.shape
        )
        if (graph_batch, graph_patches) != (batch_size, num_patches):
            raise ValueError(
                "line and graph batch/patch dimensions must match, got "
                f"line={tuple(line_tokens.shape[:2])} and "
                f"graph={tuple(graph_spatial_tokens.shape[:2])}"
            )
        if line_dim != self.line_dim:
            raise ValueError(
                f"expected line_dim={self.line_dim}, got {line_dim}"
            )
        if graph_dim != self.graph_dim:
            raise ValueError(
                f"expected graph_dim={self.graph_dim}, got {graph_dim}"
            )
        if graph_token_count < 2:
            raise ValueError(
                "graph_spatial_tokens must contain at least two K/V tokens; "
                f"got P={graph_token_count}. Cross-attention with P=1 is "
                "query-independent and therefore degenerate."
            )
        if line_tokens.device != graph_spatial_tokens.device:
            raise ValueError(
                "line_tokens and graph_spatial_tokens must be on the same "
                f"device, got {line_tokens.device} and "
                f"{graph_spatial_tokens.device}"
            )
        if line_tokens.dtype != graph_spatial_tokens.dtype:
            raise ValueError(
                "line_tokens and graph_spatial_tokens must have the same "
                f"dtype, got {line_tokens.dtype} and "
                f"{graph_spatial_tokens.dtype}"
            )
        if not line_tokens.is_floating_point():
            raise TypeError(
                "line_tokens and graph_spatial_tokens must use a floating "
                f"dtype, got {line_tokens.dtype}"
            )

        if patch_mask is None:
            return torch.ones(
                (batch_size, num_patches),
                dtype=torch.bool,
                device=line_tokens.device,
            )
        if not torch.is_tensor(patch_mask):
            patch_mask = torch.as_tensor(patch_mask, device=line_tokens.device)
        elif patch_mask.device != line_tokens.device:
            patch_mask = patch_mask.to(device=line_tokens.device)
        if patch_mask.shape != (batch_size, num_patches):
            raise ValueError(
                "patch_mask must have shape [B, N], got "
                f"{tuple(patch_mask.shape)} for B={batch_size}, N={num_patches}"
            )
        return patch_mask.to(dtype=torch.bool)

    def forward(
        self,
        line_tokens: torch.Tensor,
        graph_spatial_tokens: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
        *,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply Line-Q/Activity-Graph-KV cross-attention per window."""

        valid_mask = self._validate_inputs(
            line_tokens,
            graph_spatial_tokens,
            patch_mask,
        )
        batch_size, num_patches, _ = line_tokens.shape
        graph_token_count = graph_spatial_tokens.shape[2]

        line_valid = valid_mask.unsqueeze(-1)
        graph_valid = valid_mask.unsqueeze(-1).unsqueeze(-1)
        safe_line_tokens = torch.where(
            line_valid,
            line_tokens,
            torch.zeros_like(line_tokens),
        )
        safe_graph_tokens = torch.where(
            graph_valid,
            graph_spatial_tokens,
            torch.zeros_like(graph_spatial_tokens),
        )

        query = self.query_projection(safe_line_tokens).reshape(
            batch_size * num_patches,
            1,
            self.fusion_dim,
        )
        flat_graph = safe_graph_tokens.reshape(
            batch_size * num_patches,
            graph_token_count,
            self.graph_dim,
        )
        keys = self.key_projection(flat_graph)
        values = self.value_projection(flat_graph)
        attended, attention_weights = self.cross_attention(
            query,
            keys,
            values,
            need_weights=return_attention_weights,
            average_attn_weights=False,
        )

        hidden = self.attention_norm(
            query + self.attention_dropout(attended)
        )
        hidden = self.output_norm(hidden + self.feed_forward(hidden))
        output = hidden.squeeze(1).reshape(
            batch_size,
            num_patches,
            self.fusion_dim,
        )
        output = torch.where(
            line_valid,
            output,
            torch.zeros_like(output),
        )

        if not return_attention_weights:
            return output

        if attention_weights is None:
            raise RuntimeError("attention weights were requested but not returned")
        attention_weights = attention_weights.reshape(
            batch_size,
            num_patches,
            self.num_heads,
            1,
            graph_token_count,
        )
        attention_weights = torch.where(
            valid_mask[:, :, None, None, None],
            attention_weights,
            torch.zeros_like(attention_weights),
        )
        return output, attention_weights


__all__ = ["LineGraphCrossAttention"]
