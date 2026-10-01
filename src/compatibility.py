"""Read existing queued commands and checkpoints after the public rename.

New scripts, help text and artifacts use the canonical NeuroSigVIA names.
The aliases here let already-running experiment queues finish without restart.
"""

_PREVIOUS_OPTIONS = {
    "--timemosaic_gate_temperature": "--granularity_gate_temperature",
    "--timemosaic_selector_balance_weight": "--granularity_balance_weight",
    "--timemosaic_graph_token_grid": "--granularity_graph_token_grid",
    "--timemosaic_gate_checkpoint": "--granularity_gate_checkpoint",
    "--timemosaic_freeze_gate": "--granularity_freeze_gate",
}
# src/patch_fusion.py was src/patch_mindts.py; cache manifests and checkpoints
# written before the rename record the earlier path.
_PREVIOUS_OPTION_PREFIXES = {"--med_activity_": "--activity_graph_"}
_PREVIOUS_VALUES = {
    "--modal_interaction": {
        "patch_timemosaic_graph": "adaptive_granularity",
        "patch_mindts": "patch_fusion",
    },
    "--image_mode": {"med_activity_graph": "multiscale_activity_graph"},
}
_PREVIOUS_ARCHITECTURE = "timemosaic_adaptive_graph_crossattn_concatattn_v2"
_CURRENT_ARCHITECTURE = "neurosigvia_adaptive_graph_crossattn_concatattn_v2"
_PREVIOUS_ENCODER_WRAPPERS = {
    "src.neurosigvit.NeuroSigViT_OpenCLIP": "src.neurosigvia.NeuroSigVIA_OpenCLIP",
    "src.neurosigvit.NeuroSigViT_HF": "src.neurosigvia.NeuroSigVIA_HF",
}


def _current_option(option):
    option = _PREVIOUS_OPTIONS.get(option, option)
    for previous, current in _PREVIOUS_OPTION_PREFIXES.items():
        if option.startswith(previous):
            return current + option[len(previous):]
    return option


def normalize_cli_arguments(arguments):
    """Normalize only option names and renamed option values, never paths."""
    result = []
    pending_values = None
    for token in arguments:
        option, separator, value = token.partition("=")
        option = _current_option(option)
        if pending_values is not None and token in pending_values:
            token = pending_values[token]
        elif separator and option in _PREVIOUS_VALUES:
            token = option + separator + _PREVIOUS_VALUES[option].get(value, value)
        else:
            token = option + separator + value
        result.append(token)
        pending_values = None if separator else _PREVIOUS_VALUES.get(option)
    return result


def matches_checkpoint_architecture(recorded, expected):
    """Accept the previous name only for the same current model architecture."""
    return recorded == expected or (
        expected == _CURRENT_ARCHITECTURE and recorded == _PREVIOUS_ARCHITECTURE
    )


def is_previous_image_mode(recorded, current):
    """Recognize only the explicitly renamed image mode."""
    return (
        isinstance(recorded, str)
        and _PREVIOUS_VALUES["--image_mode"].get(recorded) == current
    )


def is_previous_encoder_wrapper(recorded, current):
    """Recognize only the two explicitly renamed visual wrapper classes."""
    return (
        isinstance(recorded, str)
        and isinstance(current, str)
        and _PREVIOUS_ENCODER_WRAPPERS.get(recorded) == current
    )
