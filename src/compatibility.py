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
_PREVIOUS_INTERACTION = "patch_timemosaic_graph"
_PREVIOUS_ARCHITECTURE = "timemosaic_adaptive_graph_crossattn_concatattn_v2"
_CURRENT_ARCHITECTURE = "neurosigvia_adaptive_graph_crossattn_concatattn_v2"
_PREVIOUS_ENCODER_WRAPPERS = {
    "src.neurosigvit.NeuroSigViT_OpenCLIP": "src.neurosigvia.NeuroSigVIA_OpenCLIP",
    "src.neurosigvit.NeuroSigViT_HF": "src.neurosigvia.NeuroSigVIA_HF",
}


def normalize_cli_arguments(arguments):
    """Normalize only option names and the selected interaction, never paths."""
    result = []
    interaction_value = False
    for token in arguments:
        option, separator, value = token.partition("=")
        option = _PREVIOUS_OPTIONS.get(option, option)
        if interaction_value and token == _PREVIOUS_INTERACTION:
            token = "adaptive_granularity"
        elif option == "--modal_interaction" and separator:
            token = option + separator + (
                "adaptive_granularity" if value == _PREVIOUS_INTERACTION else value
            )
        else:
            token = option + separator + value
        result.append(token)
        interaction_value = option == "--modal_interaction" and not separator
    return result


def matches_checkpoint_architecture(recorded, expected):
    """Accept the previous name only for the same current model architecture."""
    return recorded == expected or (
        expected == _CURRENT_ARCHITECTURE and recorded == _PREVIOUS_ARCHITECTURE
    )


def is_previous_encoder_wrapper(recorded, current):
    """Recognize only the two explicitly renamed visual wrapper classes."""
    return (
        isinstance(recorded, str)
        and isinstance(current, str)
        and _PREVIOUS_ENCODER_WRAPPERS.get(recorded) == current
    )
