"""Versioned identities for adaptive-graph static feature caches.

The adaptive classifier caches raw temporal windows plus frozen Line/OpenCLIP
and Mantis features.  Online Activity Graph rendering is deliberately outside
this identity: changing that trainable path must not force the frozen features
to be extracted again.

The legacy identity below is an exact record of the schema-10 implementation at
commit 97ade80.  It is used only to locate and verify caches written by that
known implementation before promoting them to the static-only signature.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from typing import Mapping


ADAPTIVE_STATIC_IDENTITY_SCHEME = "adaptive_static_ast_v1"
KNOWN_COMPATIBLE_STATIC_AST_MANIFEST_SHA256 = (
    "91fe9280a339a4c75a9b309d8fe650897d440346e00f9e482e2d233da744a05e"
)
KNOWN_LEGACY_ADAPTIVE_CACHE_COMMIT = (
    "97ade80e7519ab4f2f1d564dbacdd660c34919b5"
)
KNOWN_LEGACY_ADAPTIVE_CACHE_SCHEMA = 10
KNOWN_LEGACY_ADAPTIVE_ARCHITECTURE = (
    "neurosigvia_adaptive_graph_crossattn_concatattn_v2"
)
KNOWN_LEGACY_ADAPTIVE_CODE_MANIFEST_SHA256 = (
    "15068ad3f463737ad9c1a9ca8aa65489d2b0a997e61f728a4692654391776c1f"
)

# Order is significant: the legacy runner fed these path/digest pairs to one
# SHA-256 manifest in this exact sequence.
_KNOWN_LEGACY_COMPONENTS = (
    (
        "src/neurosigvia.py",
        "a458a745b28ecf21eafc53201328e49460a08c6808941a6598394735269f5982",
    ),
    (
        "src/utils.py",
        "fe2d6057956831d63bbddf433005d21d2d5f9f25be9deb92acde58f608d4484b",
    ),
    (
        "src/patch_mindts.py",
        "fa3c6e1715ca407501ee2eee1a4607ddbfe2abc7d6bfa49b8d76fa4c5072c224",
    ),
    (
        "src/line_graph_cross_attention.py",
        "f5d395828707686eb16d2672d7014cfafa8307cfec472742558c46eb0b387017",
    ),
    (
        "src/multimodal_fusion.py",
        "810b365eb86cebcb25ec0ec1168397ed55afd19b1a7c13354a0cd1fc2acde186",
    ),
    (
        "src/adaptive_graph_training.py",
        "d029225391444a439851c2a6aea6ea593add7f63b56bc83d23b245b7d203604d",
    ),
    (
        "src/activity_graph.py",
        "fda96e37644830d0e64dba1e02091eff1ac6d9a70b7fc7635a92d7d49fd374ce",
    ),
    (
        "src/temporal_granularity.py",
        "9ef0b38c8c976e497ec1b140b0ce4306931fb34e68c0450677e4d4ebe224ceb9",
    ),
    (
        "src/adaptive_activity_graph.py",
        "1cedbcb61f1fb9fdb050295ff429fb6eb2fef0abeddaf0c8ff6c09dddcdcfc48",
    ),
    (
        "src/provenance.py",
        "2edc05252fa87984abe6cb0b1bc572492887c4d88a9745191656259d3ab8cb91",
    ),
)

# Five complete seed-42 caches were written before the public class/module
# rename.  The archived launch source was checked byte-for-byte against these
# eight component hashes.  Git commit 7f38ff7 contains the same source blobs;
# the feature-producing functions differ from 97ade80 only by the mechanical
# TimeMosaic/NeuroSigViT -> adaptive/NeuroSigVIA symbol rename.
KNOWN_TIMEMOSAIC_CACHE_SOURCE_COMMIT = (
    "7f38ff72d21be2104d662ab8250f4021f33cf926"
)
KNOWN_TIMEMOSAIC_CACHE_LAUNCH_COMMIT = (
    "b78892339093bae7e7defbbf0ec06f4023889bfb"
)
KNOWN_TIMEMOSAIC_CACHE_SCHEMA = 10
KNOWN_TIMEMOSAIC_MODEL_ARCHITECTURE = (
    "timemosaic_adaptive_graph_crossattn_concatattn_v2"
)
KNOWN_TIMEMOSAIC_STATIC_CACHE_ARCHITECTURE = (
    "timemosaic_adaptive_graph_static_v1"
)
KNOWN_TIMEMOSAIC_CODE_MANIFEST_SHA256 = (
    "1c8d46c0d27314259525e7bc30d440c53ae3681779ebbaae01c45c55270935f3"
)
_KNOWN_TIMEMOSAIC_COMPONENTS = (
    (
        "src/neurosigvit.py",
        "2d9c576d46523b81ec054d4491bd6cdb073da41d82f148421b3a0f2126cd99e2",
    ),
    (
        "src/utils.py",
        "52cc322282a183a5cea41c77bcd862d0cc1d79fd4a99efc883e0a70c7cedf3a9",
    ),
    (
        "src/patch_mindts.py",
        "a9c9693494ba97a1513590dbb06ac71745142f5ad9d9c1df3e20e40eefe67f80",
    ),
    (
        "src/line_graph_cross_attention.py",
        "f5d395828707686eb16d2672d7014cfafa8307cfec472742558c46eb0b387017",
    ),
    (
        "src/timemosaic_patch_pipeline.py",
        "7d571af0f334feaf4f8a173c20804842c8c9d938a0b26a59f9ba167d8dd335f6",
    ),
    (
        "src/timemosaic_graph_training.py",
        "36cce2fe43ff757e6d81539ac40d57b763534636787b3682a3b287409b8ebad3",
    ),
    (
        "src/medformer_graph/renderer.py",
        "6da6715f8f83a80215babc8e11c3eedd1d9d0b093041bd061d2ef0bd4d595b9c",
    ),
    (
        "src/medformer_graph/timemosaic_adaptive.py",
        "d3007a77b35e56919467360d16577e5177d3d33783706b4444ef9852f37479f3",
    ),
)


_STATIC_AST_SPEC = {
    "src/neurosigvia.py": {
        "functions": {
            "_draw_waveform",
            "get_openclip_config",
            "find_openclip_checkpoint",
            "get_processor_vit",
            "render_stacked_multichannel_lineplot",
            "preprocess_stacked_multichannel_lineplot",
        },
        # Only the constructor prefix can affect the frozen visual tokens.  The
        # statements after ``med_activity_graph`` configure the online graph
        # renderer and are intentionally outside the static-cache identity.
        "function_prefixes": {
            "get_neurosigvia": {
                "argument_stop": "med_activity_patch_lengths",
                "boundary_attribute": "med_activity_graph",
            },
        },
        "methods": {
            "BaseNeuroSigVIA": {
                "__init__",
                "aggregate_hidden_representations",
                "project_pooled_representation",
            },
            "NeuroSigVIA_OpenCLIP": {
                "__init__",
                "truncate_layers",
                "forward_vit",
                "project_pooled_representation",
            },
        },
        "constants": {"OPENCLIP_LAION_MODELS"},
        "imports": {
            "AutoImageProcessor",
            "AutoModel",
            "AutoProcessor",
            "CLIPModel",
            "CLIPProcessor",
            "ViTMAEForPreTraining",
            "np",
            "open_clip",
            "os",
            "torch",
        },
    },
    "src/utils.py": {
        "functions": {"resize_mantis_input"},
        "methods": {},
        "constants": set(),
        "imports": {"F"},
    },
    "src/patch_mindts.py": {
        "functions": {
            "make_temporal_patches",
            "_encode_visual_images",
            "_line_images_for_chunk",
            "_extract_line_tokens",
            "_extract_mantis_channel_tokens",
        },
        "methods": {},
        "classes": {"TemporalPatchBatch"},
        "constants": {"PATCH_TAIL_POLICY"},
        "imports": {
            "dataclass",
            "math",
            "preprocess_stacked_multichannel_lineplot",
            "resize_mantis_input",
            "torch",
            "F",
        },
    },
    "src/adaptive_graph_training.py": {
        "functions": {
            "_one_dimensional_labels",
            "_cache_label_array",
            "_validate_static_bundle",
            "save_adaptive_graph_feature_cache",
            "load_adaptive_graph_feature_cache",
            "extract_adaptive_graph_feature_batch",
            "_extract_static_split",
            "_get_static_split",
        },
        "methods": {},
        "constants": {
            "NEUROSIGVIA_CACHE_SCHEMA_VERSION",
            "NEUROSIGVIA_CACHE_ARCHITECTURE",
            "NEUROSIGVIA_STATIC_KEYS",
        },
        "imports": {
            "Mapping",
            "Path",
            "np",
            "torch",
            "tqdm",
            "PATCH_TAIL_POLICY",
            "SequentialSampler",
            "make_temporal_patches",
            "_extract_line_tokens",
            "_extract_mantis_channel_tokens",
        },
    },
    "runners/neurosigvia.py": {
        "functions": set(),
        "methods": {},
        "constants": set(),
        # Adaptive training receives the raw Mantis8M instance from this
        # branch.  Keep its construction in the legacy compatibility guard
        # without pulling unrelated experiment orchestration into the hash.
        "mantis_mlp_factory": True,
        "imports": {"Mantis8M", "MantisTrainer"},
    },
}


def known_legacy_adaptive_code_identity() -> dict[str, object]:
    """Return the exact whole-file identity emitted by the 97ade80 runner."""

    manifest = hashlib.sha256()
    components = {}
    for relative_path, digest in _KNOWN_LEGACY_COMPONENTS:
        components[relative_path] = digest
        manifest.update(relative_path.encode("utf-8"))
        manifest.update(digest.encode("ascii"))
    actual = manifest.hexdigest()
    if actual != KNOWN_LEGACY_ADAPTIVE_CODE_MANIFEST_SHA256:
        raise RuntimeError("known legacy adaptive-cache identity is internally corrupt")
    return {
        "manifest_sha256": actual,
        "components": components,
    }


def known_timemosaic_code_identity() -> dict[str, object]:
    """Return the exact identity stored in the five archived seed-42 caches."""

    manifest = hashlib.sha256()
    components = {}
    for relative_path, digest in _KNOWN_TIMEMOSAIC_COMPONENTS:
        components[relative_path] = digest
        manifest.update(relative_path.encode("utf-8"))
        manifest.update(digest.encode("ascii"))
    actual = manifest.hexdigest()
    if actual != KNOWN_TIMEMOSAIC_CODE_MANIFEST_SHA256:
        raise RuntimeError("known TimeMosaic cache identity is internally corrupt")
    return {
        "manifest_sha256": actual,
        "components": components,
    }


def _assignment_names(node: ast.Assign | ast.AnnAssign) -> set[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return {
        target.id
        for target in targets
        if isinstance(target, ast.Name)
    }


def _canonical_ast_dump(node: ast.AST) -> str:
    """Serialize ASTs identically across Python 3.11 and 3.12+.

    Python 3.12 added ``type_params=[]`` to function and class nodes.  Empty
    fields carry no source semantics, so omit only those empty fields.  A
    non-empty type-parameter list remains in the dump and changes the identity.
    """

    canonical = copy.deepcopy(node)
    for value in ast.walk(canonical):
        fields = getattr(value, "_fields", ())
        if (
            "type_params" in fields
            and not getattr(value, "type_params", None)
        ):
            value._fields = tuple(
                field for field in fields if field != "type_params"
            )
    return ast.dump(
        canonical,
        annotate_fields=True,
        include_attributes=False,
    )


def _selected_import_descriptors(
    tree: ast.Module,
    selected_bindings: set[str],
) -> list[tuple[str, str, str | None]]:
    descriptors = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                if bound in selected_bindings:
                    descriptors.append(("import", alias.name, alias.asname))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound = alias.asname or alias.name
                if bound in selected_bindings:
                    descriptors.append((node.module or "", alias.name, alias.asname))
    return sorted(descriptors)


def _selected_file_digest(path: Path, spec: Mapping[str, object]) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    requested_functions = set(spec.get("functions", ()))
    requested_prefixes = dict(spec.get("function_prefixes", {}))
    requested_classes = set(spec.get("classes", ()))
    requested_constants = set(spec.get("constants", ()))
    requested_methods = {
        str(class_name): set(method_names)
        for class_name, method_names in dict(spec.get("methods", {})).items()
    }
    selected: list[tuple[str, object]] = []
    found = set()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in requested_functions:
                selected.append((node.name, node))
                found.add(node.name)
            if node.name in requested_prefixes:
                prefix_spec = requested_prefixes[node.name]
                argument_stop = str(prefix_spec["argument_stop"])
                boundary_attribute = str(prefix_spec["boundary_attribute"])
                positional = [*node.args.posonlyargs, *node.args.args]
                default_offset = len(positional) - len(node.args.defaults)
                arguments = []
                saw_argument_stop = False
                for index, argument in enumerate(positional):
                    if argument.arg == argument_stop:
                        saw_argument_stop = True
                        break
                    default = (
                        node.args.defaults[index - default_offset]
                        if index >= default_offset
                        else None
                    )
                    arguments.append(
                        (
                            argument.arg,
                            _canonical_ast_dump(argument.annotation)
                            if argument.annotation is not None
                            else None,
                            _canonical_ast_dump(default)
                            if default is not None
                            else None,
                        )
                    )
                boundary_index = None
                for index, statement in enumerate(node.body):
                    targets = (
                        statement.targets
                        if isinstance(statement, ast.Assign)
                        else [statement.target]
                        if isinstance(statement, ast.AnnAssign)
                        else []
                    )
                    if any(
                        isinstance(target, ast.Attribute)
                        and target.attr == boundary_attribute
                        for target in targets
                    ):
                        boundary_index = index
                        break
                if not saw_argument_stop or boundary_index is None:
                    raise RuntimeError(
                        f"cannot fingerprint {node.name} constructor prefix in "
                        f"{path}: argument_stop={argument_stop!r}, "
                        f"boundary_attribute={boundary_attribute!r}"
                    )
                prefix_payload = {
                    "arguments": arguments,
                    "body": [
                        _canonical_ast_dump(statement)
                        for statement in node.body[:boundary_index]
                    ],
                }
                selected.append(
                    (
                        f"{node.name}:constructor_prefix",
                        prefix_payload,
                    )
                )
                found.add(f"{node.name}:constructor_prefix")
        elif isinstance(node, ast.ClassDef):
            if node.name in requested_classes:
                selected.append((node.name, node))
                found.add(node.name)
            for method in node.body:
                qualified = f"{node.name}.{getattr(method, 'name', '')}"
                if (
                    isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name in requested_methods
                    and method.name in requested_methods[node.name]
                ):
                    selected.append((qualified, method))
                    found.add(qualified)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for name in _assignment_names(node) & requested_constants:
                selected.append((name, node))
                found.add(name)

    if bool(spec.get("mantis_mlp_factory", False)):
        candidates = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Attribute)
                and isinstance(node.test.value, ast.Name)
                and node.test.value.id == "args"
                and node.test.attr == "mantis"
                and node.body
                and isinstance(node.body[0], ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "network"
                    for target in node.body[0].targets
                )
            ):
                continue
            mlp_branch_index = None
            for index, statement in enumerate(node.body):
                test = statement.test if isinstance(statement, ast.If) else None
                if not (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Attribute)
                    and isinstance(test.left.value, ast.Name)
                    and test.left.value.id == "args"
                    and test.left.attr == "classifier_type"
                    and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.Eq)
                    and len(test.comparators) == 1
                    and isinstance(test.comparators[0], ast.Constant)
                    and test.comparators[0].value == "mlp"
                ):
                    continue
                mlp_branch_index = index
                candidates.append(
                    {
                        "outer_test": _canonical_ast_dump(node.test),
                        "setup": [
                            _canonical_ast_dump(value)
                            for value in node.body[:mlp_branch_index]
                        ],
                        "mlp_test": _canonical_ast_dump(test),
                        "mlp_body": [
                            _canonical_ast_dump(value)
                            for value in statement.body
                        ],
                    }
                )
                break
            if mlp_branch_index is None:
                continue
        if len(candidates) != 1:
            raise RuntimeError(
                f"cannot fingerprint unique Mantis MLP factory in {path}: "
                f"candidates={len(candidates)}"
            )
        selected.append(("mantis_mlp_factory", candidates[0]))
        found.add("mantis_mlp_factory")

    expected = requested_functions | requested_classes | requested_constants
    expected.update(
        f"{name}:constructor_prefix" for name in requested_prefixes
    )
    expected.update(
        f"{class_name}.{method_name}"
        for class_name, method_names in requested_methods.items()
        for method_name in method_names
    )
    if bool(spec.get("mantis_mlp_factory", False)):
        expected.add("mantis_mlp_factory")
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise RuntimeError(
            f"cannot fingerprint static cache code in {path}: "
            f"missing={missing}, extra={extra}"
        )

    payload = {
        "imports": _selected_import_descriptors(
            tree, set(spec.get("imports", ()))
        ),
        "nodes": [
            (
                name,
                (
                    node
                    if isinstance(node, dict)
                    else _canonical_ast_dump(node)
                ),
            )
            for name, node in sorted(selected, key=lambda item: item[0])
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def adaptive_static_extractor_code_identity(
    project_root: str | Path,
) -> dict[str, object]:
    """Hash only code that can change cached raw/Line/Mantis tensors."""

    root = Path(project_root)
    components = {}
    manifest = hashlib.sha256()
    for relative_path, spec in _STATIC_AST_SPEC.items():
        component_name = f"{relative_path}:adaptive_static_components"
        digest = _selected_file_digest(root / relative_path, spec)
        components[component_name] = digest
        manifest.update(component_name.encode("utf-8"))
        manifest.update(digest.encode("ascii"))
    return {
        "identity_scheme": ADAPTIVE_STATIC_IDENTITY_SCHEME,
        "manifest_sha256": manifest.hexdigest(),
        "components": components,
    }


def assert_known_static_extractor_compatibility(
    identity: Mapping[str, object],
) -> None:
    """Reject legacy promotion after any audited static extractor change."""

    if identity.get("identity_scheme") != ADAPTIVE_STATIC_IDENTITY_SCHEME:
        raise ValueError("unknown adaptive static extractor identity scheme")
    actual = identity.get("manifest_sha256")
    if actual != KNOWN_COMPATIBLE_STATIC_AST_MANIFEST_SHA256:
        raise ValueError(
            "legacy static-cache reuse is disabled because the current "
            "raw/Line/Mantis extractor differs from the audited implementation: "
            f"expected {KNOWN_COMPATIBLE_STATIC_AST_MANIFEST_SHA256}, got {actual}"
        )


__all__ = [
    "ADAPTIVE_STATIC_IDENTITY_SCHEME",
    "KNOWN_COMPATIBLE_STATIC_AST_MANIFEST_SHA256",
    "KNOWN_LEGACY_ADAPTIVE_ARCHITECTURE",
    "KNOWN_LEGACY_ADAPTIVE_CACHE_COMMIT",
    "KNOWN_LEGACY_ADAPTIVE_CACHE_SCHEMA",
    "KNOWN_LEGACY_ADAPTIVE_CODE_MANIFEST_SHA256",
    "KNOWN_TIMEMOSAIC_CACHE_LAUNCH_COMMIT",
    "KNOWN_TIMEMOSAIC_CACHE_SCHEMA",
    "KNOWN_TIMEMOSAIC_CACHE_SOURCE_COMMIT",
    "KNOWN_TIMEMOSAIC_CODE_MANIFEST_SHA256",
    "KNOWN_TIMEMOSAIC_MODEL_ARCHITECTURE",
    "KNOWN_TIMEMOSAIC_STATIC_CACHE_ARCHITECTURE",
    "adaptive_static_extractor_code_identity",
    "assert_known_static_extractor_compatibility",
    "known_legacy_adaptive_code_identity",
    "known_timemosaic_code_identity",
]
