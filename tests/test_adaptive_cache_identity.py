"""Static-cache identities cover frozen extraction and exclude graph training."""

import ast
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adaptive_cache_identity import (
    KNOWN_COMPATIBLE_STATIC_AST_MANIFEST_SHA256,
    KNOWN_LEGACY_ADAPTIVE_CODE_MANIFEST_SHA256,
    KNOWN_TIMEMOSAIC_CODE_MANIFEST_SHA256,
    adaptive_static_extractor_code_identity,
    assert_known_static_extractor_compatibility,
    known_legacy_adaptive_code_identity,
    known_timemosaic_code_identity,
    _canonical_ast_dump,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_SOURCE_PATHS = (
    "src/neurosigvia.py",
    "src/utils.py",
    "src/patch_mindts.py",
    "src/adaptive_graph_training.py",
    "runners/neurosigvia.py",
)


class AdaptiveCacheIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for relative_path in STATIC_SOURCE_PATHS:
            destination = self.root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT_ROOT / relative_path, destination)

    def identity(self):
        return adaptive_static_extractor_code_identity(self.root)

    def replace(self, relative_path, old, new):
        path = self.root / relative_path
        source = path.read_text(encoding="utf-8")
        self.assertIn(old, source)
        path.write_text(source.replace(old, new, 1), encoding="utf-8")

    def test_known_whole_file_manifests_are_internally_consistent(self):
        self.assertEqual(
            known_legacy_adaptive_code_identity()["manifest_sha256"],
            KNOWN_LEGACY_ADAPTIVE_CODE_MANIFEST_SHA256,
        )
        self.assertEqual(
            known_timemosaic_code_identity()["manifest_sha256"],
            KNOWN_TIMEMOSAIC_CODE_MANIFEST_SHA256,
        )

    def test_ast_dump_omits_only_empty_cross_version_type_parameters(self):
        function = ast.parse("def example():\n    pass\n").body[0]
        if "type_params" not in function._fields:
            function._fields = (*function._fields, "type_params")
        function.type_params = []
        self.assertNotIn("type_params", _canonical_ast_dump(function))
        function.type_params = [ast.Name(id="T", ctx=ast.Load())]
        self.assertIn("type_params", _canonical_ast_dump(function))

    def test_checked_in_static_extractor_matches_audited_legacy_guard(self):
        identity = self.identity()
        self.assertEqual(
            identity["manifest_sha256"],
            KNOWN_COMPATIBLE_STATIC_AST_MANIFEST_SHA256,
        )
        assert_known_static_extractor_compatibility(identity)

    def test_online_graph_constructor_changes_do_not_invalidate_static_tokens(self):
        before = self.identity()
        self.replace(
            "src/neurosigvia.py",
            "channel_mix=med_activity_channel_mix,",
            "channel_mix=0.987654321,",
        )
        self.assertEqual(self.identity(), before)

    def test_processor_change_invalidates_static_identity_and_legacy_guard(self):
        before = self.identity()
        self.replace(
            "src/neurosigvia.py",
            "pretrained=openclip_pretrained,",
            "pretrained=openclip_pretrained, image_mean=(0.5, 0.5, 0.5),",
        )
        changed = self.identity()
        self.assertNotEqual(changed, before)
        with self.assertRaisesRegex(ValueError, "legacy static-cache reuse"):
            assert_known_static_extractor_compatibility(changed)

    def test_visual_wrapper_construction_change_invalidates_legacy_guard(self):
        before = self.identity()
        self.replace(
            "src/neurosigvia.py",
            "layer_idx=model_layer,",
            "layer_idx=1,",
        )
        changed = self.identity()
        self.assertNotEqual(changed, before)
        with self.assertRaisesRegex(ValueError, "legacy static-cache reuse"):
            assert_known_static_extractor_compatibility(changed)

    def test_mantis_construction_change_invalidates_legacy_guard(self):
        before = self.identity()
        self.replace(
            "runners/neurosigvia.py",
            "network = Mantis8M(device=device)",
            "network = Mantis8M(device=device, strict=True)",
        )
        changed = self.identity()
        self.assertNotEqual(changed, before)
        with self.assertRaisesRegex(ValueError, "legacy static-cache reuse"):
            assert_known_static_extractor_compatibility(changed)

    def test_constructor_boundary_removal_fails_closed(self):
        self.replace(
            "src/neurosigvia.py",
            "neurosigvia.med_activity_graph = ActivityGraphRenderer(",
            "neurosigvia.online_graph = ActivityGraphRenderer(",
        )
        with self.assertRaisesRegex(RuntimeError, "constructor prefix"):
            self.identity()


if __name__ == "__main__":
    unittest.main()
