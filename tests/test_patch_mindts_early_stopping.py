#!/usr/bin/env python3
"""Deterministic tests for the decoupled Patch-MindTS early-stop monitor."""

import math
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.patch_mindts import _EarlyStoppingMonitor  # noqa: E402


class EarlyStoppingMonitorTests(unittest.TestCase):
    def test_raw_strategy_preserves_tuple_tie_breaker(self):
        monitor = _EarlyStoppingMonitor(
            "raw_selection_key",
            patience=2,
        )
        first = monitor.update((0.5, -0.8), 1)
        tie_break_improvement = monitor.update((0.5, -0.7), 2)
        stale_one = monitor.update((0.5, -0.7), 3)
        stopped = monitor.update((0.49, -0.6), 4)

        self.assertTrue(first["improved"])
        self.assertTrue(tie_break_improvement["improved"])
        self.assertEqual(stale_one["epochs_without_improvement"], 1)
        self.assertFalse(stale_one["should_stop"])
        self.assertTrue(stopped["should_stop"])

    def test_min_epochs_is_an_absolute_stop_floor(self):
        monitor = _EarlyStoppingMonitor(
            "ema_primary",
            patience=1,
            min_epochs=3,
            ema_decay=0.0,
            min_delta=0.0,
        )
        self.assertFalse(monitor.update((0.5,), 1)["should_stop"])
        before_floor = monitor.update((0.4,), 2)
        self.assertFalse(before_floor["eligible"])
        self.assertFalse(before_floor["should_stop"])
        at_floor = monitor.update((0.3,), 3)
        self.assertTrue(at_floor["eligible"])
        self.assertTrue(at_floor["should_stop"])

    def test_ema_primary_uses_smoothed_score_and_min_delta(self):
        monitor = _EarlyStoppingMonitor(
            "ema_primary",
            patience=2,
            ema_decay=0.5,
            min_delta=0.01,
        )
        epoch1 = monitor.update((0.5, -1.0), 1)
        epoch2 = monitor.update((0.7, -1.0), 2)
        epoch3 = monitor.update((0.51, -0.1), 3)
        epoch4 = monitor.update((0.51, -0.1), 4)

        self.assertAlmostEqual(epoch1["smoothed_score"], 0.5)
        self.assertAlmostEqual(epoch2["smoothed_score"], 0.6)
        self.assertAlmostEqual(epoch3["smoothed_score"], 0.555)
        self.assertFalse(epoch3["should_stop"])
        self.assertTrue(epoch4["should_stop"])

    def test_patience_zero_disables_stopping(self):
        monitor = _EarlyStoppingMonitor(
            "ema_primary",
            patience=0,
            ema_decay=0.5,
        )
        for epoch in range(1, 20):
            state = monitor.update((1.0 / epoch,), epoch)
            self.assertFalse(state["eligible"])
            self.assertFalse(state["should_stop"])

    def test_invalid_configuration_and_nonfinite_metric_are_rejected(self):
        with self.assertRaises(ValueError):
            _EarlyStoppingMonitor("unknown", patience=1)
        with self.assertRaises(ValueError):
            _EarlyStoppingMonitor("ema_primary", patience=-1)
        with self.assertRaises(ValueError):
            _EarlyStoppingMonitor("ema_primary", patience=1, ema_decay=1.0)
        with self.assertRaises(ValueError):
            _EarlyStoppingMonitor("ema_primary", patience=1, min_delta=-1.0)

        monitor = _EarlyStoppingMonitor("ema_primary", patience=1)
        with self.assertRaises(ValueError):
            monitor.update((math.nan,), 1)
        with self.assertRaises(ValueError):
            monitor.update((0.5,), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
