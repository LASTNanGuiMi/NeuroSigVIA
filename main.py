"""Compatibility entry point for existing queued training commands."""
import runpy

if __name__ == "__main__":
    runpy.run_module("runners.neurosigvia", run_name="__main__", alter_sys=True)
