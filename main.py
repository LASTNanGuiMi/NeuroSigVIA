"""Compatibility entry point for existing queued training commands."""
from pathlib import Path
import runpy

# 保留旧命令入口；若存在批次专用队列守卫，先由其处理已登记任务。
if __name__ == "__main__":
    guard = Path(__file__).resolve().parent / ".aris/expand_queue_20260907/queue_guard.py"
    if guard.is_file():
        runpy.run_path(str(guard))["handle"]()
    # 实际训练实现统一放在 runners.neurosigvia，避免维护两套入口逻辑。
    runpy.run_module("runners.neurosigvia", run_name="__main__", alter_sys=True)
