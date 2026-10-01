"""Re-score legacy checkpoints at window level without relabeling their selection protocol."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)

METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auroc", "macro_auprc")
DATASETS = ("shimmer10", "pads11", "apava", "tdbrain")
RUNS = {name: f"{name}_s42-43-44_20260907_211337" for name in
        ("Medformer", "PatchTST")}
RUNS.update(TimesNet="TimesNet_s42-43-44_20260908_035053",
            NeuroSigVIA="NeuroSigVIA_paperAG_s42-43-44_20260908_003550",
            NumericNoFusion="NeuroSigVIA_NumericOnly_NoFusion_s42-43-44_20260908_r1")


def recalculate(path):
    with np.load(path, allow_pickle=False) as saved:
        y, scores = saved["y_true"], saved["y_score"]
        predicted = scores.argmax(axis=1)
        np.testing.assert_array_equal(predicted, saved["y_pred"])
        onehot = np.eye(scores.shape[1])[y]
        return np.array([
            accuracy_score(y, predicted),
            precision_score(y, predicted, average="macro", zero_division=0),
            recall_score(y, predicted, average="macro", zero_division=0),
            f1_score(y, predicted, average="macro", zero_division=0),
            roc_auc_score(onehot, scores, average="macro"),
            average_precision_score(onehot, scores, average="macro"),
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    rows, aggregate = [], []
    for method, tag in RUNS.items():
        pattern = ("adaptive_graph_summary.json" if method == "NeuroSigVIA" else
                   "numeric_ablation_summary.json" if method == "NumericNoFusion" else "metrics.json")
        for dataset in DATASETS:
            matrix = []
            for seed in (42, 43, 44):
                run = args.project_root / "results" / tag / f"seed{seed}" / dataset
                candidates = list(run.rglob(pattern))
                if len(candidates) != 1:
                    raise ValueError(f"Expected exactly one result: {run}/{pattern}")
                source = candidates[0]
                payload = json.loads(source.read_text())
                if method not in ("NeuroSigVIA", "NumericNoFusion"):
                    if payload.get("status") != "COMPLETED" or not payload.get("scientific_result"):
                        raise ValueError(f"Not a completed scientific result: {source}")
                elif method == "NumericNoFusion" and payload.get("state") != "COMPLETED":
                    raise ValueError(f"Incomplete result: {source}")
                metrics = payload.get("test_metrics", payload.get("test"))
                values = np.array([metrics[key] for key in METRICS])
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite metric: {source}")
                prediction = source.parent / "test_predictions.npz"
                error = None
                if prediction.exists():
                    error = float(np.max(np.abs(recalculate(prediction) - values)))
                    if error > 1e-10:
                        raise ValueError(f"Prediction/summary mismatch: {prediction}: {error}")
                row = dict(method=method, dataset=dataset, seed=seed,
                           evaluation_unit="window", checkpoint_selection="legacy_subject_macro_f1",
                           corrected_window_training=False,
                           prediction_verified=prediction.exists(), max_metric_error=error,
                           source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                           **dict(zip(METRICS, (values * 100).tolist())))
                rows.append(row)
                matrix.append(values * 100)
            matrix = np.array(matrix)
            aggregate.append(dict(method=method, dataset=dataset, n=3,
                                  **{key: dict(mean=float(matrix[:, i].mean()),
                                              std=float(matrix[:, i].std(ddof=1)))
                                     for i, key in enumerate(METRICS)}))
    report = dict(evaluation_unit="window", unit="percent", std_ddof=1,
                  warning="Legacy subject-selected checkpoints only; NOT corrected window-selected training results.",
                  rows=rows, aggregate=aggregate)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "legacy_window_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output_dir / "legacy_window_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 旧 checkpoint 的窗口级结果（不是修正协议重训结果）", "",
             "原 checkpoint 均按受试者验证 F1 选模。本表仅改评价单位，不能当成窗口 F1 选模的新实验。",
             "单位：%；seed 42/43/44；均值 ± 样本标准差（ddof=1）。AUPRC 实际为 Macro Average Precision。", "",
             "| 方法 | 数据集 | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate:
        cells = [f"{row[key]['mean']:.2f}±{row[key]['std']:.2f}" for key in METRICS]
        lines.append(f"| {row['method']} | {row['dataset']} | " + " | ".join(cells) + " |")
    lines += ["", "证据范围：96组基线/单模态测试预测独立重算；12组主方法仅从原summary读取，未保存逐窗预测。",
              "旧文件未覆盖。全部来源、SHA256与逐seed数值见JSON/CSV。"]
    (args.output_dir / "legacy_window_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(dict(rows=len(rows), prediction_files_verified=sum(r["prediction_verified"] for r in rows),
                          max_metric_error=max(r["max_metric_error"] or 0 for r in rows), output=str(args.output_dir))))


if __name__ == "__main__":
    main()
