#!/usr/bin/env python3
import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_PATH = REPO_ROOT / "main.py"
sys.path.insert(0, str(REPO_ROOT))


def load_content_helpers():
    names = {
        "_update_content_digest",
        "_content_identity",
        "_dataset_content_identity",
        "_split_input_identity",
    }
    tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    assert {node.name for node in selected} == names
    namespace = {
        "hashlib": hashlib,
        "json": json,
        "np": np,
        "torch": torch,
    }
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            str(MAIN_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace


def make_loader(values, batch_size=2):
    return DataLoader(
        TensorDataset(values),
        batch_size=batch_size,
        shuffle=False,
    )


def main():
    helpers = load_content_helpers()
    identify = helpers["_split_input_identity"]
    dataset_identity = helpers["_dataset_content_identity"]

    train = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
    test = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    train_labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    test_labels = np.asarray([1, 0], dtype=np.int64)
    base = identify(
        make_loader(train, batch_size=1),
        train_labels,
        make_loader(test, batch_size=1),
        test_labels,
    )
    # Batch size is downstream of dataset content and must not change identity.
    assert identify(
        make_loader(train, batch_size=4),
        train_labels,
        make_loader(test, batch_size=2),
        test_labels,
    ) == base

    changed_value = train.clone()
    changed_value[1, 0, 0] += 0.5
    assert identify(
        make_loader(changed_value),
        train_labels,
        make_loader(test),
        test_labels,
    ) != base
    assert identify(
        make_loader(train.flip(0)),
        train_labels[::-1].copy(),
        make_loader(test),
        test_labels,
    ) != base
    assert identify(
        make_loader(train.to(torch.float64)),
        train_labels,
        make_loader(test),
        test_labels,
    ) != base
    changed_labels = train_labels.copy()
    changed_labels[0] = 1
    assert identify(
        make_loader(train),
        changed_labels,
        make_loader(test),
        test_labels,
    ) != base

    empty = TensorDataset(torch.empty((0, 2, 3), dtype=torch.float32))
    empty_identity = dataset_identity(empty)
    assert empty_identity["sample_count"] == 0
    assert len(empty_identity["content_sha256"]) == 64

    print("FEATURE CACHE PROVENANCE VALIDATION PASSED")


if __name__ == "__main__":
    main()
