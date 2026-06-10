"""
common/io_utils.py — output path resolution and summary-line writing.

Output layout (overridable via ADAVOTE_RESULTS env var):
    results/data/   -> CSV raw tables
    results/figs/   -> PDF vector figures
    results/e2/     -> E2 diagnostic outputs
    results/summary.txt  -> one PASS/FAIL line per experiment
"""

from __future__ import annotations

import os
import threading

_LOCK = threading.Lock()


def results_root() -> str:
    return os.environ.get(
        "ADAVOTE_RESULTS",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"),
    )


def data_path(name: str) -> str:
    p = os.path.join(results_root(), "data")
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, name)


def fig_path(name: str) -> str:
    p = os.path.join(results_root(), "figs")
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, name)


def e2_path(name: str) -> str:
    p = os.path.join(results_root(), "e2")
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, name)


def summary_file() -> str:
    root = results_root()
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, "summary.txt")


def write_summary(line: str) -> None:
    """Append a single summary line (thread/process-append safe enough for our use)."""
    with _LOCK:
        with open(summary_file(), "a") as f:
            f.write(line.rstrip("\n") + "\n")
    print("[SUMMARY] " + line.rstrip("\n"))
