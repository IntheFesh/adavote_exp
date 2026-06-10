"""
common/plotting.py — matplotlib PDF styling (vector, embedded fonts).

All figures are saved as vector PDF with Type-42 (TrueType) font embedding
(`pdf.fonttype=42`, `ps.fonttype=42`) so they pass camera-ready font checks.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("pdf")  # non-interactive vector backend
import matplotlib.pyplot as plt  # noqa: E402

# Embed fonts as editable TrueType (Type 42), not Type-3 outlines.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["grid.alpha"] = 0.3
matplotlib.rcParams["figure.autolayout"] = False


def new_fig(figsize=(5.0, 3.5)):
    """Create a fresh figure + axes."""
    return plt.subplots(figsize=figsize)


def save_pdf(fig, path: str):
    """Save a figure as a vector PDF with embedded fonts."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
