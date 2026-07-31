"""
tests/test_e1_common.py — unit tests for e1_tabular/_common.py's shared
"build a controller + certificates for a game" scaffolding.

Covered claims:
  * make_alpha_fn returns values within [lo, hi].
  * make_alpha_fn gives distinct alpha across distinct units (not constant).
  * make_alpha_fn is deterministic ACROSS SEPARATE PROCESSES for a fixed
    seed. This is the property that was broken when make_alpha_fn used
    Python's built-in hash() on a tuple containing a str: hash() of a str is
    randomized per-process (PEP 456 SipHash) unless PYTHONHASHSEED is fixed,
    so build()'s true_loss/C2 varied several percent run-to-run despite an
    identical --seed argument. Regression-tests the fix (np.random.default_rng
    with an all-integer entropy list).
"""

import subprocess
import sys

from e1_tabular._common import make_alpha_fn


def test_alpha_in_range():
    alpha_fn = make_alpha_fn(seed=0, lo=0.55, hi=0.9)
    for t in range(3):
        for s in range(3):
            for i in range(2):
                for prefix in [(), (0,), (1, 2)]:
                    a = alpha_fn(t, s, i, prefix)
                    assert 0.55 <= a <= 0.9


def test_alpha_varies_across_units():
    alpha_fn = make_alpha_fn(seed=0)
    vals = {alpha_fn(t, 0, 0, ()) for t in range(10)}
    assert len(vals) > 1, "alpha_fn should not be constant across distinct units"


def test_alpha_deterministic_within_process():
    alpha_fn = make_alpha_fn(seed=0)
    a1 = alpha_fn(1, 2, 0, (1,))
    a2 = alpha_fn(1, 2, 0, (1,))
    assert a1 == a2


def test_alpha_reproducible_across_processes():
    """Regression test for the PYTHONHASHSEED non-determinism bug: two fresh
    Python processes, identical seed/unit key, must return the identical
    alpha (bit-for-bit), not merely close."""
    snippet = (
        "from e1_tabular._common import make_alpha_fn; "
        "print(repr(make_alpha_fn(0)(1, 2, 0, (1,))))"
    )
    outs = []
    for _ in range(3):
        out = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=__file__.rsplit("/tests/", 1)[0],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        outs.append(out)
    assert len(set(outs)) == 1, f"alpha_fn not reproducible across processes: {outs}"


def test_build_true_loss_reproducible_across_processes():
    """End-to-end regression test: build()'s true_loss/C2 must be identical
    across fresh processes for a fixed gseed (previously varied by several
    percent due to the make_alpha_fn hash-randomization bug)."""
    snippet = (
        "from common.games import make_game; "
        "from e1_tabular._common import build, make_alpha_fn, const_fn; "
        "game = make_game(S=4, A=3, H=4, n=2, seed=1826701615, transition='dependent'); "
        "b = build(game, make_alpha_fn(1826701615), const_fn(5), eta=0.0, psi_rule='ref'); "
        "print(repr(b.true_loss), repr(b.chain['Ctilde']))"
    )
    outs = []
    for _ in range(3):
        out = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=__file__.rsplit("/tests/", 1)[0],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        outs.append(out)
    assert len(set(outs)) == 1, f"build() not reproducible across processes: {outs}"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
