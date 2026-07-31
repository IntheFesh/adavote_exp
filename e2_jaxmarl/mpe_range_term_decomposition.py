"""
mpe_range_term_decomposition.py — Task 7 §1.5: record that the MPE range
term is independent of B_Q's estimation method.

Because the code sets B_Q = Rmax_hat (so b0 = nH * Rmax_hat), the range term
in Theorem 3's empirical-Bernstein bound

    Bhat/Rmax = (Xbar + sqrt(2*sigma2_X*ln(2/delta_B)/m)) / Rmax
                + 7*nH*ln(2/delta_B) / (3*(m-1))

has Rmax cancel EXACTLY out of its second term, which is therefore
determined by m (and n, H, delta_B) alone -- independent of how B_Q is
estimated. This is a read-only diagnostic: no rerun, no code change to the
certificate itself, just a recorded decomposition of the three already
-committed MPE budget rows.

Outputs:
    results/e2/mpe_range_term_decomposition.json
"""

import json

import numpy as np

FILES = {
    "m150_K100":    ("results/e2/rollout_certificate_simple_spread_ippo_jensen_m150.json", "100"),
    "m1500_K8000":  ("results/e2/rollout_certificate_simple_spread_ippo_jensen_m1500.json", "8000"),
    "m3000_K25000": ("results/e2/rollout_certificate_simple_spread_ippo_jensen_m3000k25000.json", "25000"),
}


def decompose(path: str, K: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    n, T, m, delta_B = d["n"], d["T"], d["m"], d["delta_B"]
    pk = d["per_K"][K]
    Xbar, Xvar, Rmax = pk["X_bar"], pk["X_var"], d["Rmax_hat"]
    L = np.log(2.0 / delta_B)
    nH = n * T
    mean_var_term_over_Rmax = (Xbar + np.sqrt(2.0 * Xvar * L / m)) / Rmax
    range_term_over_Rmax = 7.0 * nH * L / (3.0 * (m - 1))
    total = mean_var_term_over_Rmax + range_term_over_Rmax
    return dict(
        m=m, K=int(K), n=n, T=T, delta_B=delta_B, nH=nH, Rmax_hat=Rmax,
        mean_plus_variance_term_over_Rmax=float(mean_var_term_over_Rmax),
        range_term_over_Rmax=float(range_term_over_Rmax),
        total_ratio_Rmax=float(total),
        reported_ratio_Rmax=float(pk["ratio_Rmax"]),
        range_term_pct_of_total=float(100.0 * range_term_over_Rmax / total),
        note="range_term_over_Rmax = 7*nH*ln(2/delta_B)/(3*(m-1)); Rmax cancels "
             "exactly because b0 = nH*Rmax_hat, so this term depends only on "
             "(n,T,delta_B,m), not on how B_Q/Rmax_hat is estimated.",
    )


def main():
    out = {}
    for label, (path, K) in FILES.items():
        row = decompose(path, K)
        out[label] = row

    out_path = "results/e2/mpe_range_term_decomposition.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # Self-assertion: recomputed total must match the certificate's own
    # reported ratio_Rmax for all three rows.
    for label, row in out.items():
        assert np.isclose(row["total_ratio_Rmax"], row["reported_ratio_Rmax"], rtol=1e-9), \
            f"{label}: decomposition does not sum to the reported ratio"

    with open(out_path) as f:
        check = json.load(f)
    for label in out:
        assert np.isclose(check[label]["total_ratio_Rmax"], out[label]["total_ratio_Rmax"])

    print(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
