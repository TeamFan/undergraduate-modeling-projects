"""Accuracy, convergence and variance-reduction experiments (serial part).

    python code/experiments.py        # about 5 minutes

Writes results/*.csv, results/experiments.json and results/figures/*.
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from bs import bs_call, cn_call, estimates, geometric_asian_call, moments, simulate

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
P = dict(S0=50.0, K=52.0, r=0.05, sigma=0.30)
Z99 = norm.ppf(0.995)
CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
plt.rcParams.update({"font.size": 9, "legend.fontsize": 8, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    out = {"params": P}

    # ------------------------------------------------ 1. CN accuracy for T = 1..4
    log("Crank-Nicolson accuracy")
    rows = []
    for T in [1, 2, 3, 4]:
        ex = bs_call(T=T, **P)
        rows.append({"T": T, "exact": ex,
                     "cn": cn_call(T=T, M=250, N=2000, **P),
                     "cn_plain": cn_call(T=T, M=250, N=2000, rannacher=0, **P)})
    cn = pd.DataFrame(rows)
    cn["cn_err"] = cn.cn - cn.exact
    cn["cn_plain_err"] = cn.cn_plain - cn.exact
    cn.to_csv(RES / "cn_accuracy.csv", index=False)

    # ------------------------------------------------ 2. CN grid convergence
    log("Crank-Nicolson grid refinement")
    rows = []
    ex = bs_call(T=1, **P)
    for N, M in [(50, 13), (100, 25), (200, 50), (400, 100), (800, 200), (1600, 400), (3200, 800)]:
        t = time.perf_counter()
        v = cn_call(T=1, M=M, N=N, **P)
        rows.append({"N": N, "M": M, "price": v, "abs_err": abs(v - ex), "seconds": time.perf_counter() - t})
    conv = pd.DataFrame(rows)
    conv["order"] = np.r_[np.nan, np.log2(conv.abs_err.to_numpy()[:-1] / conv.abs_err.to_numpy()[1:])]
    conv.to_csv(RES / "cn_convergence.csv", index=False)
    out["cn_observed_order_last"] = float(conv.order.iloc[-1])

    # ------------------------------------------------ 3. MC: estimator comparison, replications
    log("Monte Carlo replications (plain / antithetic / control variate)")
    T, M = 1.0, 252
    ex_e = bs_call(T=T, **P)
    reps = 40
    rows = []
    for n in [1_000, 4_000, 16_000, 64_000, 256_000]:
        for rep in range(reps):
            for anti in (False, True):
                rng = np.random.default_rng(np.random.SeedSequence([n, rep, int(anti)]))
                t = time.perf_counter()
                est = estimates(moments(simulate(rng, n, T=T, M=M, antithetic=anti, **P)), T=T, M=M, **P)
                sec = time.perf_counter() - t
                for opt in ("euro", "asian"):
                    for kind in ("plain", "cv"):
                        rows.append({"paths": n, "rep": rep, "antithetic": anti, "option": opt, "estimator": kind,
                                     "value": est[opt][kind], "se": est[opt][f"{kind}_se"], "seconds": sec})
        log(f"  n={n}")
    mc = pd.DataFrame(rows)
    mc.to_csv(RES / "mc_replications.csv", index=False)

    # reference for the arithmetic Asian: large antithetic+CV run
    log("Asian reference value")
    rng = np.random.default_rng(np.random.SeedSequence(99))
    ref = estimates(moments(simulate(rng, 4_000_000, T=T, M=M, antithetic=True, **P)), T=T, M=M, **P)
    asian_ref, asian_ref_se = ref["asian"]["cv"], ref["asian"]["cv_se"]

    truth = {"euro": ex_e, "asian": asian_ref}
    mc["err"] = mc.apply(lambda r: r.value - truth[r.option], axis=1)
    mc["covered"] = (mc.err.abs() <= Z99 * mc.se)
    summ = (mc.groupby(["option", "antithetic", "estimator", "paths"])
              .agg(rmse=("err", lambda e: float(np.sqrt(np.mean(e ** 2)))),
                   mean_se=("se", "mean"), coverage=("covered", "mean"), seconds=("seconds", "mean"))
              .reset_index())
    summ.to_csv(RES / "mc_summary.csv", index=False)
    # variance-reduction factor at the largest n, relative to plain MC (same number of paths)
    big = summ[summ.paths == summ.paths.max()]
    vrf = []
    for opt in ("euro", "asian"):
        base = big[(big.option == opt) & (~big.antithetic) & (big.estimator == "plain")].mean_se.iloc[0]
        for _, r in big[big.option == opt].iterrows():
            vrf.append({"option": opt, "antithetic": bool(r.antithetic), "estimator": r.estimator,
                        "se": r.mean_se, "vrf": (base / r.mean_se) ** 2, "coverage": r.coverage})
    pd.DataFrame(vrf).to_csv(RES / "mc_variance_reduction.csv", index=False)
    out.update({"asian_reference": asian_ref, "asian_reference_se": asian_ref_se,
                "geo_asian_exact": float(geometric_asian_call(T=T, M=M, **P)),
                "euro_exact": ex_e, "replications": reps})

    # ------------------------------------------------ 4. CI coverage for plain MC at small n
    log("Coverage of 99% intervals")
    hits, R = 0, 1000
    for rep in range(R):
        rng = np.random.default_rng(np.random.SeedSequence([7, rep]))
        e = estimates(moments(simulate(rng, 2000, T=T, M=12, **P)), T=T, M=12, **P)["euro"]
        hits += abs(e["plain"] - ex_e) <= Z99 * e["plain_se"]
    out["coverage_99_plain_n2000"] = hits / R
    out["coverage_reps"] = R
    (RES / "experiments.json").write_text(json.dumps(out, indent=2, default=float))

    # ------------------------------------------------ figures
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.loglog(conv.N, conv.abs_err, "o-", c=CB[0], label="CN with Rannacher start")
    ax.loglog(conv.N, conv.abs_err.iloc[0] * (conv.N / conv.N.iloc[0]) ** -2.0, "--", c="#999999", label="slope −2")
    ax.set_xlabel("space steps N (time steps M ≈ N/4)")
    ax.set_ylabel("|error| at S0")
    ax.legend()
    save(fig, "cn_convergence")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=True)
    styles = {(False, "plain"): ("o-", CB[0], "plain"), (True, "plain"): ("s--", CB[1], "antithetic"),
              (False, "cv"): ("^-.", CB[2], "control variate"), (True, "cv"): ("d:", CB[3], "antithetic + CV")}
    for ax, opt, title in [(axes[0], "euro", "a) European call"), (axes[1], "asian", "b) arithmetic Asian call")]:
        for (anti, kind), (ls, col, lab) in styles.items():
            s = summ[(summ.option == opt) & (summ.antithetic == anti) & (summ.estimator == kind)]
            ax.loglog(s.paths, s.rmse, ls, c=col, label=lab, ms=4)
        n = np.array([1e3, 2.56e5])
        s0 = summ[(summ.option == opt) & (~summ.antithetic) & (summ.estimator == "plain")].rmse.iloc[0]
        ax.loglog(n, s0 * (n / 1e3) ** -0.5, c="#999999", lw=0.8, label="slope −1/2")
        ax.set_xlabel("paths")
        ax.set_ylabel(f"RMSE over {reps} replications")
        ax.set_title(title, loc="left")
    axes[0].legend()
    save(fig, "mc_convergence")
    log("done")


if __name__ == "__main__":
    main()
