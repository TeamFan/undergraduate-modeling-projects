"""Method diagnostics that involve no timing and are deterministic for a fixed seed.

    python code/diagnostics.py        # about 1 minute

1. Crank-Nicolson: separate time and space refinement, truncation of the
   S-domain, and the delta/gamma near the strike with and without the
   Rannacher start (including the frozen-coefficient damping estimate).
2. Monte Carlo: sample correlations, control-variate coefficients and the
   variance-reduction factors they predict.
3. Merging of additive statistics: deviation of the merged estimate from the
   single-process estimate for several partitions, compared with averaging
   per-rank estimates; accuracy of the one-pass moment formulas.
4. Performance model: aggregate throughput and the strong-scaling speed-up
   predicted from the weak-scaling efficiency (from results/scaling_summary.csv,
   which was measured by benchmark.py; nothing is re-timed here).

Writes results/diagnostics.json, results/cn_*.csv, results/mc_correlation.csv,
results/merge_check.csv, results/performance_model.csv and
results/figures/cn_greeks.{pdf,png}.
"""

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bs import bs_call, bs_call_greeks, cn_call, cn_grid, estimates, merge, moments, simulate

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
P = dict(S0=50.0, K=52.0, r=0.05, sigma=0.30)
CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
plt.rcParams.update({"font.size": 9, "legend.fontsize": 8, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)


def greeks_fd(S, V):
    """Central-difference delta and gamma at the interior nodes."""
    h = S[1] - S[0]
    return S[1:-1], (V[2:] - V[:-2]) / (2 * h), (V[2:] - 2 * V[1:-1] + V[:-2]) / h ** 2


def cn_studies(out):
    S0, K, r, sigma = P["S0"], P["K"], P["r"], P["sigma"]
    ex = bs_call(T=1.0, **P)

    # separate refinement in time (fine space grid) and in space (fine time grid)
    rows = []
    for M in [10, 20, 40, 80, 160, 320]:
        v = cn_call(T=1.0, M=M, N=6400, **P)
        rows.append({"refine": "time", "N": 6400, "M": M, "abs_err": abs(v - ex)})
    for N in [50, 100, 200, 400, 800, 1600]:
        v = cn_call(T=1.0, M=4000, N=N, **P)
        rows.append({"refine": "space", "N": N, "M": 4000, "abs_err": abs(v - ex)})
    sep = pd.DataFrame(rows)
    sep["order"] = np.nan
    for k in ("time", "space"):
        m = sep.refine == k
        e = sep.loc[m, "abs_err"].to_numpy()
        sep.loc[m, "order"] = np.r_[np.nan, np.log2(e[:-1] / e[1:])]
    sep.to_csv(RES / "cn_refinement.csv", index=False)

    # truncation of the S-domain at fixed spacing dS = 0.1 (N = Smax / 0.1)
    rows = []
    for smax in [1.5, 2.0, 2.5, 3.0, 4.0, 6.0]:
        N = int(round(smax * S0 / 0.1))
        for T in (1.0, 4.0):
            v = cn_call(T=T, M=1000, N=N, smax=smax, **P)
            # by put-call parity the far-field condition omits the put value at Smax
            put = bs_call(smax * S0, K, T, r, sigma) - smax * S0 + K * np.exp(-r * T)
            rows.append({"smax_over_S0": smax, "N": N, "T": T, "abs_err": abs(v - bs_call(T=T, **P)),
                         "put_at_smax": put})
    pd.DataFrame(rows).to_csv(RES / "cn_domain.csv", index=False)

    # delta and gamma near the strike, T = 1, production grid N = 2000 (dS = 0.1)
    T, N, Smax = 1.0, 2000, 4.0 * S0
    dS = Smax / N
    prof = {}
    for M, tag in [(250, "M250"), (100, "M100")]:
        for ran in (0, 2):
            S, V = cn_grid(K, T, r, sigma, Smax, M=M, N=N, rannacher=ran)
            Sm, D, G = greeks_fd(S, V)
            prof[(tag, ran)] = (Sm, D, G)
    Sm = prof[("M250", 0)][0]
    d_ex, g_ex = bs_call_greeks(Sm, K, T, r, sigma)
    win = (Sm >= 0.8 * K) & (Sm <= 1.2 * K)
    pd.DataFrame({"S": Sm[win], "gamma_exact": g_ex[win],
                  "gamma_cn_M250": prof[("M250", 0)][2][win], "gamma_rannacher_M250": prof[("M250", 2)][2][win],
                  "gamma_cn_M100": prof[("M100", 0)][2][win], "gamma_rannacher_M100": prof[("M100", 2)][2][win],
                  "delta_exact": d_ex[win], "delta_cn_M250": prof[("M250", 0)][1][win],
                  "delta_rannacher_M250": prof[("M250", 2)][1][win]}).to_csv(RES / "cn_greeks_profile.csv", index=False)

    # maximal delta/gamma error on [0.8K, 1.2K] against the number of time steps
    rows = []
    for M in [50, 100, 150, 200, 250, 300, 400, 500, 700, 1000]:
        rec = {"M": M, "lambda_K": sigma ** 2 * K ** 2 * (T / M) / dS ** 2,
               "damping_estimate": float(np.exp(-2 * M ** 2 * dS ** 2 / (sigma ** 2 * K ** 2 * T)))}
        for ran, name in [(0, "cn"), (2, "rannacher")]:
            S, V = cn_grid(K, T, r, sigma, Smax, M=M, N=N, rannacher=ran)
            s, D, G = greeks_fd(S, V)
            d, g = bs_call_greeks(s, K, T, r, sigma)
            w = (s >= 0.8 * K) & (s <= 1.2 * K)
            rec[f"delta_err_{name}"] = float(np.abs(D[w] - d[w]).max())
            rec[f"gamma_err_{name}"] = float(np.abs(G[w] - g[w]).max())
            rec[f"price_err_{name}"] = abs(cn_call(T=T, M=M, N=N, rannacher=ran, **P) - ex)
        rows.append(rec)
    gm = pd.DataFrame(rows)
    gm.to_csv(RES / "cn_greeks.csv", index=False)
    out["cn_greeks_M250"] = gm[gm.M == 250].iloc[0].to_dict()
    out["gamma_exact_max_window"] = float(g_ex[win].max())

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.32})
    ax = axes[0]
    s = Sm[win]
    ax.plot(s, prof[("M250", 0)][2][win], "-", c=CB[1], lw=0.8, label="CN, M = 250")
    ax.plot(s, prof[("M250", 2)][2][win], "-", c=CB[0], lw=1.6, label="CN + Rannacher, M = 250")
    ax.plot(s, g_ex[win], ":", c="k", lw=1.0, label="exact")
    ax.set_xlim(46, 58)
    ax.set_ylim(0.0, 0.06)
    ax.set_xlabel("S")
    ax.set_ylabel("gamma")
    ax.set_title("a) gamma near the strike, T = 1", loc="left", fontsize=9)
    ax.legend(loc="upper left")
    ax = axes[1]
    ax.semilogy(gm.M, gm.gamma_err_cn, "o-", c=CB[1], ms=4, label="CN")
    ax.semilogy(gm.M, gm.gamma_err_rannacher, "s-", c=CB[0], ms=4, label="CN + Rannacher")
    ax.semilogy(gm.M, gm.damping_estimate / dS, "--", c="#999999", label=r"$\Delta S^{-1}\exp(-2M^2\Delta S^2/\sigma^2K^2T)$")
    ax.set_ylim(1e-8, 1e2)
    ax.set_xlabel("time steps M (N = 2000)")
    ax.set_ylabel("max |gamma error| on [0.8K, 1.2K]")
    ax.set_title("b) damping of the kink", loc="left", fontsize=9)
    ax.legend(loc="upper right")
    save(fig, "cn_greeks")


def mc_studies(out):
    T, M = 1.0, 252
    n = 1_000_000
    rows = []
    plain = simulate(np.random.default_rng(np.random.SeedSequence(2026)), n, T=T, M=M, **P)
    pairs = simulate(np.random.default_rng(np.random.SeedSequence(2027)), n, T=T, M=M, antithetic=True, **P)
    for opt, ctl in [("euro", "st"), ("asian", "geo")]:
        x, y = plain[opt], plain[ctl]
        xa, ya = pairs[opt], pairs[ctl]
        vx = x.var(ddof=1)
        rho = np.corrcoef(x, y)[0, 1]
        beta = np.cov(x, y)[0, 1] / y.var(ddof=1)
        # antithetic pairs: Var(pair mean) = Var(X) (1 + rho_a) / 2
        vpair = xa.var(ddof=1)
        rho_a = 2 * vpair / vx - 1
        rho_pair = np.corrcoef(xa, ya)[0, 1]
        beta_pair = np.cov(xa, ya)[0, 1] / ya.var(ddof=1)
        rows.append({"option": opt, "control": ctl, "rho_xy": rho, "beta": beta, "rho_antithetic": rho_a,
                     "rho_pair_xy": rho_pair, "beta_pair": beta_pair,
                     "vrf_pred_antithetic": 1 / (1 + rho_a),
                     "vrf_pred_cv": 1 / (1 - rho ** 2),
                     "vrf_pred_both": vx / (2 * vpair * (1 - rho_pair ** 2))})
    corr = pd.DataFrame(rows)
    x = plain["euro"]
    out["euro_payoff_zero_fraction"] = float(np.mean(x == 0.0))
    out["euro_payoff_skewness"] = float(np.mean((x - x.mean()) ** 3) / x.std() ** 3)
    obs = pd.read_csv(RES / "mc_variance_reduction.csv")

    def vrf(opt, anti, est):
        return float(obs[(obs.option == opt) & (obs.antithetic == anti) & (obs.estimator == est)].vrf.iloc[0])
    corr["vrf_obs_antithetic"] = [vrf(o, True, "plain") for o in corr.option]
    corr["vrf_obs_cv"] = [vrf(o, False, "cv") for o in corr.option]
    corr["vrf_obs_both"] = [vrf(o, True, "cv") for o in corr.option]
    corr.to_csv(RES / "mc_correlation.csv", index=False)
    out["mc_correlation_paths"] = n

    # one-pass moment formulas against two-pass reference
    rel = {}
    for k, v in plain.items():
        mu = v.sum() / v.size
        one = (np.sum(v * v) - v.size * mu * mu) / (v.size - 1)
        rel[k] = float(abs(one - v.var(ddof=1)) / v.var(ddof=1))
    out["one_pass_variance_rel_err"] = rel


def merge_check(out):
    T, M = 1.0, 252
    n = 200_003
    smp = simulate(np.random.default_rng(np.random.SeedSequence(31)), n, T=T, M=M, **P)
    ref = estimates(moments(smp), T=T, M=M, **P)
    rows = []
    for p in [1, 2, 3, 4, 6, 8, 10]:
        sizes = [n // p + (1 if q < n % p else 0) for q in range(p)]
        edges = np.r_[0, np.cumsum(sizes)]
        blocks = [{k: v[edges[q]:edges[q + 1]] for k, v in smp.items()} for q in range(p)]
        merged = estimates(merge([moments(b) for b in blocks]), T=T, M=M, **P)
        per = [estimates(moments(b), T=T, M=M, **P) for b in blocks]
        for opt in ("euro", "asian"):
            for kind in ("plain", "cv"):
                se = ref[opt][f"{kind}_se"]
                naive = np.mean([e[opt][kind] for e in per])
                naive_se = np.sqrt(np.sum([e[opt][f"{kind}_se"] ** 2 for e in per])) / p
                rows.append({"ranks": p, "option": opt, "estimator": kind,
                             "merged_minus_serial": merged[opt][kind] - ref[opt][kind],
                             "merged_se_minus_serial": merged[opt][f"{kind}_se"] - se,
                             "naive_minus_serial": naive - ref[opt][kind],
                             "naive_minus_serial_in_se": (naive - ref[opt][kind]) / se,
                             "naive_se_minus_serial": naive_se - se})
    mc = pd.DataFrame(rows)
    mc.to_csv(RES / "merge_check.csv", index=False)
    out["merge_max_abs_dev"] = float(mc.merged_minus_serial.abs().max())
    out["merge_max_abs_dev_se"] = float(mc.merged_se_minus_serial.abs().max())
    stats = moments(simulate(np.random.default_rng(1), 1000, T=T, M=M, **P))
    out["gather_payload"] = {"numbers": len(stats), "pickled_bytes": len(pickle.dumps(stats, protocol=5))}


def performance_model(out):
    s = pd.read_csv(RES / "scaling_summary.csv")
    st = s[s["mode"] == "strong"].set_index("ranks")
    wk = s[s["mode"] == "weak"].set_index("ranks")
    steps, per_rank, strong_paths = 252, 1_000_000, 8_000_000
    rows = []
    for p in st.index:
        rows.append({"ranks": p,
                     "weak_efficiency": wk.efficiency[p],
                     "throughput_weak": p * per_rank * steps / wk.t_mean[p],
                     "throughput_strong": strong_paths * steps / st.t_mean[p],
                     "speedup_pred_from_weak": p * wk.efficiency[p],
                     "speedup_obs": st.speedup[p],
                     "t_per_rank_weak": wk.t_mean[p],
                     "t_per_rank_strong_equiv": st.t_mean[p] * p * per_rank / strong_paths})
    pm = pd.DataFrame(rows)
    pm.to_csv(RES / "performance_model.csv", index=False)
    out["throughput_1_proc"] = float(pm.throughput_weak.iloc[0])


def main():
    out = {"params": P}
    cn_studies(out)
    mc_studies(out)
    merge_check(out)
    performance_model(out)
    (RES / "diagnostics.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
