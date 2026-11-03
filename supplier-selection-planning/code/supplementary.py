"""Supplementary analyses for the report: data description, an independent
multi-criteria ranking for Problem 1, stability of the SAA selection, a stress
test with correlated deliveries, the cost-compliance trade-off of the order
plan, the production-compliance curve of Problem 4 and parameter sensitivity.

    python code/supplementary.py      # about 10 minutes, fixed seed

Requires the outputs of code/run_all.py (results/results.json and the CSV files).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_all as ra  # noqa: E402
from run_all import (DEMAND, NT, SERVICE, TCAP, WEEKS, WINDOW, CB, draw_ratios, load,  # noqa: E402
                     order_lp, plt, saa_selection, save, service, simulate, stock_sd,
                     sustained_capacity, tuned_plan)

RES = ra.RES
SUP = RES / "supplementary"
SUP.mkdir(exist_ok=True)
RNG = np.random.default_rng(2022)
log = ra.log


def pools_with_cap(O, S, cap):
    out = []
    for i in range(len(O)):
        on = O[i] > 0
        r = S[i, on] / O[i, on] if on.any() else np.zeros(1)
        out.append(np.clip(r, 0, cap) if cap is not None else r)
    return out


def iterate_plan(idx, K, alpha, price, rbar, var, z, demand=DEMAND):
    x = np.repeat(K[idx][:, None], WEEKS, 1)
    for _ in range(3):
        safety = z * stock_sd(x, alpha[idx], var[idx])
        xn = order_lp(K[idx], alpha[idx], price[idx], rbar[idx], safety, 0.0, demand)
        if xn is None:
            return None, None
        x = xn
    return x, safety


def block_bootstrap_ratios(Rmat, pools, idx, n_scen, rng, block=4):
    """Ratios rho[s, k, t] taken from common historical weeks for all suppliers
    (blocks of `block` consecutive weeks), so that co-movements across suppliers
    and short-run persistence are kept.  Where a supplier had no order in the
    sampled week, its ratio is drawn from its own pool."""
    T = Rmat.shape[1]
    out = np.empty((n_scen, len(idx), WEEKS))
    for s in range(n_scen):
        weeks = np.concatenate([np.arange(b, b + block) for b in rng.integers(0, T - block + 1, WEEKS // block)])
        sub = Rmat[np.ix_(idx, weeks)]
        for k, i in enumerate(idx):
            miss = np.isnan(sub[k])
            if miss.any():
                sub[k, miss] = rng.choice(pools[i], size=miss.sum())
        out[s] = sub
    return out


def entropy_topsis(X, benefit):
    """Entropy-weight TOPSIS on indicator matrix X (n x m)."""
    Z = X.copy().astype(float)
    for j in range(Z.shape[1]):
        lo, hi = Z[:, j].min(), Z[:, j].max()
        Z[:, j] = (Z[:, j] - lo) / (hi - lo) if benefit[j] else (hi - Z[:, j]) / (hi - lo)
    P = (Z + 1e-12) / (Z + 1e-12).sum(0)
    e = -(P * np.log(P)).sum(0) / np.log(len(Z))
    w = (1 - e) / (1 - e).sum()
    V = Z * w
    dp = np.sqrt(((V - V.max(0)) ** 2).sum(1))
    dm = np.sqrt(((V - V.min(0)) ** 2).sum(1))
    return dm / (dp + dm), w


def plot_tradeoff(trade, fc):
    """Compliance and expected cost of the LP plans against the safety factor z;
    the dotted lines mark ordering every selected supplier at capacity."""
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(trade.z, trade.p_two_weeks, "o-", c=CB[0], ms=4, label="compliance (left)")
    ax.axhline(fc["p_two_weeks"], ls=":", c=CB[0], lw=0.9)
    ax.axhline(SERVICE, c="#999999", lw=0.8)
    ax.set_xlabel("safety factor z")
    ax.set_ylabel("P(two-week stock in all 24 weeks)")
    ax.set_ylim(0, 1.02)
    ax2 = ax.twinx()
    ax2.plot(trade.z, trade.cost / 1000, "s--", c=CB[1], ms=3.5, label="expected cost (right)")
    ax2.axhline(fc["expected_cost"] / 1000, ls=":", c=CB[1], lw=0.9)
    ax2.set_ylabel("expected cost over 24 weeks (thousand)")
    ax2.set_ylim(480, 550)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right")
    save(fig, "q2_tradeoff")


def main():
    t0 = time.time()
    base = json.loads((RES / "results.json").read_text())
    ids, mat, O, S, L = load()
    n = len(ids)
    alpha = np.array([ra.ALPHA[m] for m in mat])
    price = np.array([ra.PRICE[m] for m in mat])
    pools = ra.ratio_pool(O, S)
    rbar = np.array([p.mean() for p in pools])
    var = np.array([p.var() for p in pools])
    K = sustained_capacity(S, WINDOW)
    score = K * rbar / alpha
    prodS = S / alpha[:, None]
    out = {}

    # ------------------------------------------------------------ data description
    log("data description")
    on = O > 0
    allr = np.clip(S[on] / O[on], 0, 1.5)
    out["data"] = {
        "suppliers_by_material": {m: int((mat == m).sum()) for m in "ABC"},
        "suppliers_K_positive": int((K > 0).sum()),
        "suppliers_score_ge_100": int((score >= 100).sum()),
        "ratio_quantiles": dict(zip(["q05", "q25", "q50", "q75", "q95"],
                                    np.quantile(allr, [.05, .25, .5, .75, .95]).tolist())),
        "ratio_share_within_10pct": float(np.mean(np.abs(allr - 1) <= 0.1)),
        "ratio_share_above_cap": float(np.mean(S[on] / O[on] > 1.5)),
        "gini_score": float(1 - 2 * np.trapezoid(np.r_[0, np.cumsum(np.sort(score)) / score.sum()],
                                              dx=1 / n)),
        "top_k_share": {k: float(np.sort(score)[::-1][:k].sum() / score.sum()) for k in (10, 20, 50, 120)},
        "history_share_by_material": {m: float(prodS[mat == m].sum() / prodS.sum()) for m in "ABC"},
    }
    big = np.argsort(-S.sum(1))[:50]
    rel = []
    for i in big:
        m = O[i] > 0
        rel.append(np.std(S[i, m] / O[i, m]) if m.sum() > 2 else np.nan)
    out["data"]["ratio_sd_top50_median"] = float(np.nanmedian(rel))
    # carriers
    car = []
    for k in range(NT):
        pos = L[k][L[k] > 0]
        car.append({"carrier": f"T{k + 1}", "active_weeks": int((L[k] > 0).sum()), "mean_loss": float(pos.mean()),
                    "sd_loss": float(pos.std()), "q95_loss": float(np.quantile(pos, .95)),
                    "max_loss": float(pos.max())})
    pd.DataFrame(car).to_csv(SUP / "carriers.csv", index=False)
    out["carriers"] = car

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
    axs[0].hist(allr, bins=np.linspace(0, 1.5, 61), color=CB[0])
    axs[0].set_xlabel("delivered / ordered"); axs[0].set_ylabel("supplier-weeks")
    axs[0].set_yscale("log")
    srt = np.sort(score)[::-1]
    axs[1].plot(np.arange(1, n + 1), np.cumsum(srt) / srt.sum(), c=CB[0])
    for k, c_ in [(50, CB[1]), (120, CB[2])]:
        axs[1].axvline(k, ls="--", lw=0.8, c=c_, label=f"top {k}")
    axs[1].set_xscale("log"); axs[1].set_xlabel("suppliers ranked by score")
    axs[1].set_ylabel("cumulative share of score"); axs[1].legend()
    fig.tight_layout()
    save(fig, "data_ratio_pareto")

    # ------------------------------------------------------------ Q1 entropy-weight TOPSIS cross-check
    log("Q1: entropy-weight TOPSIS")
    pos_d = [S[i][S[i] > 0] for i in range(n)]
    cv = np.array([p.std() / p.mean() if len(p) > 1 else 3.0 for p in pos_d])
    X = np.c_[prodS.sum(1), K / alpha, [np.mean(p >= 0.9) for p in pools], (S > 0).mean(1), np.minimum(cv, 3.0)]
    names = ["total product delivered", "sustained capacity (product)", "share of ratios >= 0.9",
             "share of weeks with delivery", "CV of positive deliveries"]
    C, w = entropy_topsis(X, [True, True, True, True, False])
    top = np.argsort(-score)[:50]
    topT = np.argsort(-C)[:50]
    out["q1_topsis"] = {"weights": dict(zip(names, w.tolist())),
                        "kendall_tau": float(stats.kendalltau(score, C).statistic),
                        "spearman": float(stats.spearmanr(score, C).statistic),
                        "top50_overlap": len(set(top) & set(topT)),
                        "top10_overlap": len(set(top[:10]) & set(topT[:10])),
                        "only_topsis": sorted(ids[list(set(topT) - set(top))].tolist()),
                        "only_score": sorted(ids[list(set(top) - set(topT))].tolist()),
                        "share_capacity_topsis50": float(score[topT].sum() / score.sum())}
    pd.DataFrame({"id": ids, "material": mat, "score": score, "topsis": C,
                  "rank_score": stats.rankdata(-score, method="min").astype(int),
                  "rank_topsis": stats.rankdata(-C, method="min").astype(int)}).to_csv(SUP / "q1_topsis.csv", index=False)

    # ------------------------------------------------------------ Q2 SAA stability over samples
    log("Q2: SAA stability")
    sel0 = np.array(sorted(np.flatnonzero(np.isin(ids, base["q2_selection"]["suppliers"]))))
    cand = np.argsort(-score)[:120]
    rows = []
    for N, reps in [(100, 5), (200, 5), (400, 3)]:
        for r in range(reps):
            rng = np.random.default_rng(1000 + 10 * N + r)
            prod = draw_ratios(pools, cand, N, rng) * (K[cand] / alpha[cand])[None, :, None]
            sel, res = saa_selection(prod, 0.05, time_limit=600)
            idx = cand[sel]
            sv = service(simulate(np.repeat(K[idx][:, None], WEEKS, 1), alpha[idx],
                                  draw_ratios(pools, idx, 5000, RNG)))
            rows.append({"N": N, "rep": r, "n": len(idx), "mip_gap": getattr(res, "mip_gap", None),
                         "overlap_with_base": len(set(idx) & set(sel0)), "oos_p_two_weeks": sv["p_two_weeks"]})
            log(f"  N={N} rep={r}: {len(idx)} suppliers, overlap {rows[-1]['overlap_with_base']}, "
                f"oos {sv['p_two_weeks']:.3f}")
    pd.DataFrame(rows).to_csv(SUP / "q2_saa_stability.csv", index=False)

    # greedy benchmark: add suppliers by score until out-of-sample compliance >= 95%
    od = np.argsort(-score)
    greedy = []
    for m in range(14, 26):
        idx = od[:m]
        sv = service(simulate(np.repeat(K[idx][:, None], WEEKS, 1), alpha[idx], draw_ratios(pools, idx, 5000, RNG)))
        greedy.append({"n": m, "p_two_weeks": sv["p_two_weeks"]})
    pd.DataFrame(greedy).to_csv(SUP / "q2_greedy.csv", index=False)
    out["q2_greedy_first_n_95"] = next((g["n"] for g in greedy if g["p_two_weeks"] >= SERVICE), None)

    # ------------------------------------------------------------ Q2 correlated deliveries (block bootstrap)
    log("Q2: correlated-delivery stress test")
    Rmat = np.where(O > 0, np.clip(S / np.where(O > 0, O, 1), 0, 1.5), np.nan)
    sub = Rmat[sel0]
    cors = []
    for a in range(len(sel0)):
        for b in range(a + 1, len(sel0)):
            m = ~np.isnan(sub[a]) & ~np.isnan(sub[b])
            if m.sum() > 20 and sub[a, m].std() > 0 and sub[b, m].std() > 0:
                cors.append(np.corrcoef(sub[a, m], sub[b, m])[0, 1])
    ac = []
    for a in range(len(sel0)):
        v = sub[a]; m = ~np.isnan(v[:-1]) & ~np.isnan(v[1:])
        if m.sum() > 20 and v[:-1][m].std() > 0:
            ac.append(np.corrcoef(v[:-1][m], v[1:][m])[0, 1])
    xq2 = pd.read_csv(RES / "q2_order_plan.csv", index_col=0).loc[ids[sel0]].to_numpy()
    xfull = np.repeat(K[sel0][:, None], WEEKS, 1)
    stress = {"pair_corr_median": float(np.median(cors)), "pair_corr_q90": float(np.quantile(cors, .9)),
              "lag1_autocorr_median": float(np.median(ac))}
    for nm, xx in [("full_capacity", xfull), ("cheapest_plan", xq2)]:
        ind = service(simulate(xx, alpha[sel0], draw_ratios(pools, sel0, 5000, RNG)))
        for blk in (1, 4):
            bb = service(simulate(xx, alpha[sel0], block_bootstrap_ratios(Rmat, pools, sel0, 5000, RNG, blk)))
            stress[f"{nm}_block{blk}"] = bb["p_two_weeks"]
        stress[f"{nm}_independent"] = ind["p_two_weeks"]
    out["q2_stress"] = stress
    log(f"  {stress}")

    # ------------------------------------------------------------ Q2 cost-compliance trade-off over z
    log("Q2: cost-compliance trade-off")
    trade = []
    for z in np.arange(0.0, 2.01, 0.25):
        x, safety = iterate_plan(sel0, K, alpha, price, rbar, var, z)
        if x is None:
            break
        e = x * rbar[sel0][:, None]
        stock = simulate(x, alpha[sel0], draw_ratios(pools, sel0, 2000, RNG))
        sv = service(stock)
        trade.append({"z": z, "cost": float((e * price[sel0][:, None]).sum()), **sv,
                      "p_one_week": float(np.mean((stock >= DEMAND - 1e-6).all(1))),
                      "min_stock_q05_weeks": float(np.quantile(stock.min(1), .05) / DEMAND),
                      "end_safety_weeks": float(safety[-1] / DEMAND)})
    trade = pd.DataFrame(trade)
    trade.to_csv(SUP / "q2_tradeoff.csv", index=False)
    plot_tradeoff(trade, base["q2_full_capacity_plan"])

    # weekly structure of the cheapest plan
    exp_prod = (xq2 * (rbar[sel0] / alpha[sel0])[:, None]).sum(0)
    out["q2_plan_structure"] = {
        "suppliers_always_at_capacity": int((np.abs(xq2 - K[sel0][:, None]) < 1e-6).all(1).sum()),
        "swing_suppliers": ids[sel0][~(np.abs(xq2 - K[sel0][:, None]) < 1e-6).all(1)].tolist(),
        "weeks_all_at_capacity": int((np.abs(xq2 - K[sel0][:, None]) < 1e-6).all(0).sum()),
        "expected_product_week1": float(exp_prod[0]), "expected_product_min": float(exp_prod.min()),
        "expected_product_mean": float(exp_prod.mean())}
    tp = pd.read_csv(RES / "q2_transport_plan.csv")
    load_w = tp.groupby(["week", "carrier"]).expected_raw_m3.sum().unstack(fill_value=0)
    out["q2_carrier_weeks_full"] = {c: int((load_w[c] > TCAP - 1).sum()) for c in load_w.columns}
    out["q2_weekly_raw_volume"] = {"min": float(load_w.sum(1).min()), "max": float(load_w.sum(1).max())}

    # ------------------------------------------------------------ Q4 compliance curve and service level
    log("Q4: production-compliance curve")
    allidx = np.flatnonzero(K * rbar > 0)
    curve = []
    for P in np.arange(30000, 42001, 1000):
        r = tuned_plan(allidx, K, alpha, price, rbar, var, pools, RNG, demand=P, n_eval=1000)
        curve.append({"production": float(P), "p_two_weeks": r["sv"]["p_two_weeks"], "z": r["z"]})
    curve = pd.DataFrame(curve)
    curve.to_csv(SUP / "q4_curve.csv", index=False)
    lvl = []
    for target in (0.90, 0.95, 0.99):
        lo_, hi_ = 0.5 * DEMAND, float((K[allidx] * rbar[allidx] / alpha[allidx]).sum())
        for _ in range(12):
            mid = 0.5 * (lo_ + hi_)
            r = tuned_plan(allidx, K, alpha, price, rbar, var, pools, RNG, demand=mid, n_eval=500, target=target)
            if r is not None and r["sv"]["p_two_weeks"] >= target:
                lo_ = mid
            else:
                hi_ = mid
        lvl.append({"service": target, "max_production": lo_, "increase": lo_ / DEMAND - 1})
    pd.DataFrame(lvl).to_csv(SUP / "q4_service_level.csv", index=False)
    out["q4_service_level"] = lvl
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(curve.production / 1000, curve.p_two_weeks, "o-", c=CB[0], ms=4)
    ax.axhline(SERVICE, c="#999999", lw=0.8)
    ax.axvline(DEMAND / 1000, ls="--", c=CB[1], lw=0.8, label="current 28.2")
    ax.set_xlabel("weekly production (thousand m³)")
    ax.set_ylabel("P(two-week stock in all 24 weeks)")
    ax.legend(loc="lower left")
    save(fig, "q4_curve")

    # ------------------------------------------------------------ sensitivity: ratio cap and demand
    log("sensitivity: ratio cap and demand")
    sens = []
    for cap in (1.2, 1.5, 2.0, None):
        pl = pools_with_cap(O, S, cap)
        sv = service(simulate(xfull, alpha[sel0], draw_ratios(pl, sel0, 5000, RNG)))
        sens.append({"parameter": "ratio cap", "value": "none" if cap is None else cap,
                     "p_two_weeks_18_full": sv["p_two_weeks"]})
    for f in (0.95, 0.98, 1.02, 1.05):
        sv = service(simulate(xfull, alpha[sel0], draw_ratios(pools, sel0, 5000, RNG), DEMAND * f), DEMAND * f)
        sens.append({"parameter": "demand factor", "value": f, "p_two_weeks_18_full": sv["p_two_weeks"]})
    # capacity scaled: all K multiplied by a factor (measurement error in capacity)
    for f in (0.95, 1.05):
        sv = service(simulate(xfull * f, alpha[sel0], draw_ratios(pools, sel0, 5000, RNG)))
        sens.append({"parameter": "capacity factor", "value": f, "p_two_weeks_18_full": sv["p_two_weeks"]})
    pd.DataFrame(sens).to_csv(SUP / "sensitivity.csv", index=False)
    out["sensitivity"] = sens

    out["runtime_s"] = time.time() - t0
    (SUP / "supplementary.json").write_text(json.dumps(out, indent=2, default=float))
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
