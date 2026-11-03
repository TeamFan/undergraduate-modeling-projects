"""Raw-material supplier selection, ordering and transport planning (CUMCM 2021,
Problem C).

    python code/run_all.py            # about 15 minutes

Supply model
    Supplier i can be relied on for at most its *sustained capacity* K_i: the
    largest weekly quantity it delivered in every week of some run of W
    consecutive weeks (W = 4 by default), so that isolated spikes do not count.  An order x <= K_i is
    delivered as rho * x, where rho is drawn from the supplier's own history of
    delivered/ordered ratios (including zero deliveries).
Decisions
    Q2 selection: chance-constrained 0-1 programme solved by sample average
        approximation (SAA) over simulated delivery scenarios, checked out of sample.
    Orders (Q2-Q4): linear programmes on expected deliveries with a safety stock
        z * sd_t that grows with the accumulated delivery uncertainty; z is tuned
        so that a fresh simulation meets the service level.
    Transport: per-week flow MILP over the 8 carriers, replayed against the
        historical weekly loss rates.
"""

import json
import time
import warnings
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.optimize import Bounds, LinearConstraint, milp

warnings.filterwarnings("ignore", category=RuntimeWarning)  # spurious matmul warnings (numpy 2.0 + Accelerate)
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "legend.fontsize": 8, "savefig.bbox": "tight"})
CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

DEMAND = 28200.0                              # product m3 per week
ALPHA = {"A": 0.60, "B": 0.66, "C": 0.72}     # raw m3 per product m3
PRICE = {"A": 1.20, "B": 1.10, "C": 1.00}     # price per raw m3, relative to C
TCAP, NT, WEEKS = 6000.0, 8, 24
WINDOW = 4                                    # weeks for the sustained capacity
SERVICE = 0.95                                # target: P(stock >= 2 weeks in every week)
RNG = np.random.default_rng(2021)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------ data and supply model
def load():
    x = pd.read_excel(ROOT / "data" / "attachment1_suppliers.xlsx", sheet_name=None)
    o, s = list(x.values())
    t = pd.read_excel(ROOT / "data" / "attachment2_transporters.xlsx")
    return (o.iloc[:, 0].to_numpy(), o.iloc[:, 1].to_numpy(), o.iloc[:, 2:].to_numpy(float),
            s.iloc[:, 2:].to_numpy(float), t.iloc[:, 1:].to_numpy(float) / 100.0)


def sustained_capacity(S, window):
    """Largest weekly quantity the supplier delivered in *every* week of some run
    of `window` consecutive weeks: max_t min(S[t], ..., S[t+window-1])."""
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(S, window, axis=1).min(-1).max(1)


def ratio_pool(O, S):
    """Each supplier's delivered/ordered ratios on ordered weeks, capped at 1.5."""
    pools = []
    for i in range(len(O)):
        on = O[i] > 0
        pools.append(np.clip(S[i, on] / O[i, on], 0, 1.5) if on.any() else np.zeros(1))
    return pools


def draw_ratios(pools, idx, n_scen, rng):
    """Independent draws rho[s, k, t] for suppliers idx."""
    out = np.empty((n_scen, len(idx), WEEKS))
    for k, i in enumerate(idx):
        out[:, k, :] = rng.choice(pools[i], size=(n_scen, WEEKS))
    return out


def simulate(x, alpha, rho, demand=DEMAND):
    """Stock paths (product m3) for orders x[k, t] and ratio draws rho[s, k, t],
    starting with two weeks of demand."""
    supply = np.einsum("kt,skt->st", x / alpha[:, None], rho)
    return 2 * demand + np.cumsum(supply - demand, axis=1)


def service(stock, demand=DEMAND):
    return {"p_two_weeks": float(np.mean((stock >= 2 * demand - 1e-6).all(1))),
            "p_no_stockout": float(np.mean((stock >= 0).all(1))),
            "end_stock_weeks_median": float(np.median(stock[:, -1]) / demand)}


# ------------------------------------------------------------------ optimisation models
def saa_selection(prod, eps, demand=DEMAND, time_limit=300):
    """Chance-constrained selection by SAA.
    prod[s, i, t]: product-equivalent delivery of supplier i in week t of scenario s
    if it is ordered at its sustained capacity.  Choose y in {0,1}^n and scenario
    flags z_s in {0,1} with sum z_s <= eps * S:
        min sum y   s.t.  sum_{tau<=t} sum_i prod[s,i,tau] y_i >= t*D - M_st z_s  (all s, t)
    i.e. the two-week stock is kept in every week of every non-flagged scenario."""
    S_, n, W = prod.shape
    cum = np.cumsum(prod, axis=2)
    A_y = cum.transpose(0, 2, 1).reshape(S_ * W, n)     # rows ordered (s, t)
    rhs = np.tile(np.arange(1, W + 1) * demand, S_)
    A_z = sp.csr_matrix((rhs, (np.arange(S_ * W), np.repeat(np.arange(S_), W))), shape=(S_ * W, S_))
    A = sp.hstack([sp.csr_matrix(A_y), A_z]).tocsr()
    budget = sp.csr_matrix(np.r_[np.zeros(n), np.ones(S_)][None, :])
    c = np.r_[np.ones(n), np.zeros(S_)]
    res = milp(c, constraints=[LinearConstraint(A, rhs, np.inf), LinearConstraint(budget, 0, np.floor(eps * S_))],
               integrality=np.ones(n + S_), bounds=Bounds(0, 1), options={"time_limit": time_limit})
    if res.x is None:
        return None, res
    return np.flatnonzero(res.x[:n] > 0.5), res


def order_lp(K, alpha, price, rbar, safety, w_volume=0.0, demand=DEMAND, volume_cap=None, transport_cap=NT * TCAP):
    """Orders 0 <= x_kt <= K_k minimising expected purchase cost (+ w per m3 of raw
    volume) such that the expected stock exceeds the two-week level by at least
    safety_t in every week t (stock starts at two weeks):
        sum_{tau<=t} sum_k rbar_k x_k,tau / alpha_k >= t * D + safety_t,
    and expected deliveries per week fit the carriers (48,000 m3)."""
    n = len(K)
    rows, lo, hi = [], [], []
    for t in range(WEEKS):
        r = np.zeros((n, WEEKS)); r[:, :t + 1] = (rbar / alpha)[:, None]
        rows.append(r.ravel()); lo.append((t + 1) * demand + safety[t]); hi.append(np.inf)
        r = np.zeros((n, WEEKS)); r[:, t] = rbar
        rows.append(r.ravel()); lo.append(-np.inf); hi.append(transport_cap)
    if volume_cap is not None:
        rows.append(np.repeat(rbar, WEEKS)); lo.append(-np.inf); hi.append(volume_cap)
    c = np.repeat((price + w_volume) * rbar, WEEKS)
    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi), bounds=Bounds(0, np.repeat(K, WEEKS)))
    return None if res.x is None else res.x.reshape(n, WEEKS)


def stock_sd(x, alpha, var):
    """Standard deviation of cumulative product supply up to each week when
    delivery ratios are independent with variances var_k."""
    return np.sqrt(np.cumsum(((x / alpha[:, None]) ** 2 * var[:, None]).sum(0)))


def tuned_plan(idx, K, alpha, price, rbar, var, pools, rng, demand=DEMAND, w_volume=0.0, volume_cap=None,
               n_eval=2000, target=SERVICE):
    """Safety stock that grows with the accumulated delivery uncertainty:
    safety_t = z * sd_t(x), where sd_t is the standard deviation of cumulative
    supply up to week t.  Because sd depends on the plan, the LP is re-solved three
    times with sd taken from the previous plan.  z is increased in steps of 0.25
    until the plan meets `target` on freshly simulated scenarios; if no z does,
    the best plan found is returned."""
    best = None
    for z in np.arange(0.0, 6.01, 0.25):
        x = np.repeat(K[idx][:, None], WEEKS, 1)
        ok = True
        for _ in range(3):
            safety = z * stock_sd(x, alpha[idx], var[idx])
            xn = order_lp(K[idx], alpha[idx], price[idx], rbar[idx], safety, w_volume, demand, volume_cap)
            if xn is None:
                ok = False
                break
            x = xn
        if not ok:
            break
        sv = service(simulate(x, alpha[idx], draw_ratios(pools, idx, n_eval, rng), demand), demand)
        cand = {"z": float(z), "x": x, "sv": sv, "safety": safety}
        if best is None or sv["p_two_weeks"] > best["sv"]["p_two_weeks"]:
            best = cand
        if sv["p_two_weeks"] >= target:
            return cand
    return best


def transport_week(d, rate, cap=TCAP, arc_penalty=0.1):
    """min sum rate_k f_ik + arc_penalty * sum z_ik
       s.t. sum_k f_ik = d_i, f_ik <= min(d_i, cap) z_ik, sum_i f_ik <= cap.
    The small penalty makes each supplier use one carrier whenever possible."""
    idx = np.flatnonzero(d > 1e-9)
    n = len(idx)
    if n == 0:
        return [], 0.0
    nf = n * NT
    A = sp.lil_matrix((n + NT + nf, 2 * nf))
    lo, hi = [], []
    for s in range(n):
        A[s, s * NT:(s + 1) * NT] = 1
        lo.append(d[idx[s]]); hi.append(d[idx[s]])
    for k in range(NT):
        for s in range(n):
            A[n + k, s * NT + k] = 1
        lo.append(0); hi.append(cap)
    for s in range(n):
        for k in range(NT):
            r = n + NT + s * NT + k
            A[r, s * NT + k] = 1
            A[r, nf + s * NT + k] = -min(d[idx[s]], cap)
            lo.append(-np.inf); hi.append(0)
    c = np.r_[np.tile(rate, n), np.full(nf, arc_penalty)]
    res = milp(c, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=np.r_[np.zeros(nf), np.ones(nf)],
               bounds=Bounds(0, np.r_[np.full(nf, np.inf), np.ones(nf)]), options={"time_limit": 60})
    f = res.x[:nf].reshape(n, NT)
    return ([(int(idx[s]), k, float(f[s, k])) for s in range(n) for k in range(NT) if f[s, k] > 1e-6],
            float(np.sum(f * rate[None, :])))


def transport_plan(x, rbar, rate):
    sched, loss, vol = [], 0.0, 0.0
    for t in range(WEEKS):
        d = x[:, t] * rbar
        a, l = transport_week(d, rate)
        sched.append(a); loss += l; vol += d.sum()
    return sched, loss, vol


def replay_loss(sched, L, rng, n=500):
    """Loss rate of the schedule under the weekly loss rates of random historical
    24-week windows (a recorded 0 means the carrier was idle: resample a positive week)."""
    out = []
    pos = [L[k][L[k] > 0] for k in range(NT)]
    for _ in range(n):
        s0 = rng.integers(0, L.shape[1] - WEEKS + 1)
        tot = vol = 0.0
        for t, a in enumerate(sched):
            for (_, k, q) in a:
                r = L[k, s0 + t]
                tot += q * (r if r > 0 else rng.choice(pos[k]))
                vol += q
        out.append(tot / vol)
    return np.array(out)


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    ids, mat, O, S, L = load()
    n = len(ids)
    alpha = np.array([ALPHA[m] for m in mat])
    price = np.array([PRICE[m] for m in mat])
    pools = ratio_pool(O, S)
    rbar = np.array([p.mean() for p in pools])
    var = np.array([p.var() for p in pools])
    K = sustained_capacity(S, WINDOW)
    prodS = S / alpha[:, None]
    out = {"suppliers": n, "weeks": S.shape[1],
           "hist_mean_product_delivered": float(prodS.sum(0).mean()),
           "hist_mean_product_ordered": float((O / alpha[:, None]).sum(0).mean()),
           "hist_weeks_below_demand": int((prodS.sum(0) < DEMAND).sum()),
           "share_filled_orders": float(((S >= O) & (O > 0)).sum() / (O > 0).sum()),
           "share_zero_deliveries": float(((S == 0) & (O > 0)).sum() / (O > 0).sum())}
    big = np.argsort(-S.sum(1))[:20]
    corr = []
    for i in big:
        m = (O[i] > 0) & (S[i] > 0)
        corr.append(np.corrcoef(np.log(O[i, m]), np.log(S[i, m]))[0, 1])
    out["order_delivery_log_corr_top20_median"] = float(np.median(corr))
    out["capacity_total_product"] = {f"W{w}": float((sustained_capacity(S, w) * rbar / alpha).sum())
                                     for w in (3, 4, 6, 8)}
    pd.DataFrame({"id": ids, "material": mat, "alpha": alpha, "weeks_ordered": (O > 0).sum(1),
                  "mean_delivery": S.mean(1), "max_delivery": S.max(1),
                  **{f"K_W{w}": sustained_capacity(S, w) for w in (3, 4, 6, 8)},
                  "mean_ratio": rbar, "p_zero_delivery": [np.mean(p == 0) for p in pools],
                  "p_ratio_ge_090": [np.mean(p >= 0.9) for p in pools]}).to_csv(RES / "supplier_model.csv", index=False)

    # ---------------------------------------------------------- Q1
    log("Q1: importance")
    score = K * rbar / alpha            # expected weekly product the supplier can be relied on for
    rank = np.argsort(-score)
    top = rank[:50]
    variants = {"window 3 weeks": sustained_capacity(S, 3) * rbar / alpha,
                "window 6 weeks": sustained_capacity(S, 6) * rbar / alpha,
                "window 8 weeks": sustained_capacity(S, 8) * rbar / alpha,
                "reliability P(ratio>=0.9)": K * np.array([np.mean(p >= 0.9) for p in pools]) / alpha,
                "historical mean delivery": prodS.mean(1)}
    rob = [{"variant": k, "kendall_tau": float(stats.kendalltau(score, v).statistic),
            "top50_overlap": len(set(top) & set(np.argsort(-v)[:50]))} for k, v in variants.items()]
    pd.DataFrame(rob).to_csv(RES / "q1_robustness.csv", index=False)
    pd.DataFrame({"rank": np.arange(1, 51), "id": ids[top], "material": mat[top],
                  "sustained_capacity_raw_m3": K[top], "mean_ratio": rbar[top],
                  "score_product_m3_per_week": score[top]}).to_csv(RES / "q1_top50.csv", index=False)
    out["q1"] = {"top50_share_of_capacity": float(score[top].sum() / score.sum()),
                 "top50_share_of_history": float(prodS[top].sum() / prodS.sum()),
                 "materials_top50": {m: int((mat[top] == m).sum()) for m in "ABC"},
                 "top10": list(ids[rank[:10]]), "robustness": rob}

    # ---------------------------------------------------------- Q2 selection (SAA chance constraint)
    log("Q2: chance-constrained selection (SAA)")
    cand = rank[:120]
    out["q2_candidate_share_of_capacity"] = float(score[cand].sum() / score.sum())
    n_saa = 200
    prod_saa = draw_ratios(pools, cand, n_saa, RNG) * (K[cand] / alpha[cand])[None, :, None]
    rows, sel_eps = [], {}
    for eps in [0.0, 0.05, 0.10, 0.20, 0.50]:
        sel, res = saa_selection(prod_saa, eps)
        if sel is None:
            rows.append({"eps": eps, "n": None, "status": res.status})
            continue
        idx = cand[sel]
        sel_eps[eps] = idx
        sv = service(simulate(np.repeat(K[idx][:, None], WEEKS, 1), alpha[idx], draw_ratios(pools, idx, 5000, RNG)))
        rows.append({"eps": eps, "n": len(idx), "status": res.status, "mip_gap": getattr(res, "mip_gap", None),
                     "in_sample_p": 1 - eps, **{f"oos_{k}": v for k, v in sv.items()},
                     "expected_product_per_week": float((K[idx] * rbar[idx] / alpha[idx]).sum())})
        log(f"  eps={eps}: {len(idx)} suppliers, out-of-sample P(two weeks) = {sv['p_two_weeks']:.3f}")
    q2sel = pd.DataFrame(rows)
    q2sel.to_csv(RES / "q2_selection.csv", index=False)
    # benchmark ignoring uncertainty: largest expected capacities until the expected supply covers demand
    od = np.argsort(-(K * rbar / alpha))
    n_det = int(np.searchsorted(np.cumsum((K * rbar / alpha)[od]), DEMAND) + 1)
    idx = od[:n_det]
    sv = service(simulate(np.repeat(K[idx][:, None], WEEKS, 1), alpha[idx], draw_ratios(pools, idx, 5000, RNG)))
    out["q2_expected_value_benchmark"] = {"n": n_det, **sv}
    ok = q2sel[q2sel.oos_p_two_weeks >= SERVICE].sort_values(["n", "eps"], ascending=[True, False])
    eps_star = float(ok.iloc[0].eps)
    sel = sel_eps[eps_star]
    out["q2_selection"] = {"eps": eps_star, "n": len(sel), "suppliers": sorted(ids[sel]),
                           "materials": {m: int((mat[sel] == m).sum()) for m in "ABC"}}

    # ---------------------------------------------------------- Q2 orders and transport
    log("Q2: order plan and transport")
    tp_ = tuned_plan(sel, K, alpha, price, rbar, var, pools, RNG)
    keep, x, sv = tp_["z"], tp_["x"], tp_["sv"]
    exp = x * rbar[sel][:, None]
    hist_cpp = float(np.mean([(S[:, w:w + WEEKS] * price[:, None]).sum() / prodS[:, w:w + WEEKS].sum()
                              for w in range(0, S.shape[1] - WEEKS + 1)]))
    rate = np.array([L[k][L[k] > 0].mean() for k in range(NT)])
    sched, tl, tv = transport_plan(x, rbar[sel], rate)
    rl = replay_loss(sched, L, RNG)
    base = []                     # baseline: random carrier with spare room for every shipment
    for _ in range(200):
        tot = vol = 0.0
        for t in range(WEEKS):
            tload = np.zeros(NT)
            for v in x[:, t] * rbar[sel]:
                rem = v
                while rem > 1e-9:
                    k = RNG.choice(np.flatnonzero(tload < TCAP - 1e-9))
                    q = min(rem, TCAP - tload[k]); tload[k] += q; rem -= q
                    tot += q * rate[k]; vol += q
        base.append(tot / vol)
    # alternative: order every selected supplier at its sustained capacity
    xf = np.repeat(K[sel][:, None], WEEKS, 1)
    ef = xf * rbar[sel][:, None]
    svf = service(simulate(xf, alpha[sel], draw_ratios(pools, sel, 5000, RNG)))
    out["q2_full_capacity_plan"] = {**svf, "expected_cost": float((ef * price[sel][:, None]).sum()),
                                    "expected_product": float((ef / alpha[sel][:, None]).sum())}
    tp = pd.DataFrame([(t + 1, ids[sel][i], f"T{k + 1}", q) for t, a in enumerate(sched) for (i, k, q) in a],
                      columns=["week", "supplier", "carrier", "expected_raw_m3"])
    tp.to_csv(RES / "q2_transport_plan.csv", index=False)
    pd.DataFrame(np.round(x, 1), index=ids[sel], columns=[f"W{t + 1}" for t in range(WEEKS)]).to_csv(RES / "q2_order_plan.csv")
    out["q2_plan"] = {"safety_z": keep, "safety_final_weeks": float(tp_["safety"][-1] / DEMAND), **sv,
                      "expected_cost": float((exp * price[sel][:, None]).sum()),
                      "cost_per_product": float((exp * price[sel][:, None]).sum() / (exp / alpha[sel][:, None]).sum()),
                      "historical_cost_per_product": hist_cpp,
                      "expected_raw_by_material": {m: float(exp[mat[sel] == m].sum()) for m in "ABC"},
                      "carrier_mean_loss": rate.tolist(),
                      "transport_expected_loss_m3": tl, "transport_expected_rate": tl / tv,
                      "transport_replay_rate_mean": float(rl.mean()),
                      "transport_replay_rate_p95": float(np.quantile(rl, .95)),
                      "transport_random_rate_mean": float(np.mean(base)),
                      "carrier_volume": tp.groupby("carrier").expected_raw_m3.sum().round(0).to_dict(),
                      "supplier_weeks_split": int(tp.groupby(["week", "supplier"]).size().gt(1).sum()),
                      "supplier_weeks": int(tp.groupby(["week", "supplier"]).ngroups)}

    # ---------------------------------------------------------- Q3 lexicographic: volume first, then cost
    log("Q3: prefer A, reduce C (lexicographic volume, then cost)")
    allidx = np.flatnonzero(K * rbar > 0)
    t3 = tuned_plan(allidx, K, alpha, price, rbar, var, pools, RNG, w_volume=10.0)
    keep3, x3, sv3, safety3 = t3["z"], t3["x"], t3["sv"], t3["safety"]
    vmin = float((x3 * rbar[allidx][:, None]).sum())
    rows = []
    for delta in [0.0, 0.005, 0.01, 0.02, 0.05, None]:
        cap = None if delta is None else vmin * (1 + delta) + 1e-6
        xd = order_lp(K[allidx], alpha[allidx], price[allidx], rbar[allidx], safety3, 0.0, DEMAND, cap)
        e = xd * rbar[allidx][:, None]
        svd = service(simulate(xd, alpha[allidx], draw_ratios(pools, allidx, 1000, RNG)))
        rows.append({"volume_slack": -1.0 if delta is None else delta,
                     "raw_m3": float(e.sum()), **{f"{m}_m3": float(e[mat[allidx] == m].sum()) for m in "ABC"},
                     "cost": float((e * price[allidx][:, None]).sum()),
                     "suppliers_used": int((xd.sum(1) > 1e-6).sum()), **svd})
        if delta == 0.0:
            _, tl3, tv3 = transport_plan(xd, rbar[allidx], rate)
        if delta is None:
            _, tl3c, tv3c = transport_plan(xd, rbar[allidx], rate)
    q3 = pd.DataFrame(rows)
    q3.to_csv(RES / "q3_lexicographic.csv", index=False)
    out["q3"] = {"safety_z": keep3, "service_volume_min": sv3, "v_min": vmin,
                 "transport_rate_volume_min": tl3 / tv3, "transport_rate_cost_only": tl3c / tv3c}

    # ---------------------------------------------------------- Q4 maximum production
    log("Q4: maximum sustainable production")
    q4 = []
    for w in (3, 4, 6, 8):
        Kw = sustained_capacity(S, w)
        idx = np.flatnonzero(Kw * rbar > 0)
        lo_, hi_ = 0.5 * DEMAND, float((Kw[idx] * rbar[idx] / alpha[idx]).sum())
        best = None
        for _ in range(12):
            mid = 0.5 * (lo_ + hi_)
            r = tuned_plan(idx, Kw, alpha, price, rbar, var, pools, RNG, demand=mid, n_eval=500)
            if r is not None and r["sv"]["p_two_weeks"] >= SERVICE:
                lo_, best = mid, r
            else:
                hi_ = mid
        peak = float((best["x"] * rbar[idx][:, None]).sum(0).max()) if best is not None else None
        q4.append({"window": w, "capacity_expected_product": float((Kw[idx] * rbar[idx] / alpha[idx]).sum()),
                   "max_production_95": lo_, "increase": lo_ / DEMAND - 1,
                   "peak_weekly_transport_m3": peak, "safety_z": None if best is None else best["z"]})
        log(f"  window {w}: P_max = {lo_:.0f}")
    pd.DataFrame(q4).to_csv(RES / "q4_capacity.csv", index=False)
    out["q4"] = q4
    out["runtime_s"] = time.time() - t0
    (RES / "results.json").write_text(json.dumps(out, indent=2, default=float))

    # ---------------------------------------------------------- figures
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    wk = np.arange(1, S.shape[1] + 1)
    ax.plot(wk, (O / alpha[:, None]).sum(0) / 1000, "--", c=CB[1], lw=0.8, label="ordered")
    ax.plot(wk, prodS.sum(0) / 1000, c=CB[0], lw=0.8, label="delivered")
    ax.axhline(DEMAND / 1000, c="#999999", lw=0.8, label="demand 28.2")
    ax.set_xlabel("week"); ax.set_ylabel("thousand m³ of product equivalent")
    ax.legend()
    save(fig, "history")

    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ok_ = q2sel.dropna(subset=["n"])
    ax.plot(ok_.n, ok_.oos_p_two_weeks, "o-", c=CB[0], label="out of sample (5,000 scenarios)")
    ax.plot(ok_.n, ok_.in_sample_p, "s--", c=CB[1], label=r"in sample, $1-\epsilon$ (200 scenarios)")
    ax.axhline(SERVICE, c="#999999", lw=0.8)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("selected suppliers"); ax.set_ylabel("P(two-week stock in all 24 weeks)")
    ax.legend()
    save(fig, "q2_selection")

    stock = simulate(x, alpha[sel], draw_ratios(pools, sel, 2000, RNG))
    qs = np.quantile(stock / DEMAND, [0.05, 0.5, 0.95], axis=0)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    w_ = np.arange(1, WEEKS + 1)
    ax.fill_between(w_, qs[0], qs[2], color=CB[0], alpha=0.25, label="5–95% band")
    ax.plot(w_, qs[1], c=CB[0], label="median")
    ax.axhline(2, ls="--", c=CB[1], lw=0.8, label="two-week requirement")
    ax.set_xlabel("week"); ax.set_ylabel("stock (weeks of demand)")
    ax.legend()
    save(fig, "q2_stock")

    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    q3p = q3[q3.volume_slack >= 0]
    sl = q3p.volume_slack * 100
    for c_, m, mk in [(CB[0], "A", "o-"), (CB[2], "B", "s--"), (CB[1], "C", "^-.")]:
        ax.plot(sl, q3p[f"{m}_m3"] / 1000, mk, c=c_, ms=4, label=f"material {m}")
    ax.set_xlabel("allowed raw volume above the minimum (%)")
    ax.set_ylabel("expected raw m³ in 24 weeks (thousand)")
    ax.legend()
    save(fig, "q3_mix")
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
