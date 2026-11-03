"""Compositional analysis of ancient glass (CUMCM 2022 problem C).

    python code/run_all.py          # about 30 seconds

Every number and figure of the report is written to results/.
Compositions are analysed in log-ratio (CLR) coordinates, because oxide
percentages are closed data (they sum to ~100%), so raw-percentage correlations
and distances are distorted by the closure (Aitchison, 1986).
"""

import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

warnings.filterwarnings("ignore", category=RuntimeWarning)  # spurious matmul warnings (numpy 2.0 + Accelerate)
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "legend.fontsize": 8, "savefig.bbox": "tight"})
CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
OX = ["SiO2", "Na2O", "K2O", "CaO", "MgO", "Al2O3", "Fe2O3", "CuO", "PbO", "BaO", "P2O5", "SrO", "SnO2", "SO2"]
DELTA = 0.005  # replacement value (% by weight) for "not detected", half the 0.01 reporting step
RNG = np.random.default_rng(2022)
MAJOR = ["SiO2", "CaO", "Al2O3", "CuO", "PbO", "BaO", "P2O5"]  # used to score the weathering correction
MIN_CLUSTER = 3
TYPE = {"高钾": "high-K", "铅钡": "Pb-Ba"}
WEATHER = {"无风化": False, "风化": True}
COLOR = {"蓝绿": "blue-green", "浅蓝": "light blue", "紫": "purple", "深绿": "dark green", "深蓝": "dark blue",
         "浅绿": "light green", "黑": "black", "绿": "green"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------ data
def load():
    x = pd.read_excel(ROOT / "data" / "attachment.xlsx", sheet_name=None)
    s1, s2, s3 = x.values()
    art = s1.copy()
    art.columns = ["id", "decoration", "type", "color", "weathered"]
    art["type"] = art["type"].map(TYPE)
    art["weathered"] = art["weathered"].map(WEATHER)
    art["color"] = art["color"].map(COLOR)

    pts = s2.copy()
    pts.columns = ["point"] + OX
    pts[OX] = pts[OX].fillna(0.0)
    pts["id"] = pts["point"].str.extract(r"^(\d+)")[0].astype(int)
    pts["kind"] = np.select([pts.point.str.contains("未风化点"), pts.point.str.contains("严重风化点")],
                            ["unweathered spot", "severely weathered spot"], "regular")
    pts = pts.merge(art[["id", "type", "weathered", "decoration", "color"]], on="id", how="left")
    pts.loc[pts.kind == "unweathered spot", "weathered"] = False
    pts.loc[pts.kind == "severely weathered spot", "weathered"] = True
    pts["total"] = pts[OX].sum(1)
    pts["valid"] = pts.total.between(85, 105)
    pts["point_en"] = pts.point.str.replace("部位", "-part").str.replace("未风化点", "-unweathered").str.replace(
        "严重风化点", "-severe")

    unk = s3.copy()
    unk.columns = ["id", "weathered"] + OX
    unk[OX] = unk[OX].fillna(0.0)
    unk["weathered"] = unk["weathered"].map(WEATHER)
    unk["total"] = unk[OX].sum(1)
    return art, pts, unk


def closure(X):
    X = np.asarray(X, float)
    return 100 * X / X.sum(1, keepdims=True)


def clr(X, delta=None):
    X = closure(np.where(np.asarray(X, float) <= 0, DELTA if delta is None else delta, X))
    L = np.log(X)
    return L - L.mean(1, keepdims=True)


def clr_inv(Z):
    E = np.exp(Z)
    return closure(E)


def calibrate(Zu, Zw):
    """Calibrated weathering correction in clr coordinates.

    If clr(w) = clr(u) - Delta + e, with the weathering noise e independent of the unweathered state u, the best
    linear predictor of clr(u) given clr(w) is mu_u + S_u (S_u + S_e)^-1 (clr(w) - mu_w).  With a diagonal
    covariance the gain of part j is b_j = var_u,j / var_w,j, capped at 1 (b = 1 gives the plain shift).
    """
    b = np.minimum(1.0, Zu.var(0, ddof=1) / Zw.var(0, ddof=1))
    return Zu.mean(0), Zw.mean(0), b


def predict_calibrated(Zw, params):
    mu_u, mu_w, b = params
    return mu_u + b * (Zw - mu_w)


def aitchison(a, b):
    return float(np.linalg.norm(clr(a[None]) - clr(b[None])))


# ------------------------------------------------------------------ tests
def perm_chi2(table, n_perm=20000, rng=RNG):
    """Monte Carlo exact test for an r x c table: permute one factor, keep margins."""
    t = np.asarray(table)
    rows = np.repeat(np.arange(t.shape[0]), t.sum(1))
    cols = np.concatenate([np.repeat(np.arange(t.shape[1]), t[i]) for i in range(t.shape[0])])
    stat = lambda tab: stats.chi2_contingency(tab, correction=False)[0]
    obs = stat(t)
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(cols)
        tab = np.zeros_like(t)
        np.add.at(tab, (rows, perm), 1)
        cnt += stat(tab) >= obs - 1e-12
    return obs, (cnt + 1) / (n_perm + 1)


def cramers_v(t):
    t = np.asarray(t)
    chi2 = stats.chi2_contingency(t, correction=False)[0]
    return float(np.sqrt(chi2 / t.sum() / (min(t.shape) - 1)))


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    run = 0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def bh(p):
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    prev = 1.0
    for r in range(m - 1, -1, -1):
        i = order[r]
        prev = min(prev, p[i] * m / (r + 1))
        adj[i] = prev
    return adj


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    art, pts, unk = load()
    out = {"artifacts": len(art), "points": len(pts), "valid_points": int(pts.valid.sum()),
           "invalid_points": pts.loc[~pts.valid, "point_en"].tolist(),
           "missing_color": int(art.color.isna().sum())}
    V = pts[pts.valid].reset_index(drop=True)

    # ----------------------------------------------------------- Q1a: associations (artifact level)
    log("Q1a associations")
    assoc = []
    for factor in ["type", "decoration", "color"]:
        d = art.dropna(subset=[factor])
        tab = pd.crosstab(d[factor], d.weathered)
        if tab.shape == (2, 2):
            odds, p_exact = stats.fisher_exact(tab.to_numpy())
            chi2 = stats.chi2_contingency(tab.to_numpy(), correction=False)[0]
            test = "Fisher exact"
        else:
            chi2, p_exact = perm_chi2(tab.to_numpy())
            odds = np.nan
            test = "Monte Carlo exact (20,000 permutations)"
        p_asym = stats.chi2_contingency(tab.to_numpy(), correction=False)[1]
        exp = stats.chi2_contingency(tab.to_numpy(), correction=False)[3]
        assoc.append({"factor": factor, "n": int(tab.to_numpy().sum()), "levels": tab.shape[0], "chi2": chi2,
                      "p_asymptotic": p_asym, "p_exact": p_exact, "test": test,
                      "share_expected_below_5": float((exp < 5).mean()), "cramers_v": cramers_v(tab),
                      "odds_ratio": odds})
        tab.to_csv(RES / f"q1_table_{factor}.csv")
    pd.DataFrame(assoc).to_csv(RES / "q1_association.csv", index=False)
    # count artifacts (58), not sampling points (67 rows)
    out["q1a_sampling_point_count"] = 67
    weath_by_type = art.groupby("type").weathered.mean().to_dict()
    out["weathered_share_by_type"] = weath_by_type

    # ----------------------------------------------------------- Q1b: composition statistics
    log("Q1b composition by type and weathering")
    rows = []
    for ty in ["high-K", "Pb-Ba"]:
        a = V[(V.type == ty) & (~V.weathered)]
        b = V[(V.type == ty) & (V.weathered)]
        ps = []
        for ox in OX:
            p = stats.mannwhitneyu(a[ox], b[ox], alternative="two-sided").pvalue if (a[ox].any() or b[ox].any()) else 1.0
            ps.append(p)
            rows.append({"type": ty, "oxide": ox, "n_unweathered": len(a), "n_weathered": len(b),
                         "median_unweathered": a[ox].median(), "iqr_unweathered": a[ox].quantile(.75) - a[ox].quantile(.25),
                         "median_weathered": b[ox].median(), "iqr_weathered": b[ox].quantile(.75) - b[ox].quantile(.25),
                         "detected_unweathered": float((a[ox] > 0).mean()), "detected_weathered": float((b[ox] > 0).mean()),
                         "p_mwu": p})
        for r, padj in zip(rows[-len(OX):], holm(ps)):
            r["p_holm"] = padj
    q1b = pd.DataFrame(rows)
    q1b.to_csv(RES / "q1_composition_stats.csv", index=False)

    # ----------------------------------------------------------- Q1c: pre-weathering prediction
    log("Q1c weathering shift in CLR coordinates")
    Z = clr(V[OX].to_numpy())
    shift = {}
    for ty in ["high-K", "Pb-Ba"]:
        u = Z[(V.type == ty) & (~V.weathered)]
        w = Z[(V.type == ty) & (V.weathered)]
        shift[ty] = u.mean(0) - w.mean(0)

    def predict_pre(Xw, ty, sh=shift):
        return clr_inv(clr(Xw) + sh[ty])

    # validation: weathered artifacts that also have an unweathered spot (leave-one-artifact-out shift)
    val = []
    mi = [OX.index(o) for o in MAJOR]
    for aid in sorted(V.loc[V.kind == "unweathered spot", "id"].unique()):
        g = V[V.id == aid]
        ws = g[g.weathered]
        us = g[~g.weathered]
        if len(ws) == 0:
            continue
        ty = g.type.iloc[0]
        m = (V.type == ty) & (V.id != aid)
        sh = Z[m & ~V.weathered].mean(0) - Z[m & V.weathered].mean(0)
        cal = calibrate(Z[m & ~V.weathered], Z[m & V.weathered])
        target = us[OX].to_numpy().mean(0)  # average of the unweathered spots
        for _, r in ws.iterrows():
            x = r[OX].to_numpy(float)[None]
            pred = clr_inv(clr(x) + sh)[0]
            pcal = clr_inv(predict_calibrated(clr(x), cal))[0]
            val.append({"artifact": int(aid), "type": ty, "weathered_point": r.point_en,
                        "mae_no_correction": float(np.mean(np.abs(closure(x)[0][mi] - closure(target[None])[0][mi]))),
                        "mae_corrected": float(np.mean(np.abs(pred[mi] - closure(target[None])[0][mi]))),
                        "mae_calibrated": float(np.mean(np.abs(pcal[mi] - closure(target[None])[0][mi]))),
                        "SiO2_weathered": x[0][0], "SiO2_pred": pred[0], "SiO2_cal": pcal[0], "SiO2_unweathered": target[0],
                        "PbO_weathered": x[0][8], "PbO_pred": pred[8], "PbO_cal": pcal[8], "PbO_unweathered": target[8]})
    pd.DataFrame(val).to_csv(RES / "q1_prediction_validation.csv", index=False)

    # predictions for every weathered point, with bootstrap 90% intervals (resampling points within groups)
    pred_rows = []
    B = 1000
    cal_rows = []
    boots = {ty: [] for ty in shift}
    cboots = {ty: [] for ty in shift}
    for _ in range(B):
        for ty in shift:
            iu = np.flatnonzero((V.type == ty) & (~V.weathered))
            iw = np.flatnonzero((V.type == ty) & (V.weathered))
            su, sw = RNG.choice(iu, iu.size), RNG.choice(iw, iw.size)
            boots[ty].append(Z[su].mean(0) - Z[sw].mean(0))
            cboots[ty].append(calibrate(Z[su], Z[sw]))
    cal = {ty: calibrate(Z[((V.type == ty) & ~V.weathered).to_numpy()], Z[((V.type == ty) & V.weathered).to_numpy()])
           for ty in shift}
    for _, r in V[V.weathered].iterrows():
        x = r[OX].to_numpy(float)[None]
        for rows_, p, bs in [
                (pred_rows, predict_pre(x, r.type)[0], np.array([clr_inv(clr(x) + s)[0] for s in boots[r.type]])),
                (cal_rows, clr_inv(predict_calibrated(clr(x), cal[r.type]))[0],
                 np.array([clr_inv(predict_calibrated(clr(x), c))[0] for c in cboots[r.type]]))]:
            lo, hi = np.percentile(bs, [5, 95], axis=0)
            row = {"point": r.point_en, "type": r.type}
            for k, ox in enumerate(OX):
                row[ox] = p[k]
                row[f"{ox}_lo"] = lo[k]
                row[f"{ox}_hi"] = hi[k]
            rows_.append(row)
    pd.DataFrame(cal_rows).to_csv(RES / "q1_pre_weathering_predictions.csv", index=False)  # final (calibrated) model
    pd.DataFrame(pred_rows).to_csv(RES / "q1_pre_weathering_predictions_shift.csv", index=False)
    pd.DataFrame(shift, index=OX).to_csv(RES / "q1_clr_shift.csv")  # unweathered minus weathered, CLR units
    pd.DataFrame({f"{c}_{ty}": v for ty in cal for c, v in zip(["mean_unweathered", "mean_weathered", "gain"], cal[ty])},
                 index=OX).to_csv(RES / "q1_calibration.csv")

    # ----------------------------------------------------------- Q2: type rule and sub-classes
    log("Q2 classification and sub-classes")
    y = (V.type == "Pb-Ba").astype(int).to_numpy()
    groups = V.id.to_numpy()
    lr = LogisticRegression(C=1.0, max_iter=5000)
    p_cv = cross_val_predict(lr, Z, y, groups=groups, cv=LeaveOneGroupOut(), method="predict_proba")[:, 1]
    out["q2_logo_accuracy"] = float(((p_cv > 0.5) == y).mean())
    out["q2_logo_min_margin"] = float(np.min(np.where(y == 1, p_cv, 1 - p_cv)))
    # simplest separating rule on raw percentages
    pb = V.PbO + V.BaO
    out["q2_rule_PbO_BaO_max_highK"] = float(pb[V.type == "high-K"].max())
    out["q2_rule_PbO_BaO_min_PbBa"] = float(pb[V.type == "Pb-Ba"].min())
    k2o = V.K2O
    out["q2_rule_K2O_min_highK_unweathered"] = float(k2o[(V.type == "high-K") & (~V.weathered)].min())
    out["q2_rule_K2O_max_PbBa"] = float(k2o[V.type == "Pb-Ba"].max())
    lr.fit(Z, y)
    coef = pd.Series(lr.coef_[0], index=OX).sort_values()
    coef.to_csv(RES / "q2_logistic_coefficients.csv")

    # sub-classes: Ward clustering of weathering-corrected CLR compositions, per type
    Zc = Z.copy()
    for ty in shift:
        m = ((V.type == ty) & V.weathered).to_numpy()
        Zc[m] = Z[m] + shift[ty]
    feats = {"high-K": ["K2O", "CaO", "Al2O3", "Fe2O3", "CuO", "SiO2"],
             "Pb-Ba": ["PbO", "BaO", "SiO2", "CuO", "P2O5", "Al2O3"]}
    sub = {}
    sub_rows = []
    sens_rows = []
    for ty, fs in feats.items():
        idx = np.flatnonzero(V.type == ty)
        cols = [OX.index(f) for f in fs]
        X = Zc[np.ix_(idx, cols)]
        X = (X - X.mean(0)) / X.std(0)
        L = linkage(X, "ward")
        sil = {k: float(silhouette_score(X, fcluster(L, k, "maxclust"))) for k in range(2, 6)}
        ok = {k: v for k, v in sil.items() if np.bincount(fcluster(L, k, "maxclust"))[1:].min() >= MIN_CLUSTER}
        kbest = max(ok, key=ok.get) if ok else max(sil, key=sil.get)
        sub[ty] = {"k_unconstrained": max(sil, key=sil.get), "admissible_split_found": bool(ok)}
        lab = fcluster(L, kbest, "maxclust")
        sub[ty].update({"k": kbest, "silhouette": sil})
        for i, l in zip(idx, lab):
            sub_rows.append({"point": V.point_en[i], "type": ty, "subclass": f"{ty}-{l}", "weathered": bool(V.weathered[i])})
        # sensitivity 1: bootstrap stability (ARI of re-clustered resamples on the common points)
        aris = []
        for _ in range(200):
            bi = RNG.choice(len(idx), len(idx))
            u = np.unique(bi)
            lb = fcluster(linkage(X[bi], "ward"), kbest, "maxclust")
            first = {j: lb[np.flatnonzero(bi == j)[0]] for j in u}
            aris.append(adjusted_rand_score(lab[u], [first[j] for j in u]))
        # sensitivity 2: multiplicative measurement noise on the raw percentages
        raw = V.loc[idx, OX].to_numpy(float)
        for eps in [0.02, 0.05, 0.10, 0.20]:
            ar = []
            for _ in range(100):
                noisy = raw * np.exp(eps * RNG.standard_normal(raw.shape))
                Zn = clr(noisy)
                wm = V.weathered.to_numpy()[idx]
                Zn[wm] = Zn[wm] + shift[ty]
                Xn = Zn[:, cols]
                Xn = (Xn - Xn.mean(0)) / Xn.std(0)
                ar.append(adjusted_rand_score(lab, fcluster(linkage(Xn, "ward"), kbest, "maxclust")))
            sens_rows.append({"type": ty, "noise": eps, "ari_mean": float(np.mean(ar)), "ari_p10": float(np.percentile(ar, 10))})
        # sensitivity 3: no weathering correction
        Xr = Z[np.ix_(idx, cols)]
        Xr = (Xr - Xr.mean(0)) / Xr.std(0)
        sub[ty]["ari_bootstrap_median"] = float(np.median(aris))
        sub[ty]["ari_bootstrap_p10"] = float(np.percentile(aris, 10))
        sub[ty]["ari_without_weathering_correction"] = float(adjusted_rand_score(
            lab, fcluster(linkage(Xr, "ward"), kbest, "maxclust")))
        # cluster profiles (median raw %)
        prof = V.loc[idx, OX].groupby(lab).median()
        prof.index = [f"{ty}-{l}" for l in prof.index]
        prof["n"] = np.bincount(lab)[1:]
        prof.to_csv(RES / f"q2_subclass_profiles_{ty}.csv")
        sub[ty]["Z"] = X
        sub[ty]["labels"] = lab
    pd.DataFrame(sub_rows).to_csv(RES / "q2_subclasses.csv", index=False)
    pd.DataFrame(sens_rows).to_csv(RES / "q2_sensitivity.csv", index=False)
    out["q2_subclasses"] = {ty: {k: v for k, v in d.items() if k not in ("Z", "labels")} for ty, d in sub.items()}

    # ----------------------------------------------------------- Q3: unknown artifacts
    log("Q3 unknown artifacts")
    Zu = clr(unk[OX].to_numpy())
    p_unk = lr.predict_proba(Zu)[:, 1]
    res3 = []
    for i, r in unk.iterrows():
        x = r[OX].to_numpy(float)
        flips = {}
        for eps in [0.05, 0.10, 0.20, 0.40]:
            pp = lr.predict_proba(clr(x[None] * np.exp(eps * RNG.standard_normal((2000, len(OX))))))[:, 1]
            flips[f"agree_{int(100 * eps)}pct"] = float(np.mean((pp > 0.5) == (p_unk[i] > 0.5)))
        res3.append({"id": r.id, "weathered": bool(r.weathered), "total": r.total, "PbO+BaO": r.PbO + r.BaO,
                     "K2O": r.K2O, "p_PbBa": float(p_unk[i]),
                     "type": "Pb-Ba" if p_unk[i] > 0.5 else "high-K", **flips})
    pd.DataFrame(res3).to_csv(RES / "q3_unknown_classification.csv", index=False)

    # ----------------------------------------------------------- Q4: correlation structure
    log("Q4 correlation structure")
    present = [o for o in OX if (V[o] > 0).mean() >= 0.5]
    corr = {}
    for ty in ["high-K", "Pb-Ba"]:
        m = (V.type == ty).to_numpy()
        corr[ty] = pd.DataFrame(np.corrcoef(Zc[m][:, [OX.index(o) for o in present]].T), index=present, columns=present)
        corr[ty].to_csv(RES / f"q4_clr_correlation_{ty}.csv")
    nK = int((V.type == "high-K").sum())
    nP = int((V.type == "Pb-Ba").sum())
    diff_rows = []
    for i, a in enumerate(present):
        for j, b in enumerate(present):
            if j <= i:
                continue
            r1, r2 = corr["high-K"].loc[a, b], corr["Pb-Ba"].loc[a, b]
            z = (np.arctanh(np.clip(r1, -0.999, 0.999)) - np.arctanh(np.clip(r2, -0.999, 0.999))) / np.sqrt(
                1 / (nK - 3) + 1 / (nP - 3))
            diff_rows.append({"pair": f"{a}-{b}", "r_highK": r1, "r_PbBa": r2, "z": z, "p": 2 * stats.norm.sf(abs(z))})
    dif = pd.DataFrame(diff_rows)
    dif["q_bh"] = bh(dif.p.to_numpy())
    dif.sort_values("p").to_csv(RES / "q4_correlation_differences.csv", index=False)
    out["q4_components"] = present
    out["q4_n"] = {"high-K": nK, "Pb-Ba": nP}
    out["q4_pairs"] = len(dif)
    out["q4_significant_bh_0.05"] = int((dif.q_bh < 0.05).sum())
    out["runtime_s"] = time.time() - t0
    (RES / "results.json").write_text(json.dumps(out, indent=2, default=float))

    # ----------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.6))
    for ax, ox in zip(axes, ["SiO2", "K2O", "PbO", "P2O5"]):
        data, labs = [], []
        for ty in ["high-K", "Pb-Ba"]:
            for w in [False, True]:
                data.append(V.loc[(V.type == ty) & (V.weathered == w), ox])
                labs.append(f"{'K' if ty == 'high-K' else 'PbBa'}\n{'W' if w else 'U'}")
        ax.boxplot(data, widths=0.6, medianprops={"color": CB[1]})
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(labs, fontsize=7)
        ax.set_title(ox, fontsize=8)
        ax.set_ylabel("wt %" if ox == "SiO2" else None)
    save(fig, "q1_boxplots")


    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), gridspec_kw={"wspace": 0.35})
    D = corr["high-K"].to_numpy() - corr["Pb-Ba"].to_numpy()
    panels = [(corr["high-K"].to_numpy(), f"a) high-K (n = {nK})", 1), (corr["Pb-Ba"].to_numpy(), f"b) Pb-Ba (n = {nP})", 1),
              (D, "c) difference a - b", 2)]
    for t, (ax, (M, title, vmax)) in enumerate(zip(axes, panels)):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(present))); ax.set_xticklabels(present, rotation=90, fontsize=6)
        ax.set_yticks(range(len(present))); ax.set_yticklabels(present if t == 0 else [], fontsize=6)
        ax.set_title(title, loc="left", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.7, ticks=[-vmax, 0, vmax])
    save(fig, "q4_correlations")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, ty in zip(axes, ["high-K", "Pb-Ba"]):
        X = sub[ty]["Z"]
        U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
        P = U[:, :2] * S[:2]
        for t, l in enumerate(np.unique(sub[ty]["labels"])):
            m = sub[ty]["labels"] == l
            ax.plot(P[m, 0], P[m, 1], "osd^v"[t], c=CB[t], ls="none", ms=4, label=f"{ty}-{l}")
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        ax.set_title(f"{'a' if ty == 'high-K' else 'b'}) {ty}, k = {sub[ty]['k']}", loc="left")
        ax.legend()
    save(fig, "q2_subclasses")

    extended(art, V, unk, Z, Zc, shift, sub, feats, out)
    out["runtime_s"] = time.time() - t0
    (RES / "results.json").write_text(json.dumps(out, indent=2, default=float))
    log(f"done in {time.time() - t0:.0f} s")


# ------------------------------------------------------------------ extended analyses
# A separate random stream, so that adding or removing an analysis here leaves the results above unchanged.
RNG2 = np.random.default_rng(2023)


def chi2_stat(f, w, nf):
    """Pearson chi-square of an nf x 2 table built from factor codes f and boolean labels w."""
    t = np.zeros((nf, 2))
    np.add.at(t, (f, w.astype(int)), 1)
    t = t[t.sum(1) > 0]
    e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
    m = e > 0
    return float((((t - e) ** 2)[m] / e[m]).sum())


def stratified_perm(d, factor, n_perm=20000, rng=RNG2):
    """Exact conditional test of factor vs weathering given glass type: labels are permuted within each type."""
    codes = {v: i for i, v in enumerate(sorted(d[factor].unique()))}
    strata = [(g[factor].map(codes).to_numpy(), g.weathered.to_numpy(bool)) for _, g in d.groupby("type")]
    nf = len(codes)
    obs = sum(chi2_stat(f, w, nf) for f, w in strata)
    cnt = 0
    for _ in range(n_perm):
        cnt += sum(chi2_stat(f, rng.permutation(w), nf) for f, w in strata) >= obs - 1e-12
    return obs, (cnt + 1) / (n_perm + 1)


def ward_labels(X, k):
    return fcluster(linkage(X, "ward"), k, "maxclust")


def within_ss(X, lab):
    return float(sum(((X[lab == l] - X[lab == l].mean(0)) ** 2).sum() for l in np.unique(lab)))


def gap_statistic(X, kmax=6, B=100, rng=RNG2):
    """Gap statistic (Tibshirani et al., 2001) for Ward clustering; reference data uniform on the PCA-aligned box."""
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt.T
    lo, hi = P.min(0), P.max(0)
    rows = []
    for k in range(1, kmax + 1):
        lw = np.log(within_ss(X, ward_labels(X, k) if k > 1 else np.ones(len(X), int)))
        ref = []
        for _ in range(B):
            R = rng.uniform(lo, hi, size=P.shape) @ Vt
            ref.append(np.log(within_ss(R, ward_labels(R, k) if k > 1 else np.ones(len(R), int))))
        ref = np.array(ref)
        rows.append({"k": k, "gap": float(ref.mean() - lw), "s": float(ref.std() * np.sqrt(1 + 1 / B))})
    g = pd.DataFrame(rows)
    khat = next((int(g.k[i]) for i in range(len(g) - 1) if g.gap[i] >= g.gap[i + 1] - g.s[i + 1]), int(g.k.iloc[-1]))
    return g, khat


def shift_of(Zm, V, ty, exclude=None):
    m = (V.type == ty).to_numpy()
    if exclude is not None:
        m = m & (V.id != exclude).to_numpy()
    w = V.weathered.to_numpy(bool)
    return Zm[m & ~w].mean(0) - Zm[m & w].mean(0)


def paired_validation(V, Zm, delta=None):
    """Leave-one-artefact-out check on artefacts 49 and 50 (weathered point plus unweathered spot)."""
    mi = [OX.index(o) for o in MAJOR]
    res = {}
    for aid in (49, 50):
        g = V[V.id == aid]
        x = g.loc[g.weathered, OX].to_numpy(float)[:1]
        target = closure(g.loc[~g.weathered, OX].to_numpy(float).mean(0)[None])[0]
        sh = shift_of(Zm, V, "Pb-Ba", exclude=aid)
        pred = clr_inv(clr(x, delta) + sh)[0]
        m = ((V.type == "Pb-Ba") & (V.id != aid)).to_numpy()
        w = V.weathered.to_numpy(bool)
        pcal = clr_inv(predict_calibrated(clr(x, delta), calibrate(Zm[m & ~w], Zm[m & w])))[0]
        res[aid] = (float(np.mean(np.abs(closure(x)[0][mi] - target[mi]))), float(np.mean(np.abs(pred[mi] - target[mi]))),
                    float(np.mean(np.abs(pcal[mi] - target[mi]))))
    return res


def core_results(V, unk, delta, C=1.0, kind="mean"):
    """The main quantitative results recomputed for one detection-limit replacement and one regularisation."""
    Zd = clr(V[OX].to_numpy(), delta)
    sh = {ty: shift_of(Zd, V, ty) for ty in ["high-K", "Pb-Ba"]}
    if kind == "median":
        w = V.weathered.to_numpy(bool)
        sh = {ty: np.median(Zd[(V.type == ty).to_numpy() & ~w], 0) - np.median(Zd[(V.type == ty).to_numpy() & w], 0)
              for ty in sh}
    val = paired_validation(V, Zd, delta)
    y = (V.type == "Pb-Ba").astype(int).to_numpy()
    lr = LogisticRegression(C=C, max_iter=5000)
    p_cv = cross_val_predict(lr, Zd, y, groups=V.id.to_numpy(), cv=LeaveOneGroupOut(), method="predict_proba")[:, 1]
    lr.fit(Zd, y)
    pu = lr.predict_proba(clr(unk[OX].to_numpy(), delta))[:, 1]
    Zc = Zd.copy()
    for ty in sh:
        m = ((V.type == ty) & V.weathered).to_numpy()
        Zc[m] += sh[ty]
    feats = {"high-K": ["K2O", "CaO", "Al2O3", "Fe2O3", "CuO", "SiO2"],
             "Pb-Ba": ["PbO", "BaO", "SiO2", "CuO", "P2O5", "Al2O3"]}
    clus = {}
    for ty, fs in feats.items():
        idx = np.flatnonzero(V.type == ty)
        X = Zc[np.ix_(idx, [OX.index(f) for f in fs])]
        X = (X - X.mean(0)) / X.std(0)
        L = linkage(X, "ward")
        sil = {k: silhouette_score(X, fcluster(L, k, "maxclust")) for k in range(2, 6)}
        ok = {k: v for k, v in sil.items() if np.bincount(fcluster(L, k, "maxclust"))[1:].min() >= MIN_CLUSTER}
        clus[ty] = (max(ok, key=ok.get) if ok else None, fcluster(L, 4, "maxclust"))
    present = [o for o in OX if (V[o] > 0).mean() >= 0.5]
    ci = [OX.index(o) for o in present]
    mK, mP = (V.type == "high-K").to_numpy(), (V.type == "Pb-Ba").to_numpy()
    RK, RP = np.corrcoef(Zc[mK][:, ci].T), np.corrcoef(Zc[mP][:, ci].T)
    iu = np.triu_indices(len(present), 1)
    z = (np.arctanh(np.clip(RK[iu], -.999, .999)) - np.arctanh(np.clip(RP[iu], -.999, .999))) / np.sqrt(
        1 / (mK.sum() - 3) + 1 / (mP.sum() - 3))
    q = bh(2 * stats.norm.sf(np.abs(z)))
    return {"delta": delta, "C": C, "shift": kind, "mae49": val[49][1], "mae50": val[50][1], "mae49_cal": val[49][2], "mae50_cal": val[50][2],
            "logo_accuracy": float(((p_cv > 0.5) == y).mean()),
            "unknown_labels": "".join("P" if p > 0.5 else "K" for p in pu), "p_A5": float(pu[4]),
            "k_PbBa": clus["Pb-Ba"][0], "k_highK_admissible": clus["high-K"][0],
            "PbBa_labels_k4": clus["Pb-Ba"][1], "q4_significant": int((q < 0.05).sum()),
            "q4_significant_pairs": ";".join(f"{present[i]}-{present[j]}" for i, j, qq in zip(*iu, q) if qq < 0.05)}


def extended(art, V, unk, Z, Zc, shift, sub, feats, out):
    log("extended analyses")
    ext = {}
    # ---------------- Q1a: associations within each glass type, and conditional on type
    rows = []
    for ty in ["high-K", "Pb-Ba"]:
        for factor in ["decoration", "color"]:
            d = art[(art.type == ty)].dropna(subset=[factor])
            tab = pd.crosstab(d[factor], d.weathered)
            if tab.shape == (2, 2):
                chi2 = stats.chi2_contingency(tab.to_numpy(), correction=False)[0]
                p = stats.fisher_exact(tab.to_numpy())[1]
                test = "Fisher exact"
            else:
                chi2, p = perm_chi2(tab.to_numpy(), rng=RNG2)
                test = "Monte Carlo exact"
            rows.append({"stratum": ty, "factor": factor, "n": int(tab.to_numpy().sum()), "levels": tab.shape[0],
                         "chi2": chi2, "p_exact": p, "cramers_v": cramers_v(tab), "test": test})
    for factor in ["decoration", "color"]:
        d = art.dropna(subset=[factor])
        chi2, p = stratified_perm(d, factor)
        rows.append({"stratum": "within type (pooled)", "factor": factor, "n": len(d), "levels": d[factor].nunique(),
                     "chi2": chi2, "p_exact": p, "cramers_v": np.nan, "test": "stratified permutation"})
    hk = art[art.type == "high-K"]
    tabB = pd.crosstab(hk.decoration == "B", hk.weathered).to_numpy()
    rows.append({"stratum": "high-K", "factor": "decoration B vs A/C", "n": int(tabB.sum()), "levels": 2,
                 "chi2": stats.chi2_contingency(tabB, correction=False)[0], "p_exact": stats.fisher_exact(tabB)[1],
                 "cramers_v": cramers_v(tabB), "test": "Fisher exact"})
    pd.DataFrame(rows).to_csv(RES / "q1_association_by_type.csv", index=False)

    # ---------------- Q1b: compositional centre of weathered vs unweathered points (permutation test)
    rows = []
    w_all = V.weathered.to_numpy(bool)
    for ty in ["high-K", "Pb-Ba"]:
        m = (V.type == ty).to_numpy()
        Zt, wt = Z[m], w_all[m]
        stat = lambda lab: float(((Zt[~lab].mean(0) - Zt[lab].mean(0)) ** 2).sum())
        obs = stat(wt)
        perm = np.array([stat(RNG2.permutation(wt)) for _ in range(10000)])
        rows.append({"type": ty, "n_unweathered": int((~wt).sum()), "n_weathered": int(wt.sum()),
                     "aitchison_distance_centres": np.sqrt(obs), "p_perm": (np.sum(perm >= obs) + 1) / (len(perm) + 1)})
    pd.DataFrame(rows).to_csv(RES / "q1_centre_test.csv", index=False)

    # ---------------- Q1c: direction of weathering, alternative correction models, distance to unweathered glass
    rows = []
    for aid in (49, 50):
        g = V[V.id == aid]
        zw = Z[g.index[g.weathered.to_numpy(bool)][0]]
        zu = Z[g.index[~g.weathered.to_numpy(bool)]].mean(0)
        sh = shift_of(Z, V, "Pb-Ba", exclude=aid)
        v = zw - zu
        rows.append({"artifact": aid, "comparison": "weathered minus unweathered spot",
                     "cosine_with_weathering_direction": float(v @ -sh / np.linalg.norm(v) / np.linalg.norm(sh)),
                     "length": float(np.linalg.norm(v)), "scale_lambda": float(-(v @ sh) / (sh @ sh)),
                     "shift_length": float(np.linalg.norm(sh))})
    for aid in (8, 26, 54):
        g = V[V.id == aid]
        zr = Z[g.index[(g.kind == "regular").to_numpy()][0]]
        zs = Z[g.index[(g.kind == "severely weathered spot").to_numpy()][0]]
        sh = shift_of(Z, V, "Pb-Ba", exclude=aid)
        v = zs - zr
        rows.append({"artifact": aid, "comparison": "severe spot minus regular weathered point",
                     "cosine_with_weathering_direction": float(v @ -sh / np.linalg.norm(v) / np.linalg.norm(sh)),
                     "length": float(np.linalg.norm(v)), "scale_lambda": float(-(v @ sh) / (sh @ sh)),
                     "shift_length": float(np.linalg.norm(sh))})
    direc = pd.DataFrame(rows)
    direc.to_csv(RES / "q1_weathering_direction.csv", index=False)
    # chance level: cosine between random directions in the 13-dimensional clr hyperplane
    rnd = RNG2.standard_normal((20000, 2, len(OX)))
    rnd -= rnd.mean(2, keepdims=True)
    cos = np.einsum("ij,ij->i", rnd[:, 0], rnd[:, 1]) / np.linalg.norm(rnd[:, 0], axis=1) / np.linalg.norm(rnd[:, 1], axis=1)
    ext["q1_random_cosine_p95"] = float(np.percentile(np.abs(cos), 95))

    mi = [OX.index(o) for o in MAJOR]
    raw = V[OX].to_numpy(float)
    rows = []
    for aid in (49, 50):
        g = V[V.id == aid]
        iw = g.index[g.weathered.to_numpy(bool)][0]
        x = raw[iw][None]
        target = closure(raw[g.index[~g.weathered.to_numpy(bool)]].mean(0)[None])[0]
        m = ((V.type == "Pb-Ba") & (V.id != aid)).to_numpy()
        sh = shift_of(Z, V, "Pb-Ba", exclude=aid)
        shmed = np.median(Z[m & ~w_all], 0) - np.median(Z[m & w_all], 0)
        add = closure(np.clip(x + raw[m & ~w_all].mean(0) - raw[m & w_all].mean(0), 0, None))
        cal = calibrate(Z[m & ~w_all], Z[m & w_all])
        preds = {"no correction": closure(x), "additive shift (wt%)": add,
                 "clr median shift": clr_inv(clr(x) + shmed), "clr mean shift": clr_inv(clr(x) + sh),
                 "calibrated clr shift": clr_inv(predict_calibrated(clr(x), cal))}
        for name, pr in preds.items():
            pr = pr[0]
            rows.append({"artifact": aid, "model": name, "mae_major": float(np.mean(np.abs(pr[mi] - target[mi]))),
                         "mae_all": float(np.mean(np.abs(pr - target))),
                         "aitchison": float(np.linalg.norm(clr(pr[None])[0] - clr(target[None])[0])),
                         "SiO2": pr[0], "PbO": pr[8], "BaO": pr[9], "P2O5": pr[10],
                         "SiO2_true": target[0], "PbO_true": target[8], "BaO_true": target[9], "P2O5_true": target[10]})
    models = pd.DataFrame(rows)
    models.to_csv(RES / "q1_correction_models.csv", index=False)

    # plausibility: share of predicted major-oxide values inside the range of the unweathered points of the type
    rows = []
    for ty in ["high-K", "Pb-Ba"]:
        m = (V.type == ty).to_numpy()
        lo_, hi_ = raw[m & ~w_all].min(0), raw[m & ~w_all].max(0)
        X = raw[m & w_all]
        Zt = Z[m & w_all]
        cal = calibrate(Z[m & ~w_all], Z[m & w_all])
        add = closure(np.clip(X + raw[m & ~w_all].mean(0) - X.mean(0), 0, None))
        preds = {"no correction": closure(X), "additive shift (wt%)": add,
                 "clr median shift": clr_inv(Zt + np.median(Z[m & ~w_all], 0) - np.median(Zt, 0)),
                 "clr mean shift": clr_inv(Zt + shift[ty]), "calibrated clr shift": clr_inv(predict_calibrated(Zt, cal))}
        jj = [OX.index(o) for o in ["SiO2", "K2O", "CaO", "Al2O3", "CuO", "PbO", "BaO", "P2O5"]]
        for name, P in preds.items():
            inside = (P[:, jj] >= lo_[jj] - 1e-9) & (P[:, jj] <= hi_[jj] + 1e-9)
            rows.append({"type": ty, "model": name, "n_points": len(P), "share_values_in_range": float(inside.mean()),
                         "points_all_in_range": int(inside.all(1).sum())})
    pd.DataFrame(rows).to_csv(RES / "q1_correction_plausibility.csv", index=False)


    # ---------------- Q2: separability of single oxides, decision tree, random forest
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    y = (V.type == "Pb-Ba").astype(int).to_numpy()
    R = V[OX].copy()
    R["PbO+BaO"] = V.PbO + V.BaO
    rows = []
    for c in R.columns:
        a, b = R.loc[y == 0, c].to_numpy(), R.loc[y == 1, c].to_numpy()
        vals = np.unique(R[c])
        thr = (vals[:-1] + vals[1:]) / 2
        acc = [max(((R[c] > t) == y).mean(), ((R[c] <= t) == y).mean()) for t in thr] if len(thr) else [0.5]
        rows.append({"variable": c, "highK_min": a.min(), "highK_max": a.max(), "PbBa_min": b.min(), "PbBa_max": b.max(),
                     "gap": max(b.min() - a.max(), a.min() - b.max()), "best_single_threshold_accuracy": float(max(acc))})
    sep = pd.DataFrame(rows).sort_values("gap", ascending=False)
    sep.to_csv(RES / "q2_separability.csv", index=False)
    tree = DecisionTreeClassifier(max_depth=2, random_state=2022)
    p_tree = cross_val_predict(tree, V[OX], y, groups=V.id, cv=LeaveOneGroupOut())
    tree.fit(V[OX], y)
    rf = RandomForestClassifier(n_estimators=500, random_state=2022)
    p_rf = cross_val_predict(rf, V[OX], y, groups=V.id, cv=LeaveOneGroupOut())
    rf.fit(V[OX], y)
    imp = pd.Series(rf.feature_importances_, index=OX).sort_values(ascending=False)
    imp.to_csv(RES / "q2_random_forest_importance.csv", header=["importance"])
    ext["q2_tree"] = {"root_feature": OX[tree.tree_.feature[0]], "root_threshold": float(tree.tree_.threshold[0]),
                      "logo_accuracy": float((p_tree == y).mean())}
    ext["q2_random_forest"] = {"logo_accuracy": float((p_rf == y).mean()),
                               "top5": {k: float(v) for k, v in imp.head(5).items()}}
    ext["q2_rule_threshold_midpoint"] = float((out["q2_rule_PbO_BaO_max_highK"] + out["q2_rule_PbO_BaO_min_PbBa"]) / 2)

    # ---------------- Q2: rationality of the sub-classes
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    rows, gaps = [], []
    for ty, fs in feats.items():
        X, lab, k = sub[ty]["Z"], sub[ty]["labels"], sub[ty]["k"]
        idx = np.flatnonzero(V.type == ty)
        alt = {"k-means": KMeans(k, n_init=50, random_state=2022).fit_predict(X),
               "average linkage": fcluster(linkage(X, "average"), k, "maxclust"),
               "complete linkage": fcluster(linkage(X, "complete"), k, "maxclust"),
               "Gaussian mixture (diagonal)": GaussianMixture(k, covariance_type="diag", n_init=10,
                                                              random_state=2022).fit(X).predict(X)}
        X14 = Zc[idx]
        X14 = (X14 - X14.mean(0)) / X14.std(0)
        alt["Ward, all 14 clr coordinates"] = ward_labels(X14, k)
        for name, l2 in alt.items():
            rows.append({"type": ty, "k": k, "alternative": name, "ari_with_ward": float(adjusted_rand_score(lab, l2)),
                         "smallest_cluster": int(np.bincount(pd.factorize(l2)[0]).min())})
        g, khat = gap_statistic(X)
        g["type"] = ty
        gaps.append(g)
        ext.setdefault("q2_gap_k", {})[ty] = khat
    pd.DataFrame(rows).to_csv(RES / "q2_rationality.csv", index=False)
    # sub-class profiles of the weathering-corrected compositions (median wt%), easier to read chemically
    Xcorr = clr_inv(Zc)
    for ty in feats:
        idx = np.flatnonzero(V.type == ty)
        lab = sub[ty]["labels"]
        prof = pd.DataFrame(Xcorr[idx], columns=OX).groupby(lab).median()
        prof.index = [f"{ty}-{l}" for l in prof.index]
        prof["n"] = np.bincount(lab)[1:]
        prof["n_weathered"] = pd.Series(V.weathered.to_numpy(bool)[idx]).groupby(lab).sum().to_numpy()
        prof.to_csv(RES / f"q2_subclass_profiles_corrected_{ty}.csv")
    pd.concat(gaps).to_csv(RES / "q2_gap_statistic.csv", index=False)

    # ---------------- Q3: sub-class of the unknown Pb-Ba artefacts and distance classifier
    ty = "Pb-Ba"
    idx = np.flatnonzero(V.type == ty)
    cols = [OX.index(f) for f in feats[ty]]
    Xr = Zc[np.ix_(idx, cols)]
    mu, sd = Xr.mean(0), Xr.std(0)
    lab = sub[ty]["labels"]
    cent = np.array([((Xr - mu) / sd)[lab == l].mean(0) for l in np.unique(lab)])
    cK = Zc[(V.type == "high-K").to_numpy()].mean(0)
    cP = Zc[(V.type == "Pb-Ba").to_numpy()].mean(0)
    rows = []
    for i, r in unk.iterrows():
        x = r[OX].to_numpy(float)[None]

        def corrected(xx, t):
            zz = clr(xx)
            return zz + shift[t] if r.weathered else zz

        dK = np.linalg.norm(corrected(x, "high-K")[0] - cK)
        dP = np.linalg.norm(corrected(x, "Pb-Ba")[0] - cP)
        row = {"id": r.id, "weathered": bool(r.weathered), "aitchison_to_highK": float(dK),
               "aitchison_to_PbBa": float(dP), "distance_type": "Pb-Ba" if dP < dK else "high-K"}
        if r.PbO + r.BaO > ext["q2_rule_threshold_midpoint"]:
            def nearest(xx):
                q = (corrected(xx, ty)[:, cols] - mu) / sd
                dd = np.linalg.norm(q[:, None, :] - cent[None], axis=2)
                return dd.argmin(1) + 1, dd
            l0, dd = nearest(x)
            srt = np.sort(dd[0])
            row.update({"subclass": f"{ty}-{l0[0]}", "distance_ratio": float(srt[0] / srt[1])})
            for eps in [0.05, 0.10, 0.20]:
                ln, _ = nearest(x * np.exp(eps * RNG2.standard_normal((2000, len(OX)))))
                row[f"subclass_agree_{int(100 * eps)}pct"] = float(np.mean(ln == l0[0]))
        rows.append(row)
    pd.DataFrame(rows).to_csv(RES / "q3_unknown_subclass.csv", index=False)

    # ---------------- Q4: variation matrix, within-type significance, closure effect, bootstrap intervals
    present = out["q4_components"]
    ci = [OX.index(o) for o in present]
    rows, boot_rows = [], []
    rawc = {}
    for ty in ["high-K", "Pb-Ba"]:
        m = (V.type == ty).to_numpy()
        Zt = Zc[m][:, ci]
        n = len(Zt)
        Rm = np.corrcoef(Zt.T)
        Rraw = np.corrcoef(V.loc[m, present].to_numpy(float).T)
        rawc[ty] = (Rm, Rraw)
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                r = Rm[a, b]
                t = r * np.sqrt((n - 2) / max(1 - r * r, 1e-12))
                rows.append({"type": ty, "pair": f"{present[a]}-{present[b]}", "tau": float(np.var(Zt[:, a] - Zt[:, b], ddof=1)),
                             "r_clr": float(r), "r_raw": float(Rraw[a, b]), "p": float(2 * stats.t.sf(abs(t), n - 2))})
    q4 = pd.DataFrame(rows)
    q4["q_bh"] = np.nan
    for ty in ["high-K", "Pb-Ba"]:
        m = q4.type == ty
        q4.loc[m, "q_bh"] = bh(q4.loc[m, "p"].to_numpy())
    q4.to_csv(RES / "q4_within_type_pairs.csv", index=False)
    ext["q4_within_significant_bh_0.05"] = {ty: int(((q4.type == ty) & (q4.q_bh < 0.05)).sum()) for ty in ["high-K", "Pb-Ba"]}
    # pairs whose correlation changes sign between raw percentages and clr coordinates (|r| > 0.3 in either)
    chg = (np.sign(q4.r_raw) != np.sign(q4.r_clr)) & (np.maximum(q4.r_raw.abs(), q4.r_clr.abs()) > 0.3)
    ext["q4_sign_changes_raw_vs_clr"] = {ty: int((chg & (q4.type == ty)).sum()) for ty in ["high-K", "Pb-Ba"]}
    for pair in ["Fe2O3-CuO", "SiO2-Fe2O3", "CaO-CuO", "CaO-SrO"]:
        a, b = [present.index(o) for o in pair.split("-")]
        for ty in ["high-K", "Pb-Ba"]:
            m = np.flatnonzero((V.type == ty).to_numpy())
            bs = []
            for _ in range(2000):
                s = RNG2.choice(m, len(m))
                bs.append(np.corrcoef(Zc[s, ci[a]], Zc[s, ci[b]])[0, 1])
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            boot_rows.append({"pair": pair, "type": ty, "r": float(np.corrcoef(Zc[m, ci[a]], Zc[m, ci[b]])[0, 1]),
                              "ci_low": float(lo), "ci_high": float(hi)})
    pd.DataFrame(boot_rows).to_csv(RES / "q4_bootstrap_intervals.csv", index=False)

    # ---------------- sensitivity to the detection-limit replacement and to the regularisation
    log("sensitivity to delta and C")
    ref = core_results(V, unk, DELTA)
    rows = []
    for delta in [0.001, 0.0025, 0.005, 0.01, 0.02]:
        r = core_results(V, unk, delta)
        r["ari_PbBa_k4_vs_reference"] = float(adjusted_rand_score(ref["PbBa_labels_k4"], r.pop("PbBa_labels_k4")))
        rows.append(r)
    for C in [0.01, 0.1, 10.0, 100.0]:
        r = core_results(V, unk, DELTA, C=C)
        r.pop("PbBa_labels_k4")
        rows.append(r)
    r = core_results(V, unk, DELTA, kind="median")
    r["ari_PbBa_k4_vs_reference"] = float(adjusted_rand_score(ref["PbBa_labels_k4"], r.pop("PbBa_labels_k4")))
    r["mae49"], r["mae50"], r["mae49_cal"], r["mae50_cal"] = [np.nan] * 4  # see q1_correction_models.csv
    rows.append(r)
    pd.DataFrame(rows).to_csv(RES / "q5_sensitivity_delta_C.csv", index=False)

    out["extended"] = ext

    # ---------------- figures: weathering correction in clr space, dendrograms
    from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
    set_link_color_palette(CB)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, ty in zip(axes, ["high-K", "Pb-Ba"]):
        m = (V.type == ty).to_numpy()
        Zt, wt = Z[m], w_all[m]
        c = Zt.mean(0)
        _, _, Vt = np.linalg.svd(Zt - c, full_matrices=False)
        P = (Zt - c) @ Vt[:2].T
        Pc = (Zt + shift[ty] - c) @ Vt[:2].T
        for j in np.flatnonzero(wt):
            ax.annotate("", Pc[j], P[j], arrowprops={"arrowstyle": "->", "color": "#bbbbbb", "lw": 0.6})
        ax.plot(P[~wt, 0], P[~wt, 1], "o", c=CB[0], ms=4, ls="none", label="unweathered")
        ax.plot(P[wt, 0], P[wt, 1], "s", c=CB[1], ms=4, ls="none", label="weathered")
        ax.plot(Pc[wt, 0], Pc[wt, 1], "s", mfc="none", mec=CB[1], ms=4, ls="none", label="weathered, corrected")
        ax.set_xlabel("clr PC 1"); ax.set_ylabel("clr PC 2")
        ax.set_title(f"{'a' if ty == 'high-K' else 'b'}) {ty}", loc="left")
        ax.legend(fontsize=6.5)
    save(fig, "q1_clr_correction")

    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    names = list(dict.fromkeys(models.model))
    for t, aid in enumerate((49, 50)):
        mm = models[models.artifact == aid].set_index("model").loc[names]
        ax.bar(np.arange(len(names)) + (t - 0.5) * 0.38, mm.mae_major, 0.38, color=CB[t], label=f"artefact {aid}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" (wt%)", "\n(wt%)").replace("clr ", "clr\n").replace("calibrated\n", "calibrated\n")
                        .replace("no correction", "no\ncorrection") for n in names], fontsize=7)
    ax.set_ylabel("MAE of major oxides (wt %)")
    ax.legend()
    save(fig, "q1_prediction_validation")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, ty in zip(axes, ["high-K", "Pb-Ba"]):
        X, k = sub[ty]["Z"], sub[ty]["k"]
        L = linkage(X, "ward")
        cut = (L[-k, 2] + L[-k + 1, 2]) / 2
        dendrogram(L, ax=ax, color_threshold=cut, no_labels=True, above_threshold_color="#999999")
        ax.axhline(cut, ls=":", c="#555555", lw=0.8)
        ax.set_ylabel("Ward merge height")
        ax.set_title(f"{'a' if ty == 'high-K' else 'b'}) {ty}, cut at k = {k}", loc="left")
    save(fig, "q2_dendrograms")


if __name__ == "__main__":
    main()
