"""Supplementary experiments: FBP filter windows, convergence of the primal-dual TV
solver, limited-angle data and computational cost.  Requires results/best_by_method.csv
(from sweep.py) for the TV parameters.

    python code/ablations.py

Writes results/fbp_windows.csv, results/tv_convergence.csv, results/limited_angle.csv
and results/timing.csv."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import time
import warnings

import numpy as np
import pandas as pd

from common import (ANGLES, LSQR_GRID, LSQR_KMAX, METHODS, N, NOISE, PHANTOMS, RES, SEEDS, TIK_LAMBDAS,
                    TV_ITERS, TV_LAMBDAS, add_noise, coarse, make_data, phantoms_fine)
from tomo import Projector, angles, fbp, lsqr_path, metrics, op_norm, system_matrix, tikhonov, tv_pdhg

warnings.filterwarnings("ignore", category=RuntimeWarning)
WINDOWS = ["ram-lak", "shepp-logan", "hann"]
LIMITED_SPANS = [180, 150, 120, 90]
LIMITED_M = 90


def fbp_windows(ph, truth):
    rows = []
    for m in ANGLES:
        A, nd = system_matrix(N, angles(m))
        P = Projector(A)
        for name in PHANTOMS:
            b0 = make_data(ph[name], m, nd)
            for eta in NOISE:
                for seed in (SEEDS if eta > 0 else [0]):
                    b, _ = add_noise(b0, eta, m, seed)
                    for w in WINDOWS:
                        rows.append({"phantom": name, "angles": m, "noise": eta, "seed": seed, "window": w,
                                     **metrics(fbp(P, b, N, m, nd, window=w), truth[name])})
    raw = pd.DataFrame(rows)
    return raw.groupby(["phantom", "angles", "noise", "window"]).agg(
        rel_err=("rel_err", "mean"), ssim=("ssim", "mean")).reset_index()


def tv_convergence(ph, truth, best):
    rows = []
    for m in (10, 180):
        A, nd = system_matrix(N, angles(m))
        P = Projector(A)
        LA = op_norm(P)
        for name in PHANTOMS:
            lam = float(best[(best.phantom == name) & (best.angles == m) & (best.noise == 0.01)
                             & (best.method == "TV")].param.iloc[0])
            b, _ = add_noise(make_data(ph[name], m, nd), 0.01, m, 0)
            hist = []
            tv_pdhg(P, b, N, lam, iters=3000, LA=LA, history=hist, x_true=truth[name])
            rows += [{"phantom": name, "angles": m, "lam": lam, "iter": it, "objective": o, "rel_err": e}
                     for it, o, e in hist]
    return pd.DataFrame(rows)


def limited_angle(ph, truth):
    """m = 90 angles spread over [0, span) degrees, 1% noise, oracle parameters on the same grids."""
    rows = []
    for span in LIMITED_SPANS:
        th = angles(LIMITED_M, span)
        A, nd = system_matrix(N, th)
        P = Projector(A)
        LA = op_norm(P)
        for name in PHANTOMS:
            x_true = truth[name]
            b0 = make_data(ph[name], LIMITED_M, nd, span=span)
            for seed in SEEDS:
                b, _ = add_noise(b0, 0.01, LIMITED_M, seed)
                base = {"phantom": name, "span": span, "seed": seed}
                rows.append({**base, "method": "FBP", "param": np.nan,
                             **metrics(fbp(P, b, N, LIMITED_M, nd, span=span), x_true)})
                its, _, _ = lsqr_path(P, b, LSQR_KMAX, record=LSQR_GRID)
                rows += [{**base, "method": "LSQR", "param": k, **metrics(its[k], x_true)} for k in LSQR_GRID]
                rows += [{**base, "method": "Tikhonov", "param": lam, **metrics(tikhonov(A, b, lam), x_true)}
                         for lam in np.sqrt(LIMITED_M) * np.array(TIK_LAMBDAS)]
                rows += [{**base, "method": "TV", "param": lam,
                          **metrics(tv_pdhg(P, b, N, lam, iters=TV_ITERS, LA=LA), x_true)}
                         for lam in LIMITED_M * np.array(TV_LAMBDAS)]
        print(f"[{time.strftime('%H:%M:%S')}] limited angle span={span} done", flush=True)
    raw = pd.DataFrame(rows)
    agg = raw.groupby(["phantom", "span", "method", "param"], dropna=False).agg(
        rel_err=("rel_err", "mean"), ssim=("ssim", "mean")).reset_index()
    return agg.loc[agg.groupby(["phantom", "span", "method"]).rel_err.idxmin()]


def timing(ph, truth, best, repeats=3):
    m = 180
    A, nd = system_matrix(N, angles(m))
    P = Projector(A)
    b, _ = add_noise(make_data(ph["digits"], m, nd), 0.01, m, 0)
    sel = best[(best.phantom == "digits") & (best.angles == m) & (best.noise == 0.01)].set_index("method").param
    LA = op_norm(P)
    tasks = {"FBP": lambda: fbp(P, b, N, m, nd),
             "LSQR": lambda: lsqr_path(P, b, int(sel["LSQR"])),
             "Tikhonov": lambda: tikhonov(A, b, float(sel["Tikhonov"])),
             "TV": lambda: tv_pdhg(P, b, N, float(sel["TV"]), iters=TV_ITERS, LA=LA)}
    rows = []
    for meth in METHODS:
        ts = []
        for _ in range(repeats):
            t = time.perf_counter(); tasks[meth](); ts.append(time.perf_counter() - t)
        rows.append({"method": meth, "param": sel[meth], "seconds": float(np.median(ts))})
    v = np.ones(N * N)
    t = time.perf_counter()
    for _ in range(100):
        P.adj(P @ v)
    rows.append({"method": "A and A^T product", "param": np.nan, "seconds": (time.perf_counter() - t) / 100})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    ph = phantoms_fine()
    truth = {k: coarse(v) for k, v in ph.items()}
    best = pd.read_csv(RES / "best_by_method.csv")
    timing(ph, truth, best).to_csv(RES / "timing.csv", index=False)
    fbp_windows(ph, truth).to_csv(RES / "fbp_windows.csv", index=False)
    print(f"[{time.strftime('%H:%M:%S')}] FBP windows done", flush=True)
    tv_convergence(ph, truth, best).to_csv(RES / "tv_convergence.csv", index=False)
    print(f"[{time.strftime('%H:%M:%S')}] TV convergence done", flush=True)
    limited_angle(ph, truth).to_csv(RES / "limited_angle.csv", index=False)
    print(f"ablations finished in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
