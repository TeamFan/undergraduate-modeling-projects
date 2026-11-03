"""Main experiment: four reconstruction methods over angles, noise levels, noise
realisations and regularisation parameters, for data simulated on a finer grid
("fine") and, for comparison, with the reconstruction operator itself ("crime").

    python code/sweep.py

Writes results/all_runs.csv, results/lsqr_paths.csv, results/best_by_method.csv,
results/discrepancy.csv, results/inverse_crime.csv and results/recon_examples.npz."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import (ANGLES, CRIME_ANGLES, CRIME_NOISE, LSQR_GRID, LSQR_KMAX, METHODS, N, NOISE,
                    PHANTOMS, RES, SEEDS, TAU, TIK_LAMBDAS, TV_ITERS, TV_LAMBDAS, add_noise, coarse,
                    make_data, phantoms_fine)
from tomo import Projector, angles, fbp, lsqr_path, metrics, op_norm, system_matrix, tikhonov, tv_pdhg

warnings.filterwarnings("ignore", category=RuntimeWarning)
EXAMPLES = {("digits", 10, 0.01), ("digits", 180, 0.01), ("shepp_logan", 10, 0.01), ("shepp_logan", 180, 0.01)}


def job(args):
    mode, m, name = args
    hi = phantoms_fine()[name]
    x_true = coarse(hi)
    A, nd = system_matrix(N, angles(m))
    P = Projector(A)
    LA = op_norm(P)
    b_clean = make_data(hi, m, nd) if mode == "fine" else A @ x_true.ravel()
    model_err = float(np.linalg.norm(b_clean - A @ x_true.ravel()) / np.linalg.norm(b_clean))
    noise_levels = NOISE if mode == "fine" else CRIME_NOISE
    rows, paths, examples = [], [], {}
    for eta in noise_levels:
        for seed in (SEEDS if eta > 0 else [0]):
            b, e = add_noise(b_clean, eta, m, seed)
            base = {"mode": mode, "phantom": name, "angles": m, "noise": eta, "seed": seed,
                    "model_err": model_err, "delta_noise": float(np.linalg.norm(e)),
                    "delta_total": float(np.linalg.norm(b - A @ x_true.ravel())), "norm_b": float(np.linalg.norm(b))}

            def add(method, param, x):
                r = float(np.linalg.norm(A @ x.ravel() - b))
                rows.append({**base, "method": method, "param": param, "residual": r, **metrics(x, x_true)})
                return rows[-1]["rel_err"]

            best = {}
            x = fbp(P, b, N, m, nd)
            add("FBP", np.nan, x)
            best["FBP"] = (0.0, x)
            its, res, err = lsqr_path(P, b, LSQR_KMAX, record=LSQR_GRID, x_true=x_true)
            if mode == "fine":
                paths.append(pd.DataFrame({"phantom": name, "angles": m, "noise": eta, "seed": seed,
                                           "k": np.arange(1, LSQR_KMAX + 1), "rel_err": err, "residual": res,
                                           "delta_noise": base["delta_noise"], "delta_total": base["delta_total"]}))
            for k in LSQR_GRID:
                best.setdefault("LSQR", []).append((add("LSQR", k, its[k]), its[k]))
            for lam in np.sqrt(m) * np.array(TIK_LAMBDAS):
                xk = tikhonov(A, b, lam)
                best.setdefault("Tikhonov", []).append((add("Tikhonov", lam, xk), xk))
            for lam in m * np.array(TV_LAMBDAS):
                xk = tv_pdhg(P, b, N, lam, iters=TV_ITERS, LA=LA)
                best.setdefault("TV", []).append((add("TV", lam, xk), xk))
            if mode == "fine" and seed == 0 and (name, m, eta) in EXAMPLES:
                for meth in METHODS:
                    xb = best[meth][1] if meth == "FBP" else min(best[meth], key=lambda t: t[0])[1]
                    examples[f"{name}_{m}_{int(100 * eta)}_{meth}"] = np.asarray(xb).reshape(N, N)
                examples[f"{name}_truth"] = x_true
    print(f"[{time.strftime('%H:%M:%S')}] {mode} m={m} {name} done", flush=True)
    return rows, (pd.concat(paths) if paths else None), examples


def discrepancy(raw, paths):
    """Discrepancy principle: the most regularised solution with ||A x - b|| <= TAU * delta,
    for delta = ||e|| (noise only) and delta = ||b - A x_true|| (noise plus model error)."""
    out = []
    keys = ["phantom", "angles", "noise", "seed"]
    fine = raw[(raw["mode"] == "fine") & (raw.noise > 0)]
    for dtype in ("delta_noise", "delta_total"):
        for (ph, m, eta, seed), g in paths[paths.noise > 0].groupby(keys):
            g = g.sort_values("k")
            ok = g.residual.values <= TAU * g[dtype].values
            i = int(np.argmax(ok)) if ok.any() else len(g) - 1
            out.append({"phantom": ph, "angles": m, "noise": eta, "seed": seed, "method": "LSQR",
                        "delta": dtype, "param": int(g.k.values[i]), "attained": bool(ok.any()),
                        "rel_err": float(g.rel_err.values[i])})
        for (ph, m, eta, seed, meth), g in fine[fine.method.isin(["Tikhonov", "TV"])].groupby(keys + ["method"]):
            g = g.sort_values("param")
            ok = g.residual.values <= TAU * g[dtype].values
            i = int(np.nonzero(ok)[0].max()) if ok.any() else 0
            out.append({"phantom": ph, "angles": m, "noise": eta, "seed": seed, "method": meth, "delta": dtype,
                        "param": float(g.param.values[i]), "attained": bool(ok.any()),
                        "rel_err": float(g.rel_err.values[i])})
    d = pd.DataFrame(out)
    agg = d.groupby(["phantom", "angles", "noise", "method", "delta"]).agg(
        rel_err=("rel_err", "mean"), rel_err_std=("rel_err", "std"), attained=("attained", "mean"),
        param=("param", "median")).reset_index()
    return agg


def main():
    t0 = time.time()
    jobs = [("fine", m, ph) for m in ANGLES[::-1] for ph in PHANTOMS]
    jobs += [("crime", m, ph) for m in CRIME_ANGLES[::-1] for ph in PHANTOMS]
    with Pool(min(4, os.cpu_count() or 1)) as pool:
        results = pool.map(job, jobs, chunksize=1)
    raw = pd.DataFrame([r for rows, _, _ in results for r in rows])
    paths = pd.concat([p for _, p, _ in results if p is not None])
    examples = {k: v for _, _, ex in results for k, v in ex.items()}
    RES.mkdir(exist_ok=True)
    raw.to_csv(RES / "all_runs.csv", index=False, float_format="%.6g")
    paths.to_csv(RES / "lsqr_paths.csv", index=False, float_format="%.6g")
    np.savez_compressed(RES / "recon_examples.npz", **examples)

    # oracle parameter per (mode, phantom, angles, noise, method): best mean relative error over seeds
    g = raw.groupby(["mode", "phantom", "angles", "noise", "method", "param"], dropna=False)
    agg = g.agg(rel_err=("rel_err", "mean"), rel_err_std=("rel_err", "std"), psnr=("psnr", "mean"),
                psnr_std=("psnr", "std"), ssim=("ssim", "mean"), ssim_std=("ssim", "std"),
                residual=("residual", "mean"), model_err=("model_err", "first")).reset_index()
    best = agg.loc[agg.groupby(["mode", "phantom", "angles", "noise", "method"]).rel_err.idxmin()]
    best[best["mode"] == "fine"].drop(columns="mode").to_csv(RES / "best_by_method.csv", index=False)
    crime = best[best.angles.isin(CRIME_ANGLES) & best.noise.isin(CRIME_NOISE)]
    crime.pivot_table(index=["phantom", "angles", "noise", "method"], columns="mode",
                      values="rel_err").reset_index().to_csv(RES / "inverse_crime.csv", index=False)
    discrepancy(raw, paths).to_csv(RES / "discrepancy.csv", index=False)
    print(f"sweep finished in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
