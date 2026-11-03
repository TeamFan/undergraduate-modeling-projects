"""Strong- and weak-scaling benchmark of mpi_price.py, plus the T = 1..4 price table.

    python code/benchmark.py          # about 10 minutes on a 10-core laptop

Each configuration is run REPEATS times as a fresh `mpiexec` job.  We record
(i) the compute time reported by the program (max over ranks, from the first
barrier to the end of the reduction) and (ii) the wall time of the whole job
including process start-up and imports.
"""

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
MPIEXEC = str(Path(sys.executable).with_name("mpiexec"))
REPEATS = 5
STRONG_PATHS = 8_000_000
WEAK_PATHS = 1_000_000
CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
plt.rcParams.update({"font.size": 9, "legend.fontsize": 8, "savefig.bbox": "tight"})


def run(p, *args):
    cmd = [MPIEXEC, "-n", str(p), sys.executable, str(CODE / "mpi_price.py"), *args]
    t = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    wall = time.perf_counter() - t
    out = json.loads(res.stdout.strip().splitlines()[-1])
    out["wall"] = wall
    return out


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)


def machine():
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for key in ["machdep.cpu.brand_string", "hw.ncpu", "hw.perflevel0.physicalcpu", "hw.perflevel1.physicalcpu"]:
        try:
            info[key] = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
        except OSError:
            pass
    import mpi4py
    import numpy
    info["mpi4py"] = mpi4py.__version__
    info["numpy"] = numpy.__version__
    from mpi4py import MPI
    info["mpi"] = MPI.Get_library_version().splitlines()[0].strip()
    return info


def plot_scaling(s, info, f_hat):
    st = s[s["mode"] == "strong"]
    wk = s[s["mode"] == "weak"]
    p = st.ranks.to_numpy(float)
    n_perf = float(info.get("hw.perflevel0.physicalcpu", 0) or 0)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    ax.plot(p, p, ":", c="#999999", label="ideal")
    ax.errorbar(st.ranks, st.speedup, yerr=st.speedup * st.t_std / st.t_mean, fmt="o-", c=CB[0], ms=4,
                capsize=2, label="compute time")
    ax.plot(st.ranks, st.wall_speedup, "s--", c=CB[1], ms=4, label="wall time of job")
    ax.plot(p, 1 / ((1 - f_hat) + f_hat / p), "-.", c=CB[2], label=f"Amdahl fit, f = {f_hat:.2f}")
    if n_perf:
        ax.axvline(n_perf + 0.5, c="#cccccc", lw=0.8)
    ax.set_xlabel("MPI processes")
    ax.set_ylabel("speed-up")
    ax.set_title(f"a) strong scaling, {STRONG_PATHS / 1e6:.0f}M paths", loc="left", fontsize=9)
    ax.legend(loc="upper left")
    ax = axes[1]
    ax.errorbar(wk.ranks, wk.efficiency, yerr=wk.efficiency * wk.t_std / wk.t_mean, fmt="o-", c=CB[0], ms=4,
                capsize=2, label="compute time")
    ax.plot(wk.ranks, wk.wall_efficiency, "s--", c=CB[1], ms=4, label="wall time of job")
    if n_perf:
        ax.axvline(n_perf + 0.5, c="#cccccc", lw=0.8)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("MPI processes")
    ax.set_ylabel("efficiency $T_1/T_p$")
    ax.set_title(f"b) weak scaling, {WEAK_PATHS / 1e6:.0f}M paths per process", loc="left", fontsize=9)
    ax.legend()
    save(fig, "scaling")


def main():
    procs = list(range(1, 11))
    rows = []
    log(f"strong scaling, {STRONG_PATHS} paths")
    run(2, "--paths", "100000")  # warm-up (file cache, imports)
    for rep in range(REPEATS):
        for p in procs:
            o = run(p, "--paths", str(STRONG_PATHS))
            rows.append({"mode": "strong", "ranks": p, "rep": rep, "t_compute": o["t_compute_max"],
                         "t_compute_min": o["t_compute_min"], "wall": o["wall"]})
        log(f"  repeat {rep + 1}/{REPEATS}")
    log(f"weak scaling, {WEAK_PATHS} paths per rank")
    for rep in range(REPEATS):
        for p in procs:
            o = run(p, "--paths-per-rank", str(WEAK_PATHS))
            rows.append({"mode": "weak", "ranks": p, "rep": rep, "t_compute": o["t_compute_max"],
                         "t_compute_min": o["t_compute_min"], "wall": o["wall"]})
        log(f"  repeat {rep + 1}/{REPEATS}")
    raw = pd.DataFrame(rows)
    raw.to_csv(RES / "scaling_raw.csv", index=False)

    s = raw.groupby(["mode", "ranks"]).agg(t_mean=("t_compute", "mean"), t_std=("t_compute", "std"),
                                           wall_mean=("wall", "mean"), wall_std=("wall", "std"),
                                           imbalance=("t_compute_min", "mean")).reset_index()
    s["imbalance"] = 1 - s.imbalance / s.t_mean
    for mode in ("strong", "weak"):
        m = s["mode"] == mode
        t1 = s.loc[m & (s.ranks == 1), "t_mean"].iloc[0]
        w1 = s.loc[m & (s.ranks == 1), "wall_mean"].iloc[0]
        if mode == "strong":
            s.loc[m, "speedup"] = t1 / s.loc[m, "t_mean"]
            s.loc[m, "efficiency"] = s.loc[m, "speedup"] / s.loc[m, "ranks"]
            s.loc[m, "wall_speedup"] = w1 / s.loc[m, "wall_mean"]
            s.loc[m, "wall_efficiency"] = s.loc[m, "wall_speedup"] / s.loc[m, "ranks"]
        else:
            s.loc[m, "efficiency"] = t1 / s.loc[m, "t_mean"]
            s.loc[m, "wall_efficiency"] = w1 / s.loc[m, "wall_mean"]
    s.to_csv(RES / "scaling_summary.csv", index=False)

    # Amdahl fit on the strong-scaling compute times: T_p = T_1 ((1-f) + f/p)
    st = s[s["mode"] == "strong"]
    p = st.ranks.to_numpy(float)
    y = st.t_mean.to_numpy() / st.t_mean.iloc[0]
    f_hat = float(np.clip(np.sum((1 / p - 1) * (y - 1)) / np.sum((1 / p - 1) ** 2), 0, 1))

    # price table T = 1..4 on 8 ranks, 1e6 paths + antithetic
    log("price table")
    z = norm.ppf(0.995)
    table = []
    for T in [1, 2, 3, 4]:
        for anti in (False, True):
            args = ["--paths", "1000000", "--T", str(T)] + (["--antithetic"] if anti else [])
            o = run(8, *args)
            for est in ("plain", "cv"):
                v, se = o["euro"][est], o["euro"][f"{est}_se"]
                table.append({"T": T, "antithetic": anti, "estimator": est, "price": v, "se": se,
                              "ci_low": v - z * se, "ci_high": v + z * se, "exact": o["exact_euro"],
                              "abs_err": abs(v - o["exact_euro"]),
                              "covered": v - z * se <= o["exact_euro"] <= v + z * se,
                              "asian": o["asian"][est], "asian_se": o["asian"][f"{est}_se"]})
    pd.DataFrame(table).to_csv(RES / "price_table.csv", index=False)

    info = machine()
    info.update({"repeats": REPEATS, "strong_paths": STRONG_PATHS, "weak_paths_per_rank": WEAK_PATHS,
                 "steps": 252, "amdahl_parallel_fraction": f_hat})
    (RES / "benchmark.json").write_text(json.dumps(info, indent=2))

    plot_scaling(s, info, f_hat)
    log("done")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "figures":
        s_ = pd.read_csv(RES / "scaling_summary.csv")
        info_ = json.loads((RES / "benchmark.json").read_text())
        plot_scaling(s_, info_, info_["amdahl_parallel_fraction"])
    else:
        main()
