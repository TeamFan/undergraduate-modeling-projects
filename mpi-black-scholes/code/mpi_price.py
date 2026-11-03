"""MPI-parallel Monte Carlo pricing of European and arithmetic Asian calls.

    mpiexec -n 8 python code/mpi_price.py --paths 2000000 --steps 252 --T 1

Each rank simulates its share of the paths with an independent random stream
(numpy SeedSequence.spawn), reduces additive sufficient statistics to rank 0,
and rank 0 prints one JSON line with the estimates and the timings.  The
result for a fixed seed depends on the number of ranks (the paths are split
into different streams) but not on the order in which ranks finish.
"""

import argparse
import json

import numpy as np
from mpi4py import MPI

from bs import bs_call, estimates, geometric_asian_call, merge, moments, simulate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S0", type=float, default=50.0)
    ap.add_argument("--K", type=float, default=52.0)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--r", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=0.30)
    ap.add_argument("--steps", type=int, default=252)
    ap.add_argument("--paths", type=int, default=1_000_000, help="total paths (strong scaling)")
    ap.add_argument("--paths-per-rank", type=int, default=None, help="paths per rank (weak scaling)")
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--antithetic", action="store_true")
    ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    total = a.paths_per_rank * size if a.paths_per_rank else a.paths
    share = total // size + (1 if rank < total % size else 0)
    rng = np.random.default_rng(np.random.SeedSequence(a.seed).spawn(size)[rank])
    prm = dict(S0=a.S0, K=a.K, T=a.T, r=a.r, sigma=a.sigma)

    comm.Barrier()
    t0 = MPI.Wtime()
    stats = []
    done = 0
    while done < share:
        n = min(a.chunk, share - done)
        n -= n % 2 if a.antithetic else 0
        if n == 0:
            break
        stats.append(moments(simulate(rng, n, M=a.steps, antithetic=a.antithetic, **prm)))
        done += n
    local = merge(stats) if stats else {}
    t_compute = MPI.Wtime() - t0
    gathered = comm.gather(local, root=0)
    t_total = MPI.Wtime() - t0
    t_compute_max = comm.reduce(t_compute, op=MPI.MAX, root=0)
    t_compute_min = comm.reduce(t_compute, op=MPI.MIN, root=0)

    if rank == 0:
        st = merge([g for g in gathered if g])
        est = estimates(st, M=a.steps, **prm)
        out = {"ranks": size, "paths": int(total), "steps": a.steps, "T": a.T, "antithetic": a.antithetic,
               "samples": est["n"], "seed": a.seed,
               "exact_euro": float(bs_call(**prm)),
               "exact_geo_asian": float(geometric_asian_call(M=a.steps, **prm)),
               "euro": {k: float(v) for k, v in est["euro"].items()},
               "asian": {k: float(v) for k, v in est["asian"].items()},
               "t_compute_max": t_compute_max, "t_compute_min": t_compute_min, "t_total": t_total}
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
