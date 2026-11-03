# Parallel Monte Carlo and finite-difference pricing in the Black–Scholes model (MPI)

Course project, *Large-Scale Scientific Computing Infrastructure* (Southwest Petroleum University, 2024).

**Report:** [report/main.pdf](report/main.pdf)

## Main results

- Crank–Nicolson (with Rannacher start) converges at second order in time and space (observed order 2.00). It matches the closed-form European call to within 4 × 10⁻⁵ for maturities of 1–4 years (S₀ = 50, K = 52, r = 5%, σ = 30%).
- Without the Rannacher start the price is equally accurate, but the gamma at the strike is off by more than half its value on the same grid (N = 2000, M = 250); a frozen-coefficient damping estimate predicts this error without fitted constants.
- At 2.56 × 10⁵ paths, antithetic sampling plus the terminal-price control variate reduces the variance of the European estimator 27-fold. The geometric-Asian control variate reduces the variance of the arithmetic Asian estimator about 500-fold. The factors predicted from three measured correlations agree with the observed ones to within 1%.
- On a ten-core laptop (4 performance and 6 efficiency cores), strong scaling with 8 × 10⁶ paths reaches 3.49× on 8 processes (efficiency 44%). Weak scaling shows that shared memory bandwidth and clock limits, not MPI communication, cause the loss: load imbalance is at most 4.4%, there is a single gather of 213 bytes per process, and the speed-up is predicted by p × (weak-scaling efficiency).
- Merging additive statistics reproduces the single-process estimates to within 1.4 × 10⁻¹⁴ for any partition of the paths.

## Contents

| Path | Content |
|---|---|
| `code/bs.py` | Closed forms (European, geometric Asian), path simulation, moment merging, Crank–Nicolson |
| `code/mpi_price.py` | MPI pricer (one JSON line of results on rank 0) |
| `code/experiments.py` | Accuracy, CN convergence, variance reduction, CI coverage (serial) |
| `code/benchmark.py` | Strong/weak scaling and the T = 1..4 price table (runs `mpiexec`) |
| `code/diagnostics.py` | CN refinement, domain and Greeks studies; MC correlations; merge check; performance model (no timing) |
| `results/` | CSV/JSON outputs; `results/figures/` PDF + PNG figures |
| `report/main.tex`, `report/main.pdf` | Report |

## Reproduce

```bash
pip install -r ../requirements.txt          # includes the mpich wheel, which provides mpiexec
python code/experiments.py                  # about 1 minute
python code/benchmark.py                    # about 10 minutes; run on an otherwise idle machine
python code/diagnostics.py                  # about 15 seconds; reads results/scaling_summary.csv
mpiexec -n 8 python code/mpi_price.py --paths 1000000 --T 1 --antithetic
```

| Report item | Output file |
|---|---|
| Table 1 (prices T = 1..4) | `results/price_table.csv`, `results/cn_accuracy.csv` |
| Fig. 1 (CN convergence) | `results/cn_convergence.csv` |
| Tables 2–3 (time/space refinement, domain truncation) | `results/cn_refinement.csv`, `results/cn_domain.csv` |
| Fig. 2 (gamma and Rannacher start) | `results/cn_greeks.csv`, `results/cn_greeks_profile.csv` |
| Fig. 3, Table 4 (variance reduction) | `results/mc_summary.csv`, `results/mc_variance_reduction.csv`, `results/experiments.json` |
| Table 5 (correlations, predicted VRFs) | `results/mc_correlation.csv`, `results/diagnostics.json` |
| Table 6 (merge check) | `results/merge_check.csv`, `results/diagnostics.json` |
| Fig. 4, Table 7 (scaling) | `results/scaling_raw.csv`, `results/scaling_summary.csv`, `results/benchmark.json` |
| Table 8 (performance model) | `results/performance_model.csv` |
