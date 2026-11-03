# Sparse-view image reconstruction from Radon projections

Course project, *Scientific Computing Applications and Practice* (Southwest Petroleum University, 2023).
The parallel-beam projection matrix of a 100 × 100 image is assembled explicitly for 5 to 180 angles, so that its rank, null space and singular values can be computed exactly. Filtered back-projection (FBP), LSQR, Tikhonov regularisation and total-variation (TV) regularisation (Chambolle–Pock) are compared on data simulated on a four-times finer grid, with oracle parameters and with the discrepancy principle, for two phantoms, three noise levels and three noise realisations.

**Report:** [report/main.pdf](report/main.pdf)

## Main results

- With 10 angles the 1,430 equations have rank 1,243, so 88% of the 10,000-dimensional image space is invisible to the measurements. With 90 and 180 angles the systems have full column rank, but condition numbers of 4.3 × 10⁶ and 1.4 × 10⁴.
- Every method whose output lies in the range of Aᵀ (FBP before clipping, LSQR from zero, Tikhonov) has a relative error at least equal to the null-space fraction of the object (Proposition 3.1). With 10 angles this bound is 0.734 for the "0038" digits phantom and 0.511 for Shepp–Logan; LSQR and Tikhonov reach 0.742/0.741 and 0.521/0.519.
- TV is the only method that beats the bound: 0.246 (digits) and 0.301 (Shepp–Logan) at 10 angles with 1% noise, against 1.21 and 0.74 for FBP. At 180 angles FBP, LSQR and Tikhonov reach 0.19–0.23 on the digits phantom and TV 0.10.
- The discretisation (model) error of 6–8% (digits) and 2–3% (Shepp–Logan) dominates 1% noise. The discrepancy principle with the noise level alone fails (relative errors near 1 for LSQR and Tikhonov at 180 angles); with the total data error it matches the oracle to within 0.01 in almost all cases.
- Generating the data with the reconstruction operator itself (inverse crime) would have lowered the TV error at 180 angles from 0.101 to 0.013.
- A restricted angular range is more damaging than few angles: 90 angles over 90° give TV 0.429 on the digits phantom, worse than 10 angles over 180°.

## Contents

| Path | Content |
|---|---|
| `code/tomo.py` | Pixel-driven projector, phantoms, FBP (Ram–Lak, Shepp–Logan, Hann), LSQR, Tikhonov, TV (primal–dual), error measures |
| `code/common.py` | Grids, angle sets, noise levels, parameter grids, data simulation on the fine grid |
| `code/sweep.py` | Main experiment: all methods, angles, noise levels, seeds and parameters; oracle and discrepancy selection; inverse-crime comparison |
| `code/spectral.py` | Rank, singular values, null-space bound and Picard coefficients on the 100 × 100 grid |
| `code/ablations.py` | FBP filter windows, limited-angle data, TV convergence, timing |
| `code/figures.py` | All figures from the files in `results/` |
| `code/run_all.py` | Runs the four scripts above in order |
| `results/` | CSV, NPZ and figure outputs (`results/figures/`, PDF and PNG) |
| `report/main.tex`, `report/main.pdf` | Report |

## Reproduce

```bash
pip install -r ../requirements.txt
python code/run_all.py            # about 45 minutes on 4 cores; noise seeds fixed
python code/run_all.py figures    # redraw all figures from results/ (seconds)
```

`sweep.py` runs its 18 jobs in a pool of four processes (about 18 minutes); `spectral.py` needs a few GB of memory for the dense 10,000 × 10,000 Gram matrices (about 10 minutes); `ablations.py` reads the oracle TV parameters from `results/best_by_method.csv` (about 16 minutes). The digits phantom is rendered with the DejaVu Sans font shipped with matplotlib, so it is the same on every platform. Timings in `results/timing.csv` (report Section 5.8) are single-core measurements and depend on the machine.

| Report item | Script | Output file |
|---|---|---|
| Table 1, Fig. 1(a), Fig. 2 | `spectral.py` | `results/system_sizes.csv`, `results/singular_values.csv`, `results/nullspace_bound.csv`, `results/nullspace_digits_10.npz` |
| Fig. 1(b) (Picard plot), residual floors | `spectral.py` | `results/picard.csv`, `results/nullspace_bound.csv` |
| Fig. 3, Table 2 | `sweep.py` | `results/best_by_method.csv` (every run: `results/all_runs.csv`) |
| Fig. 4 | `sweep.py` | `results/recon_examples.npz` |
| Fig. 5 | `sweep.py` | `results/lsqr_paths.csv`, `results/all_runs.csv` |
| Table 3 (discrepancy principle) | `sweep.py` | `results/discrepancy.csv` |
| Table 4 (inverse crime) | `sweep.py` | `results/inverse_crime.csv` |
| Table 5 (FBP windows) | `ablations.py` | `results/fbp_windows.csv` |
| Table 6 (limited angle) | `ablations.py` | `results/limited_angle.csv` |
| Fig. 6, Section 5.8 | `ablations.py` | `results/tv_convergence.csv`, `results/timing.csv` |
