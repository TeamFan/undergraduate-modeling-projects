# Undergraduate modelling and scientific-computing projects

Five projects from my B.Sc. in Information and Computing Sciences at Southwest Petroleum University (2021–2025), covering facility location and vehicle routing, stochastic supply planning, parallel option pricing, tomographic reconstruction and compositional data analysis. Each folder contains the input data, the code, every generated table and figure, and a report (LaTeX source and compiled PDF).

The bachelor thesis has its own repository: [thyroid-cancer-recurrence](https://github.com/TeamFan/thyroid-cancer-recurrence).

| Project | Setting | Methods | Report |
|---|---|---|---|
| [medical-warehouse-location](medical-warehouse-location/) | SWPU modelling contest 2023, Problem A (team, third prize) | Spherical Weber problem, genetic algorithm, capacitated range-constrained VRP (MILP), ε-constraint distance–loss trade-off | [PDF](medical-warehouse-location/report/main.pdf) |
| [supplier-selection-planning](supplier-selection-planning/) | CUMCM 2021 Problem C (training exercise, 2023) | Chance-constrained supplier selection (SAA), LP with uncertainty-dependent safety stock, carrier assignment MILP, out-of-sample simulation | [PDF](supplier-selection-planning/report/main.pdf) |
| [mpi-black-scholes](mpi-black-scholes/) | Course project, Large-Scale Scientific Computing (2024) | MPI-parallel Monte Carlo, antithetic and control variates, Asian options, Crank–Nicolson, strong/weak scaling | [PDF](mpi-black-scholes/report/main.pdf) |
| [radon-reconstruction](radon-reconstruction/) | Course project, Scientific Computing Applications (2023) | Over/underdetermined systems, SVD, FBP, LSQR, Tikhonov, TV (Chambolle–Pock) | [PDF](radon-reconstruction/report/main.pdf) |
| [ancient-glass-composition](ancient-glass-composition/) | CUMCM 2022 Problem C (training exercise, 2023) | Compositional (log-ratio) data analysis, exact tests, logistic classification, Ward clustering with stability analysis | [PDF](ancient-glass-composition/report/main.pdf) |

## Requirements

- Python 3.9 with the pinned packages in [`requirements.txt`](requirements.txt). The `mpich` wheel provides `mpiexec` for the MPI project, so no system MPI is needed.
- Optional, for rebuilding the reports: a TeX distribution with pdfLaTeX and `latexmk` (TeX Live 2023 or later).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the results

Every script writes its outputs to `<project>/results/` (CSV/JSON) and `<project>/results/figures/` (PDF and PNG). Random seeds are fixed in every script.

| Project | Command | Run time |
|---|---|---|
| medical-warehouse-location | `python medical-warehouse-location/code/run_all.py` | about 1 hour (MILPs) |
| | `python medical-warehouse-location/code/further_analysis.py` | about 30 minutes (MILPs) |
| supplier-selection-planning | `python supplier-selection-planning/code/run_all.py` | about 20 minutes |
| | `python supplier-selection-planning/code/supplementary.py` | about 25 minutes |
| mpi-black-scholes | `python mpi-black-scholes/code/experiments.py` | about 1 minute |
| | `python mpi-black-scholes/code/benchmark.py` | about 10 minutes |
| | `python mpi-black-scholes/code/diagnostics.py` | about 15 seconds |
| radon-reconstruction | `python radon-reconstruction/code/run_all.py` | about 45 minutes on 4 cores |
| ancient-glass-composition | `python ancient-glass-composition/code/run_all.py` | about 1 minute |

Run times were measured with Python 3.9.6 on macOS 27.2 (Apple M5, 10 cores). MILPs with time limits and the scaling benchmarks can give slightly different values on other machines; the project READMEs say where. Each project README maps every table, figure and number in the report to the script and output file that produce it.

## Building the reports

```bash
cd <project>/report && latexmk -pdf main.tex
```

Figures are read from `../results/figures/`, so the reports build from the committed results without rerunning the code.

## Repository layout

```text
<project>/
├── README.md        # main results, contents, report item → output file
├── code/            # scripts (run_all.py or equivalent)
├── data/            # input data (contest attachments where applicable)
├── results/         # generated CSV/JSON and figures (PDF + PNG)
└── report/          # main.tex and compiled main.pdf
```
