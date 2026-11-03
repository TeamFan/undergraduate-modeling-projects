# Supplier selection, ordering and transport planning under uncertain deliveries (CUMCM 2021, Problem C)

Undergraduate mathematical-modelling training exercise (Southwest Petroleum University, 2023).

**Report:** [report/main.pdf](report/main.pdf)

## Main results

- **Supply model.** Each supplier has a *sustained capacity*: the largest quantity it delivered in every week of some 4-week run. Its delivered-to-ordered ratio is random and resampled from its own history.
- **Q1.** The 50 highest-ranked suppliers hold 98.8% of the sustained capacity. The ranking is stable across capacity windows (Kendall τ 0.62–0.99, top-50 overlap 48–50), and an independent entropy-weight TOPSIS evaluation shares 42 of the top 50.
- **Q2.** A chance-constrained 0–1 programme solved by sample average approximation selects **18 suppliers**. When every selected supplier is ordered at its estimated capacity, they keep the two-week stock in all 24 weeks in 96.7% of 5,000 new simulated scenarios (delivery ratios resampled from the fitted supply model with a seed independent of the selection). Ignoring uncertainty would suggest 15 suppliers, which achieve 40.1%. Thirteen repeated approximations with 100–400 scenarios select 17 or 18 suppliers (12 of 13 select 18), each sharing at least 15 with this set; the selection keeps about 97% compliance under a bootstrap of common historical weeks.
- **Q2 plans.** For the same 18 suppliers, the cheapest order plan reaches a lower compliance of 91.6% at the historical cost per unit of product; ordering at capacity reaches the 96.7% above for 8.8% more cost. Optimised carrier assignment gives an expected loss rate of 0.58% (0.59% when replayed against historical loss rates), compared with 1.38% for random feasible carrier assignments.
- **Q3.** Minimising raw volume first, which means preferring A, cuts material C by 61% and total volume by 4.0% at a 0.2% higher purchase cost.
- **Q4.** About +34% production (37,700 m³/week) is sustainable at 95% compliance with the base capacity (+36% at 90%, +31% at 99%). The answer ranges from −11% to +101% across capacity windows, so it depends mainly on the capacity definition.

## Contents

| Path | Content |
|---|---|
| `data/attachment1_suppliers.xlsx`, `data/attachment2_transporters.xlsx` | Official contest attachments (402 suppliers × 240 weeks; 8 carriers × 240 weeks) |
| `code/run_all.py` | Supply model and the four problems (HiGHS MILP/LP via SciPy, scenario simulation) |
| `code/supplementary.py` | Data description, TOPSIS cross-check, SAA stability, correlated-delivery stress test, cost–compliance trade-off, Q4 curve, sensitivity |
| `results/` | CSV/JSON outputs; `results/supplementary/` outputs of the supplementary analyses; `results/figures/` PDF + PNG figures |
| `report/main.tex`, `report/main.pdf` | Report |

The contest attachments were obtained from the public repository `Jin-zd/Mathematical-Modeling`. Two independent copies there have the same SHA-1 checksum (`54710a31…` for attachment 1).

## Reproduce

```bash
pip install -r ../requirements.txt
python code/run_all.py         # about 20 minutes, fixed seed
python code/supplementary.py   # about 25 minutes, fixed seed; reads the outputs of run_all.py
```

| Report item | Output file |
|---|---|
| Section 4, Figs. 2–3, Table 2 | `results/supplier_model.csv`, `results/results.json`, `results/supplementary/supplementary.json` (`data`), `results/supplementary/carriers.csv` |
| Tables 3–4 (Q1) | `results/q1_top50.csv`, `results/q1_robustness.csv`, `results/supplementary/q1_topsis.csv` |
| Table 5, Fig. 4 | `results/q2_selection.csv` |
| Table 6, greedy check | `results/supplementary/q2_saa_stability.csv`, `results/supplementary/q2_greedy.csv` |
| Q2 plans, Figs. 5–6 | `results/q2_order_plan.csv`, `results/q2_transport_plan.csv`, `results/results.json` (`q2_plan`, `q2_full_capacity_plan`), `results/supplementary/q2_tradeoff.csv` |
| Table 7, Fig. 7 | `results/q3_lexicographic.csv` |
| Table 8, Fig. 8 | `results/q4_capacity.csv`, `results/supplementary/q4_curve.csv`, `results/supplementary/q4_service_level.csv` |
| Tables 9–10 | `results/supplementary/supplementary.json` (`q2_stress`), `results/supplementary/sensitivity.csv` |
