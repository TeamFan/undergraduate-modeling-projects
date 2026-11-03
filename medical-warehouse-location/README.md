# Medical-supply depot location and helicopter routing

Southwest Petroleum University Mathematical Modelling Contest 2023, Problem A (team entry, third prize).

**Report:** [report/main.pdf](report/main.pdf)

## Main results

- **Q1.** The depot that minimises the total great-circle distance to the 21 hospitals is exactly the Ziyang hospital site (3,847.95 km in total), certified for this spherical distance objective by applying Kuhn's vertex condition (resultant 0.735 ≤ 1). Any move of the depot increases the total distance by at least 0.27 km per km; a genetic algorithm with 20 seeds always ends within 0.83 km of the optimum.
- **Q2.** A capacitated, range-constrained routing MILP solved to optimality shows that one Mi-26 cannot serve all hospitals: the shortest tour is 2,505.5 km against a 2,000 km range. **Two helicopters** are optimal (2,593.7 km in total), 4.4% shorter than the Clarke–Wright savings heuristic.
- **Q3.** The loss data follow 0.12·(t/300)³ to the printed precision, and the loss-optimal plans are independent of the cruise speed. The distance-optimal plan loses 433 kg; the efficient plans range down to 7.7 kg with ten helicopters at twice the distance. The compromise plan (closest to the ideal point) is a **feasible three-helicopter plan, 2,800 km, 86 kg**, found within the 300 s limit with a remaining MIP gap of 76.9%, so it is not certified as a point of the efficient set.
- **Q4.** Both routing problems are re-solved at 47 candidate sites. The minimum loss is certified at all of them and the shortest routes at the 33 central sites; the 14 peripheral route problems keep gaps of 3–30% and their lower bounds lie below Ziyang's 2,593.7 km, so they are not ranked. Two certified sites 16–18 km north-west are slightly shorter (2,592.4 and 2,591.3 km, at most 0.1%), and moving to Chengdu would add 1.1%. **The Ziyang depot is appropriate** within these margins.

## Contents

| Path | Content |
|---|---|
| `data/` | Contest attachments 1–3 and the problem statement (Chinese) |
| `code/model.py` | Distances, GA, Weber point, loss fit, routing MILP (HiGHS via SciPy) |
| `code/run_all.py` | Models I–IV: `python code/run_all.py q123` (Q1–Q3, ~35 min), `... q4` (~50 min), no argument for both |
| `code/further_analysis.py` | Model checks and sensitivity: `python code/further_analysis.py slow` (additional budgets, range/payload grid, ~1 h), then `... fast` (~1 min; needs `results/results.json`) |
| `results/` | CSV/JSON outputs; `results/figures/` PDF + PNG figures |
| `report/main.tex`, `report/main.pdf` | Report |

## Reproduce

```bash
pip install -r ../requirements.txt
python code/run_all.py
python code/further_analysis.py
```

MILPs use HiGHS 1.x as bundled with SciPy 1.13.1. The ε-constrained loss problems stop after 300 s and record their MIP gap; the candidate-site problems of Q4 stop after 60 s. Because of these time limits, plans that are not proven optimal can differ slightly between runs and machines.

| Report item | Output file |
|---|---|
| Fig. 2, Q1 numbers | `results/q1_ga_runs.csv`, `results/results.json` (`q1`) |
| Tables 1–2, Fig. 3 (vertex test, landscape) | `results/q1_vertex_test.csv`, `results/q1_flatness.csv`, `results/further_analysis.json` (`q1`) |
| Table 3, Fig. 4 | `results/q2_fleet_size.csv`, `results/results.json` (`q2`) |
| Table 9, savings heuristic | `results/q2_route_detail.csv`, `results/further_analysis.json` (`q2`) |
| Fig. 5 (loss fit) | `results/results.json` (`loss_fit`) |
| Table 4, Figs. 6–8 | `results/q3_pareto.csv`, `results/q3_front_extra.json`, `results/q3_tradeoff.csv`, `results/q3_weight_choice.csv` |
| Table 10 | `results/q3_knee_route_detail.csv` |
| Table 5 (speed) | `results/q3_speed_sensitivity.csv`, `results/further_analysis.json` (`q2.speed_scaling`) |
| Table 6 (range, payload) | `results/q2_range_payload.csv` |
| Table 7 (loss law) | `results/q3_loss_law.csv` |
| GA parameters | `results/q1_ga_parameters.csv` |
| Fig. 9 (Model IV) | `results/q4_candidates.csv`, `results/results_q4.json` |
