# Ancient glass composition analysis (CUMCM 2022, Problem C)

Undergraduate mathematical-modelling training project (Southwest Petroleum University, 2023). All analyses are implemented in Python and treat the oxide percentages as compositional (log-ratio) data; artefact attributes are tested on artefacts, and every model is validated by leaving whole artefacts out.

**Report:** [report/main.pdf](report/main.pdf)

## Main results

- Surface weathering is associated with glass type (Fisher exact p = 0.011; 70% of Pb–Ba vs 33% of high-K artefacts weathered). Decoration looks unrelated in the pooled table (p = 0.10), but within high-K glass the six weathered artefacts are exactly the six decoration-B artefacts (p = 5.4 × 10⁻⁵); colour is unrelated in both types.
- Weathering raises SiO₂ in high-K glass (median 65.5% → 93.5%) and lowers it in Pb–Ba glass (54.6% → 25.0%) while enriching PbO (20.1% → 44.1%) and P₂O₅ (0.19% → 4.97%).
- A calibrated weathering correction in clr coordinates (best linear predictor under a perturbation model with artefact-specific noise) predicts the pre-weathering composition. On the two artefacts with both weathered and unweathered spots it lowers the mean absolute error of the major oxides from 7.30 to 3.40 and from 8.09 to 4.62 percentage points, and 23 of 26 corrected Pb–Ba points fall inside the range of unweathered glass.
- PbO + BaO separates the two types without error (high-K ≤ 4.27%, Pb–Ba ≥ 14.34%; threshold 9.3%), confirmed by a classification tree, a random forest and a log-ratio logistic model in leave-one-artefact-out validation (67, 67 and 66 of 67 points).
- Pb–Ba glass divides into four sub-classes (copper-free, impurity-bearing, barium-free, low-impurity) that k-means reproduces exactly and that are stable to measurement noise (mean ARI 0.91 at 10%); the gap statistic shows that they partition a graded continuum. High-K glass has no supported sub-classes.
- The 8 unknown artefacts get the same type from the rule, the logistic model and an Aitchison-distance classifier (A1, A6, A7 high-K; the others Pb–Ba), unchanged under 40% multiplicative noise and all detection-limit and regularisation settings.
- Two of 55 oxide-pair correlations differ between the types after FDR control, most clearly Fe₂O₃–CuO (r = 0.62 in high-K vs −0.49 in Pb–Ba).

## Contents

| Path | Content |
|---|---|
| `data/attachment.xlsx` | Contest data (sheet 1: 58 artefacts; sheet 2: 69 sampling points; sheet 3: 8 unknowns) |
| `data/problem_C_2022.pdf` | Problem statement (Chinese) |
| `code/run_all.py` | Complete analysis; writes every table and figure |
| `results/` | CSV/JSON outputs; `results/figures/` PDF + PNG figures |
| `report/main.tex`, `report/main.pdf` | Report |

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r ../requirements.txt
python code/run_all.py      # about 1 minute
cd report && latexmk -pdf main.tex
```

Fixed seeds (2022 for the main analysis, 2023 for the additional checks). Tested with Python 3.9 on macOS 27.2 (Apple M5) and on Linux x86-64 with the pinned versions; outputs agree up to floating-point rounding.

| Report item | Output file (in `results/`) |
|---|---|
| Table 1 (associations, pooled and by type) | `q1_association.csv`, `q1_association_by_type.csv`, `q1_table_*.csv` |
| Tables 2 and 9, Fig. 2 (composition by type and weathering) | `q1_composition_stats.csv`, `q1_centre_test.csv`, `figures/q1_boxplots.pdf` |
| Fig. 3, weathering shift and calibration gains | `q1_clr_shift.csv`, `q1_calibration.csv`, `figures/q1_clr_correction.pdf` |
| Table 3, Fig. 4 (test of the corrections) | `q1_correction_models.csv`, `q1_correction_plausibility.csv`, `q1_prediction_validation.csv`, `q1_weathering_direction.csv`, `figures/q1_prediction_validation.pdf` |
| Table 10 (pre-weathering predictions with 90% intervals) | `q1_pre_weathering_predictions.csv` (calibrated), `q1_pre_weathering_predictions_shift.csv` (plain shift) |
| Table 4, type rule and classifiers | `q2_separability.csv`, `q2_random_forest_importance.csv`, `q2_logistic_coefficients.csv`, `results.json` |
| Table 5, Figs. 5–6 (sub-classes) | `q2_subclasses.csv`, `q2_subclass_profiles_*.csv`, `figures/q2_dendrograms.pdf`, `figures/q2_subclasses.pdf` |
| Table 6, gap statistic, stability | `q2_rationality.csv`, `q2_gap_statistic.csv`, `q2_sensitivity.csv`, `results.json` |
| Table 7 (unknown artefacts) | `q3_unknown_classification.csv`, `q3_unknown_subclass.csv` |
| Fig. 7, Problem 4 | `q4_within_type_pairs.csv`, `q4_clr_correlation_*.csv`, `q4_correlation_differences.csv`, `q4_bootstrap_intervals.csv`, `figures/q4_correlations.pdf` |
| Table 8 (sensitivity to δ, correction estimator and C) | `q5_sensitivity_delta_C.csv` |
