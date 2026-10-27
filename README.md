# ANSR-DT: Lightweight HNSRM PoC on C-MAPSS FD001

This repository is a lightweight **hybrid neuro-symbolic reasoning and monitoring (HNSRM)** proof of concept built on NASA C-MAPSS **FD001**. It keeps the core idea of ANSR-DT, neural perception followed by explicit symbolic and probabilistic reasoning, while removing the heavy components from the original framework.

The following are intentionally out of scope in this PoC:

- GraphRAG, Neo4j and knowledge-graph storage
- LLM-based decision making
- PPO/RL and control policies
- SKAB and the synthetic digital-twin data path
- TensorFlow and CNN-LSTM models

## Pipeline

```text
30-cycle sensor window (14 FD001 sensors)
        |
        +---------------------------+
        |                           |
        v                           v
RandomForest (default)       LightGBM (complement)
StandardScaler + flatten     explicit temporal features
                             + optional raw window
                             + auxiliary RUL regressor
        |                           |
        +--> calibration on training data (auto: sigmoid/isotonic)
        |                           |
        +--> validation-only decision threshold
        |                           |
        |                           +--> validation-only logistic fusion
        |                                (neural, symbolic, ProbLog, RUL)
        |                           |
        +--> full window -> SWI-Prolog HPC + temporal rules -> insights
                                    |
                       active conditions + learned HITL rules
                                    |
                                    v
                        ProbLog model -> P(failure_risk)
                                    |
                                    v
                        alert confirmation (k consecutive / smoothing)
                                    |
                                    v
                        HITL queue (uncertain / conflicting)
                                    |
                       human label -> rules_learned.pl
                                    -> hitl_log.jsonl
```

The neural model detects degradation from sliding windows. RandomForest remains the default architecture. LightGBM is a complementary path that receives interpretable temporal statistics in addition to the optional flattened window, and can use an out-of-fold RUL prediction as an extra feature. Probabilities are calibrated on training data, and the decision threshold is selected on validation data only. The symbolic layer explains which HPC conditions are active, including trend and persistence rules over the full window. ProbLog combines those conditions into a failure-risk estimate, optionally boosted by validated HITL rules. For LightGBM, a logistic model fitted on validation data only fuses the neural probability, symbolic score, ProbLog risk and predicted RUL. Alerts can require consecutive positive windows or a smoothed risk score. Cases where the neural probability and symbolic evidence disagree, or where the probability is close to the decision boundary, are placed in a human review queue.

## Reliability and evaluation safeguards

- **Probability calibration:** sigmoid or isotonic calibration with stratified folds is fitted on training data only. `calibration.method: auto` evaluates both on validation and selects the lower Brier score; the test split never drives the choice. Reports are written to `results/calibration_report.json` (and architecture-specific copies during comparison).
- **Decision threshold:** selected on validation scores, never on the test set. The default policy maximises recall subject to validation precision `>= 0.75` (`thresholding.strategy: recall_at_precision`). When fusion is active, the reported threshold is the fusion threshold and the neural threshold is kept alongside it.
- **Explicit temporal features:** the LightGBM path builds 13 interpretable statistics per sensor (first/last value, mean, standard deviation, min/max, slope, relative change, moving-average statistics and trailing persistence). Percentile thresholds are fitted on training windows only, and the optional flattened raw window is appended in sensor-major then cycle order (`results/temporal_feature_metadata.json`).
- **Auxiliary RUL regression:** a LightGBM regressor is trained on the training split. Its out-of-fold predictions are generated with unit-grouped `GroupKFold`, so windows from the same engine never appear on both sides of a fold. Predicted RUL is appended to the classifier features and is also a fusion input.
- **Validation-only fusion:** `ProbabilityFusion` standardises `[neural_probability, symbolic_rule_score, problog_failure_risk, predicted_rul]` and fits logistic regression on the validation split only. The fusion threshold is selected on validation as well; coefficients are recorded for auditability in `results/cmapss_results.json`.
- **Alert confirmation:** `results/alerting_analysis*.json` compares immediate alerts with `k` consecutive positive windows and trailing moving-average smoothing, including false alarms, recall, precision and lead-time deltas.
- **Model comparison:** `python main.py --compare-models` evaluates RandomForest and LightGBM on the same engine-level split. The selected model is chosen from validation metrics only; the comparison report records `selection_split: "validation"`.
- **Engine-level metrics:** per-engine recall, precision, false alarms, first alert cycle, lead time and anticipation are exported to `results/engine_metrics_validation.csv` and `results/engine_metrics_test.csv`, plus fusion and per-architecture copies.
- **Temporal rules:** slopes, relative changes, moving averages, standard deviation and persistence are computed per window. Their thresholds come from training windows only (`results/temporal_thresholds.json`). They are computed before fusion so training and inference use the same symbolic contract.
- **Learned-rule validation:** at the start of a pipeline run, existing HITL rules are checked on validation windows before being used. A rule must reach precision `>= 0.90`, support `>= 5` and false-positive rate `<= 0.05`; rejected rules are removed automatically. The report is `results/hitl_rule_validation.json`.
- **Input quality:** NaN/Inf values are either replaced by zero (default) or rejected, controlled by `preprocessing.non_finite_policy`.

## Dataset and labels

- **Training data:** `data/raw/CMAPSS/FD001_processed.parquet` from the Hugging Face TwinnableAgent processed FD001 dataset (includes a `rul` column).
- **External test:** official NASA C-MAPSS `test_FD001.txt` and `RUL_FD001.txt`.
- **Fallback:** raw `train_FD001.txt` is also supported when the processed parquet is unavailable.
- **Selected sensors (14):** `s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21`.
- **Label:** a window is positive when its final-cycle RUL is `<= 30`.
- **Low-data protocol:** 20% of engines for training, 20% for validation, and the 100 official test engines for evaluation.

Place the data files under `data/raw/CMAPSS/`. The directory is ignored by git because the files are downloaded separately.

### Symbolic thresholds

Thresholds are derived from training percentiles, not from the test set:

| Condition | Threshold | Interpretation |
|---|---:|---|
| `s4` elevated | `> 1414.555` | HPC outlet temperature elevated (75th percentile) |
| `s4` critical | `> 1425.670` | HPC outlet temperature critical (95th percentile) |
| `s11` high | `> 47.700` | HPC static pressure high |
| `s7` low | `< 552.810` | HPC outlet pressure low |
| `s12` low | `< 520.960` | Fuel flow low |
| `s21` low | `< 23.222` | Coolant bleed low |

Spearman analysis on the training set shows that `s4`, `s11`, `s15`, `s17`, `s2` and `s3` rise with degradation, while `s7`, `s12`, `s20` and `s21` fall.

## Installation

Requirements:

- Python 3.10+ (tested with Python 3.13)
- SWI-Prolog (tested with 10.0.2)
- `pip install -r requirements.txt`

On Windows, make SWI-Prolog available before running any test or pipeline command:

```powershell
$env:Path = "C:\Program Files\swipl\bin;$env:Path"
swipl --version
```

On Linux or macOS, install SWI-Prolog with your package manager (for example `sudo apt install swi-prolog` or `brew install swi-prolog`).

Then install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Quick start

Default RandomForest run:

```bash
python main.py
```

LightGBM run with temporal features, auxiliary RUL regression and validation-only fusion:

```bash
python main.py --architecture lightgbm
```

Controlled RandomForest vs LightGBM comparison (selection on validation only):

```bash
python main.py --compare-models
```

The pipeline writes the run summary, metrics, calibration/threshold reports, alerting analysis, HITL queue and architecture-specific artifacts. Key files:

- `results/final_run_results.json` and `results/cmapss_results.json`
- `results/model_comparison.json` (after `--compare-models`)
- `results/calibration_report.json`
- `results/decision_threshold.json` (neural) and `results/fusion_threshold.json` (LightGBM fusion)
- `results/alerting_analysis.json` and `results/alerting_analysis_fusion.json`
- `results/temporal_feature_metadata.json`
- `results/engine_metrics_validation.csv` and `results/engine_metrics_test.csv`
- `results/temporal_thresholds.json`
- `results/hitl_rule_validation.json`
- `results/hitl_queue.csv`, `results/hitl_queue.json` and `results/hitl_log.jsonl` (after feedback)
- `results/models/light_classifier.joblib` (RandomForest) or `results/models/lightgbm_classifier.joblib` (LightGBM)
- `results/models/temporal_features.joblib`, `results/models/lightgbm_rul_regressor.joblib` and `results/models/probability_fusion.joblib` (LightGBM path)
- `results/scaler.pkl` (RandomForest path)
- `results/visualization/*.png`

Evaluate the standalone ProbLog examples:

```bash
python -m src.reasoning.prob_query
```

Run inference on a raw C-MAPSS file (no true RUL file required):

```bash
# RandomForest artifact from config
python -m src.inference.inference --mode cmapss \
  --data_file data/raw/CMAPSS/test_FD001.txt \
  --output results/inference_cmapss.json

# LightGBM artifact, using the fusion model when its artifacts exist
python -m src.inference.inference --mode cmapss --model lightgbm \
  --data_file data/raw/CMAPSS/test_FD001.txt \
  --fusion auto --output results/inference_lightgbm.json
```

`--fusion auto` is the default. It applies validation-fitted fusion for LightGBM when `probability_fusion.joblib` and `fusion_threshold.json` are present, and falls back to the neural threshold otherwise. Use `--fusion on` to fail if the artifacts are missing, or `--fusion off` to reproduce raw neural decisions. Inference rebuilds temporal features, RUL predictions, symbolic scores and ProbLog risk from the persisted artifacts, so its decisions match the training pipeline.

Review the first five queued cases interactively:

```bash
python -m src.hitl.feedback --queue results/hitl_queue.json --max-cases 5
```

Run the test suite (with SWI-Prolog on `PATH`):

```bash
python -m unittest discover -s tests -q
```

## Current results

The reported run is `python main.py --architecture lightgbm`, using 20 engines for training, 20 for validation and the 100 official test engines. The neural threshold is `0.0733` and the validation-selected fusion threshold is `0.2220`.

### RandomForest vs LightGBM comparison

`python main.py --compare-models` selects LightGBM using validation `fbeta`; the test split is shown for reporting only and does not drive selection. The comparison scores each architecture's calibrated neural model; fusion is evaluated separately in the selected-model run below.

| Model | Validation F1 | Validation recall | Test F1 | Test recall | Test precision |
|---|---:|---:|---:|---:|---:|
| RandomForest | 0.8297 | 0.9274 | 0.7135 | 0.7801 | 0.6574 |
| LightGBM | 0.8540 | 0.9903 | 0.7616 | 0.9910 | 0.6184 |

### LightGBM with fusion

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg. precision |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0.9414 | 0.7503 | 0.9839 | 0.8514 | 0.9920 | 0.9610 |
| Official test | 0.9821 | 0.6556 | 0.9518 | 0.7764 | 0.9952 | 0.8703 |

Test confusion matrix: `[[9698, 166], [16, 316]]`.

Fusion slightly lowers validation F1 relative to the raw LightGBM neural model (`0.8540` to `0.8514`) but improves official-test precision (`0.6184` to `0.6556`), F1 (`0.7616` to `0.7764`) and false alarms (`203` to `166`). Recall decreases from `0.9910` to `0.9518`, so the trade-off is explicit rather than uniformly better. The official test windows are highly imbalanced (`3.26%` positive), so accuracy alone is misleading; F1, recall, average precision and ROC-AUC are the more useful indicators.

### Calibration

`calibration.method: auto` selected sigmoid on validation for this run.

| Split | Brier score | ECE |
|---|---:|---:|
| Validation | 0.0275 | 0.0185 |
| Official test | 0.0100 | 0.0073 |

The fused probabilities have validation Brier `0.0318` and test Brier `0.0112`.

### Engine-level metrics (official test, fusion)

| Metric | Value |
|---|---:|
| Engines with at least one positive window | 25 / 100 |
| Mean per-engine recall | 0.9095 |
| Mean per-engine precision | 0.6407 |
| Total false alarms | 166 |
| Mean false alarms per engine | 1.66 |
| Mean lead time before last observed cycle | 12.29 cycles |
| Mean anticipation at first correct alert | 29.54 cycles |
| Alerted before end-of-life among relevant engines | 0.96 |

Most FD001 test trajectories never enter the `RUL <= 30` failure zone, hence the separate denominator for "relevant" engines.

### Alert confirmation

| Mode | Total false alarms | Mean recall | Mean precision |
|---|---:|---:|---:|
| Immediate | 166 | 0.9095 | 0.6407 |
| Consecutive 2 | 148 | 0.9064 | 0.6613 |
| Consecutive 3 | 146 | 0.8725 | 0.6483 |
| Smoothed 3 | 153 | 0.9046 | 0.6706 |

Consecutive confirmation removes isolated alerts at a small recall cost. Smoothing gives the best precision in this run.

### Auxiliary RUL regression

| Split | RMSE | MAE | Spearman |
|---|---:|---:|---:|
| Train (out-of-fold) | 20.16 | 14.70 | 0.806 |
| Validation | 19.14 | 14.70 | 0.799 |
| Official test | 49.46 | 35.59 | 0.633 |

RUL is an auxiliary signal, not the primary target. The test error is higher because the official test engines follow different trajectories from the training engines.

### ProbLog examples

The standalone examples in `src/reasoning/prob_query.py` produce:

| Example | `P(failure_risk)` |
|---|---:|
| Normal | 0.0000 |
| Degraded | 0.4318 |
| Critical | 0.6452 |

The representative test windows selected by the LightGBM pipeline are:

| Window | Fused probability | ProbLog risk |
|---|---:|---:|
| Normal | 0.0000 | 0.0000 |
| Degraded / near threshold | 0.2221 | 0.4318 |
| Critical | 0.9910 | 0.4318 |

The degraded window is a useful HITL example: its fused probability is at the decision threshold while ProbLog also reports an active degradation pattern.

## Human-in-the-loop

The queue is built from test windows with:

- calibrated neural probability inside the uncertainty band centered on the selected threshold, or
- high neural probability with no active HPC rule (rule/probability conflict), or
- low neural probability with an active HPC rule (rule/probability conflict).

The queue API also supports a ProbLog-risk conflict reason.

Current LightGBM + fusion run: **200 queued cases** from 776 evaluated candidates (`287` eligible).

| Reason | Cases |
|---|---:|
| Uncertain probability | 89 |
| Rule/probability conflict (low neural, rule active) | 111 |
| Rule/probability conflict (high neural, no strong rule) | 0 |

Workflow:

1. `python main.py` writes `results/hitl_queue.csv` and `results/hitl_queue.json`.
2. `python -m src.hitl.feedback` prompts for `failure` or `normal` labels.
3. Each feedback is appended to `results/hitl_log.jsonl`.
4. For confirmed failures, the active symbolic conditions are saved as a `hitl_rule_N` clause in `src/reasoning/rules_learned.pl`.
5. On the next pipeline run, learned rules are validated on held-out validation windows; rules that are too broad or imprecise are rejected.
6. Validated rules are consulted by Prolog and injected into ProbLog with per-rule weight `0.15`, capped at `3` active learned rules.

## Project layout

```text
configs/
  config.yaml
data/raw/CMAPSS/                  # downloaded FD001 files (ignored by git)
main.py
src/
  ansrdt/
    core.py                       # perception -> symbols -> state
  data/
    cmapss_loader.py              # FD001 parquet/txt loader and windowing
    data_loader.py                # legacy synthetic loader + factory
  evaluation/
    calibration.py                # Brier score, ECE, reliability curves
    thresholding.py               # validation-only threshold selection
    engine_metrics.py             # per-engine recall, false alarms, lead time
    alerting.py                   # consecutive confirmation and smoothing
  features/
    temporal_features.py          # explicit temporal statistics for LightGBM
  fusion/
    probability_fusion.py         # validation-only logistic fusion
  hitl/
    queue.py                      # uncertain/conflict queue, CSV/JSON export
    feedback.py                   # labels, learned rules, JSONL log, CLI
  inference/
    inference.py                  # legacy/raw C-MAPSS inference, optional fusion
  integration/
    adaptive_controller.py        # legacy synthetic-only controller (isolated)
  models/
    light_classifier.py           # RandomForest / MLP / LightGBM + calibration
    rul_regressor.py              # auxiliary LightGBM RUL and grouped OOF
  pipeline/
    pipeline.py                   # training, comparison, fusion, alerting, HITL
  reasoning/
    rules.pl                      # HPC symbolic rules, thresholds and temporal rules
    prob_rules.pl                 # ProbLog failure-risk model
    prob_query.py                 # ProbLog examples and evaluation
    reasoning.py                  # PySWIP bridge, rule learning, condition queries
    temporal.py                   # train-only temporal features and thresholds
    rules_learned.pl              # rules learned from HITL feedback
    rule_learning.py
    state_tracker.py
  utils/
    model_utils.py                # scaler and model persistence
tests/                            # 81 unit and integration tests
results/                          # generated artifacts (ignored by git)
```

## Testing

The suite covers the C-MAPSS loader, RandomForest/LightGBM classifiers, calibration, thresholding, engine metrics, temporal features, RUL regression, alert confirmation, probability fusion, config validation, learned-rule validation, ProbLog, the pipeline, inference, the HITL queue and HITL feedback:

```bash
$env:Path = "C:\Program Files\swipl\bin;$env:Path"
python -m unittest discover -s tests -q
```

Expected result: `81 tests ... OK`.

## Limitations

- FD001 only, with a binary `RUL <= 30` label.
- The official test set is highly imbalanced; accuracy is not a sufficient metric.
- Calibration is a post-hoc sigmoid/isotonic wrapper around RandomForest/MLP/LightGBM, not a sequence model trained with a probabilistic loss.
- LightGBM uses engineered window statistics and a flattened window, not a learned sequence encoder.
- The auxiliary RUL regressor is trained on training engines; its error on the official test engines is substantially higher than on validation, so it is a supporting feature rather than a standalone RUL estimator.
- The probability fusion is a four-input logistic model fitted on validation data. Its coefficients and threshold are specific to this FD001 split and should be recalibrated for other datasets or operating conditions.
- The decision threshold implements a recall-oriented policy; changing the precision constraint changes the precision/recall trade-off.
- HITL feedback can add symbolic and ProbLog rules, but it does not retrain the neural classifier.
- Temporal thresholds are fitted on training windows; learned-rule validation uses validation data. Neither is tuned on the official test set.
- No RL, control actions, LLM reasoning, GraphRAG or Neo4j.
- `AdaptiveController` is legacy synthetic-only code and is isolated from the FD001 pipeline.
- Data files are not distributed with the repository.
- This is a research proof of concept, not a production digital twin.

## Original ANSR-DT publications

This repository is a lightweight derivative for FD001 experimentation. The original framework is described in:

```bibtex
@article{hakim2025ansr,
  title={ANSR-DT: An Adaptive Neuro-Symbolic Learning and Reasoning Framework for Digital Twins},
  author={Hakim, Safayat Bin and Adil, Muhammad and Velasquez, Alvaro and Song, Houbing Herbert},
  journal={arXiv preprint arXiv:2501.08561},
  year={2025}
}
```

S. B. Hakim, M. Adil, A. Velasquez and H. H. Song, "An Explainable Neuro-Symbolic Rule Extraction Framework for Digital Twins," 2025 IEEE Smart World Congress (SWC), Calgary, AB, Canada, 2025, pp. 1042-1047. DOI: `10.1109/SWC65939.2025.00168`.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
