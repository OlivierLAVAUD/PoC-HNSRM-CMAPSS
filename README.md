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
        v
StandardScaler -> LightClassifier (RandomForest by default)
        |
        +--> probability calibration (sigmoid, 5-fold) -> P(failure)
        |                                                 |
        |                     validation-only threshold --+
        |                                                 |
        +--> full window -> SWI-Prolog HPC + temporal rules -> insights
                                                          |
                             active conditions + learned HITL rules
                                                          |
                                                          v
                                      ProbLog model -> P(failure_risk)
                                                          |
                                                          v
                                      HITL queue (uncertain / conflicting)
                                                          |
                                             human label -> rules_learned.pl
                                                          -> hitl_log.jsonl
```

The neural model detects degradation from sliding windows. Probabilities are calibrated on training data and the decision threshold is selected on validation data only. The symbolic layer explains which HPC conditions are active, including trend and persistence rules over the full window. ProbLog combines those conditions into a failure-risk estimate, optionally boosted by validated HITL rules. Cases where the neural probability and symbolic evidence disagree, or where the probability is close to the decision boundary, are placed in a human review queue.

## Reliability and evaluation safeguards

- **Probability calibration:** sigmoid calibration with 5 stratified folds is fitted on training data only. Validation/test Brier scores and expected calibration error (ECE) are written to `results/calibration_report.json`.
- **Decision threshold:** selected on validation scores, never on the test set. The default policy maximises recall subject to validation precision `>= 0.75` (`thresholding.strategy: recall_at_precision`).
- **Engine-level metrics:** per-engine recall, precision, false alarms, first alert cycle, lead time and anticipation are exported to `results/engine_metrics_validation.csv` and `results/engine_metrics_test.csv`.
- **Temporal rules:** slopes, relative changes, moving averages, standard deviation and persistence are computed per window. Their thresholds come from training windows only (`results/temporal_thresholds.json`).
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

```bash
python main.py
```

This trains the lightweight classifier, evaluates it on the official FD001 test engines, runs ProbLog on representative windows, builds the HITL queue and writes:

- `results/final_run_results.json`
- `results/cmapss_results.json`
- `results/calibration_report.json`
- `results/decision_threshold.json`
- `results/engine_metrics_validation.csv`
- `results/engine_metrics_test.csv`
- `results/temporal_thresholds.json`
- `results/hitl_rule_validation.json`
- `results/hitl_queue.csv`
- `results/hitl_queue.json`
- `results/hitl_log.jsonl` (after feedback)
- `results/models/light_classifier.joblib`
- `results/scaler.pkl`
- `results/visualization/*.png`

Evaluate the standalone ProbLog examples:

```bash
python -m src.reasoning.prob_query
```

Run inference on a raw C-MAPSS file (no true RUL file required):

```bash
python -m src.inference.inference --mode cmapss \
  --data_file data/raw/CMAPSS/test_FD001.txt \
  --output results/inference_cmapss.json
```

Review the first five queued cases interactively:

```bash
python -m src.hitl.feedback --queue results/hitl_queue.json --max-cases 5
```

Run the test suite (with SWI-Prolog on `PATH`):

```bash
python -m unittest discover -s tests -q
```

## Current results

The reported run uses 20 engines for training, 20 for validation and the 100 official test engines. The selected threshold is `0.1296`, chosen on validation only.

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg. precision |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0.9354 | 0.7503 | 0.9306 | 0.8308 | 0.9843 | 0.9330 |
| Official test | 0.9798 | 0.6599 | 0.7831 | 0.7163 | 0.9903 | 0.8049 |

Test confusion matrix: `[[9730, 134], [72, 260]]`.

The official test windows are highly imbalanced (`3.26%` positive), so accuracy alone is misleading; F1, recall, average precision and ROC-AUC are the more useful indicators.

The threshold policy intentionally favours recall while keeping validation precision at or above `0.75`. Compared with the earlier fixed-`0.5` baseline (test precision `0.7248`, recall `0.7139`, F1 `0.7193`), the calibrated model raises test recall to `0.7831` and lowers precision to `0.6599`, with a similar F1 (`0.7163`).

### Calibration

| Split | Brier score | ECE |
|---|---:|---:|
| Validation | 0.0415 | 0.0201 |
| Official test | 0.0131 | 0.0023 |

### Engine-level metrics (official test)

| Metric | Value |
|---|---:|
| Engines with at least one positive window | 25 / 100 |
| Mean per-engine recall | 0.7075 |
| Mean per-engine precision | 0.7221 |
| Total false alarms | 134 |
| Mean false alarms per engine | 1.34 |
| Mean lead time before last observed cycle | 12.74 cycles |
| Mean anticipation at first correct alert | 28.53 cycles |
| Alerted before end-of-life among relevant engines | 0.76 |

Most FD001 test trajectories never enter the `RUL <= 30` failure zone, hence the separate denominator for "relevant" engines.

### ProbLog examples

The standalone examples in `src/reasoning/prob_query.py` produce:

| Example | `P(failure_risk)` |
|---|---:|
| Normal | 0.0000 |
| Degraded | 0.4318 |
| Critical | 0.6452 |

The representative test windows selected by the pipeline are:

| Window | Neural probability | ProbLog risk |
|---|---:|---:|
| Normal | 0.0049 | 0.0000 |
| Degraded / near threshold | 0.1295 | 0.0000 |
| Critical | 0.9698 | 0.4318 |

The degraded window is a useful HITL example: its calibrated probability is essentially at the decision threshold while the ProbLog risk remains zero because the temperature condition is still inactive.

## Human-in-the-loop

The queue is built from test windows with:

- calibrated neural probability inside the uncertainty band centered on the selected threshold, or
- high neural probability with no active HPC rule (rule/probability conflict), or
- low neural probability with an active HPC rule (rule/probability conflict).

The queue API also supports a ProbLog-risk conflict reason.

Current run: **200 queued cases** from 866 evaluated candidates (`377` eligible).

| Reason | Cases |
|---|---:|
| Uncertain probability | 89 |
| Rule/probability conflict (low neural, rule active) | 108 |
| Rule/probability conflict (high neural, no strong rule) | 3 |

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
  hitl/
    queue.py                      # uncertain/conflict queue, CSV/JSON export
    feedback.py                   # labels, learned rules, JSONL log, CLI
  inference/
    inference.py                  # legacy and raw C-MAPSS inference
  integration/
    adaptive_controller.py        # legacy synthetic-only controller (isolated)
  models/
    light_classifier.py           # RandomForest / MLP classifier + calibration
  pipeline/
    pipeline.py                   # FD001 training, evaluation and HITL queue
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
tests/                            # 63 unit and integration tests
results/                          # generated artifacts (ignored by git)
```

## Testing

The suite covers the C-MAPSS loader, the lightweight classifier, calibration, thresholding, engine metrics, temporal reasoning, learned-rule validation, ProbLog, the pipeline, inference, the HITL queue and HITL feedback:

```bash
$env:Path = "C:\Program Files\swipl\bin;$env:Path"
python -m unittest discover -s tests -q
```

Expected result: `63 tests ... OK`.

## Limitations

- FD001 only, with a binary `RUL <= 30` label.
- The official test set is highly imbalanced; accuracy is not a sufficient metric.
- Calibration is a post-hoc sigmoid wrapper around a RandomForest/MLP, not a sequence model trained with a probabilistic loss.
- The decision threshold implements a recall-oriented policy; changing the precision constraint changes the precision/recall trade-off.
- HITL feedback can add symbolic and ProbLog rules, but it does not retrain the neural classifier.
- Temporal thresholds and learned-rule validation use validation data; neither is tuned on the official test set.
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
