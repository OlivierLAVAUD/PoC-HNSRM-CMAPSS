# ANSR-DT: HNSRM PoC on C-MAPSS FD001

This repository is a **hybrid neuro-symbolic reasoning and monitoring (HNSRM)** proof of concept built on NASA C-MAPSS **FD001**. It keeps the core idea of ANSR-DT, neural perception followed by explicit symbolic and probabilistic reasoning, while removing the heavy components from the original framework.

## Scope

This PoC focuses on **tabular anomaly detection + symbolic reasoning + HITL** on FD001. Compared to the original ANSR-DT framework:

- **Removed**: PPO/control (FD001 has no control loop), native SKAB path.
- **Added**: calibration, decision threshold validation-only, logistic fusion, symbolic provenance blocking, HITL rule validation, audit trail, drift, multi-seed, threshold sensitivity, operational policy, and **CNN-LSTM** neural architecture.

The default architecture is RandomForest. LightGBM is a complementary path that receives interpretable temporal statistics in addition to the optional flattened window, and can use an out-of-fold RUL prediction as an extra feature. **CNN-LSTM** is a sequential architecture with Conv1D → LSTM → MultiHeadAttention that processes raw 3D sensor windows directly.


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
        |
        v
validation-only decision threshold
        |
        v
validation-only logistic fusion
        (neural, symbolic, ProbLog, RUL)
        |
        v
full window -> SWI-Prolog HPC + temporal rules -> insights
        |
        v
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
        v
human label -> rules_learned.pl
        -> hitl_log.jsonl

**CNN-LSTM** processes raw 3D windows (no flattening, no StandardScaler). Calibration is skipped for sequential architectures; the validation-selected decision threshold is used directly. Probabilities flow through `positive_proba()` which handles 3D input natively.
```

The neural model detects degradation from sliding windows. **RandomForest** remains the default architecture. **LightGBM** is a complementary path that receives interpretable temporal statistics in addition to the optional flattened window, and can use an out-of-fold RUL prediction as an extra feature. **CNN-LSTM** is a sequential architecture with Conv1D → LSTM → MultiHeadAttention that processes raw 3D sensor windows directly; probabilities are calibrated using validation-selected thresholds.

Probabilities are calibrated on training data (sigmoid or isotonic), and the decision threshold is selected on validation data only. The symbolic layer explains which HPC conditions are active, including trend and persistence rules over the full window. ProbLog combines those conditions into a failure-risk estimate, optionally boosted by validated HITL rules. For LightGBM, a logistic model fitted on validation data only fuses the neural probability, symbolic score, ProbLog risk and predicted RUL. For CNN-LSTM, the validation-selected threshold is used directly.

**Knowledge Graph (Optional)**

An optional neuro-symbolic knowledge graph can be generated alongside the pipeline. When enabled via `knowledge_graph.enabled: true` in `configs/config.yaml`:

- Maps 14 FD001 sensors to feature groups (temperature, pressure, vibration, flow, misc) using `SENSOR_GROUPS` definitions
- Builds rule lists from `reasoner.learned_rules`
- Renders `results/knowledge_graph.png` using `KnowledgeGraphGenerator` with `plot_type='focused_ieee'`
- Uses `try/except ImportError` for graceful degradation if Graphviz is absent
- Disabled by default; enable in `configs/config.yaml` to test end-to-end generation of `results/knowledge_graph.png`

Alerts can require consecutive positive windows or a smoothed risk score. Cases where the neural probability and symbolic evidence disagree, or where the probability is close to the decision boundary, are placed in a human review queue.

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
- **Learned-rule validation:** existing HITL rules are checked on validation windows before fusion, explanations and HITL queue construction. A rule must reach precision `>= 0.90`, support `>= 5` and false-positive rate `<= 0.05`; rejected rules are removed automatically. The report is `results/hitl_rule_validation.json`.
- **Input quality:** NaN/Inf values in the train, validation and test splits are either replaced by zero (default) or rejected, controlled by `preprocessing.non_finite_policy`.
- **Symbolic switch:** `symbolic_reasoning.enabled: false` skips the Prolog layer (temporal rules, learned-rule validation, explanations and rule-conflict HITL reasons); fusion then receives zero symbolic and ProbLog signals.
- **Multi-seed validation:** `python main.py --multi-seed --seeds 42,43,44,45,46` re-splits the training engines per seed, re-derives the symbolic thresholds from that seed's own training engines, retrains, recalibrates, re-selects the neural and fusion thresholds on validation, refits the fusion and evaluates on the unchanged official test engines. It writes `results/multi_seed_report.json`, `results/multi_seed_summary.csv` and `results/visualization/multi_seed_distributions.png`; per-seed artifacts stay under `results/multi_seed/seed_<seed>/`.
- **Threshold sensitivity:** `python main.py --threshold-sensitivity` sweeps `recall_at_precision` precision constraints, F-beta weights and F1, still selecting every threshold on validation only. Outputs `results/threshold_sensitivity.json`, `.csv` and `results/visualization/threshold_sensitivity.png`.
- **Operational alert policy:** `python main.py --operational-policy` prices false negatives and false alarms (`operational_policy.false_negative_cost`, `false_alarm_cost`), compares immediate, consecutive and smoothed alerts, and selects a mode on validation. The decision and its rationale are written to `results/operational_policy.json`.
- **Fusion robustness:** `fusion.validation` runs an engine-grouped cross-validation over the validation engines and an ablation study, reporting coefficient and threshold stability, relative input importance and the validation delta of removing each input (`results/fusion_diagnostics.json`, `results/fusion_ablation.json`). This is diagnostics only: the production fusion remains a single validation-fitted logistic model.
- **Drift detection:** `monitoring.drift` compares incoming windows with training percentiles, bounds, NaN/Inf share and the predicted signal distributions, then reports `ok`, `warning` or `critical` in `results/drift_report.json`. A critical level never blocks inference: it warns and recommends routing the windows to the HITL queue.
- **Decision audit:** every decided window gets a `decision_id`, artifact versions and SHA-256 digests (model, scaler, temporal metadata, `rules.pl`, `rules_learned.pl`, `prob_rules.pl`), both thresholds, the four fusion signals, the final decision and the data-quality context. The journal is `results/decision_audit.jsonl`; `python -m src.utils.audit --results results` verifies it.

## Dataset and labels

- **Training data:** `data/raw/CMAPSS/FD001_processed.parquet` from the Hugging Face TwinnableAgent processed FD001 dataset (includes a `rul` column).
- **External test:** official NASA C-MAPSS `test_FD001.txt` and `RUL_FD001.txt`.
- **Fallback:** raw `train_FD001.txt` is also supported when the processed parquet is unavailable.
- **Selected sensors (14):** `s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21`.
- **Label:** a window is positive when its final-cycle RUL is `<= training.rul_threshold` (default `30`).
- **Low-data protocol:** 20% of engines for training, 20% for validation, and the 100 official test engines for evaluation. When `paths.test_file` is not configured, the remaining engines are used as the held-out test split and a warning is emitted.

Place the data files under `data/raw/CMAPSS/`. The directory is ignored by git because the files are downloaded separately.

### Symbolic thresholds

Thresholds are derived from the 20 training engines selected by the pipeline's 20/20/60 unit split (`random_state=42`), before any validation or held-out engine is seen:

| Condition | Threshold | Interpretation |
|---|---:|---|
| `s4` elevated | `> 1414.993` | HPC outlet temperature elevated (75th percentile) |
| `s4` critical | `> 1426.015` | HPC outlet temperature critical (95th percentile) |
| `s11` high | `> 47.710` | HPC static pressure high |
| `s7` low | `< 552.777` | HPC outlet pressure low |
| `s12` low | `< 520.930` | Fuel flow low |
| `s21` low | `< 23.218` | Coolant bleed low |

Recompute the facts from the same training split (the command prints a Prolog block; add `--write` to update `src/reasoning/rules.pl` in place):

```bash
python -m src.reasoning.thresholds \
  --data data/raw/CMAPSS/FD001_processed.parquet
# add --write to persist the values
```

The defaults match the FD001 protocol (`--validation-split 0.2`, `--test-split 0.6`, `--random-state 42`). Keep these options aligned with `training` and `paths` in `configs/config.yaml`; `tests/test_rule_thresholds.py` checks the committed values against this split when the parquet is available.

Every C-MAPSS run also re-derives these facts from the training engines and compares them with `rules.pl` before the symbolic layer is used. A mismatch aborts the run with the recompute command instead of silently feeding validation or test statistics into fusion; the check is recorded under `symbolic_rule_provenance` in `results/final_run_results.json` (`tests/test_symbolic_provenance.py`).

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

Robustness experiments and the operational decision (all selections stay on validation):

```bash
# Re-split the training engines for several seeds; official test untouched
python main.py --multi-seed --seeds 42,43,44,45,46

# Sweep validation-selected decision thresholds
python main.py --threshold-sensitivity

# Price false negatives against false alarms and pick an alert mode
python main.py --operational-policy

# Verify the audit trail of the last run
python -m src.utils.audit --results results
```

The new analyses are also configurable: `experiments.multi_seed`, `threshold_sensitivity` and `operational_policy` are disabled by default and enabled by their flag (or with `enabled: true`), while `monitoring.drift`, `audit` and `fusion.validation` are enabled in `configs/config.yaml` and run with every normal pipeline run.

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
- `results/operational_policy.json` (after `--operational-policy`)
- `results/threshold_sensitivity.json`, `.csv` and `results/visualization/threshold_sensitivity.png` (after `--threshold-sensitivity`)
- `results/multi_seed_report.json`, `results/multi_seed_summary.csv`, `results/visualization/multi_seed_distributions.png` and `results/multi_seed/seed_<seed>/` (after `--multi-seed`)
- `results/fusion_diagnostics.json` and `results/fusion_ablation.json` (LightGBM fusion path)
- `results/drift_reference.json` and `results/drift_report.json`
- `results/decision_audit.jsonl`

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

The reported run is `python main.py --architecture lightgbm`, using 20 engines for training, 20 for validation and the 100 official test engines. The neural threshold is `0.0733` and the validation-selected fusion threshold is `0.2196`.

### RandomForest vs LightGBM comparison

`python main.py --compare-models` selects LightGBM using validation `fbeta`; the test split is shown for reporting only and does not drive selection. The comparison scores each architecture's calibrated neural model; fusion is evaluated separately in the selected-model run below.

| Model | Validation F1 | Validation recall | Test F1 | Test recall | Test precision |
|---|---:|---:|---:|---:|---:|
| RandomForest | 0.8297 | 0.9274 | 0.7135 | 0.7801 | 0.6574 |
| LightGBM | 0.8540 | 0.9903 | 0.7616 | 0.9910 | 0.6184 |

### LightGBM with fusion

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg. precision |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 0.9412 | 0.7500 | 0.9823 | 0.8506 | 0.9920 | 0.9593 |
| Official test | 0.9813 | 0.6448 | 0.9458 | 0.7668 | 0.9950 | 0.8627 |

Test confusion matrix: `[[9691, 173], [18, 314]]`.

Fusion slightly lowers validation F1 relative to the raw LightGBM neural model (`0.8540` to `0.8506`) but improves official-test precision (`0.6184` to `0.6448`), F1 (`0.7616` to `0.7668`) and false alarms (`203` to `173`). Recall decreases from `0.9910` to `0.9458`, so the trade-off is explicit rather than uniformly better. The official test windows are highly imbalanced (`3.26%` positive), so accuracy alone is misleading; F1, recall, average precision and ROC-AUC are the more useful indicators.

### Calibration

`calibration.method: auto` selected sigmoid on validation for this run.

| Split | Brier score | ECE |
|---|---:|---:|
| Validation | 0.0275 | 0.0185 |
| Official test | 0.0100 | 0.0073 |

The fused probabilities have validation Brier `0.0316` and test Brier `0.0114`.

### Engine-level metrics (official test, fusion)

| Metric | Value |
|---|---:|
| Engines with at least one positive window | 25 / 100 |
| Mean per-engine recall | 0.8882 |
| Mean per-engine precision | 0.6041 |
| Total false alarms | 173 |
| Mean false alarms per engine | 1.73 |
| Mean lead time before last observed cycle | 12.29 cycles |
| Mean anticipation at first correct alert | 29.54 cycles |
| Alerted before end-of-life among relevant engines | 0.96 |

Most FD001 test trajectories never enter the `RUL <= 30` failure zone, hence the separate denominator for "relevant" engines.

### Alert confirmation

| Mode | Total false alarms | Mean recall | Mean precision |
|---|---:|---:|---:|
| Immediate | 173 | 0.8882 | 0.6041 |
| Consecutive 2 | 153 | 0.8637 | 0.6400 |
| Consecutive 3 | 141 | 0.8565 | 0.6730 |
| Smoothed 3 | 156 | 0.9216 | 0.6615 |

Consecutive confirmation removes isolated alerts at a small recall cost. Requiring three consecutive alerts gives the best precision in this run.

### Multi-seed validation

`python main.py --multi-seed --seeds 42,43,44,45,46` re-splits the 40 training engines five times (20 train / 20 validation per seed) while the official test split stays byte-identical across seeds (fingerprint `aac8a6c7…` in `results/multi_seed_report.json`). Aggregate over the five seeds:

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| Neural threshold (validation-selected) | 0.0432 | 0.0189 | 0.0241 | 0.0733 |
| Fusion threshold (validation-selected) | 0.1941 | 0.0815 | 0.0869 | 0.3084 |
| Test F1 | 0.7522 | 0.0173 | 0.7305 | 0.7668 |
| Test recall | 0.9386 | 0.0282 | 0.9187 | 0.9849 |
| Test precision | 0.6280 | 0.0204 | 0.6064 | 0.6531 |
| Test ROC-AUC | 0.9953 | 0.0013 | 0.9934 | 0.9970 |
| Test Brier | 0.0100 | 0.0012 | 0.0080 | 0.0114 |
| Engine mean recall | 0.9114 | 0.0404 | 0.8857 | 0.9822 |
| Total false alarms | 185 | 16.5 | 162 | 198 |
| Mean anticipation cycles | 29.40 | 0.30 | 29.08 | 29.84 |

The chosen alert mode was `consecutive_3` for every seed. The main split sensitivity is in the fusion threshold (0.087–0.308 across seeds), which is why the threshold is re-selected on validation for every run rather than hard-coded. The 95% confidence intervals for each metric are in `results/multi_seed_report.json` (`aggregate.*.ci95_low/ci95_high`).

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
| Degraded | 0.5218 |
| Critical | 0.6939 |

The representative test windows selected by the LightGBM pipeline are:

| Window | Neural probability | ProbLog risk |
|---|---:|---:|
| Normal | 0.0000 | 0.0000 |
| Degraded / near threshold | 0.2193 | 0.1963 |
| Critical | 0.9910 | 0.3120 |

The ProbLog model keeps a dedicated pressure/flow pathway (high static pressure, low outlet pressure, low fuel flow), so strong pressure and fuel evidence raises the failure risk even when the outlet temperature is not elevated.

The degraded window is a useful HITL example: its neural probability sits on the validation-selected fusion threshold (`0.2196`) while ProbLog also reports an active degradation pattern.

## Human-in-the-loop

The queue is built from test windows with:

- calibrated neural probability inside the uncertainty band centered on the selected threshold, or
- high neural probability with no active HPC rule (rule/probability conflict), or
- low neural probability with an active HPC rule (rule/probability conflict).

The queue API also supports a ProbLog-risk conflict reason.

Current LightGBM + fusion run: **200 queued cases** from 777 evaluated candidates (`273` eligible).

| Reason | Cases |
|---|---:|
| Uncertain probability | 104 |
| Rule/probability conflict (low neural, rule active) | 96 |
| Rule/probability conflict (high neural, no strong rule) | 0 |

Workflow:

1. `python main.py` writes `results/hitl_queue.csv` and `results/hitl_queue.json`.
2. `python -m src.hitl.feedback` prompts for `failure` or `normal` labels.
3. Each feedback is appended to `results/hitl_log.jsonl`.
4. For confirmed failures, the active symbolic conditions are saved as a `hitl_rule_N` clause in `src/reasoning/rules_learned.pl`.
5. On the next pipeline run, learned rules are validated on held-out validation windows; rules that are too broad or imprecise are rejected.
6. Validated rules are consulted by Prolog and injected into ProbLog with per-rule weight `0.15`, capped at `3` active learned rules.

## Operational recommendation

This section states the defaults supported by the current FD001 results, and what to review before changing them.

- **Architecture:** LightGBM with explicit temporal features plus the validation-fitted probability fusion. It was selected over RandomForest on validation (`fbeta`), and the fusion improves test precision and false alarms at a known recall cost.
- **Decision threshold:** do not hard-code a threshold across deployments. The official seed-42 split gives `0.2196` (`recall_at_precision` with `min_precision: 0.75`), but the multi-seed sweep shows the validation-selected fusion threshold varies between `0.0869` and `0.3084` depending on the training/validation split. Always re-run threshold selection on validation for a new split; the threshold sensitivity sweep (`results/threshold_sensitivity.csv`) shows the validation-selected thresholds remain within their precision constraints on test (e.g. `rec@0.75`: validation precision `0.7500` / test `0.6448`), so the lever works but precision itself degrades out of distribution.
- **Alert mode:** `consecutive_3` (three consecutive alert-positives). Under the current cost assumption (`false_negative_cost: 10`, `false_alarm_cost: 1`) it has the lowest validation expected cost (296 vs 313 for immediate alerts), costs about `-0.10` cycles of mean anticipation and recovers precision from `0.7500` to `0.7755`. Re-run `python main.py --operational-policy` whenever the cost weights change: with a false-negative cost far above 10 the trade-off moves toward `smoothed_3` or immediate alerts.
- **Human review triggers:** route a window to HITL when (a) its probability falls in the uncertainty band around the threshold, (b) the rule layer and the probability disagree (active rule with low probability, or high probability with no active rule), or (c) drift is at `critical`. At `warning` keep operating and watch the drift report over the next cycles; at `critical` stop reviewing only the affected windows on a best-effort basis — drift never blocks inference.
- **Drift interpretation:** `ok` means incoming windows sit inside training bounds; `warning` (`shares within 5%–15%`) means out-of-bound features are below 15% of windows, and `critical` means out-of-bound rates or NaN/Inf rates exceed 15% — check data plumbing first, then assume an operating-condition shift and plan a recalibration on new labelled data. On the official FD001 test engines a non-`ok` level is expected (test trajectories run to failure while training trajectories are truncated), so judge drift on deployment traffic, not on the offline test run.
- **Production metrics to watch:** recall and precision on labelled events, false alarms per engine and per cycle, calibration drift (Brier/ECE recomputed on any new labelled batch), mean anticipation, HITL queue rate per 1,000 windows and drift level. A rising HITL rate without a recall drop means the uncertainty band is drifting off-centre: re-select the threshold before retraining.

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
    loader_factory.py             # picks the C-MAPSS loader for parquet/txt input
  evaluation/
    calibration.py                # Brier score, ECE, reliability curves
    thresholding.py               # validation-only threshold selection
    engine_metrics.py             # per-engine recall, false alarms, lead time
    alerting.py                   # consecutive confirmation and smoothing
    operational_policy.py         # alert modes priced by maintenance cost
  experiments/
    multi_seed.py                 # per-seed protocol, aggregates, CSV, plot
    threshold_sensitivity.py      # threshold grid, CSV/JSON, plot
  features/
    temporal_features.py          # explicit temporal statistics for LightGBM
  fusion/
    probability_fusion.py         # validation-only logistic fusion
    diagnostics.py                # grouped CV, stability, ablation
  hitl/
    queue.py                      # uncertain/conflict queue, CSV/JSON export
    feedback.py                   # labels, learned rules, JSONL log, CLI
  inference/
    inference.py                  # legacy/raw C-MAPSS inference, optional fusion
  models/
    light_classifier.py           # RandomForest / MLP / LightGBM + calibration
    rul_regressor.py              # auxiliary LightGBM RUL and grouped OOF
  monitoring/
    drift.py                      # training reference vs incoming windows
  pipeline/
    pipeline.py                   # training, comparison, fusion, alerting, HITL
  reasoning/
    rules.pl                      # HPC symbolic rules, thresholds and temporal rules
    prob_rules.pl                 # ProbLog failure-risk model
    prob_query.py                 # ProbLog examples and evaluation
    thresholds.py                 # recompute rules.pl facts from the training split
    reasoning.py                  # PySWIP bridge, rule learning, condition queries
    temporal.py                   # train-only temporal features and thresholds
    rules_learned.pl              # rules learned from HITL feedback
  utils/
    audit.py                      # decision ids, artifact hashes, JSONL, CLI
    model_utils.py                # scaler and model persistence
legacy/
  synthetic_data_loader.py        # .npz synthetic loader (not FD001)
  adaptive_controller.py          # synthetic-only control loop (not FD001)
  data/                           # synthetic .npz sample used by the legacy path
tests/                            # 177 unit and integration tests
results/                          # generated artifacts (ignored by git)
```

### Legacy modules

The `legacy/` package is not part of the C-MAPSS FD001 pipeline. These modules
are kept only so the synthetic demonstration path and its tests keep working:

- `legacy/synthetic_data_loader.py`: loader for the synthetic `.npz` sample
  (inputs `temperature`, `vibration`, `pressure`, `efficiency_index`, ...);
- `legacy/adaptive_controller.py`: the synthetic control loop;
- `legacy/data/synthetic_sensor_data_with_anomalies.npz`: the sample file.

These modules are not imported by `main.py`, the FD001 pipeline, the fusion, the
monitoring layer or the inference path used on C-MAPSS files. `src/data/loader_factory.py`
still falls back to `legacy.synthetic_data_loader.DataLoader` when
`paths.data_file` does not point at a parquet or text file, which is the only
remaining link and is exercised by `tests/test_data_loader.py`.

## Testing

The suite covers the C-MAPSS loader, RandomForest/LightGBM classifiers, calibration, thresholding, engine metrics, temporal features, RUL regression, alert confirmation, probability fusion, config validation, learned-rule validation, symbolic-threshold provenance, the published README metrics, ProbLog, the pipeline, inference, the HITL queue and HITL feedback. The new analyses have their own tests: `test_multi_seed.py`, `test_threshold_sensitivity.py`, `test_operational_policy.py`, `test_fusion_diagnostics.py`, `test_drift.py`, `test_audit.py` and `test_pipeline_analyses.py`.

```bash
$env:Path = "C:\Program Files\swipl\bin;$env:Path"
python -m unittest discover -s tests -q
```

Expected result: `177 tests ... OK`. The FD001 threshold, provenance and classifier tests, plus the README metric check, are skipped when the parquet or run artifacts are missing; the symbolic tests are skipped when SWI-Prolog is not available.

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
- Multi-seed validation only re-splits the 40 training engines: five seeds give 20 training and 20 validation engines each. It measures split sensitivity, not a different dataset.
- The fusion diagnostics use engine-grouped cross-validation on the 20 validation engines, so fold estimates are coarse (four engines per held-out fold) and only detect large effects.
- Drift on the official test engines is legitimately different from training: FD001 test trajectories run to failure while the training trajectories are truncated, so a non-`ok` drift level there is expected and is not a defect.
- The decision audit journal stores one compact record per decided window, with symbolic insights only for the riskiest windows (`audit.include_insights`). It is an audit trail, not a full provenance database.
- The operational cost weights (`false_negative_cost`, `false_alarm_cost`) are configurable assumptions, not measured maintenance costs.
- `AdaptiveController` is legacy synthetic-only code, now under `legacy/`, isolated from the FD001 pipeline.
- Data files are not distributed with the repository.
- This is a research proof of concept, not a production digital twin.

## Original ANSR-DT publications

This repository is a derivative for FD001 experimentation. The original framework is described in:

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
