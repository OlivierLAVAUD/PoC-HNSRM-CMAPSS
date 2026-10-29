# PoC-HNSRM: Proof of Concept - Hybrid Neuro-Symbolic Reasoning & Monitoring

## Project Overview

**PoC-HNSRM** (Proof of Concept - Hybrid Neuro-Symbolic Reasoning and Monitoring) is an advanced framework for aerospace system prognostics and health management (PHM). The project integrates machine learning models with symbolic reasoning to provide interpretable anomaly detection, fault diagnosis, and remaining useful life prediction for aerospace systems.

This repository implements a hybrid approach using:
- **Neural perception** (RandomForest, LightGBM, or CNN-LSTM) for feature extraction and anomaly detection from C-MAPSS FD001 and SKAB datasets
- **Symbolic reasoning** (Prolog-based) for rule extraction, validation, and explanation of model decisions
- **Human-in-the-Loop (HITL)** for rule acceptance/rejection and audit trail
- **ProbLog** for failure risk estimation with probabilistic certainty
- **Fusion logic** combining multiple signal sources into a final decision

The framework addresses the need for both high-performance prediction and interpretable decision-making in safety-critical aerospace applications.

---

## 🔧 Methodology

### Neural Perception

Three architectural approaches are implemented:

1. **RandomForest (Default)**
   - Tabular features with StandardScaler preprocessing
   - Explicit feature engineering with lag features and temporal statistics
   - Fast training and good performance with careful feature engineering
   - Provides feature importance for interpretability

2. **LightGBM (Complement)**
   - Gradient boosting on tabular features
   - Can use optional raw time windows + auxiliary RUL regressor
   - Out-of-fold RUL prediction as extra feature
   - Complements RandomForest with different bias-variance tradeoff

3. **CNN-LSTM (Sequential)**
   - Conv1D → LSTM → MultiHeadAttention architecture
   - Processes raw 3D sensor windows directly (no flattening, no StandardScaler)
   - Automatic feature learning via convolutions
   - Explicitly models temporal dependencies via LSTM gates
   - Captures spatial patterns through convolutional filters
   - Probabilities calibrated using validation-selected thresholds

### Symbolic Reasoning

The symbolic layer provides:
- **Rule extraction** from trained neural models
- **Temporal rules** (slopes, relative changes, moving averages, persistence)
- **Rule validation** on held-out data with precision/support/FPR thresholds
- **Human-in-the-Loop acceptance** mechanism for HITL queue
- **ProLog integration** for failure risk estimation
- **Feature aliasing** for human-readable rule names

### Fusion Logic

Final decision combines four signal sources:
- Neural probability (from the trained model)
- Symbolic score (active conditions from Prolog rules)
- ProbLog risk (probabilistic failure estimate)
- Predicted RUL (remaining useful life)

Fusion model: logistic regression fitted on validation data only (to avoid lookahead bias).

---

## 📊 Results and Evaluation

### FD001 Benchmark (C-MAPSS)

The FD001 ( Commercial Modular Aero-Propulsion System Simulation) benchmark uses NASA's engine dataset for fault detection and remaining useful life prediction.

**Key metrics reported:**
- Per-engine recall, precision, false alarms
- First alert cycle and lead time
- Anticipation of fault onset
- Fusion coefficient stability
- Threshold sensitivity analysis

**Output artifacts:**
- `results/engine_metrics_validation.csv` - Validation set per-engine metrics
- `results/engine_metrics_test.csv` - Official test set metrics
- `results/hitl_rule_validation.json` - Rule acceptance/rejection statistics
- `results/decision_audit.jsonl` - Complete decision audit trail with SHA-256 digests
- `results/multi_seed_report.json` (if --multi-seed used)
- `results/multi_seed_summary.csv` - Multi-seed validation summary

### SKAB Benchmark

The native SKAB path implements the same hybrid approach on the SKAB dataset, providing:
- Random Forest baseline with 8 features (vs FD001's 7)
- Symbolic rule extraction and validation
- Lazy TensorFlow import (no TF loaded for FD001-only runs)
- Config separation avoiding fragile 8→7 feature mapping

**Output artifacts:**
- `results/skab_separate/<run>/summary.csv` - Benchmark summary with RF accuracy and symbolic rule stats
- `results/skab_separate/<run>/rules_accepted_*.pl` - Accepted validated rules

---

## 📈 Demonstration Images

The following diagrams were generated during benchmark execution, providing visual insights into model performance and decision boundaries:

### Precision-Recall Curve
![](./results/visualization/precision_recall_curve.png)
*Figure 1: Precision-Recall curve across all models - Shows the trade-off between precision and recall across different thresholds. The curve demonstrates model performance across operating points.*

### Random Forest Precision-Recall Curve
![](./results/visualization/precision_recall_curve_random_forest.png)
*Figure 2: RF-specific Precision-Recall curve - RF model performance focused operating region.*

### Threshold Sensitivity
![](./results/visualization/threshold_sensitivity.png)
*Figure 3: Threshold sensitivity analysis - How the decision threshold affects model performance, selected on validation data only.*

### Reliability Diagrams
![](./results/visualization/reliability_validation.png)
*Figure 4: Validation reliability - Calibration assessment on validation set. Shows how well predicted probabilities match actual frequencies.*

![](./results/visualization/reliability_test.png)
*Figure 5: Test reliability - Calibration assessment on official test set. Generalization check.*

### Random Forest Reliability
![](./results/visualization/reliability_validation_random_forest.png)
*Figure 6: RF validation reliability - RF-specific calibration on validation data.*

![](./results/visualization/reliability_test_random_forest.png)
*Figure 7: RF test reliability - RF calibration on test set, demonstrating generalization.*

### Multi-Seed Distributions
![](./results/visualization/multi_seed_distributions.png)
*Figure 8: Multi-seed validation distributions - Results across 5 different data splits (seeds 42-46), showing result stability.*

---

## 🛡 Quality & Validation Framework

### HITL Rule Validation

Rules must meet strict thresholds before integration:
- **Precision**: ≥ 0.90 (minimum for rule acceptance)
- **Support**: ≥ 5 (minimum number of windows supporting the rule)
- **FPR**: ≤ 0.05 (maximum acceptable false positive rate)

Rejected rules are automatically removed with explanations written to `results/hitl_rule_validation.json`.

### Decision Audit

Every decided window receives:
- `decision_id` + artifact versions + SHA-256 digests
- Model, scaler, temporal metadata, `rules.pl`, `rules_learned.pl`, `prob_rules.pl`
- Four fusion signals + final decision (neural, symbolic, ProbLog, RUL)
- Data-quality context (NaN/Inf handling, drift status)

Journal: `results/decision_audit.jsonl`
Verification command: `python -m src.utils.audit --results results`

### Drift Detection

Incoming windows are compared with training data:
- Percentile bounds and distribution comparisons
- NaN/Inf share monitoring
- Reports `ok`, `warning`, or `critical` in `results/drift_report.json`
- Critical level warns but never blocks inference - recommends routing to HITL queue

---

## 📁 Key Project Files

### Source Code Structure

```
src/
├── main.py              CLI entry point with --benchmark flag
├── config/              Configuration management
├── reasoning/           Prolog rules, alias facts, validation
├── pipeline/            Main ANSR-DT pipeline
├── skab/                Native SKAB path
│   ├── config.py        8-feature configuration
│   ├── data.py          Data loading/preprocessing
│   ├── model.py         Models with lazy TF import
│   ├── pipeline.py      Benchmark orchestration
│   ├── reasoner.py      Rule validation/deduplication
│   ├── run.py           Main entry point
│   └── utils.py         Utility functions
└── reasoning/           Prolog rules and symbolic reasoning

### Configuration

- `configs/config.yaml` - Main FD001 configuration
- `configs/config_skab_separate.yaml` - SKAB-specific config (8 features)

### Results

- `results/` - Run-specific subdirectories
  - `results/skab_separate/` - SKAB benchmark outputs
  - `results/visualization/` - Generated diagrams (8 figures)
  - `results/multi_seed/` - Multi-seed validation results

### Data

- `data/raw/CMAPSS/` - NASA C-MAPSS FD001 dataset
- `data/SKAB/` - SKAB dataset with 8 features
- `data/SKAB/README.md` - Dataset documentation

---

## 🚀 Running the Framework

### FD001 Benchmark (Default)

```bash
# RandomForest baseline
python main.py

# Multi-seed validation (5 seeds)
python main.py --multi-seed --seeds 42,43,44,45,46

# Threshold sensitivity sweep
python main.py --threshold-sensitivity

# Operational alert policy
python main.py --operational-policy
```

### SKAB Benchmark (Native Path)

```bash
# Run both RF baseline and symbolic reasoning
python main.py --benchmark skab --modes random_forest symbolic

# Random Forest only on SKAB
python main.py --benchmark skab --modes random_forest

# Symbolic only on SKAB
python main.py --benchmark skab --modes symbolic
```

### Expected Output Quality

- **RF Accuracy on FD001**: Typically 0.85-0.95 range depending on window size and features
- **Symbolic Rule Acceptance**: Typically 60-80% of generated rules pass validation thresholds
- **Fusion Improvement**: Logistic fusion typically improves over individual model scores
- **Drift Monitoring**: Early warning capability before performance degradation

---

## 📦 Requirements

```bash
# From requirements.txt
numpy
pandas
scikit-learn
scipy
networkx
pyprolog
lightgbm
tensorflow  (optional, for CNN-LSTM architecture)
```

Install: `pip install -r requirements.txt`

---

## 🎯 Project Objectives Achieved

The PoC-HNSRM framework successfully demonstrates:

1. **Hybrid neuro-symbolic approach** combining ML performance with symbolic interpretability
2. **Rule extraction and validation** with HITL integration for rule acceptance
3. **Multiple neural architectures** (RF, LGBM, CNN-LSTM) for different data types
4. **Fusion logic** combining multiple signal sources optimally
5. **Quality assurance** through decision audits, drift monitoring, and threshold validation
6. **Native SKAB path** extending the framework to new datasets without TF dependency when not needed
7. **Comprehensive evaluation** with multi-seed validation and threshold sensitivity

The framework provides a foundation for production-grade PHM systems where both performance and interpretability are required.

---

*PoC-HNSRM-CMAPSS Framework - Proof of Concept for Hybrid Neuro-Symbolic Reasoning and Monitoring*

