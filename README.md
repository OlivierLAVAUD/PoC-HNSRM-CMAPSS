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
        |                    |
        |                    +--> P(failure) ----------------------+
        |                                                          |
        +--> raw last cycle -> SWI-Prolog HPC rules -> insights ---+
                                                                   |
                                         ProbLog model -> P(failure_risk)
                                                                   |
                                                                   v
                                      HITL queue (uncertain / conflicting)
                                                                   |
                                             human label -> rules_learned.pl
                                                          -> hitl_log.jsonl
```

The neural model detects degradation from sliding windows. The symbolic layer explains which HPC conditions are active. ProbLog combines those conditions into a failure-risk estimate. Cases where the neural probability and symbolic evidence disagree, or where the probability is close to the decision boundary, are placed in a human review queue.

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
- `results/hitl_queue.csv`
- `results/hitl_queue.json`
- `results/hitl_log.jsonl` (after feedback)
- `results/models/light_classifier.joblib`
- `results/scaler.pkl`

Evaluate the standalone ProbLog examples:

```bash
python -m src.reasoning.prob_query
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

The reported run uses 20 engines for training, 20 for validation and the 100 official test engines.

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Validation | 0.9414 | 0.7997 | 0.8758 | 0.8360 | 0.9833 |
| Official test | 0.9819 | 0.7248 | 0.7139 | 0.7193 | 0.9898 |

Test confusion matrix: `[[9774, 90], [95, 237]]`.

The official test windows are highly imbalanced (`3.26%` positive), so accuracy alone is misleading; F1, recall and ROC-AUC are the more useful indicators.

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
| Normal | 0.0001 | 0.0000 |
| Borderline | 0.5000 | 0.0000 |
| Critical | 0.9945 | 0.6452 |

The borderline window is a useful HITL example: the neural model is uncertain while no strong symbolic rule fires.

## Human-in-the-loop

The queue is built from test windows with:

- neural probability in `[0.4, 0.6]` (uncertain), or
- high neural probability with no active HPC rule (rule/probability conflict), or
- low neural probability with an active HPC rule (rule/probability conflict).

The queue API also supports a ProbLog-risk conflict reason.

Current run: **192 queued cases** from 685 evaluated candidates.

| Reason | Cases |
|---|---:|
| Uncertain probability | 125 |
| Rule/probability conflict (low neural, rule active) | 64 |
| Rule/probability conflict (high neural, no strong rule) | 3 |

Workflow:

1. `python main.py` writes `results/hitl_queue.csv` and `results/hitl_queue.json`.
2. `python -m src.hitl.feedback` prompts for `failure` or `normal` labels.
3. Each feedback is appended to `results/hitl_log.jsonl`.
4. For confirmed failures, the active symbolic conditions are saved as a `hitl_rule_N` clause in `src/reasoning/rules_learned.pl` through the reasoner.
5. Learned rules are consulted by subsequent Prolog reasoning and appear as `Learned Rule Activated: hitl_rule_N` insights.

Learned HITL rules currently influence symbolic insights. They are not yet injected into the ProbLog probability model.

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
  hitl/
    queue.py                      # uncertain/conflict queue, CSV/JSON export
    feedback.py                   # labels, learned rules, JSONL log, CLI
  models/
    light_classifier.py           # RandomForest / MLP classifier
  pipeline/
    pipeline.py                   # FD001 training, evaluation and HITL queue
  reasoning/
    rules.pl                      # HPC symbolic rules and thresholds
    prob_rules.pl                 # ProbLog failure-risk model
    prob_query.py                 # ProbLog examples and evaluation
    reasoning.py                  # PySWIP bridge, rule learning, condition queries
    rules_learned.pl              # rules learned from HITL feedback
    rule_learning.py
    state_tracker.py
  utils/
    model_utils.py                # scaler and model persistence
tests/                            # 37 unit and integration tests
results/                          # generated artifacts (ignored by git)
```

## Testing

The suite covers the C-MAPSS loader, the lightweight classifier, symbolic reasoning, ProbLog, the pipeline, the HITL queue and HITL feedback:

```bash
$env:Path = "C:\Program Files\swipl\bin;$env:Path"
python -m unittest discover -s tests -q
```

Expected result: `37 tests ... OK`.

## Limitations

- FD001 only, with a binary `RUL <= 30` label.
- The official test set is highly imbalanced; accuracy is not a sufficient metric.
- HITL feedback currently updates Prolog rules, not the neural model or ProbLog probabilities.
- The neural model is a small RandomForest/MLP, not a calibrated sequence model.
- No RL, control actions, LLM reasoning, GraphRAG or Neo4j.
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
