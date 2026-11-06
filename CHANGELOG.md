# Changelog

All notable changes to `poc-hnsrm-cmapss` are documented here. The project is a
research proof of concept; entries describe evaluation-protocol changes rather
than production releases.

## [1.8.0] - 2026-11-06

### Added

- **Label-free cross-condition adaptation ablation.**
  `experiments.cross_condition.adaptation` and the CLI flags
  `--cross-normalization` / `--cross-threshold` add a 2x2 ablation:
  source vs per-condition standardisation, crossed with the source threshold
  vs a threshold matched to the source validation alert rate. Per-condition
  statistics are fitted on unlabelled target operating settings and windows;
  target labels remain reporting-only. `ConditionNormalizer` clusters the
  three operating settings and standardises each condition separately;
  `select_alert_rate_threshold` chooses the achievable threshold closest to
  the requested rate, accounting for tied scores.
- `CMAPSSLoader.window_op_settings` exposes the median operating settings of
  each window. Cross-condition reports now include `variants`, `baseline_*`
  columns, achieved target alert rate and the adaptation metadata.

### Finding

- On the real FD001 -> FD002/FD003/FD004 run, per-condition scaling alone
  produced zero alerts on FD002/FD004 and slightly degraded ROC-AUC. Matching
  the source alert rate improved F1 on FD002/FD004 only by cutting recall to
  `0.28-0.32` (thousands of false-alarm episodes remained), and degraded FD003
  from F1 `0.493` to `0.175`. Simple label-free adaptation is not a solution;
  the ranking itself is near chance on the multi-condition targets.

### Tests

- Condition-normalizer fitting/transform, single-condition fallback, input
  validation, alert-rate threshold behaviour and the full cross-condition
  ablation metadata.

## [1.7.0] - 2026-11-06

### Added

- **Episode-level alert policy comparison.** `src/evaluation/alert_policy_comparison.py`
  compares `immediate`, `consecutive_2`, `consecutive_3`, `smoothed_3` and
  `hysteresis` on validation and test. Every mode reports real/detected
  episodes, episode recall/precision, false-alarm episodes, window false
  positives, first correct alert, mean anticipation, mean alert-to-episode-end
  delay and episode/window/engine cost. Selection is validation-only; test rows
  are explicitly `reporting_only`. Artifacts:
  `results/alert_policy_comparison.json` and `.csv`.
- **Episode selection unit for the operational policy.**
  `operational_policy.selection_unit` accepts `window`, `engine` or `episode`
  (default `window`, unchanged). An episode false negative is an undetected
  degradation episode; an episode false alarm is an alert episode without a
  matching real episode. `results/operational_policy.json` now includes the
  episode counts, per-unit costs and a readable rationale.
- **Cost-aware HITL prioritization.** `hitl.prioritization` gains
  `strategy` (`uncertainty`, `expected_cost`, `hybrid`), `false_negative_cost`,
  `false_alarm_cost` and `rul_threshold`. Expected-cost ordering combines the
  failure probability, ProbLog risk, predicted RUL, optional RUL uncertainty
  and drift level. Priority fields (`hitl_priority`, `priority_score`,
  `priority_reason`, `expected_cost`, `inspection_cost`) are exported in
  `results/hitl_queue.csv`/`.json` and in the decision audit. Prioritization
  never changes the decision or bypasses learned-rule validation.
- **FD002-FD004 benchmark support.** `python main.py --cmapss-dataset fd00X`
  selects an overlay configuration that extends `configs/config.yaml`.
  `src/data/cmapss_datasets.py` validates the dataset id, sensor names/count,
  input shape and path conventions, and reports missing data with the expected
  `FD00X_processed.parquet`, `test_FD00X.txt` and `RUL_FD00X.txt` layout. FD001
  remains the default with unchanged results.
- **Cross-condition generalisation.** `experiments.cross_condition` and
  `--cross-condition` train/calibrate/threshold on one source dataset and apply
  the frozen artifacts to target datasets. Reports include F1, precision,
  recall, ROC-AUC, average precision, Brier score, ECE, engine and episode
  metrics, false-alarm episodes and relative degradation against source
  validation. Artifacts: `results/cross_condition_report.json` and
  `results/cross_condition_summary.csv`.
- **Symbolic contribution diagnostics.** `results/symbolic_contribution.json`
  reports signal-level ablation and, when rule activations are available,
  per-rule activation frequency, validation precision/recall, marginal F1 and
  episode-recall contribution, neural conflict rate and rules to remove or
  examine. The report is diagnostic and keeps the production fusion intact.
- **Robustness dossier.** `python main.py --robustness-report` aggregates
  multi-seed, threshold sensitivity, fusion/nested validation, fusion and
  symbolic ablation, alert policies, drift and HITL ordering into
  `results/robustness_report.json`, `.csv`, `.md` and
  `results/visualization/robustness_report.png`.
- **Configuration layering.** `load_config` supports `extends: config.yaml`
  with recursive deep merge, circular-chain detection and clear errors.

### Documentation

- README adds `### Alert policy comparison` (regenerated FD001 RandomForest
  artifacts), `### Cross-condition generalisation`, `### Symbolic contribution`
  and `## Robustness report`, plus the new CLI commands, artifacts, HITL
  strategies and updated project layout.
- `poc.md` documents the new commands, overlay configurations and artifacts.
- This changelog.

### Tests

- New coverage for alert-policy mode completeness, JSON/CSV coherence,
  hysteresis threshold validity and validation-only selection.
- Operational-policy selection units, missed episodes, false-alarm episodes,
  cost ties and invalid configuration values.
- HITL cost/hybrid ordering, high-cost and conflicting cases, disabled
  prioritization, export/reload compatibility and strategy validation.
- Dataset overlays, dataset selection without real data, sensor/path validation
  and missing-data messages.
- Cross-condition Brier/ECE, relative degradation and protocol metadata.
- Per-rule symbolic ablation on a controlled synthetic dataset.
- Robustness JSON/CSV/PNG artifact generation.

### Notes

- No published LightGBM + fusion metrics were regenerated. The README alert
  policy table was produced by an isolated default FD001 RandomForest run and
  copied into `results/alert_policy_comparison.{json,csv}`; existing
  `results/final_run_results.json` and the LightGBM fusion tables are unchanged.
- A real FD001 → FD002/FD003/FD004 cross-condition run was added after the NASA
  data download. It is documented in the README as a negative transfer result:
  naive single-condition scaling/thresholding collapses precision on FD002 and
  FD004. Artifacts: `results/cross_condition_report.json` and
  `results/cross_condition_summary.csv`.
- SWI-Prolog is optional. Without it, the symbolic, ProbLog, HITL rule and
  symbolic-contribution tests are skipped and the pipeline must run with
  `symbolic_reasoning.enabled: false`.
