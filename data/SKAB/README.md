# SKAB Dataset

The SKAB (System for Knowledge-based Analysis of Budgets) dataset is used for 
benchmarking the native SKAB path in the PoC-HNSRM-CMAPSS framework.

## Structure

```
data/SKAB/
├── README.md
└── *.csv              # Raw SKAB measurement data files
```

## Features

The SKAB dataset contains 8 features measured during industrial processes:

1. **air_temperature** - Ambient air temperature sensor reading
2. **process_temperature** - Process temperature sensor reading  
3. **rotational_speed** - Rotational speed of the machinery
4. **torque** - Torque measurement sensor
5. **tool_wear** - Tool wear indicator sensor
6. **vibration_x** - Vibration component along X axis
7. **vibration_y** - Vibration component along Y axis
8. **vibration_z** - Vibration component along Z axis

## Format

Each CSV file in the data/SKAB directory should contain:
- Rows: Individual time steps or measurements
- Columns: The 8 features listed above, plus a `target` column for
  classification/regression tasks
- No header required (will be inferred), but recommended

## Usage

The SKAB dataset is used with the native SKAB benchmark path:

```bash
python main.py --benchmark skab --modes random_forest symbolic
```

This will:
1. Load the SKAB dataset from `data/SKAB/`
2. Train a Random Forest baseline model
3. Extract symbolic rules using the symbolic reasoner
4. Write results to `results/skab_separate/<run>/summary.csv`

## Differences from FD001

- SKAB has 8 features (vs FD001's 7)
- Different sensor naming convention
- Config uses `configs/config_skab_separate.yaml` to avoid
  fragile 8→7 feature mapping
- TensorFlow dependencies are lazily imported to avoid
  loading when running FD001 benchmark
```

