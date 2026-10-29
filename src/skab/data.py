"""SKAB data module.

Handles loading and preprocessing of the SKAB dataset.
"""

import os
import numpy as np
import pandas as pd
from src.skab.config import SKAB_DATA_PATH, SKAB_FEATURES


def load_skab_data(path: str = None) -> tuple:
    """Load SKAB dataset.
    
    Args:
        path: Path to SKAB data directory. Uses default from config if None.
    
    Returns:
        Tuple of (X, y, feature_names)
    """
    if path is None:
        path = SKAB_DATA_PATH
    
    # Look for CSV files in the data directory
    csv_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.csv'):
                csv_files.append(os.path.join(root, f))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in SKAB path: {path}")
    
    # Load all CSV files
    dfs = []
    for csv_file in csv_files:
        dfs.append(pd.read_csv(csv_file))
    
    if not dfs:
        raise ValueError("No dataframes loaded")
    
    # Concatenate all data
    data = pd.concat(dfs, ignore_index=True)
    
    # Separate features and target
    if 'target' in data.columns:
        y = data['target'].values
        X = data.drop(columns=['target']).values
    else:
        # Assume last column is the target
        y = data.iloc[:, -1].values
        X = data.iloc[:, :-1].values
    
    # Ensure feature names
    feature_names = SKAB_FEATURES[:X.shape[1]]
    
    # Basic preprocessing: handle NaN values
    X = np.nan_to_num(X)
    
    return X, y, feature_names


def get_feature_names() -> list:
    """Return the list of SKAB feature names."""
    return SKAB_FEATURES.copy()


def describe_data(X: np.ndarray, y: np.ndarray, feature_names: list = None) -> dict:
    """Generate a description of the SKAB dataset."""
    if feature_names is None:
        from src.skab.config import SKAB_FEATURES
        feature_names = SKAB_FEATURES
    
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    description = {
        'n_samples': n_samples,
        'n_features': n_features,
        'feature_names': feature_names,
        'y_distribution': {
            'class_0': int(np.sum(y == 0)),
            'class_1': int(np.sum(y == 1)),
        },
        'X_stats': {
            'mean': X.mean(axis=0).tolist(),
            'std': X.std(axis=0).tolist(),
        }
    }
    
    return description
