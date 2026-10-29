"""scikit-learn adapter around a Keras model for the CNN-LSTM path.

Exposes the ``fit`` / ``predict_proba`` / ``predict`` / ``classes_`` contract
expected by :class:`src.models.light_classifier.LightClassifier` and by the
rest of the FD001 pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class KerasSklearnAdapter(BaseEstimator, ClassifierMixin):
    """Wrap a compiled Keras model as a scikit-learn classifier.

    The Keras model is built by
    :func:`src.models.cnn_lstm_model.create_cnn_lstm_model` and passed in at
    construction time. The adapter owns the training loop (epochs, batch size,
    early stopping) and exposes class probabilities.
    """

    def __init__(
        self,
        model: Any,
        epochs: int = 30,
        batch_size: int = 32,
        patience: int = 5,
        verbose: int = 0,
    ):
        self.model = model
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.verbose = int(verbose)
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KerasSklearnAdapter":
        """Fit the Keras model on 3D windows ``(batch, window, features)``."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        if X.ndim != 3:
            raise ValueError(
                "KerasSklearnAdapter expects 3D input (batch, window, "
                f"features), got shape {X.shape}."
            )
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")

        callbacks = []
        if self.patience and self.patience > 0:
            from tensorflow.keras.callbacks import EarlyStopping

            callbacks.append(
                EarlyStopping(
                    monitor="loss",
                    patience=self.patience,
                    restore_best_weights=True,
                    verbose=0,
                )
            )

        self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=self.verbose,
        )
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities with columns aligned to ``[0, 1]``."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(
                f"KerasSklearnAdapter expects 3D input, got shape {X.shape}."
            )
        p = np.asarray(
            self.model.predict(X, verbose=0), dtype=np.float64
        ).ravel()
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels using the 0.5 decision boundary."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_params(self, deep: bool = True) -> dict:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "verbose": self.verbose,
        }

    def set_params(self, **params: Any) -> "KerasSklearnAdapter":
        for key, value in params.items():
            setattr(self, key, value)
        return self




def calibrate_keras_model(
        build_fn,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        method: str = "sigmoid",
        n_splits: int = 5,
        epochs: int = 20,
        batch_size: int = 32,
        verbose: int = 0):
    """Out-of-fold probability calibration for a Keras model builder.

    ``build_fn`` is a callable returning a fresh compiled Keras model. A
    GroupKFold split (grouped by engine) produces out-of-fold probabilities;
    an ``IsotonicRegression`` (or ``LogisticRegression``) calibrator is then
    fitted on those out-of-fold scores. This mirrors
    :func:`src.models.rul_regressor.out_of_fold_predictions`.

    Returns the fitted calibrator (``IsotonicRegression`` or
    ``LogisticRegression``) mapping raw scores to calibrated probabilities.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=float).ravel()
    groups = np.asarray(groups).ravel()
    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError("X, y and groups must have the same length.")
    n_groups = int(len(np.unique(groups)))
    n_splits = int(min(max(2, n_splits), n_groups))

    oof = np.zeros(len(y), dtype=float)
    if n_splits < 2:
        model = build_fn()
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        oof = np.asarray(model.predict(X, verbose=0), dtype=float).ravel()
    else:
        from sklearn.model_selection import GroupKFold
        for train_index, valid_index in GroupKFold(
                n_splits=n_splits).split(X, y, groups):
            fold_model = build_fn()
            fold_model.fit(
                X[train_index], y[train_index],
                epochs=epochs, batch_size=batch_size, verbose=verbose,
            )
            oof[valid_index] = np.asarray(
                fold_model.predict(X[valid_index], verbose=0), dtype=float
            ).ravel()

    oof = np.clip(oof, 0.0, 1.0)
    if str(method).lower() == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        calibrator = IsotonicRegression(out_of_bounds="clip")
    else:
        from sklearn.linear_model import LogisticRegression
        calibrator = LogisticRegression()
    calibrator.fit(oof.reshape(-1, 1), y)
    return calibrator

