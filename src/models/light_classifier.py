# src/models/light_classifier.py

import logging
import os
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from src.models.keras_adapter import KerasSklearnAdapter


class LightClassifier:
    """Lightweight sklearn classifier for windowed sensor data.

    Flattens ``(samples, window_size, features)`` inputs internally so the
    pipeline can keep the same 3D sequence contract as the original CNN-LSTM.
    """

    def __init__(self, architecture: str = "random_forest", random_state: int = 42,
                 logger: Optional[logging.Logger] = None, **kwargs):
        """Build the requested sklearn estimator."""
        self.logger = logger or logging.getLogger(__name__)
        self.architecture = architecture
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = self._build_estimator(architecture, random_state, kwargs)
        self.window_size: Optional[int] = None
        self.n_features: Optional[int] = None
        self.decision_threshold: float = 0.5
        self.is_calibrated: bool = False
        self.calibration_method: Optional[str] = None
        self.calibration_cv_folds: Optional[int] = None
        self.temporal_feature_extractor: Any = None
        self.rul_regressor: Any = None
        self.feature_names_out: Optional[list] = None
        self.uses_temporal_features: bool = False
        self.uses_rul_feature: bool = False

    @staticmethod
    def _build_estimator(architecture: str, random_state: int, kwargs: dict) -> Any:
        """Create the sklearn estimator matching the architecture name."""
        name = (architecture or "random_forest").lower()
        if name in ("random_forest", "rf"):
            return RandomForestClassifier(
                n_estimators=int(kwargs.get("n_estimators", 100)),
                max_depth=kwargs.get("max_depth", 10),
                min_samples_leaf=int(kwargs.get("min_samples_leaf", 1)),
                class_weight=kwargs.get("class_weight", "balanced"),
                random_state=random_state,
                n_jobs=-1,
            )
        if name in ("mlp", "mlp_classifier"):
            return MLPClassifier(
                hidden_layer_sizes=tuple(kwargs.get("hidden_layer_sizes", (64, 32))),
                max_iter=int(kwargs.get("max_iter", 300)),
                random_state=random_state,
            )
        if name in ("lightgbm", "lgbm", "lightgbm_classifier"):
            try:
                from lightgbm import LGBMClassifier
            except ImportError as error:  # pragma: no cover - install guidance
                raise ImportError(
                    "LightGBM is required for architecture='lightgbm'. "
                    "Install it with `pip install 'lightgbm>=4.5,<5'`."
                ) from error

            params = {
                "n_estimators": int(kwargs.get("n_estimators", 300)),
                "learning_rate": float(kwargs.get("learning_rate", 0.05)),
                "num_leaves": int(kwargs.get("num_leaves", 31)),
                "max_depth": int(kwargs.get("max_depth", -1)),
                "min_child_samples": int(kwargs.get("min_child_samples", 20)),
                "subsample": float(kwargs.get("subsample", 0.9)),
                "colsample_bytree": float(kwargs.get("colsample_bytree", 0.9)),
                "reg_lambda": float(kwargs.get("reg_lambda", 0.0)),
                "class_weight": kwargs.get("class_weight", "balanced"),
                "random_state": random_state,
                "n_jobs": -1,
                "verbosity": -1,
            }
            if params["subsample"] < 1.0:
                params["subsample_freq"] = 1
            return LGBMClassifier(**params)
        if name in ("cnn_lstm", "cnn-lstm"):
            from src.models.cnn_lstm_model import create_cnn_lstm_model

            input_shape = kwargs.get("input_shape")
            if input_shape is None:
                raise ValueError(
                    "cnn_lstm architecture requires input_shape=(window_size, n_features) "
                    "in model configuration."
                )
            keras_model = create_cnn_lstm_model(
                input_shape=tuple(input_shape),
                learning_rate=float(kwargs.get("learning_rate", 0.001)),
                use_attention=bool(kwargs.get("use_attention", True)),
                attention_key_dim=int(kwargs.get("attention_key_dim", 50)),
                attention_heads=int(kwargs.get("attention_heads", 4)),
                cnn_filters=int(kwargs.get("cnn_filters", 64)),
                cnn_kernel_size=int(kwargs.get("cnn_kernel_size", 3)),
                lstm_units=int(kwargs.get("lstm_units", 64)),
                dropout=float(kwargs.get("dropout", 0.2)),
            )
            return KerasSklearnAdapter(
                model=keras_model,
                epochs=int(kwargs.get("epochs", 30)),
                batch_size=int(kwargs.get("batch_size", 32)),
                patience=int(kwargs.get("patience", 5)),
            )
        raise ValueError(f"Unsupported light classifier architecture: {architecture}")

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        """Flatten 3D windows to a 2D matrix, leaving 2D input untouched."""
        X = np.asarray(X, dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError(
                "LightClassifier received NaN or infinite values. "
                "Apply the configured data_quality policy before prediction."
            )
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        if X.ndim == 2:
            return X
        raise ValueError(f"Expected 2D or 3D input, got shape {X.shape}")

    def _is_sequential(self) -> bool:
        """Return True when the architecture consumes raw 3D windows."""
        return str(self.architecture).lower() in ("cnn_lstm", "cnn-lstm")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightClassifier":
        """Fit the classifier on windowed sensor data.

        Sequential architectures (CNN-LSTM) consume raw 3D windows
        ``(batch, window, features)`` directly; tabular architectures flatten
        them to ``(batch, window * features)``.
        """
        array = np.asarray(X, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError(
                "LightClassifier received NaN or infinite values. "
                "Apply the configured data_quality policy before training."
            )
        if self._is_sequential():
            if array.ndim != 3:
                raise ValueError(
                    f"Sequential architecture '{self.architecture}' expects 3D "
                    f"windows (batch, window, features), got shape {array.shape}."
                )
            self.window_size = int(array.shape[1])
            self.n_features = int(array.shape[2])
            self.model.fit(array, np.asarray(y).ravel())
            n_samples, n_feat = array.shape[0], array.shape[1] * array.shape[2]
        else:
            X_flat = self._flatten(array)
            self.n_features = X_flat.shape[1]
            if array.ndim == 3:
                self.window_size = int(array.shape[1])
            self.model.fit(X_flat, np.asarray(y).ravel())
            n_samples, n_feat = X_flat.shape
        self.logger.info(
            "LightClassifier fitted on %d samples with %d features (%s).",
            n_samples,
            n_feat,
            self.architecture,
        )
        return self


    def calibrate(self, X: np.ndarray, y: np.ndarray, method: str = "sigmoid",
                  cv_folds: int = 5) -> "LightClassifier":
        """Calibrate probabilities with stratified cross-validation on training data.

        The calibration is performed with ``CalibratedClassifierCV`` and never
        sees the official test set. ``sigmoid`` is the safer default for the
        small, imbalanced FD001 training split; ``isotonic`` is available when
        enough samples are present.

        Sequential architectures (CNN-LSTM) are not wrapped in
        ``CalibratedClassifierCV`` (Keras models are not scikit-learn clonable);
        rely on the validation-selected decision threshold instead.
        """
        if self._is_sequential():
            self.logger.warning(
                "Calibration skipped for sequential architecture '%s'; using "
                "the validation-selected decision threshold instead.",
                self.architecture,
            )
            return self
        method = str(method).lower()
        if method not in {"sigmoid", "isotonic"}:
            raise ValueError(f"Unsupported calibration method: {method!r}")

        X_flat = self._flatten(X)
        y_flat = np.asarray(y).ravel().astype(int)
        if len(np.unique(y_flat)) < 2:
            self.logger.warning(
                "Calibration skipped: training labels contain a single class."
            )
            return self

        class_counts = np.bincount(y_flat, minlength=2)
        n_splits = int(min(max(2, cv_folds), class_counts.min()))
        if n_splits < 2:
            self.logger.warning(
                "Calibration skipped: not enough samples per class (%s).", class_counts.tolist()
            )
            return self

        base_estimator = self._build_estimator(
            self.architecture, self.random_state, self.kwargs
        )
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        calibrated = CalibratedClassifierCV(
            estimator=base_estimator, method=method, cv=cv, n_jobs=-1
        )
        calibrated.fit(X_flat, y_flat)
        self.model = calibrated
        self.is_calibrated = True
        self.calibration_method = method
        self.calibration_cv_folds = n_splits
        self.logger.info(
            "LightClassifier calibrated with method=%s and %d stratified folds.",
            method, n_splits,
        )
        return self

    def set_threshold(self, threshold: float) -> "LightClassifier":
        """Set the positive-class decision threshold used by :meth:`predict`."""
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Decision threshold must be in [0, 1], got {threshold}")
        self.decision_threshold = threshold
        self.logger.info("LightClassifier decision threshold set to %.4f.", threshold)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels using the configured threshold."""
        threshold = float(getattr(self, "decision_threshold", 0.5))
        return (self.positive_proba(X) >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities with columns aligned to ``self.classes_``.

        Sequential architectures receive raw 3D windows; tabular architectures
        receive flattened windows.
        """
        array = np.asarray(X, dtype=np.float32)
        if self._is_sequential():
            return self.model.predict_proba(array)
        return self.model.predict_proba(self._flatten(array))

    def positive_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the probability of the positive (failure) class."""
        proba = self.predict_proba(X)
        classes = list(getattr(self.model, "classes_", []))
        if 1 in classes:
            return proba[:, classes.index(1)]
        return np.zeros(len(proba), dtype=np.float32)

    def prepare_base_features(self, X: np.ndarray) -> np.ndarray:
        """Apply temporal extraction/flattening without the RUL feature."""
        array = np.asarray(X, dtype=np.float32)
        if self.temporal_feature_extractor is not None:
            if array.ndim != 3:
                raise ValueError(
                    "This model requires raw 3D windows for temporal feature "
                    f"extraction, got shape {array.shape}."
                )
            return self.temporal_feature_extractor.transform(array)
        if array.ndim == 3:
            return array.reshape(array.shape[0], -1)
        return array

    def prepare_features(self, X: np.ndarray) -> np.ndarray:
        """Apply the persisted temporal/RUL feature bundle to raw windows.

        Training code passes already engineered matrices and therefore calls
        the estimator directly. Inference calls this helper with raw 3D windows
        so saved LightGBM artifacts remain self-contained.
        """
        array = self.prepare_base_features(X)
        if self.rul_regressor is not None and self.uses_rul_feature:
            predicted_rul = self.rul_regressor.predict(array).reshape(-1, 1)
            array = np.hstack([array, predicted_rul])
        return array

    @property
    def classes_(self) -> np.ndarray:
        """Expose the fitted estimator's classes."""
        return getattr(self.model, "classes_", np.array([]))

    def save(self, path: str) -> None:
        """Persist the wrapper (estimator + metadata) with joblib."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        self.logger.info("LightClassifier saved to %s", path)

    @classmethod
    def load(cls, path: str, logger: Optional[logging.Logger] = None) -> "LightClassifier":
        """Load a persisted LightClassifier from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"LightClassifier not found at {path}")
        instance = joblib.load(path)
        if not isinstance(instance, cls):
            raise TypeError(f"Object at {path} is not a LightClassifier")
        if logger is not None:
            instance.logger = logger
        return instance
