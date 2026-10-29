"""SKAB model module.

Contains TensorFlow/Keras model definitions with lazy imports
to avoid tensorflow dependency when not needed (Task 8.3).
"""

import logging

logger = logging.getLogger(__name__)

# Lazy import flag - tensorflow will be imported on demand
_tf = None
_keras = None
_nn = None


def _ensure_tf():
    """Lazy import of tensorflow."""
    global _tf, _keras, _nn
    if _tf is None:
        try:
            import tensorflow as tf
            from tensorflow import keras
            _tf = tf
            _keras = keras
            _nn = tf.keras.layers.Layer
            logger.info("TensorFlow imported lazily")
        except ImportError:
            logger.warning("TensorFlow not available")
            _tf = None
            _keras = None
            _nn = None
    return _tf, _keras, _nn


class SKABClassifier:
    """SKAB classification model using TensorFlow/Keras."""
    
    def __init__(self, input_dim: int, hidden_units: list = None, 
                 num_classes: int = 2, learning_rate: float = 0.001):
        self.input_dim = input_dim
        self.hidden_units = hidden_units or [64, 32]
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = None
        self._history = None
    
    def build(self) -> None:
        """Build the Keras model. Must be called before fit."""
        tf, keras, _ = _ensure_tf()
        if tf is None:
            raise ImportError("TensorFlow is required to build the SKAB model")
        
        model = keras.Sequential()
        model.add(keras.layers.Dense(
            self.hidden_units[0], 
            activation='relu', 
            input_dim=self.input_dim
        ))
        
        for units in self.hidden_units[1:]:
            model.add(keras.layers.Dense(units, activation='relu'))
        
        model.add(keras.layers.Dense(self.num_classes, activation='softmax'))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info(f"SKAB model built: input_dim={self.input_dim}, "
                    f"hidden_units={self.hidden_units}, classes={self.num_classes}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            validation_split: float = 0.2, epochs: int = 50,
            batch_size: int = 32, verbose: int = 0):
        """Fit the model to training data."""
        tf, keras, _ = _ensure_tf()
        if tf is None:
            raise ImportError("TensorFlow is required to fit the SKAB model")
        if self.model is None:
            self.build()
        
        self._history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        logger.info(f"SKAB model trained: {epochs} epochs, "
                    f"final loss: {self._history.history['loss'][-1]:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        tf, keras, _ = _ensure_tf()
        if tf is None:
            raise ImportError("TensorFlow is required to predict with the SKAB model")
        if self.model is None:
            raise ValueError("Model not built yet. Call build() first.")
        
        predictions = self.model.predict(X, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        tf, keras, _ = _ensure_tf()
        if tf is None:
            raise ImportError("TensorFlow is required to predict probabilities")
        if self.model is None:
            raise ValueError("Model not built yet. Call build() first.")
        
        probabilities = self.model.predict(X, verbose=0)
        return probabilities


class RandomForestModel:
    """Random Forest baseline model for SKAB."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self._imported = False
    
    def _ensure_sklearn(self):
        """Lazy import of sklearn."""
        if not self._imported:
            try:
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state
                )
                self._imported = True
                logger.info("Scikit-learn RandomForest initialized lazily")
            except ImportError:
                logger.warning("Scikit-learn not available")
                self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit Random Forest model."""
        self._ensure_sklearn()
        if self.model is not None:
            self.model.fit(X, y)
            logger.info("Random Forest model trained successfully")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        self._ensure_sklearn()
        if self.model is not None:
            return self.model.predict(X)
        return np.zeros(len(X), dtype=int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        self._ensure_sklearn()
        if self.model is not None:
            return self.model.predict_proba(X)[:, 1]
        return np.zeros(len(X))


def create_skab_model(model_type: str = "random_forest", **kwargs):
    """Factory function to create SKAB models.
    
    Args:
        model_type: "keras" or "random_forest"
        **kwargs: Model-specific parameters
    
    Returns:
        Initialized model instance
    """
    if model_type == "keras":
        from src.skab.model import SKABClassifier
        return SKABClassifier(**kwargs)
    elif model_type == "random_forest":
        from src.skab.model import RandomForestModel
        return RandomForestModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
