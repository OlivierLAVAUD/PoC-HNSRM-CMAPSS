"""CNN-LSTM + MultiHeadAttention sequential classifier for FD001 windows.

Optional TensorFlow/Keras model. Importing this module does not require
TensorFlow; only calling :func:`create_cnn_lstm_model` does.
"""

from __future__ import annotations

from typing import Optional, Tuple

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    _TF_AVAILABLE = True
except Exception:  # pragma: no cover - environment without TensorFlow
    tf = None  # type: ignore[assignment]
    keras = None  # type: ignore[assignment]
    layers = None  # type: ignore[assignment]
    _TF_AVAILABLE = False


def _require_tf() -> None:
    if not _TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is required for architecture='cnn_lstm'. "
            "Install it with `pip install tensorflow`."
        )


def create_cnn_lstm_model(
    input_shape: Tuple[int, int],
    learning_rate: float = 0.001,
    use_attention: bool = True,
    attention_key_dim: int = 50,
    attention_heads: int = 4,
    cnn_filters: int = 64,
    cnn_kernel_size: int = 3,
    lstm_units: int = 64,
    dropout: float = 0.2,
) -> "keras.Model":
    """Build a CNN-LSTM (+ optional MultiHeadAttention) binary classifier.

    Parameters
    ----------
    input_shape:
        ``(window_size, n_features)`` of the 3D input windows.
    learning_rate:
        Adam learning rate.
    use_attention:
        Whether to apply a MultiHeadAttention layer over the LSTM output.
    attention_key_dim:
        Key dimension for the attention layer.
    attention_heads:
        Number of attention heads.
    cnn_filters / cnn_kernel_size:
        Conv1D parameters applied to the raw window.
    lstm_units:
        LSTM hidden units.
    dropout:
        Dropout rate applied after CNN and before the classification head.

    Returns
    -------
    A compiled Keras model with a single sigmoid output.
    """
    _require_tf()

    window_size, n_features = int(input_shape[0]), int(input_shape[1])

    inputs = keras.Input(
        shape=(window_size, n_features), name="sensor_window"
    )

    # Temporal feature extraction over the window.
    x = layers.Conv1D(
        filters=cnn_filters,
        kernel_size=cnn_kernel_size,
        padding="same",
        activation="relu",
        name="conv1d",
    )(inputs)
    x = layers.BatchNormalization(name="conv_bn")(x)
    x = layers.Dropout(dropout, name="conv_dropout")(x)

    # Recurrent encoding of temporal dependencies.
    x = layers.LSTM(
        lstm_units,
        return_sequences=True,
        name="lstm",
    )(x)

    # Optional self-attention to weigh the most informative cycles.
    if use_attention:
        attention_output = layers.MultiHeadAttention(
            num_heads=attention_heads,
            key_dim=attention_key_dim,
            name="multihead_attention",
        )(x, x)
        x = layers.Add(name="attention_add")([x, attention_output])
        x = layers.LayerNormalization(name="attention_norm")(x)
        x = layers.Dropout(dropout, name="attention_dropout")(x)

    # Global pooling over the time axis -> fixed-size representation.
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(32, activation="relu", name="dense")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(
        1, activation="sigmoid", name="failure_probability"
    )(x)

    model = keras.Model(
        inputs=inputs, outputs=outputs, name="cnn_lstm_attention"
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model
