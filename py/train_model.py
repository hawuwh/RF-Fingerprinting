import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# --- Hyperparameters ---
BATCH_SIZE = 32  # Reduced for memory stability
EPOCHS = 25
LEARNING_RATE = 1e-5  # Conservative rate for stability
CLIP_VALUE = 0.5  # Gradient clipping threshold
DATA_PATH = "processed_wisig.npz"
MODEL_SAVE_PATH = "best_rf_model.keras"


def build_cnn_model(input_shape, num_classes):
    """
    Constructs the 1D Convolutional Neural Network architecture.
    """
    model = Sequential(
        [
            # Feature Extraction Block 1
            Conv1D(64, 3, activation="relu", input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(2),
            # Feature Extraction Block 2
            Conv1D(64, 3, activation="relu"),
            BatchNormalization(),
            MaxPooling1D(2),
            # Feature Extraction Block 3
            Conv1D(128, 3, activation="relu"),
            BatchNormalization(),
            MaxPooling1D(2),
            # Classification Head
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),  # Regularization
            Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed data {DATA_PATH} not found.")

    print("--- Initializing Training Pipeline ---")

    # Load Data
    data = np.load(DATA_PATH)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    # Memory Management: Explicitly remove file handle
    del data
    gc.collect()

    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = y_train.shape[1]

    print(f" [Info] Training Samples: {len(X_train)}")
    print(f" [Info] Class Count: {num_classes}")

    # Model Compilation
    model = build_cnn_model(input_shape, num_classes)

    # Optimizer Configuration with Gradient Clipping
    optimizer = Adam(learning_rate=LEARNING_RATE, clipvalue=CLIP_VALUE)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Callbacks for robust training
    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]

    # Training Execution
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    # Save Training History Graph
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("RF Fingerprinting Training Performance")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig("training_history.png")
    print(" [Success] Training complete. Model and history saved.")


if __name__ == "__main__":
    main()
