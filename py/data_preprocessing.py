import gc
import os
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# --- Configuration Parameters ---
RAW_DATA_PATH = "ManyTx.pkl"
OUTPUT_FILENAME = "processed_wisig.npz"
SIGNAL_LEN = 256
TEST_SPLIT = 0.2
RANDOM_SEED = 42


def recursive_flatten(nested_structure):
    """
    Recursively traverses the nested dataset structure to extract
    valid signal batches matching the target dimensions.
    """
    extracted_signals = []

    # Recursion for lists or object arrays
    if isinstance(nested_structure, (list, tuple)) or (
        isinstance(nested_structure, np.ndarray) and nested_structure.dtype == "O"
    ):
        for item in nested_structure:
            extracted_signals.extend(recursive_flatten(item))

    # Base case: valid numpy array found
    elif isinstance(nested_structure, np.ndarray):
        # Validation: Check for shape (N, 256, 2)
        if (
            nested_structure.ndim == 3
            and nested_structure.shape[1] == SIGNAL_LEN
            and nested_structure.shape[2] == 2
        ):
            if nested_structure.shape[0] > 0:
                for signal in nested_structure:
                    extracted_signals.append(signal)

    return extracted_signals


def normalize_data(X):
    """
    Applies Min-Max normalization to scale signals between -1 and 1.
    Also handles NaN/Inf sanitization to prevent gradient explosion.
    """
    print(" [Status] Sanitizing and normalizing data...")

    # 1. Sanitize: Replace NaNs with 0 and Infs with boundary values
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    # 2. Clip: Enforce strict bounds [-1, 1]
    X = np.clip(X, -1.0, 1.0)

    return X


def main():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw dataset {RAW_DATA_PATH} not found.")

    print(f"--- Processing Dataset: {RAW_DATA_PATH} ---")

    # Load raw pickle file
    with open(RAW_DATA_PATH, "rb") as f:
        dataset = pickle.load(f, encoding="latin1")

    raw_data = dataset["data"]
    labels_list = dataset["tx_list"]

    X_list = []
    y_list = []

    # Parse dataset structure
    print(f" [Status] Parsing {len(labels_list)} device classes...")
    for i, _ in enumerate(labels_list):
        device_signals = recursive_flatten(raw_data[i])

        if len(device_signals) > 0:
            X_list.extend(device_signals)
            y_list.extend([i] * len(device_signals))

    # Convert to Numpy arrays
    X = np.array(X_list)
    y = np.array(y_list)

    # Clear memory
    del dataset, raw_data
    gc.collect()

    print(f" [Info] Total signals extracted: {len(X)}")

    # Preprocessing pipeline
    X = normalize_data(X)

    # One-hot encoding of labels
    y_categorical = to_categorical(y, num_classes=len(labels_list))

    # Stratified Train-Test Split
    print(" [Status] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
    )

    # Save processed artifacts
    print(f" [Status] Saving to {OUTPUT_FILENAME}...")
    np.savez_compressed(
        OUTPUT_FILENAME, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )
    print(" [Success] Data preprocessing complete.")


if __name__ == "__main__":
    main()
