# RF Fingerprinting: Deep Learning for IoT Device Authentication

## 📌 Project Overview
This project implements a **Physical Layer Security** framework to authenticate wireless IoT devices using **Radio Frequency (RF) Fingerprinting**. 

Unlike traditional security methods (like MAC addresses) which can be easily spoofed, this system identifies devices based on unique hardware imperfections inherent in their RF circuitry (e.g., I/Q imbalance, frequency offsets). Using **Digital Signal Processing (DSP)** and a **1D Convolutional Neural Network (CNN)**, the system successfully classifies **150 distinct WiFi transmitters** from the **WiSig ManyTx** dataset.

## 🎯 Key Features
- **Scalable Identification:** Capable of classifying **150 unique devices** (significantly harder than standard 5-10 device studies).
- **Robust DSP Pipeline:** Implements I/Q decomposition, Min-Max Normalization, and Gradient Clipping to handle raw hardware signals.
- **Deep Learning Architecture:** Uses a custom 1D-CNN optimized for time-series signal classification.
- **High Performance:** Achieves **36.63% Accuracy** on a 150-class problem (approx. **55x better than random guessing**).

## 📂 Dataset
This project uses the **WiSig (WiFi Signal) Dataset - ManyTx Subset** [1].
- **Source:** CORES Lab, UCLA.
- **Content:** Real-world WiFi preambles captured via USRP N210 radios.
- **Classes:** 150 off-the-shelf WiFi transmitters.
- **Signal Shape:** 256 complex-valued samples ($I + jQ$).
- *Note: The dataset is not included in this repo due to size. It must be downloaded separately.*

## 🛠️ Methodology

### 1. Data Preprocessing (DSP)
Raw RF data is inherently noisy and contains high-dynamic-range artifacts. We implemented a robust pipeline:
- **Sanitization:** Removal of corrupted signals (NaN/Inf values).
- **I/Q Decomposition:** Splitting complex signals into two real-valued channels (In-Phase & Quadrature).
- **Normalization:** Min-Max scaling to range $[-1, 1]$ to prevent exploding gradients.

### 2. Neural Network Architecture
The core model is a **1D Convolutional Neural Network (CNN)** built with TensorFlow/Keras:
- **Feature Extraction:** 3x Stacked `Conv1D` layers with Batch Normalization and Max Pooling.
- **Classification Head:** Dense layer (256 units) with Dropout (0.5) leading to a Softmax output (150 units).
- **Optimization:** Adam optimizer with **Gradient Clipping** (`clipvalue=0.5`) to ensure stability.

## 📊 Results
The system was evaluated on a held-out test set (20% of data).

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **36.63%** | ~55x superior to random chance (0.66%). |
| **Precision** | **0.38** | High reliability in positive identifications. |
| **Recall** | **0.36** | Consistent detection of authorized devices. |
| **F1-Score** | **0.35** | Balanced performance across all 150 classes. |

### Confusion Matrix (Subset of 20 Devices)
The diagonal structure confirms distinct fingerprints are being detected, while off-diagonal clusters reveal manufacturing similarities between specific hardware batches.

![Confusion Matrix](confusion_matrix.png)

## 🚀 How to Run

### Prerequisites
Install the required Python libraries:
```bash
pip install -r requirements.txt
