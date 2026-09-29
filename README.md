# Acoustic Biomarker Classifier

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract
Early detection of respiratory conditions often requires costly diagnostic equipment and specialized clinical expertise. This repository provides a deep learning framework that converts raw time-domain audio recordings into 128-band Log-Mel Spectrograms to classify respiratory abnormalities using a MobileNetV3 backbone. Evaluated with 5-fold `StratifiedGroupKFold` cross-validation grouped strictly by patient ID to eliminate speaker data leakage, the model achieves an average **0.86 ROC-AUC**, complete with Grad-CAM spectral heatmaps for interpretable inference.

---

## Key Features
* **Leakage-Free Validation:** Strict patient-level split (`StratifiedGroupKFold`) ensures no audio samples from the same patient appear across both training and test sets.
* **Robust Audio Preprocessing:** Standardized 16 kHz mono resampling with Log-Mel Spectrogram transformation and dynamic `SpecAugment` (time and frequency masking).
* **Explainable AI (XAI):** Integrated Grad-CAM heatmaps overlaying time-frequency activation patterns directly onto input spectrograms.
* **Web Deployment:** Interactive Streamlit web interface for single-file upload, real-time microphone recording, and diagnostic probability visualization.

---

## Directory Structure

```text
acoustic-biomarker-classifier/
├── data/
│   ├── raw/                   # Raw dataset (.wav / metadata)
│   └── splits.json            # Patient-grouped cross-validation index mapping
├── models/
│   └── baseline_mobilenet.pt  # Saved model weights
├── reports/
│   ├── figures/               # Grad-CAM outputs and ROC curves
│   └── metrics_summary.csv    # Per-fold evaluation metrics
├── src/
│   ├── data/
│   │   └── clean_metadata.py  # Dataset parsing and filtering
│   ├── features/
│   │   └── audio_transforms.py# Resampling & Log-Mel spectrogram generation
│   ├── models/
│   │   └── train.py           # PyTorch training pipeline with early stopping
│   └── explainability.py      # Grad-CAM heatmap generation
├── app.py                     # Interactive Streamlit application
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## Dataset & Preprocessing

The primary pipeline supports the **ICBHI 2017 Respiratory Sound Database** and **COUGHVID** dataset.

* **Audio Format:** 16,000 Hz Mono
* **Time-Frequency Representation:** Log-Mel Spectrogram (128 Mel bands, FFT window size: 1024, Hop length: 512)
* **Data Augmentation:** `SpecAugment` (Frequency Masking $F_{\text{max}}=15$, Time Masking $T_{\text{max}}=35$)

---

## Methodology & Training

The architecture utilizes a transfer learning approach leveraging a lightweight MobileNetV3-Small backbone pre-trained on ImageNet and fine-tuned on spectrogram embeddings.

```text
Raw Audio (.wav) ──> Resample (16kHz Mono) ──> Log-Mel Spectrogram ──> SpecAugment ──> MobileNetV3 ──> BCEWithLogitsLoss
```

### Hyperparameters

| Parameter | Value |
| :--- | :--- |
| **Backbone Architecture** | `mobilenet_v3_small` |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$) |
| **Learning Rate** | $1 \times 10^{-4}$ with Cosine Annealing |
| **Batch Size** | 32 |
| **Loss Function** | `BCEWithLogitsLoss` (Weighted for class imbalance) |
| **Early Stopping** | Validation ROC-AUC (Patience = 10 epochs) |

---

## Experimental Results

Evaluated across a 5-fold cross-validation setup grouped by patient ID:

| Metric | Fold Mean | Target Threshold |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.862 ± 0.02** | $\ge 0.85$ |
| **Sensitivity (Recall)** | **0.914 ± 0.03** | Target $\ge 0.90$ |
| **Specificity** | **0.798 ± 0.04** | -- |
| **F1-Score** | **0.835 ± 0.02** | -- |

### Interpretability (Grad-CAM)
Model decision-making is validated using Gradient-weighted Class Activation Mapping (Grad-CAM) to ensure classifications rely on physiological acoustic markers (e.g., crackles, wheezes, high-frequency cough bursts) rather than background noise artifacts.

---

## Quickstart & Installation

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/your-username/acoustic-biomarker-classifier.git
cd acoustic-biomarker-classifier

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Data Pipeline & Train Model

```bash
# Generate leakage-free patient splits
python src/data/clean_metadata.py --metadata_path data/raw/metadata.csv --output data/splits.json

# Train model across cross-validation splits
python src/models/train.py --config configs/train_config.yaml
```

### 3. Launch Local Web App

```bash
streamlit run app.py
```

---

## Deployment
The interactive web demo is hosted on Hugging Face Spaces / Streamlit Community Cloud. Users can upload standard `.wav` or `.mp3` files or record directly via microphone to receive real-time abnormality probabilities and diagnostic visual explanations.
