# Acoustic Biomarker Classifier

> **Status: early-stage research prototype.** No models have been trained and no results have been produced yet. This project is for research on respiratory-sound screening and classification only. It is **not** a medical device and makes no clinical or diagnostic claims.

## Research Question

**How well can respiratory-sound classifiers generalize to completely unseen patients and degraded real-world audio, and can uncertainty estimation identify predictions that should not be trusted?**

Many reported results on respiratory-sound benchmarks mix recordings from the same patient across training and test sets, and evaluate only on clean audio. This project measures performance under stricter conditions:

1. **Patient-independent evaluation:** no patient contributes audio to both training and test data.
2. **Robustness:** how performance degrades under realistic noise and recording artifacts.
3. **Trustworthiness:** whether calibrated probabilities and uncertainty estimates can flag predictions that should be deferred rather than acted on.

---

## Dataset

**Primary dataset:** [ICBHI 2017 Respiratory Sound Database](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)

- Auscultation recordings from multiple patients, annotated at the respiratory-cycle level for the presence of crackles and/or wheezes.
- Recordings come from different chest locations, stethoscopes, and recording devices, with varying sample rates.
- Raw audio is not distributed with this repository. Download it separately and place it under `data/raw/`. That directory is git-ignored.

---

## Planned Scope

| Component | Description | Status |
| :--- | :--- | :--- |
| Leakage-free splitting | Patient-grouped train/validation/test splits | Planned |
| Respiratory sound classification | Cycle-level classification (e.g., normal / crackle / wheeze / both) | Planned |
| Baseline CNN | CNN trained on Log-Mel spectrograms | Planned |
| Pretrained audio model | Comparison against a pretrained audio representation model | Planned |
| Noise robustness | Evaluation under synthetic and realistic audio degradations | Planned |
| Probability calibration | Reliability analysis and post-hoc calibration | Planned |
| Uncertainty & abstention | Uncertainty estimates and selective prediction (abstaining on low-confidence inputs) | Planned |
| Explainability | Saliency / attribution maps over time-frequency inputs | Planned |
| Error analysis | Analysis of failure modes by patient, device, class, and noise condition | Planned |
| Interactive demo | Research demo for inspecting predictions, uncertainty, and explanations | Planned |

---

## Methodology (Planned)

```text
Raw audio ──> Resample / segment into respiratory cycles ──> Log-Mel spectrogram
          ──> Model (baseline CNN | pretrained audio model)
          ──> Calibrated probabilities + uncertainty ──> Predict or abstain
```

**Evaluation principles**
- All splits are grouped by patient ID. Split definitions will be saved to `data/splits/` so experiments can be reproduced.
- Models are compared on clean audio and under controlled degradations (e.g., additive noise at varying SNRs, band-limiting, and other recording artifacts).
- Metrics will cover discrimination (e.g., ROC-AUC, sensitivity, specificity), calibration (e.g., ECE, reliability diagrams), and selective prediction (risk-coverage curves).

Specific architectures, hyperparameters, and augmentation settings will be documented here once they are implemented.

---

## Results

**TBD: not yet evaluated.** No models have been trained. This section will report patient-independent results, robustness curves, calibration, and abstention analysis once experiments are complete.

| Experiment | Status |
| :--- | :--- |
| Baseline CNN (clean audio) | TBD |
| Pretrained audio model (clean audio) | TBD |
| Noise robustness | TBD |
| Calibration | TBD |
| Uncertainty & abstention | TBD |
| Explainability & error analysis | TBD |

---

## Repository Structure

```text
acoustic-biomarker-classifier/
├── app/                # Interactive research demo
├── data/
│   ├── raw/            # ICBHI 2017 audio and annotations (git-ignored, not distributed)
│   └── splits/         # Patient-grouped split definitions
├── reports/
│   └── figures/        # Generated plots and visualizations
├── src/
│   ├── data/           # Dataset parsing, annotation handling, splitting
│   ├── evaluation/     # Metrics, calibration, robustness, and abstention analysis
│   ├── features/       # Audio preprocessing and spectrogram generation
│   └── models/         # Model definitions and training
├── tests/              # Unit tests
├── .gitignore
└── README.md
```

---

## Setup

The codebase is under active development. Dependencies, training scripts, and usage instructions will be added as components are implemented.

```bash
git clone https://github.com/justinyyueh/acoustic-biomarker-classifier.git
cd acoustic-biomarker-classifier

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

---

## Disclaimer

This repository is a research prototype for studying respiratory-sound classification, generalization, and model reliability. It is not intended for clinical use, diagnosis, or medical decision-making.
