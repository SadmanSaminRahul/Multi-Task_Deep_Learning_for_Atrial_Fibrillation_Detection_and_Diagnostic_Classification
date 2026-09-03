# Multi-Task Deep Learning for Atrial Fibrillation Detection and Diagnostic Classification of 12-Lead ECG with Explainable Attribution

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Thesis%20Complete-brightgreen)

Undergraduate thesis (EEE 400, July 2025 term), Department of Electrical and
Electronic Engineering, Bangladesh University of Engineering and Technology
(BUET), Dhaka, Bangladesh — June 2026.

A CNN–BiLSTM detector for atrial fibrillation (AFIB) from raw 12-lead ECG,
trained and validated on PTB-XL with external validation on
Chapman–Shaoxing, paired with a multi-method explainability pipeline
(Integrated Gradients, Grad-CAM, Occlusion, Feature Ablation) that turns
attribution maps into short natural-language explanations.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Datasets](#datasets)
- [Model Architecture](#model-architecture)
- [Methodology](#methodology)
- [Results](#results)
- [Explainability](#explainability)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Limitations](#limitations)
- [Authors](#authors)
- [Citation](#citation)
- [References](#references)
- [License](#license)

---

## Overview

Atrial fibrillation is the most common sustained cardiac arrhythmia and a
major risk factor for stroke, yet it is frequently missed on a single
resting ECG. This project trains a deep detector to flag AFIB directly from
the raw twelve-lead signal, while treating interpretability as part of the
method rather than an afterthought: every prediction can be traced back to
the leads and time windows that drove it, and that evidence is condensed
into a short, plain-language explanation.

The model reads the signal at two scales at once — a convolutional front
end for local beat morphology, and a bidirectional LSTM for the
longer-range rhythm irregularity that defines fibrillation — and is
trained jointly on the binary AFIB label and the five PTB-XL diagnostic
superclasses (NORM, MI, CD, HYP, STTC).

## Key Features

- **CNN + BiLSTM detector** — a 1-D convolutional encoder distills each of
  the 12 leads into a compact feature sequence, which a 2-layer
  bidirectional LSTM reads for rhythm context.
- **Class-imbalance handling** — a weighted random sampler (bucketed by
  superclass × AFIB label, with a mild AFIB boost and a weight cap) plus a
  class-weighted loss, since AFIB accounts for under 7% of PTB-XL.
- **Internal + external validation** — evaluated on the held-out PTB-XL
  test fold and, without any retraining or threshold refitting, on the
  independent Chapman–Shaoxing database.
- **Multi-method explainability** — Integrated Gradients, Grad-CAM,
  Occlusion, and Feature Ablation are run together and cross-checked, with
  a deletion/insertion faithfulness analysis to quantify how much each map
  is actually trusted.
- **Automated explanation generation** — attribution summaries (top leads,
  top time windows) plus rhythm-derived signal features (RR-interval
  statistics, RMSSD, pNN50, LF/HF ratio) are assembled into a structured
  prompt and rendered as a short natural-language explanation.

## Datasets

| Dataset | Role | Records | AFIB positive | AFIB % |
|---|---|---|---|---|
| [PTB-XL](https://physionet.org/content/ptb-xl/) v1.0.1 | Train | 17,100 | 1,157 | 6.77% |
| PTB-XL | Validation | 2,155 | 147 | 6.82% |
| PTB-XL | Test (fold 10) | 2,162 | 144 | 6.66% |
| [Chapman–Shaoxing](https://physionet.org/content/ecg-arrhythmia/) | External test only | 10,247 | 1,780 | 17.4% |

PTB-XL provides 21,837 twelve-lead recordings at 500 Hz; records without a
mapped diagnostic superclass (420 of them) are excluded, since the
imbalance-handling scheme buckets by superclass. The predefined stratified
folds are used as given (folds 1–8 train, fold 9 validation, fold 10 test)
so that recordings from the same patient never cross a split. Chapman–
Shaoxing is used only for evaluation, with AFIB positivity taken from the
SNOMED-CT rhythm code `164889003`.

## Model Architecture

| Stage | Layer | Output channels |
|---|---|---|
| Input | 12-lead ECG (12 × 5000) | 12 |
| Encoder | ConvBlock (k=9, s=2, p=4) | 64 |
| | ConvBlock (k=7, s=1, p=3) | 64 |
| | MaxPool (/2) | 64 |
| | ConvBlock (k=7, s=2, p=3) | 128 |
| | ConvBlock (k=5, s=1, p=2) | 128 |
| | MaxPool (/2) | 128 |
| | ConvBlock (k=5, s=2, p=2) | 192 |
| | ConvBlock (k=3, s=1, p=1) | 192 |
| Recurrent | BiLSTM (2 layers, hidden 128/direction) | 256 |
| Pooling | Mean over time | 256 |
| Head | Linear → GELU → Dropout(0.4) → Linear | 1 (AFIB logit) |

Each `ConvBlock` is a 1-D convolution followed by batch normalisation,
GELU, and dropout (0.15). Signals are 10 s (5,000 samples at 500 Hz),
z-scored per lead after a 0.5 Hz high-pass filter to remove baseline
wander.

## Methodology

1. **Preprocessing** — resample to 500 Hz, crop/pad to 5,000 samples,
   0.5 Hz high-pass filter, per-lead z-score normalization. Identical
   pipeline applied to PTB-XL and Chapman–Shaoxing.
2. **Augmentation** (train only) — random amplitude scaling, additive
   Gaussian noise, circular time shift, random lead dropout, and random
   time-window masking, each applied stochastically.
3. **Imbalance handling** — a weighted random sampler over 10 buckets
   (5 superclasses × AFIB label), with a 1.5× boost on AFIB-positive
   buckets and weights capped at 5.0, combined with a class-weighted
   binary cross-entropy loss.
4. **Training** — AdamW (lr 2×10⁻³, weight decay 1×10⁻⁴), batch size 32,
   gradient clipping, mixed-precision, `ReduceLROnPlateau` scheduling,
   and early stopping on validation performance. The decision threshold is
   chosen on the validation set by maximising Youden's J statistic rather
   than fixed at 0.5.
5. **Explainability** — four Captum attribution methods (Integrated
   Gradients, Grad-CAM, Occlusion, Feature Ablation) are run on the frozen
   model; their outputs are condensed into top contributing leads and
   half-second time windows, cross-checked, and validated with a
   deletion/insertion faithfulness test.
6. **Explanation generation** — a frozen pre-trained language model
   verbalises the attribution summary and rhythm-derived signal features
   through a fixed template, so the wording is fluent but every factual
   claim traces directly to the computed evidence.

## Results

**PTB-XL held-out test set (fold 10):**

| Metric | Value |
|---|---|
| ROC-AUC | 0.9791 |
| F1-score (AFIB class) | 0.8272 |
| Precision (AFIB class) | 0.7444 |
| Recall (AFIB class) | 0.9306 |
| F1-score (non-AFIB class) | 0.9860 |
| Overall accuracy | 0.9741 |
| Macro-average F1 | 0.9066 |

**External validation on Chapman–Shaoxing** (frozen model, no
fine-tuning, identical preprocessing):

| Metric | Val-threshold | Threshold 0.5 |
|---|---|---|
| ROC-AUC | 0.9610 | 0.9610 |
| Precision (AFIB class) | 0.552 | 0.602 |
| Recall (AFIB class) | 0.973 | 0.942 |
| F1-score (AFIB class) | 0.704 | 0.734 |
| Overall accuracy | 0.858 | 0.882 |
| Macro-average F1 | 0.805 | 0.829 |

The detector is tuned to favour recall on the view that, for a screening
aid, a missed arrhythmia is costlier than a false alarm. ROC-AUC transfers
well to the unseen Chapman population (0.979 → 0.961); precision drops
more, which is the expected effect of carrying a threshold tuned on one
population's prevalence over to another with a higher AFIB base rate
(17.4% vs. 6.8%).

**Attribution faithfulness** (deletion/insertion AUC, averaged over 30
AFIB-positive test recordings — lower deletion and higher insertion both
indicate a more faithful attribution):

| Method | Deletion AUC ↓ | Insertion AUC ↑ |
|---|---|---|
| Integrated Gradients | 0.712 | 0.891 |
| Grad-CAM | 0.760 | 0.715 |
| Occlusion | 0.570 | 0.868 |
| Feature Ablation | — | — |
| Fused (average) | 0.721 | 0.825 |
| Random (control) | 0.722 | 0.725 |

All four methods beat the random control, indicating the highlighted
regions genuinely drive the prediction rather than merely coinciding with
it.

## Explainability

For a correctly detected AFIB case, the pipeline produces:

1. A 12-lead attribution heatmap and per-lead importance ranking.
2. The top-3 most influential half-second time windows.
3. A generated explanation, e.g.:

   > *"The model predicts AF with high confidence (predicted probability =
   > 0.989). The strongest evidence is concentrated in leads V1, aVR, I.
   > The most influential time regions are around 4.00–4.50 s, 8.50–9.00 s.
   > These regions were consistently highlighted by input-gradient and
   > perturbation-based explainability methods."*

Attribution indicates *where* the model attends, not whether that
attention is clinically correct — see [Limitations](#limitations).

## Repository Structure

```
.
├── README.md
├── notebooks/
│   └── thesismultihead.ipynb      # main training + evaluation + XAI pipeline
├── docs/
│   └── thesis.pdf                 # full thesis document
├── figures/                       # exported plots used in the thesis
├── checkpoints/                   # saved model weights (not tracked in git)
├── requirements.txt
└── LICENSE
```

Adjust paths above to match your actual layout before pushing.

## Getting Started

### Prerequisites

- Python 3.10+
- A CUDA-capable GPU is recommended for training (CPU works for inference)

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0
numpy
pandas
scikit-learn
scipy
matplotlib
captum
```

### Data

1. Download [PTB-XL](https://physionet.org/content/ptb-xl/1.0.1/) and
   [Chapman–Shaoxing](https://physionet.org/content/ecg-arrhythmia/1.0.0/)
   from PhysioNet.
2. Update the dataset paths (`BASE_DIR`, `CHAPMAN_DIR`, etc.) at the top of
   the notebook to point to your local copies.

## Usage

Open `notebooks/thesismultihead.ipynb` and run top-to-bottom:

1. **Setup & data loading** — imports, seeding, PTB-XL label construction.
2. **Preprocessing & sampler** — signal pipeline and imbalance handling.
3. **Model & training** — defines the CNN–BiLSTM and runs the training
   loop with early stopping.
4. **Evaluation** — internal PTB-XL test metrics and external
   Chapman–Shaoxing validation.
5. **Explainability** — attribution methods, faithfulness analysis, and
   natural-language explanation generation.

## Limitations

- Attribution methods show where the model attends, not whether its
  reasoning is clinically correct; the methods can disagree at the
  margins, and generated explanations have not been validated against
  expert annotation.
- The model is trained on a single population and recording protocol;
  Chapman–Shaoxing tempers but does not remove this concern.
- The fixed 10-second window discards signal content outside it.
- This is a decision-support research prototype, not a validated clinical
  tool, and is not intended for unsupervised clinical use.

## Authors

- **Sadman Samin Rahul** (2006055)
- **Md. Shahriar Hasan** (2006164)

**Supervised by:** Dr. Hafiz Imtiaz, Professor, Department of Electrical
and Electronic Engineering, BUET

## Citation

If you use this work, please cite:

```bibtex
@thesis{rahul_hasan_2026_afib,
  title   = {Multi-Task Deep Learning for Atrial Fibrillation Detection
             and Diagnostic Classification of 12-Lead ECG with
             Explainable Attribution},
  author  = {Rahul, Sadman Samin and Hasan, Md. Shahriar},
  school  = {Bangladesh University of Engineering and Technology},
  year    = {2026},
  type    = {Bachelor's Thesis},
  address = {Dhaka, Bangladesh}
}
```

## References

Key works this project builds on — see the thesis for the full list:

- Wagner et al., ["PTB-XL, a Large Publicly Available Electrocardiography Dataset,"](https://doi.org/10.1038/s41597-020-0495-6) *Scientific Data*, 2020.
- Strodthoff et al., ["Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL,"](https://doi.org/10.1109/JBHI.2020.3022989) *IEEE JBHI*, 2021.
- Zheng et al., ["A 12-Lead Electrocardiogram Database for Arrhythmia Research,"](https://doi.org/10.1038/s41597-020-0386-x) *Scientific Data*, 2020.
- Sundararajan et al., ["Axiomatic Attribution for Deep Networks,"](https://arxiv.org/abs/1703.01365) *ICML*, 2017.
- Selvaraju et al., ["Grad-CAM: Visual Explanations from Deep Networks,"](https://arxiv.org/abs/1610.02391) *ICCV*, 2017.
- Kokhlikyan et al., ["Captum: A Unified and Generic Model Interpretability Library for PyTorch,"](https://arxiv.org/abs/2009.07896) 2020.

## License

This project is licensed under the MIT License — see the
[LICENSE](LICENSE) file for details, or replace with your institution's
preferred license for academic code release.
