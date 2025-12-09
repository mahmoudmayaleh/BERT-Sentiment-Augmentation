# Enhancing Sentiment Classification on Small Datasets through Data Augmentation and Transfer Learning

This repository contains the code and experiments supporting the study  
**“Enhancing Sentiment Classification on Small Datasets through Data Augmentation and Transfer Learning”** by Mahmoud Mayaleh and Samer Mayaleh.

## Overview

The project investigates how different text data augmentation techniques affect sentiment classification performance in **low-resource settings**. It provides a **unified, reproducible benchmark** comparing:

- Easy Data Augmentation (EDA)  
- NLPaug (contextual token substitution)  
- Back-Translation (Helsinki-NLP MarianMT, en↔fr)  

These methods are evaluated with both:

- Classical models: **Logistic Regression**, **Random Forest**  
- Transformer model: **BERT** (fine-tuned)

All experiments are conducted on a **5,000-sample IMDb subset**, with **balanced 100% augmentation** (one augmented sample per original, 10,000 training instances) for each augmentation method.

## Key Features

- **Balanced augmentation pipeline**  
  - EDA, NLPaug, and Back-Translation all applied at a 1:1 ratio (100% coverage)  
  - Back-Translation performed via **Helsinki-NLP MarianMT** models (local, no external API limits)

- **Classical ML baselines with rigor**
  - Stratified 10-fold cross-validation on Logistic Regression and Random Forest  
  - Metrics: Accuracy, F1-score, AUC  
  - 95% confidence intervals across folds  
  - Cohen’s *d* effect sizes and paired *t*-tests (p < 0.001) comparing original vs augmented data

- **BERT fine-tuning with multi-seed evaluation**
  - Fine-tuning on:
    - Original dataset (IMDb 5k)  
    - Original + NLPaug-augmented dataset (10k)  
  - Evaluation across **three random seeds** (12, 44, 127)  
  - Metrics: Accuracy, F1-score, AUC, with mean and standard deviation across seeds  
  - Per-epoch tracking of validation loss, accuracy, and AUC

- **Augmentation quality analysis**
  - Sentence-level cosine similarity between original and augmented samples using `sentence-transformers` (all-MiniLM-L6-v2)  
  - Quantifies semantic drift for EDA, NLPaug, and Back-Translation  
  - Exported examples of semantic drift and artifacts for each method

- **Visualization and reporting**
  - Plots of:
    - Validation loss, accuracy, and AUC for BERT (Original vs Orig+NLPaug) across seeds  
    - Overall accuracy / F1 / AUC for all models and augmentation conditions  
  - Confusion matrices for Logistic Regression, Random Forest, and BERT (per seed) on the test set  
  - CSV export of aggregated metrics (`metrics_df.csv`) for reproducible analysis

## Getting Started

### Requirements

- Python 3.8+  
- Recommended: GPU (e.g., NVIDIA T4) for BERT fine-tuning  

Core libraries:
pip install scikit-learn pandas torch transformers nlpaug
sentence-transformers datasets matplotlib seaborn

### Data

- Dataset: **IMDb** sentiment dataset (loaded via `datasets`):
  - 5,000 reviews sampled uniformly at random from the training split (`random_state=42`)  
  - Balanced positive/negative label distribution

## Usage

The main workflow in `v2_enhancing_sentiment_classification_on_small_datasets.py` is:

1. **Install dependencies** (Colab or local environment)
2. **Load and sample IMDb** (5,000 reviews)
3. **Apply augmentation methods**:
   - EDA (custom implementation: random deletion / swap / insertion)
   - NLPaug (`naw.RandomWordAug`)
   - Back-Translation (Helsinki-NLP MarianMT, pre-generated in practice)
4. **Train and evaluate classical models**:
   - Logistic Regression and Random Forest with stratified 10-fold CV
   - Compute accuracy, F1-score, AUC, and 95% CIs
   - Compute Cohen’s *d* and paired *t*-tests vs original baseline
5. **Fine-tune BERT**:
   - Train on original data and on original+NLPaug
   - Repeat for seeds 12, 44, 127
   - Track per-epoch validation metrics and final test metrics
6. **Analyze augmentation quality**:
   - Compute cosine similarity between original and augmented samples
   - Optionally inspect example pairs for semantic drift
7. **Generate artifacts**:
   - Save plots:
     - `bert_val_loss_all_seeds.png`
     - `bert_val_acc_all_seeds.png`
     - `bert_val_auc_all_seeds.png`
     - `overall_multi_metric.png`
   - Save aggregated metrics to `metrics_df.csv`

## Reproducibility

The code is designed for reproducibility:

- Fixed random seeds:
  - Dataset sampling: `random_state=42`
  - BERT runs: `{12, 44, 127}`
- Stratified 10-fold cross-validation for classical models
- Augmentation applied **only to training folds** to avoid leakage
- All key hyperparameters (learning rate, batch size, epochs, etc.) are fixed in the script

## Citation

If you use this work, please cite:

> Mahmoud S. Mayaleh and Samer A. Mayaleh,  
> *“Enhancing Sentiment Classification on Small Datasets through Data Augmentation and Transfer Learning,”* 2025.

## License

MIT License  
Copyright (c) 2025 Mahmoud Mayaleh
