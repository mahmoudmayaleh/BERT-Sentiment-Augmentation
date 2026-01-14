# Enhancing Sentiment Classification on Small Datasets through Data Augmentation and Transfer Learning

This repository contains the code and experiments for the paper  
**“Enhancing Sentiment Classification on Small Datasets through Data Augmentation and Transfer Learning”**  
by Mahmoud S. Mayaleh and Samer A. Mayaleh, published in *Discover Artificial Intelligence* (Springer), 2026.

---

## Overview

The project investigates how different text data augmentation strategies affect sentiment classification in **low-resource settings** and provides a **unified, reproducible benchmark** for three widely used methods:

- **Easy Data Augmentation (EDA)**
- **Contextual token substitution (NLPaug-style)**
- **Back-translation** with Helsinki-NLP MarianMT models (English↔French)

These augmentors are evaluated under identical experimental conditions with both:

- Classical models: **Logistic Regression**, **Random Forest**
- Transformer model: **BERT** (bert-base-uncased, fine-tuned)

All experiments use a **5,000-sample, class-balanced IMDb subset**, with **100% augmentation coverage** (1:1 ratio, one augmented sample per original) for each augmentation method, yielding 10,000 training instances per configuration.

---

## Main Contributions

- **Unified augmentation benchmark:**  
  Direct, apples-to-apples comparison of EDA, back-translation, and contextual augmentation across traditional and transformer models on the same data, splits, and hyperparameters.

- **Cross-family model evaluation:**  
  Stratified 10-fold cross-validation for Logistic Regression and Random Forest, plus multi-seed BERT fine-tuning, to study how each augmentation interacts with model class.

- **Statistical rigor and effect sizes:**  
  Accuracy, F1-score, and AUC with 95% confidence intervals for classical models, together with paired *t*-tests and Cohen’s *d* to quantify the practical impact of augmentation.

- **Augmentation quality analysis:**  
  Sentence-level cosine similarity (all-MiniLM-L6-v2) between original and augmented reviews to characterize semantic drift and the diversity–fidelity trade-off of each method.

- **Open and reproducible framework:**  
  Fixed seeds, controlled preprocessing, and export of all metrics and plots, enabling extension to new datasets, languages, and hybrid augmentation pipelines.

---

## Key Findings

- All three augmentation strategies significantly improve sentiment classification performance compared with training on the original 5k IMDb subset alone.
- **Contextual augmentation (NLPaug)** provides the most consistent gains for BERT, pushing test accuracy to roughly **97%** when training on original + contextual-augmented data.
- **EDA** and **back-translation** offer larger relative improvements for traditional models, especially Random Forest, which reaches accuracies above **98%** with augmented training data.
- Back-translation with MarianMT exhibits extremely high cosine similarity to the original sentences (≈0.9999), indicating near-identity paraphrases that preserve sentiment but limit lexical diversity compared to EDA and NLPaug.

---

## Requirements

- Python 3.8+
- Recommended: NVIDIA GPU (e.g., Tesla T4) for BERT fine-tuning.

Install core dependencies:

```bash
pip install \
  scikit-learn pandas torch transformers nlpaug \
  sentence-transformers datasets matplotlib seaborn
```
## License

MIT License  
Copyright (c) 2026 Mahmoud Mayaleh
