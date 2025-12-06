# Enhancing Sentiment Classification on Small Datasets through Data Augmentation and Transfer Learning

This repository contains code and experiments supporting the study **"Enhancing Sentiment Classification on Small Datasets through Data Augmentation and Transfer Learning"** by Mahmoud Mayaleh and Samer Mayaleh.

## Overview

The project investigates the impact of various data augmentation techniques on sentiment classification performance when training data is limited. It compares:

- Easy Data Augmentation (EDA)  
- NLPaug (contextual augmentation)  
- Back-Translation  

Classical machine learning models (Logistic Regression, Random Forest) and a fine-tuned BERT model are evaluated on original and augmented datasets.

## Key Features

- Data augmentation methods for text classification  
- Training and evaluation of classical ML models with cross-validation  
- Transfer learning with BERT for sentiment classification  
- Performance metrics: Accuracy, F1-score, AUC  
- Confidence interval calculations for robustness analysis  

## Getting Started

### Requirements

- Python 3.7+  
- Libraries: `scikit-learn`, `pandas`, `transformers`, `torch`, `nlpaug`, `datasets`  

## Usage

- Prepare dataset (IMDB small subset used here)  
- Apply augmentation techniques  
- Train and evaluate classical ML models  
- Fine-tune BERT model with transfer learning  
- Generate performance reports with confidence intervals  

## Citation

If you use this work, please cite:

Mahmoud Mayaleh and Samer Mayaleh,  
*"Enhancing Sentiment Classification on Small Datasets through Data Augmentation and Transfer Learning,"* 2025.

## License
MIT License
Copyright (c) 2025 Mahmoud Mayaleh
