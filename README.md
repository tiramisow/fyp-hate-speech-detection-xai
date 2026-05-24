# fyp-hate-speech-detection-xai
# FYP Hate Speech Detection with XAI

This repository contains the source code, experiments, trained model, and proof-of-concept web application for my Final Year Project:

**Context-Aware Multi-Class Bullying and Hate Speech Detection using Transformer Models and XAI**

## Project Overview

This project focuses on multi-label bullying and hate speech detection using traditional machine learning models and Transformer-based models. The task involves classifying online comments into six toxicity categories:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

The project compares traditional TF-IDF based baselines with Transformer-based models and integrates Explainable AI (XAI) to interpret model predictions.

## Main Features

- Multi-label toxicity classification
- Traditional ML baselines:
  - Logistic Regression
  - Linear SVM
- Transformer models:
  - BERT
  - RoBERTa
- Threshold tuning for multi-label classification
- Per-label performance analysis
- Priority-based 6×6 confusion matrix visualization
- LIME-based XAI explanation
- SHAP-based additional XAI experiment
- React + FastAPI proof-of-concept web application

## Final Model Selection

BERT was selected as the main model for the proof-of-concept system because it achieved strong overall performance and slightly better macro F1-score compared with RoBERTa.

| Model | Micro F1 | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.6053 | 0.4960 | 0.7218 |
| Linear SVM | 0.7196 | 0.5919 | 0.7295 |
| BERT | 0.7951 | 0.6865 | 0.7977 |
| RoBERTa | 0.7959 | 0.6855 | 0.7977 |

## Repository Structure

```text
fyp-hate-speech-detection-xai/
│
├── notebooks/
│   ├── baseline_final.ipynb
│   ├── MultiLabel_2.ipynb
│   ├── roberta_multilabel.ipynb
│   ├── final_results_xai_fairness.ipynb
│   └── XAI_SHAP.ipynb
│
├── results/
│   ├── final_model_comparison_table.csv
│   ├── final_model_performance_comparison.png
│   ├── bert_roberta_per_label_f1_heatmap.png
│   └── confusion_matrix_figures/
│
├── backend/
│   ├── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── package.json
│   └── src/
│
├── models/
   └── bert_multilabel_grid_search_final/
       ├── best_bert_full_checkpoint.pt
       ├── bert_tokenizer/
       ├── best_config.json
       └── best_thresholds.json

