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

Dataset
The experiment is based on a multi-label toxic comment dataset containing six toxicity labels. Due to dataset size and licensing considerations, the raw dataset is not included in this repository.

Expected dataset format:

comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

Each label column should contain binary values:

0 = label not present
1 = label present

Methodology
1. Data Splitting

A fixed multi-label stratified train-validation-test split was used. This was chosen to preserve the distribution of all six labels across the training, validation, and test sets.

K-Fold cross-validation was not used for Transformer models due to computational cost, as BERT and RoBERTa require long fine-tuning time. The same split strategy was applied across all models to ensure fair comparison.

2. Baseline Models

The traditional baselines used TF-IDF features with One-vs-Rest classification:

Logistic Regression
Linear SVM

Hyperparameter tuning was performed using validation macro F1-score.

3. Transformer Models

The Transformer models were fine-tuned for multi-label classification using:

BERT
RoBERTa

The output layer uses sigmoid activation for six independent labels. Tuned thresholds were applied to determine the final predicted labels.

4. Explainable AI

LIME was used as the main post-hoc XAI method. It identifies words that positively or negatively contribute to a selected label prediction.

SHAP was also explored as an additional token-level XAI method.

Proof-of-Concept Web Application

The proof-of-concept system uses:

Frontend: React with Vite
Backend: FastAPI
Model: Fine-tuned BERT
XAI: LIME

The system allows users to enter a comment and returns:

predicted toxicity labels
probability scores for all six labels
tuned decision thresholds
LIME important words
one-sentence XAI explanation
How to Run the Backend

Open a terminal:

cd backend
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload

Backend URL:

http://127.0.0.1:8000

Health check:

http://127.0.0.1:8000/health
How to Run the Frontend

Open another terminal:

cd frontend
npm install
npm run dev

If PowerShell blocks npm, use:

npm.cmd install
npm.cmd run dev

Frontend URL:

http://localhost:5173
Example Inputs
Non-toxic Example
I hope you have a nice day.

Expected behavior:
No toxicity label detected.

Toxic / Threat Example
I will kill you

Expected behavior:
Predicted labels may include toxic, threat, and insult.
The XAI explanation should highlight words such as "kill".
XAI Explanation Example

For the input:
I will kill you

The system may generate an explanation such as:
LIME indicates that words such as "kill" increased the threat prediction, which explains why the model classified the comment as threatening.
Limitations

The model performs better on explicit toxic language than subtle or indirect bullying. Comments that require deeper conversational context, sarcasm understanding, or implicit intent may not always be detected.

The system is a proof-of-concept and should not be used as a final moderation decision tool.

Technologies Used
Python
PyTorch
Transformers
Scikit-learn
LIME
SHAP
FastAPI
React
Vite
Pandas
Matplotlib

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

