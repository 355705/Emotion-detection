# Emotion-detection
# 🎭 Emotion Detection from Text 

This project is a complete NLP-based pipeline for detecting **emotions from textual input**. It includes preprocessing, model training, evaluation, optimization, and an interactive web dashboard built with **Streamlit**.

---

## 📌 Project Overview

- Preprocess raw text (including emojis).
- Train and compare multiple machine learning models.
- Evaluate performance using F1-Score, Accuracy, and Confusion Matrix.
- Deploy an interactive app for real-time and batch predictions.

---

## 🛠️ Features

### 1. Text Preprocessing
- Converts emojis to words (😊 → "smiling_face").
- Removes punctuation and numbers.
- Lowercases text and removes stopwords.
- Applies **lemmatization** using NLTK.

### 2. Model Training & Evaluation
- Models used:
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Random Forest
  - Naive Bayes
  - Gradient Boosting

- TF-IDF is used for text vectorization.
- Metrics:
  - Accuracy
  - F1 Score (weighted)
  - Confusion Matrix
  - Training Time

### 3. Model Optimization
- Uses `GridSearchCV` to tune the best performing model.
- Saves best models to `.pkl` files for reuse.
- Saves evaluation metrics to `model_metrics.json`.

### 4. Streamlit Dashboard
- Enter a single sentence or upload a `.txt` file.
- Real-time emotion prediction using all trained models.
- Visual outputs:
  - Prediction table with probabilities
  - Pie chart of tuned model predictions
  - Model performance comparisons
  - Confusion matrix display
- Supports downloading results as `.csv`.

---
## 📊 Example Output
Real-time text prediction with emotion label.

Confidence scores across models.

Pie chart for emotion probabilities.

Bar charts for model comparison.

Confusion matrix visualization.

---
## 📁 Folder Structure


emotion-detection-nlp/

    model.py              # Model training & evaluation script
    app.py                # Streamlit dashboard
    training.txt
    model_metrics.json    # Saved evaluation results
    *.pkl                 # Saved model/vectorizer files
    *.png                 # Visualization files
    train.txt / val.txt / test.txt

---
## 🧰 Technologies Used

Python

scikit-learn

NLTK

TF-IDF

Streamlit

Joblib

Matplotlib / Seaborn / Plotly



