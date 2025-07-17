import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
import joblib
import time
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json
import os
import logging
import emoji

# Setup logging
logging.basicConfig(filename='training.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    logging.info("Downloaded NLTK stopwords and wordnet")

# Text preprocessing functions
def remove_punctuation(text):
    return ''.join(char for char in text if char not in string.punctuation)

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def convert_emojis(text):
    return emoji.demojize(text, delimiters=('', ''))

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    if not isinstance(text, str):
        logging.warning(f"Invalid input text: {text}")
        return ""
    try:
        text = convert_emojis(text)
        text = text.lower()
        text = remove_punctuation(text)
        text = remove_numbers(text)
        # Skipped correct_spelling to speed up processing
        tokens = text.split()
        filtered_words = [word for word in tokens if word not in stop_words]
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
        processed_text = " ".join(lemmatized_words)
        logging.debug(f"Processed text: {processed_text}")
        return processed_text
    except Exception as e:
        logging.error(f"Error in preprocessing: {str(e)}")
        return ""

# Clean old files
def clean_old_files():
    files_to_delete = [
        f for f in os.listdir() if f.endswith('.pkl') or f.endswith('.json') or f.endswith('.png')
    ]
    for f in files_to_delete:
        try:
            os.remove(f)
            logging.info(f"Deleted old file: {f}")
        except Exception as e:
            logging.error(f"Error deleting {f}: {str(e)}")

# Load data
logging.info("Loading data...")
try:
    train_data = pd.read_csv("train.txt", header=None, sep=";")
    val_data = pd.read_csv("val.txt", header=None, sep=";")
    test_data = pd.read_csv("test.txt", header=None, sep=";")
    
    train_data.rename(columns={0: "text", 1: "emotion"}, inplace=True)
    val_data.rename(columns={0: "text", 1: "emotion"}, inplace=True)
    test_data.rename(columns={0: "text", 1: "emotion"}, inplace=True)
    
    train_data = train_data.drop_duplicates()
    val_data = val_data.drop_duplicates()
    test_data = test_data.drop_duplicates()
    
    train_combined = pd.concat([train_data, val_data], ignore_index=True)
    
    logging.info("Preprocessing text...")
    train_combined["text"] = train_combined["text"].apply(preprocess)
    test_data["text"] = test_data["text"].apply(preprocess)
    
    # Remove empty texts
    train_combined = train_combined[train_combined["text"].str.strip() != ""]
    test_data = test_data[test_data["text"].str.strip() != ""]
    
    X_train = train_combined["text"].values
    y_train = train_combined["emotion"].values
    X_test = test_data["text"].values
    y_test = test_data["emotion"].values
    
    logging.info(f"Training data shape: {X_train.shape}")
    logging.info(f"Testing data shape: {X_test.shape}")
    logging.info(f"Emotion classes: {np.unique(y_train)}")
    
    print("\nClass distribution in training data:")
    print(train_combined["emotion"].value_counts())
    print("\nClass distribution in test data:")
    print(test_data["emotion"].value_counts())
    
except FileNotFoundError:
    logging.error("Data files not found: train.txt, val.txt, test.txt")
    print("Error: Data files not found. Please make sure train.txt, val.txt, and test.txt are in the current directory.")
    exit(1)
except Exception as e:
    logging.error(f"Error loading data: {str(e)}")
    print(f"Error loading data: {str(e)}")
    exit(1)

# Clean old files before saving new ones
clean_old_files()

# Feature extraction
logging.info("Extracting features with TF-IDF...")
try:
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
except Exception as e:
    logging.error(f"Error in TF-IDF vectorization: {str(e)}")
    print(f"Error in TF-IDF vectorization: {str(e)}")
    exit(1)

# Model training and evaluation
def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    logging.info(f"Training {model_name}...")
    try:
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        logging.info(f"Evaluating {model_name}...")
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"{model_name} Accuracy: {accuracy:.4f}")
        print(f"{model_name} F1 Score (weighted): {f1:.4f}")
        print(f"{model_name} Training Time: {training_time:.2f} seconds")
        
        return {
            'model': model,
            'name': model_name,
            'accuracy': accuracy,
            'f1_score': f1,
            'report': report,
            'predictions': y_pred,
            'training_time': training_time
        }
    except Exception as e:
        logging.error(f"Error training/evaluating {model_name}: {str(e)}")
        print(f"Error training/evaluating {model_name}: {str(e)}")
        return None

# Initialize models
models = [
    (SVC(kernel='linear', probability=True), "Support Vector Machine (Linear)"),
    (LogisticRegression(max_iter=1000, C=1.0, solver='liblinear'), "Logistic Regression"),
    (RandomForestClassifier(n_estimators=100), "Random Forest"),
    (MultinomialNB(), "Naive Bayes"),
    (GradientBoostingClassifier(n_estimators=100), "Gradient Boosting")
]

# Train and evaluate all models
results = []
for model, model_name in models:
    result = train_and_evaluate_model(model, model_name, X_train_tfidf, y_train, X_test_tfidf, y_test)
    if result:
        results.append(result)
        # Debug: Check model type before saving
        logging.info(f"Preparing to save {model_name}: Type={type(result['model'])}, Has predict={hasattr(result['model'], 'predict')}")
        print(f"Preparing to save {model_name}: Type={type(result['model'])}, Has predict={hasattr(result['model'], 'predict')}")
        # Save each model
        if hasattr(result['model'], 'predict'):
            try:
                joblib.dump(result['model'], f"best_model_{model_name.replace(' ', '_').lower()}.pkl")
                logging.info(f"Model saved as best_model_{model_name.replace(' ', '_').lower()}.pkl")
                print(f"Model saved as best_model_{model_name.replace(' ', '_').lower()}.pkl")
            except Exception as e:
                logging.error(f"Error saving {model_name}: {str(e)}")
                print(f"Error saving {model_name}: {str(e)}")
        else:
            logging.error(f"Model {model_name} does not have a predict method. Not saved.")
            print(f"Error: Model {model_name} does not have a predict method. Not saved.")

# Save metrics to JSON
try:
    metrics_to_save = [
        {
            'name': result['name'],
            'f1_score': result['f1_score'],
            'accuracy': result['accuracy'],
            'training_time': result['training_time']
        } for result in results
    ]
    with open("model_metrics.json", "w") as f:
        json.dump(metrics_to_save, f)
    logging.info("Model metrics saved as model_metrics.json")
    print("Model metrics saved as model_metrics.json")
except Exception as e:
    logging.error(f"Error saving metrics: {str(e)}")
    print(f"Error saving metrics: {str(e)}")

# Find the best model based on F1 score
try:
    best_model = max(results, key=lambda x: x['f1_score'])
    logging.info(f"Best Model: {best_model['name']} with F1 score {best_model['f1_score']:.4f}")
    print(f"\nBest Model: {best_model['name']} with F1 score {best_model['f1_score']:.4f}")
except Exception as e:
    logging.error(f"Error finding best model: {str(e)}")
    print(f"Error finding best model: {str(e)}")

# Save TF-IDF vectorizer
try:
    joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
    logging.info("TF-IDF vectorizer saved as tfidf_vectorizer.pkl")
    print("TF-IDF vectorizer saved as tfidf_vectorizer.pkl")
except Exception as e:
    logging.error(f"Error saving TF-IDF vectorizer: {str(e)}")
    print(f"Error saving TF-IDF vectorizer: {str(e)}")

# Detailed report for the best model
try:
    logging.info(f"Detailed Classification Report for {best_model['name']}:")
    print("\nDetailed Classification Report for the Best Model:")
    print(classification_report(y_test, best_model['predictions']))
except Exception as e:
    logging.error(f"Error generating classification report: {str(e)}")
    print(f"Error generating classification report: {str(e)}")

# Plot confusion matrix for the best model
try:
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, best_model['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {best_model["name"]}')
    plt.savefig(f'confusion_matrix_{best_model["name"].replace(" ", "_").lower()}.png')
    logging.info(f"Confusion matrix saved as confusion_matrix_{best_model['name'].replace(' ', '_').lower()}.png")
    print(f"Confusion matrix saved as confusion_matrix_{best_model['name'].replace(' ', '_').lower()}.png")
except Exception as e:
    logging.error(f"Error saving confusion matrix: {str(e)}")
    print(f"Error saving confusion matrix: {str(e)}")

# Compare model performances
try:
    plt.figure(figsize=(12, 6))
    f1_scores = [result['f1_score'] for result in results]
    model_names = [result['name'] for result in results]
    plt.bar(model_names, f1_scores)
    plt.xlabel('Model')
    plt.ylabel('F1 Score (weighted)')
    plt.title('Model F1 Score Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('model_comparison_f1.png')
    logging.info("Model F1 score comparison chart saved as model_comparison_f1.png")
    print("Model F1 score comparison chart saved as model_comparison_f1.png")
except Exception as e:
    logging.error(f"Error saving F1 score comparison: {str(e)}")
    print(f"Error saving F1 score comparison: {str(e)}")

# Compare training times
try:
    plt.figure(figsize=(12, 6))
    times = [result['training_time'] for result in results]
    plt.bar(model_names, times)
    plt.xlabel('Model')
    plt.ylabel('Training Time (seconds)')
    plt.title('Model Training Time Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('training_time_comparison.png')
    logging.info("Training time comparison chart saved as training_time_comparison.png")
    print("Training time comparison chart saved as training_time_comparison.png")
except Exception as e:
    logging.error(f"Error saving training time comparison: {str(e)}")
    print(f"Error saving training time comparison: {str(e)}")

# Hyperparameter tuning for the best model
logging.info(f"Performing hyperparameter tuning for {best_model['name']}...")
try:
    if best_model['name'] == "Support Vector Machine (Linear)":
        param_grid = {
            'C': [0.1, 1],
            'kernel': ['linear'],
            'gamma': ['scale']
        }
        tuned_model = SVC(probability=True)
    elif best_model['name'] == "Logistic Regression":
        param_grid = {
            'C': [0.1, 1],
            'solver': ['liblinear'],
            'penalty': ['l2']
        }
        tuned_model = LogisticRegression(max_iter=1000)
    elif best_model['name'] == "Random Forest":
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        }
        tuned_model = RandomForestClassifier()
    elif best_model['name'] == "Gradient Boosting":
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
        tuned_model = GradientBoostingClassifier()
    else:  # Naive Bayes
        param_grid = {
            'alpha': [0.5, 1.0]
        }
        tuned_model = MultinomialNB()

    grid_search = GridSearchCV(tuned_model, param_grid, cv=3, scoring='f1_weighted', n_jobs=2)
    grid_search.fit(X_train_tfidf, y_train)

    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    tuned_model = grid_search.best_estimator_
    y_tuned_pred = tuned_model.predict(X_test_tfidf)
    tuned_f1 = f1_score(y_test, y_tuned_pred, average='weighted')
    tuned_accuracy = accuracy_score(y_test, y_tuned_pred)

    logging.info(f"Tuned model F1 score: {tuned_f1:.4f}")
    logging.info(f"Tuned model Accuracy: {tuned_accuracy:.4f}")
    logging.info(f"Improvement over original model: {tuned_f1 - best_model['f1_score']:.4f}")
    print(f"Tuned model F1 score: {tuned_f1:.4f}")
    print(f"Tuned model Accuracy: {tuned_accuracy:.4f}")
    print(f"Improvement over original model: {tuned_f1 - best_model['f1_score']:.4f}")

    # Debug: Check tuned model type before saving
    logging.info(f"Preparing to save Tuned Best Model: Type={type(tuned_model)}, Has predict={hasattr(tuned_model, 'predict')}")
    print(f"Preparing to save Tuned Best Model: Type={type(tuned_model)}, Has predict={hasattr(tuned_model, 'predict')}")
    # Save tuned model
    if hasattr(tuned_model, 'predict'):
        try:
            joblib.dump(tuned_model, "tuned_best_model.pkl")
            logging.info("Tuned best model saved as tuned_best_model.pkl")
            print("Tuned best model saved as tuned_best_model.pkl")
        except Exception as e:
            logging.error(f"Error saving tuned model: {str(e)}")
            print(f"Error saving tuned model: {str(e)}")
    else:
        logging.error("Tuned model does not have a predict method. Not saved.")
        print("Error: Tuned model does not have a predict method. Not saved.")

    # Plot and save confusion matrix for Tuned Best Model
    try:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_tuned_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix - Tuned Best Model')
        plt.savefig('confusion_matrix_tuned_best_model.png')
        logging.info("Confusion matrix saved as confusion_matrix_tuned_best_model.png")
        print("Confusion matrix saved as confusion_matrix_tuned_best_model.png")
        plt.close()
    except Exception as e:
        logging.error(f"Error saving confusion matrix for Tuned Best Model: {str(e)}")
        print(f"Error saving confusion matrix for Tuned Best Model: {str(e)}")

    # Save tuned model metrics to JSON
    metrics_to_save.append({
        'name': 'Tuned Best Model',
        'f1_score': tuned_f1,
        'accuracy': tuned_accuracy,
        'training_time': None
    })
    with open("model_metrics.json", "w") as f:
        json.dump(metrics_to_save, f)
    logging.info("Updated model metrics saved as model_metrics.json")
    print("Updated model metrics saved as model_metrics.json")

except Exception as e:
    logging.error(f"Error in hyperparameter tuning: {str(e)}")
    print(f"Error in hyperparameter tuning: {str(e)}")

logging.info("Model comparison complete!")
print("\nModel comparison complete!")