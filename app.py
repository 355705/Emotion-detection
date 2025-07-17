import streamlit as st
import pandas as pd
import joblib
import json
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import os
import emoji
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Setup logging
logging.basicConfig(filename='streamlit.log', level=logging.INFO,
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
        tokens = text.split()
        if len(tokens) <= 2:  # Skip stop words for short texts
            filtered_words = tokens
        else:
            filtered_words = [word for word in tokens if word not in stop_words]
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
        processed_text = " ".join(lemmatized_words)
        logging.debug(f"Processed text: {processed_text}")
        return processed_text
    except Exception as e:
        logging.error(f"Error in preprocessing: {str(e)}")
        return ""

# Load models and vectorizer
@st.cache_resource
def load_models_and_vectorizer():
    models = {}
    model_files = [
        ("Support Vector Machine (Linear)", "best_model_support_vector_machine_(linear).pkl"),
        ("Logistic Regression", "best_model_logistic_regression.pkl"),
        ("Random Forest", "best_model_random_forest.pkl"),
        ("Naive Bayes", "best_model_naive_bayes.pkl"),
        ("Gradient Boosting", "best_model_gradient_boosting.pkl"),
        ("Tuned Best Model", "tuned_best_model.pkl")
    ]
    for model_name, model_file in model_files:
        try:
            if os.path.exists(model_file):
                model = joblib.load(model_file)
                if hasattr(model, 'predict'):
                    models[model_name] = model
                    logging.info(f"Loaded {model_name} from {model_file}")
                else:
                    logging.error(f"Model {model_name} does not have predict method")
            else:
                logging.error(f"Model file {model_file} not found")
        except Exception as e:
            logging.error(f"Error loading {model_name}: {str(e)}")

    vectorizer = None
    try:
        if os.path.exists("tfidf_vectorizer.pkl"):
            vectorizer = joblib.load("tfidf_vectorizer.pkl")
            logging.info("Loaded TF-IDF vectorizer")
        else:
            logging.error("TF-IDF vectorizer file not found")
    except Exception as e:
        logging.error(f"Error loading vectorizer: {str(e)}")

    return models, vectorizer

# Load metrics
@st.cache_data
def load_metrics():
    try:
        if os.path.exists("model_metrics.json"):
            with open("model_metrics.json", "r") as f:
                metrics = json.load(f)
            df = pd.DataFrame(metrics)
            df = df[['name', 'f1_score', 'accuracy', 'training_time']]
            df.columns = ['Model', 'F1 Score', 'Accuracy', 'Training Time (s)']
            logging.info("Loaded model metrics")
            return df
        else:
            logging.error("Metrics file model_metrics.json not found")
            return pd.DataFrame(columns=['Model', 'F1 Score', 'Accuracy', 'Training Time (s)'])
    except Exception as e:
        logging.error(f"Error loading metrics: {str(e)}")
        return pd.DataFrame(columns=['Model', 'F1 Score', 'Accuracy', 'Training Time (s)'])

# Streamlit app
def main():
    st.set_page_config(page_title="Emotion Detection Dashboard", layout="wide")

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px;}
    .stButton>button:hover {background-color: #45a049;}
    .sidebar .sidebar-content {background-color: #ffffff;}
    h1, h2, h3 {color: #2c3e50;}
    </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.title("🎭 Emotion Detection Dashboard")
    st.markdown("""
    Welcome to the Emotion Detection Dashboard! Analyze emotions in text using advanced machine learning models.  
    - **Enter a text** or **upload a file** to predict emotions.  
    - Explore **model performance** and **visualizations** with interactive buttons.  
    """)

    # Sidebar for file status and controls
    with st.sidebar:
        st.header("📊 Dashboard Controls")
        st.subheader("File Status")
        model_files = [
            "best_model_support_vector_machine_(linear).pkl",
            "best_model_logistic_regression.pkl",
            "best_model_random_forest.pkl",
            "best_model_naive_bayes.pkl",
            "best_model_gradient_boosting.pkl",
            "tuned_best_model.pkl",
            "tfidf_vectorizer.pkl",
            "model_metrics.json"
        ]
        for file in model_files:
            if os.path.exists(file):
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file)).strftime('%Y-%m-%d %H:%M:%S')
                st.write(f"✅ {file} (Last modified: {file_mtime})")
            else:
                st.write(f"❌ {file} (Not found)")

        # Button to show metrics table
        show_metrics = st.button("📈 Show Model Metrics Table")

    # Load models and vectorizer
    models, vectorizer = load_models_and_vectorizer()
    metrics_df = load_metrics()

    # Input section
    st.header("✍️ Enter Text for Emotion Prediction")
    input_text = st.text_area("Type your text here:", height=100, placeholder="e.g., I am so happy today! 😊")
    predict_button = st.button("🔍 Predict Emotion")

    # File upload section
    st.header("📂 Upload Text File")
    uploaded_file = st.file_uploader("Upload a .txt file with multiple texts (one per line)", type="txt")
    
    # Prediction results
    if predict_button and input_text:
        st.header("🔮 Prediction Results")
        processed_text = preprocess(input_text)
        if len(input_text.split()) <= 2:
            st.warning("⚠️ Input text is very short, predictions may be less accurate.")
        if not processed_text:
            st.warning("⚠️ Processed text is empty, using original text for prediction.")
            processed_text = input_text.lower()
        try:
            X_tfidf = vectorizer.transform([processed_text])
            predictions = []
            for model_name, model in models.items():
                try:
                    prediction = model.predict(X_tfidf)[0]
                    prob_dict = {}
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(X_tfidf)[0]
                        classes = model.classes_
                        prob_dict = {cls: round(prob, 2) for cls, prob in zip(classes, probabilities)}
                    predictions.append({
                        'Model': model_name,
                        'Predicted Emotion': prediction,
                        'Probabilities': prob_dict
                    })
                    logging.info(f"Prediction for {model_name}: {prediction}, Probabilities: {prob_dict}")
                except Exception as e:
                    predictions.append({
                        'Model': model_name,
                        'Predicted Emotion': f"Error: {str(e)}",
                        'Probabilities': {}
                    })
                    logging.error(f"Error predicting with {model_name}: {str(e)}")

            # Display predictions in a table
            pred_df = pd.DataFrame(predictions)
            pred_df['Probabilities'] = pred_df['Probabilities'].apply(
                lambda x: ', '.join([f"{k}: {v}" for k, v in x.items()]) if x else 'N/A')
            st.table(pred_df[['Model', 'Predicted Emotion', 'Probabilities']])

            # Pie chart for Tuned Best Model probabilities
            tuned_model = models.get("Tuned Best Model")
            if tuned_model and hasattr(tuned_model, 'predict_proba'):
                probs = tuned_model.predict_proba(X_tfidf)[0]
                prob_df = pd.DataFrame({
                    'Emotion': tuned_model.classes_,
                    'Probability': probs
                })
                fig = px.pie(prob_df, names='Emotion', values='Probability', 
                             title='Emotion Probabilities (Tuned Best Model)',
                             color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig, use_container_width=True)

            # Download predictions as CSV
            csv = pred_df.to_csv(index=False)
            st.download_button("💾 Download Predictions as CSV", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"🚨 Error in vectorization or prediction: {str(e)}")
            logging.error(f"Error in vectorization or prediction: {str(e)}")

    # File upload predictions
    if uploaded_file:
        st.header("📋 Batch Prediction Results")
        try:
            texts = uploaded_file.read().decode("utf-8").splitlines()
            predictions = []
            for text in texts:
                processed = preprocess(text)
                if processed:
                    X_tfidf = vectorizer.transform([processed])
                    pred = models["Tuned Best Model"].predict(X_tfidf)[0]
                    predictions.append({"Text": text, "Predicted Emotion": pred})
                else:
                    predictions.append({"Text": text, "Predicted Emotion": "Empty after processing"})
            batch_df = pd.DataFrame(predictions)
            st.table(batch_df)
            # Download batch predictions
            csv = batch_df.to_csv(index=False)
            st.download_button("💾 Download Batch Predictions as CSV", csv, "batch_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"🚨 Error processing file: {str(e)}")
            logging.error(f"Error processing file: {str(e)}")

    # Model performance section
    st.header("📊 Model Performance")
    if not metrics_df.empty:
        # Interactive metric selection for bar chart
        metric = st.selectbox("Select Metric to Display:", ['F1 Score', 'Accuracy', 'Training Time (s)'])
        fig = px.bar(metrics_df, x='Model', y=metric, title=f'Model {metric} Comparison',
                     color='Model', text=metric, color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.update_traces(texttemplate='%{text:.4f}', textposition='auto')
        st.plotly_chart(fig, use_container_width=True)

        # Show metrics table on button click
        if show_metrics:
            st.subheader("Detailed Model Metrics")
            st.dataframe(metrics_df.style.format({
                'F1 Score': '{:.4f}',
                'Accuracy': '{:.4f}',
                'Training Time (s)': '{:.2f}'
            }))

    else:
        st.warning("⚠️ No metrics available. Please run the training script first.")

    # Button to show confusion matrix
    st.header("🧠 Advanced Visualizations")
    if st.button("🗺️ Show Confusion Matrix (Tuned Best Model)"):
        confusion_matrix_file = "confusion_matrix_tuned_best_model.png"
        if os.path.exists(confusion_matrix_file):
            st.image(confusion_matrix_file, caption="Confusion Matrix for Tuned Best Model", use_container_width =True)
        else:
            st.warning("⚠️ Confusion matrix image not found. Please run the training script.")

if __name__ == "__main__":
    main()