import streamlit as st
import requests
import pyrebase
import json
import time
import pandas as pd
import logging
from streamlit.components.v1 import html
from textblob import TextBlob
from fuzzywuzzy import fuzz, process  # Added for fuzzy matching

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Custom CSS for improved UI (unchanged)
st.markdown("""
    <style>
        /* All existing CSS remains exactly the same */
        .stApp > header {
            display: none !important;
        }
        .stApp {
            margin-top: -50px !important;
            padding-top: 0 !important;
        }
        .fixed-input-container {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: white;
            padding: 1rem;
            z-index: 999;
            border-bottom: 1px solid #e0e0e0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .chat-history-container {
            margin-top: 100px;
            padding: 1rem;
            overflow-y: auto;
            max-height: calc(100vh - 150px);
        }
        .custom-title {
            margin: 70px 0 10px 1rem !important;
            padding: 0 !important;
        }
        .auth-button {
            width: 100%;
            margin: 5px 0;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
        }
        @media (max-width: 768px) {
            .chat-history-container {
                margin-top: 80px;
                max-height: calc(100vh - 130px);
            }
        }
        .recommendation-btn {
            background-color: #f0f2f6;
            color: #1e88e5;
            border: 1px solid #1e88e5;
            border-radius: 4px;
            padding: 8px 12px;
            margin: 5px 0;
            cursor: pointer;
        }
        .recommendation-btn:hover {
            background-color: #e3f2fd;
        }
        .follow-up-question {
            background-color: #f0f2f6;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 8px 12px;
            margin: 5px 0;
            cursor: pointer;
        }
        .follow-up-question:hover {
            background-color: #e3f2fd;
        }
    </style>
""", unsafe_allow_html=True)

# Firebase Configuration (unchanged)
firebase_config = {
    "apiKey": "AIzaSyB2tpQPqv35WdPNP2MgFlM7rE6SYeVUVtI",
    "authDomain": "aitutorbot-bb549.firebaseapp.com",
    "databaseURL": "https://aitutorbot-bb549-default-rtdb.firebaseio.com",
    "projectId": "aitutorbot-bb549",
    "storageBucket": "aitutorbot-bb549.appspot.com",
    "messagingSenderId": "1032407725286",
    "appId": "1:1032407725286:web:1285bb2cf87f8613497727",
    "measurementId": "G-ZZ3YL5M41C"
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
db = firebase.database()

API_URL = "https://ai-tutor-chatbot-fkjr.onrender.com/chat"

# AI/ML Keywords List (unchanged, keeping all original keywords)
AI_ML_KEYWORDS = [
    # Core concepts
    "ai", "ml", "machine learning", "deep learning", "artificial intelligence", "neural networks", 
    "computer vision", "natural language processing", "nlp", "reinforcement learning",
    "supervised learning", "unsupervised learning", "semi-supervised learning",
    "transfer learning", "ensemble learning", "active learning", "online learning",
    "feature engineering", "model training", "hyperparameter tuning", "overfitting",
    "underfitting", "bias-variance", "regularization", "optimization", "gradient descent",
    "backpropagation", "feedforward", "convolution", "attention", "transformer",
    
    # Algorithms & Techniques
    "linear regression", "logistic regression", "decision trees", "random forest",
    "svm", "support vector machine", "k-means", "knn", "k-nearest neighbors", 
    "naive bayes", "xgboost", "lightgbm", "catboost", "boosting", "bagging",
    "cnn", "convolutional neural network", "rnn", "recurrent neural network", 
    "lstm", "long short-term memory", "transformer", "gan", "generative adversarial network",
    "autoencoder", "attention mechanism", "self-attention", "vae", "variational autoencoder",
    
    # Applications
    "predictive modeling", "pattern recognition", "anomaly detection", "recommendation systems",
    "time series analysis", "sentiment analysis", "object detection", "speech recognition",
    "text generation", "image generation", "data mining", "predictive analytics",
    "fraud detection", "chatbot development", "autonomous vehicles", "robotics",
    "face recognition", "text classification", "named entity recognition", "ner",
    "machine translation", "question answering", "summarization", "image segmentation",
    
    # Tools & Frameworks
    "python", "scikit-learn", "sklearn", "tensorflow", "pytorch", "keras", "opencv", 
    "nltk", "spacy", "huggingface", "transformers", "pandas", "numpy", "matplotlib", 
    "seaborn", "jupyter", "colab", "google colab", "mlflow", "kubeflow", "airflow", 
    "docker", "fastapi", "streamlit", "flask", "django", "plotly", "dash",
    
    # Data Concepts
    "data science", "data engineering", "data preprocessing", "data cleaning",
    "feature selection", "dimensionality reduction", "pca", "principal component analysis",
    "eda", "exploratory data analysis", "data augmentation", "cross-validation",
    "train-test split", "data pipeline", "big data", "data warehouse", "data lake",
    "feature extraction", "data normalization", "data scaling", "one-hot encoding",
    "label encoding", "imputation", "missing data", "outlier detection",
    
    # Advanced Topics
    "generative ai", "llm", "large language model", "gpt", "bert", "stable diffusion", 
    "graph neural networks", "meta learning", "few-shot learning", "zero-shot learning",
    "self-supervised learning", "quantum machine learning", "explainable ai", "xai",
    "ai ethics", "mlops", "model deployment", "model monitoring", "model serving",
    "feature store", "model registry", "hyperparameter optimization", "neural architecture search",
    "federated learning", "differential privacy", "adversarial attacks", "model robustness",
    
    # Mathematical Foundations
    "linear algebra", "calculus", "statistics", "probability", "bayesian inference",
    "information theory", "algorithm complexity", "numerical methods", "activation function",
    "sigmoid", "relu", "tanh", "softmax", "loss function", "cross entropy", "mse",
    "mean squared error", "gradient", "derivative", "matrix", "vector", "eigenvalue",
    "eigenvector", "probability distribution", "normal distribution", "bayes theorem",
    
    # Industry Terms
    "ai model", "ml pipeline", "model inference", "model serving", "feature store",
    "model registry", "hyperparameter optimization", "neural architecture search",
    "data drift", "concept drift", "model retraining", "continuous integration",
    "continuous deployment", "ci/cd", "ab testing", "model versioning",
    
    # New additions
    "attention mechanism", "self-supervised learning", "contrastive learning",
    "knowledge distillation", "model pruning", "quantization", "onnx",
    "tensorrt", "coreml", "tf lite", "pytorch lightning", "ray tune",
    "optuna", "hyperopt", "automl", "auto-sklearn", "tpot", "h2o.ai",
    "data labeling", "active learning", "weak supervision", "snorkel",
    "label studio", "prodigy", "data version control", "dvc",
    "feature importance", "shap", "lime", "partial dependence plots",
    "model interpretability", "fairness metrics", "bias detection"
]

def correct_spelling(text):
    """Correct spelling in the given text without showing the correction message"""
    blob = TextBlob(text)
    return str(blob.correct())

def is_ai_ml_related(question: str, threshold: int = 75) -> bool:
    """
    Uses fuzzy matching to check if question is related to AI/ML topics
    threshold: 0-100, higher means more strict matching
    """
    question = question.lower()
    
    # First check for exact matches (more efficient)
    if any(keyword in question for keyword in AI_ML_KEYWORDS):
        return True
        
    # Then use fuzzy matching for variations
    best_match, score = process.extractOne(question, AI_ML_KEYWORDS)
    if score >= threshold:
        return True
        
    return False

def validate_response(question: str, response: str) -> bool:
    """
    Uses fuzzy matching to validate if response is relevant to question
    """
    # Check if any important keywords from question appear in response
    question_keywords = [word for word in question.split() if len(word) > 3]  # Only consider longer words
    
    if not question_keywords:
        return True
        
    # Check if response contains any of the question keywords with fuzzy matching
    for keyword in question_keywords:
        if fuzz.partial_ratio(keyword.lower(), response.lower()) > 75:
            return True
            
    return False

# All other existing functions remain exactly the same (parse_firebase_error, google_sign_in, 
# handle_google_sign_in, animate_response, save_chat_to_firebase, auto_scroll_script,
# show_follow_up_questions, main_chat_interface, process_study_plan_request)

def process_input():
    user_message = st.session_state.get("user_input", "")
    if user_message:
        try:
            # Correct spelling internally first
            corrected_message = correct_spelling(user_message)
            if corrected_message != user_message:
                user_message = corrected_message
            
            # Check if question is AI/ML related using fuzzy matching
            if not is_ai_ml_related(user_message):
                st.warning("⚠️ This chatbot specializes in AI/ML topics. I'll try to answer, but for best results, ask about AI, Machine Learning, or Data Science.")
            
            headers = {"Authorization": f"Bearer {st.session_state.user_token}"}
            response = requests.post(API_URL, 
                                    json={"user_message": user_message}, 
                                    headers=headers, 
                                    verify=False)
            
            if response.status_code == 200:
                bot_response = response.json().get("response", "No response available.")
                
                # Validate the response using fuzzy matching
                if not validate_response(user_message, bot_response):
                    # If response seems irrelevant, get a more focused answer
                    focused_response = requests.post(API_URL, 
                                                  json={
                                                      "user_message": f"Please answer this specifically about AI/ML: {user_message}",
                                                      "context": "You are an AI tutor. Provide a technical answer focused on artificial intelligence and machine learning."
                                                  }, 
                                                  headers=headers, 
                                                  verify=False)
                    if focused_response.status_code == 200:
                        bot_response = focused_response.json().get("response", bot_response)
                
                formatted_response = bot_response.replace('\n', '\n\n')
                animate_response(formatted_response)
                st.session_state.chat_history.append((user_message, formatted_response))
                save_chat_to_firebase(st.session_state.user_email, st.session_state.chat_history)
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
        except Exception as e:
            logging.error("Chatbot request failed", exc_info=True)
            st.error("❌ Failed to connect to the chatbot service.")
        finally:
            st.session_state.process_input = False

# Main App Logic (unchanged)
st.title("🎓 AI Tutor Chatbot")

# Initialize session state variables if they don't exist
if "user_token" not in st.session_state:
    st.session_state.update({
        "chat_history": [],
        "show_recommendation": False,
        "current_topic": "",
        "asked_for_study_plan": False
    })

# Handle Google Sign-In response
if 'credential' in st.session_state:
    handle_google_sign_in(st.session_state.credential)

if "user_token" not in st.session_state:
    with st.sidebar:
        st.header("Authentication")
        
        # Google Sign-In
        if st.button("Google Sign-In", key="google_signin_btn", help="Sign in with your Google account"):
            google_sign_in()
            google_data = html(
                """
                <script>
                window.addEventListener('message', (event) => {
                    if (event.data.type === 'streamlit:setComponentValue') {
                        Streamlit.setComponentValue(event.data.data);
                    }
                });
                </script>
                """, 
                height=0
            )
            if google_data and 'credential' in google_data:
                st.session_state.credential = google_data['credential']
                st.rerun()
        
        # Email-based authentication
        auth_option = st.radio("Email Auth", ["Login", "Sign Up"], horizontal=True)
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        
        if auth_option == "Sign Up":
            if st.button("Create Account", key="signup_btn"):
                try:
                    auth.create_user_with_email_and_password(email, password)
                    db.child("users").child(email.replace(".", "_")).set({
                        "email": email, 
                        "created_at": time.ctime()
                    })
                    st.success("✅ Account created! Please log in.")
                except Exception as e:
                    st.error(f"❌ Error: {parse_firebase_error(e)}")
        
        elif auth_option == "Login":
            if st.button("Login", key="login_btn"):
                try:
                    user = auth.sign_in_with_email_and_password(email, password)
                    st.session_state.update({
                        "user_token": user["idToken"],
                        "user_email": user["email"],
                        "chat_history": [],
                        "last_activity": time.time(),
                        "show_recommendation": False,
                        "current_topic": "",
                        "asked_for_study_plan": False
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {parse_firebase_error(e)}")

else:
    main_chat_interface()
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

# Session timeout handling
SESSION_TIMEOUT = 1800
if "last_activity" in st.session_state and time.time() - st.session_state.last_activity > SESSION_TIMEOUT:
    st.session_state.clear()
    st.sidebar.warning("Session expired. Please log in again.")

# Download Chat History
if "user_token" in st.session_state and st.sidebar.button("Download Chat History"):
    chat_df = pd.DataFrame(st.session_state.chat_history, columns=["User", "AI Tutor"])
    st.sidebar.download_button("📥 Download Chat", chat_df.to_csv(index=False), "chat_history.csv", "text/csv")