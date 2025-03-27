import streamlit as st
import requests
import pyrebase
import json
import time
import pandas as pd
import logging
from streamlit.components.v1 import html
from textblob import TextBlob
from fuzzywuzzy import fuzz, process

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Custom CSS for improved UI
st.markdown("""
    <style>
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

# Firebase Configuration
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

# AI/ML Keywords List
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

def is_ai_ml_related(question: str, threshold: int = 80) -> bool:
    """Check if question is related to AI/ML topics with improved matching"""
    question = question.lower()
    
    # Check for exact matches with word boundaries
    if any(f' {keyword} ' in f' {question} ' for keyword in AI_ML_KEYWORDS):
        return True
        
    # Fuzzy matching with higher threshold
    best_match, score = process.extractOne(question, AI_ML_KEYWORDS)
    return score >= threshold

def validate_response(question: str, response: str) -> bool:
    """Validate response relevance using fuzzy matching"""
    question_keywords = [word for word in question.split() if len(word) > 3]
    
    if not question_keywords:
        return True
        
    for keyword in question_keywords:
        if fuzz.partial_ratio(keyword.lower(), response.lower()) > 75:
            return True
            
    return False

def parse_firebase_error(error):
    """Parse Firebase error messages"""
    error_str = str(error)
    if "INVALID_EMAIL" in error_str:
        return "Invalid email address format."
    elif "EMAIL_NOT_FOUND" in error_str:
        return "No account found with this email."
    elif "INVALID_PASSWORD" in error_str:
        return "Incorrect password."
    elif "EMAIL_EXISTS" in error_str:
        return "This email is already registered."
    elif "TOO_MANY_ATTEMPTS_TRY_LATER" in error_str:
        return "Too many attempts. Please try again later."
    elif "WEAK_PASSWORD" in error_str:
        return "Password should be at least 6 characters."
    else:
        return "An error occurred. Please try again."

def google_sign_in():
    """Render Google Sign-In button"""
    html_code = """
    <html>
    <head>
        <script src="https://accounts.google.com/gsi/client" async defer></script>
    </head>
    <body>
        <div id="g_id_onload"
            data-client_id="1032407725286-7v8mh2q5j7q3q3q3q3q3q3q3q3q3q.apps.googleusercontent.com"
            data-callback="handleCredentialResponse"
            data-auto_prompt="false">
        </div>
        <div class="g_id_signin"
            data-type="standard"
            data-size="large"
            data-theme="outline"
            data-text="sign_in_with"
            data-shape="rectangular"
            data-logo_alignment="left">
        </div>
        <script>
            function handleCredentialResponse(response) {
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    data: {credential: response.credential}
                }, '*');
            }
        </script>
    </body>
    </html>
    """
    html(html_code, height=70)

def handle_google_sign_in(credential):
    """Handle Google Sign-In response"""
    try:
        user = auth.sign_in_with_oauth_credential(credential)
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

def animate_response(response_text):
    """Animate the chatbot's response"""
    response_placeholder = st.empty()
    full_response = ""
    
    for chunk in response_text.split(" "):
        full_response += chunk + " "
        response_placeholder.markdown(full_response)
        time.sleep(0.05)
    
    return full_response

def save_chat_to_firebase(email, chat_history):
    """Save chat history to Firebase"""
    try:
        if not email or not chat_history:
            return
            
        email_key = email.replace(".", "_")
        db.child("chats").child(email_key).set({
            "email": email,
            "chat_history": json.dumps(chat_history),
            "last_updated": time.ctime()
        })
    except Exception as e:
        logging.error("Failed to save chat to Firebase", exc_info=True)

def auto_scroll_script():
    """Improved auto-scrolling JavaScript"""
    return """
    <script>
        function scrollToBottom() {
            const chatHistory = document.querySelector('.chat-history-container');
            if (chatHistory) {
                chatHistory.scrollTop = chatHistory.scrollHeight;
            }
        }
        
        // Initial scroll
        setTimeout(scrollToBottom, 100);
        
        // Create observer for dynamic content
        const observer = new MutationObserver(scrollToBottom);
        observer.observe(document.querySelector('.chat-history-container'), {
            childList: true,
            subtree: true
        });
        
        // Attach scroll on window load
        window.addEventListener('load', scrollToBottom);
    </script>
    """

def show_follow_up_questions(topic):
    """Display follow-up questions with proper state management"""
    follow_ups = {
        "machine learning": [
            "What are the different types of machine learning?",
            "Can you explain supervised vs unsupervised learning?",
            "How do I evaluate a machine learning model?"
        ],
        "deep learning": [
            "What's the difference between CNN and RNN?",
            "How does backpropagation work?",
            "What are some common activation functions?"
        ],
        "natural language processing": [
            "What are transformer models?",
            "How does tokenization work in NLP?",
            "What's the difference between BERT and GPT?"
        ],
        "data science": [
            "What's the typical data science workflow?",
            "How do I handle missing data?",
            "What are some common data visualization techniques?"
        ]
    }
    
    default_follow_ups = [
        "Can you explain this in simpler terms?",
        "What are some practical applications of this?",
        "How does this compare to similar concepts?"
    ]
    
    questions = follow_ups.get(topic.lower(), default_follow_ups)
    
    st.markdown("**Follow-up Questions:**")
    for q in questions:
        if st.button(q, key=f"followup_{hash(q)}", help="Click to ask this follow-up"):
            st.session_state.pending_question = q
            st.rerun()

def process_study_plan_request(topic):
    """Process study plan requests"""
    if not topic:
        return "Please specify a topic for the study plan."
    
    headers = {"Authorization": f"Bearer {st.session_state.user_token}"}
    try:
        response = requests.post(
            API_URL,
            json={
                "user_message": f"Create a detailed study plan for {topic}",
                "context": "You are an AI tutor creating a structured learning path."
            },
            headers=headers,
            verify=False
        )
        
        if response.status_code == 200:
            return response.json().get("response", "Failed to generate study plan.")
        return f"Error: {response.text}"
    except Exception as e:
        logging.error("Study plan request failed", exc_info=True)
        return "Failed to connect to the chatbot service."

def main_chat_interface():
    """Main chat interface with proper state handling"""
    st.markdown(f'<h2 class="custom-title">Welcome, {st.session_state.user_email.split("@")[0]}!</h2>', unsafe_allow_html=True)
    
    # Handle pending questions
    if 'pending_question' in st.session_state:
        st.session_state.user_input = st.session_state.pending_question
        del st.session_state.pending_question
    
    # Input container
    with st.container():
        st.markdown('<div class="fixed-input-container">', unsafe_allow_html=True)
        user_input = st.text_input(
            "Ask anything about AI, Machine Learning, or Data Science:",
            key="user_input",
            placeholder="Type your question here...",
            label_visibility="collapsed",
            value=st.session_state.get("user_input", "")
        )
        if st.button("Send"):
            st.session_state.process_input = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display warnings above chat history
    if st.session_state.get("show_warning", False):
        st.warning("⚠️ This chatbot specializes in AI/ML topics. For best results, ask about AI, Machine Learning, or Data Science.")
        st.session_state.show_warning = False
    
    # Chat history display
    with st.container():
        st.markdown('<div class="chat-history-container">', unsafe_allow_html=True)
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            
        for user_msg, bot_msg in st.session_state.chat_history:
            st.markdown(f"**You:** {user_msg}")
            st.markdown(f"**AI Tutor:** {bot_msg}")
            st.markdown("---")
            
        st.markdown('</div>', unsafe_allow_html=True)
        html(auto_scroll_script(), height=0)
    
    # Process input
    if st.session_state.get("process_input", False):
        process_input()
    
    # Recommendations and follow-ups
    if st.session_state.get("show_recommendation", False) and st.session_state.current_topic:
        topic = st.session_state.current_topic
        st.sidebar.markdown(f"**Topic:** {topic}")
        
        if st.sidebar.button("Generate Study Plan"):
            study_plan = process_study_plan_request(topic)
            st.session_state.chat_history.append((
                f"Please create a study plan for {topic}",
                study_plan
            ))
            st.rerun()
        
        show_follow_up_questions(topic)
    
    st.session_state.last_activity = time.time()

def process_input():
    """Process user input with safe state handling"""
    try:
        user_message = st.session_state.user_input.strip()
        if not user_message:
            return
            
        # Correct spelling
        corrected = correct_spelling(user_message)
        if corrected != user_message:
            st.session_state.user_input = corrected
            st.rerun()
        
        # AI/ML check
        if not is_ai_ml_related(user_message):
            st.session_state.show_warning = True
        
        # API call
        headers = {"Authorization": f"Bearer {st.session_state.user_token}"}
        response = requests.post(API_URL, 
                               json={"user_message": user_message}, 
                               headers=headers,
                               verify=False)
        
        if response.status_code == 200:
            bot_response = response.json().get("response", "")
            
            # Validate response
            if not validate_response(user_message, bot_response):
                focused_response = requests.post(API_URL, 
                                              json={
                                                  "user_message": f"Please answer this specifically about AI/ML: {user_message}",
                                                  "context": "Provide a technical AI/ML-focused answer."
                                              }, 
                                              headers=headers,
                                              verify=False)
                if focused_response.ok:
                    bot_response = focused_response.json().get("response", bot_response)
            
            formatted_response = bot_response.replace('\n', '\n\n')
            animate_response(formatted_response)
            st.session_state.chat_history.append((user_message, formatted_response))
            save_chat_to_firebase(st.session_state.user_email, st.session_state.chat_history)
            
            # Set topic
            if not st.session_state.current_topic:
                for keyword in AI_ML_KEYWORDS:
                    if keyword in user_message.lower():
                        st.session_state.current_topic = keyword
                        st.session_state.show_recommendation = True
                        break
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
        
    except Exception as e:
        logging.error("Processing failed", exc_info=True)
        st.error("❌ Processing failed")
    finally:
        st.session_state.process_input = False
        st.session_state.user_input = ""
        st.rerun()

# Main App Logic
st.title("🎓 AI Tutor Chatbot")

# Initialize session state
if "user_token" not in st.session_state:
    st.session_state.update({
        "chat_history": [],
        "show_recommendation": False,
        "current_topic": "",
        "asked_for_study_plan": False,
        "show_warning": False
    })

# Handle authentication
if 'credential' in st.session_state:
    handle_google_sign_in(st.session_state.credential)

if "user_token" not in st.session_state:
    with st.sidebar:
        st.header("Authentication")
        
        # Google Sign-In
        if st.button("Google Sign-In"):
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
        
        # Email auth
        auth_option = st.radio("Email Auth", ["Login", "Sign Up"], horizontal=True)
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        
        if auth_option == "Sign Up":
            if st.button("Create Account"):
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
            if st.button("Login"):
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

# Session timeout
SESSION_TIMEOUT = 1800
if "last_activity" in st.session_state and time.time() - st.session_state.last_activity > SESSION_TIMEOUT:
    st.session_state.clear()
    st.sidebar.warning("Session expired. Please log in again.")

# Download chat
if "user_token" in st.session_state and st.sidebar.button("Download Chat History"):
    chat_df = pd.DataFrame(st.session_state.chat_history, columns=["User", "AI Tutor"])
    st.sidebar.download_button("📥 Download Chat", chat_df.to_csv(index=False), "chat_history.csv", "text/csv")