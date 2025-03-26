import streamlit as st
import requests
import pyrebase
import json
import time
import pandas as pd
import logging
from streamlit.components.v1 import html
from textblob import TextBlob  # For spelling correction

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Custom CSS for improved UI
st.markdown("""
    <style>
        /* Remove all default Streamlit headers/spacing */
        .stApp > header {
            display: none !important;
        }
        .stApp {
            margin-top: -50px !important;
            padding-top: 0 !important;
        }
        
        /* Fixed input container */
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
        
        /* Chat history container */
        .chat-history-container {
            margin-top: 100px;
            padding: 1rem;
            overflow-y: auto;
            max-height: calc(100vh - 150px);
        }
        
        /* Custom title styling */
        .custom-title {
            margin: 70px 0 10px 1rem !important;
            padding: 0 !important;
        }
        
        /* Auth button styling */
        .auth-button {
            width: 100%;
            margin: 5px 0;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
        }
        
        /* Mobile optimization */
        @media (max-width: 768px) {
            .chat-history-container {
                margin-top: 80px;
                max-height: calc(100vh - 130px);
            }
        }
        
        /* Recommendation button styling */
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

# Expanded AI/ML Keywords List
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
    """Correct spelling in the given text"""
    blob = TextBlob(text)
    corrected = str(blob.correct())
    if corrected.lower() != text.lower():  # Only return if correction was made
        return corrected
    return text

def is_ai_ml_related(question: str) -> bool:
    """Check if question is related to AI/ML topics with spelling correction"""
    corrected_question = correct_spelling(question)
    if corrected_question.lower() != question.lower():
        st.info(f"🔍 Did you mean: '{corrected_question}'?")
        question = corrected_question
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in AI_ML_KEYWORDS)

def parse_firebase_error(e):
    try:
        error_json = json.loads(e.args[1])
        error_message = error_json['error']['message']
        errors = {
            "EMAIL_NOT_FOUND": "Email not found. Please sign up first.",
            "INVALID_PASSWORD": "Incorrect password. Please try again.",
            "EMAIL_EXISTS": "This email is already registered.",
            "TOO_MANY_ATTEMPTS_TRY_LATER": "Too many failed attempts. Try again later."
        }
        return errors.get(error_message, "Authentication error. Please try again.")
    except Exception as parse_error:
        logging.error(f"Error parsing Firebase error: {parse_error}")
        return "An unexpected error occurred. Please try again."

def google_sign_in():
    google_sign_in_html = """
    <script src="https://accounts.google.com/gsi/client" async defer></script>
    <div id="g_id_onload"
         data-client_id="1032407725286-50mpttmbjtojch9qbvn011jt1sej5c80.apps.googleusercontent.com"
         data-callback="handleCredentialResponse">
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
    """
    html(google_sign_in_html, height=50)

def handle_google_sign_in(credential):
    try:
        user = auth.sign_in_with_google(credential)
        st.session_state.update({
            "user_token": user["idToken"],
            "user_email": user["email"],
            "chat_history": [],
            "last_activity": time.time(),
            "show_recommendation": False,
            "current_topic": ""
        })
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"❌ Error: {parse_firebase_error(e)}")

def animate_response(response):
    placeholder = st.empty()
    animated_text = ""
    paragraphs = response.split('\n\n')
    for para in paragraphs:
        words = para.split()
        for word in words:
            animated_text += word + " "
            placeholder.markdown(animated_text + "▌", unsafe_allow_html=True)
            time.sleep(0.05)
        animated_text += "\n\n"
        placeholder.markdown(animated_text + "▌", unsafe_allow_html=True)
    placeholder.markdown(animated_text, unsafe_allow_html=True)

def save_chat_to_firebase(user_email, chat_history):
    try:
        db.child("chats").child(user_email.replace(".", "_")).set(chat_history)
    except Exception as e:
        logging.error(f"Error saving chat history: {e}")

def auto_scroll_script():
    scroll_js = """
    <script>
    function scrollToBottom() {
        const chatContainer = document.querySelector('.chat-history-container');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }
    // Initial scroll with delay for rendering
    setTimeout(scrollToBottom, 300);
    // Create observer for dynamic content
    const observer = new MutationObserver(() => {
        scrollToBottom();
    });
    observer.observe(chatContainer, {
        childList: true,
        subtree: true
    });
    </script>
    """
    html(scroll_js, height=0)

def generate_recommendation_buttons(topic):
    """Generate buttons for study recommendations"""
    st.markdown("""
    <div style="margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
        <p><strong>Would you like me to suggest a learning path for this topic?</strong></p>
        <button class="recommendation-btn" onclick="Streamlit.setComponentValue('beginner')">Beginner Level</button>
        <button class="recommendation-btn" onclick="Streamlit.setComponentValue('intermediate')">Intermediate Level</button>
        <button class="recommendation-btn" onclick="Streamlit.setComponentValue('advanced')">Advanced Level</button>
    </div>
    """, unsafe_allow_html=True)
    
    # Handle button clicks
    button_value = html(
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
    
    if button_value in ['beginner', 'intermediate', 'advanced']:
        return get_study_recommendation(topic, button_value)
    return None

def get_study_recommendation(topic, level):
    """Generate study recommendations based on topic and level"""
    recommendations = {
        "beginner": f"""
        ### Beginner Learning Path for {topic}:
        1. **Introduction to Concepts**: Start with basic definitions and applications
        2. **Fundamental Math**: Review relevant linear algebra and statistics
        3. **Simple Implementations**: Try basic implementations with scikit-learn
        4. **Online Courses**: Take introductory courses on Coursera/edX
        5. **Practice Projects**: Work on small datasets to apply concepts
        """,
        "intermediate": f"""
        ### Intermediate Learning Path for {topic}:
        1. **Deep Dive**: Study algorithms and architectures in detail
        2. **Advanced Implementations**: Work with TensorFlow/PyTorch
        3. **Research Papers**: Read seminal papers in the field
        4. **Kaggle Competitions**: Participate in relevant competitions
        5. **Optimization Techniques**: Learn hyperparameter tuning and model optimization
        """,
        "advanced": f"""
        ### Advanced Learning Path for {topic}:
        1. **Current Research**: Follow latest arXiv papers and conferences
        2. **Custom Architectures**: Design and implement novel solutions
        3. **Production Deployment**: Learn MLOps and model serving
        4. **Specialization**: Focus on niche applications or optimizations
        5. **Contribution**: Contribute to open-source projects or publish research
        """
    }
    return recommendations.get(level, "No recommendation available.")

def main_chat_interface():
    st.write(f"👋 Welcome, {st.session_state.user_email}!")
    
    # Fixed input at top
    with st.container():
        st.markdown('<div class="fixed-input-container">', unsafe_allow_html=True)
        user_message = st.text_input("Ask me anything:", key="user_input", 
                                   on_change=lambda: st.session_state.update(process_input=True))
        
        if st.button("Get Answer") or st.session_state.get("process_input"):
            process_input()
        st.markdown('</div>', unsafe_allow_html=True)

    # Scrollable chat history below
    with st.container():
        if st.session_state.chat_history:
            st.markdown('<div class="chat-history-container">', unsafe_allow_html=True)
            for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
                st.markdown(f"**👤 You:** {user_msg}")
                st.markdown(f"**🤖 AI Tutor:**  \n{bot_msg}", unsafe_allow_html=True)
                
                # Check if this is the last message and we should show recommendations
                if i == len(st.session_state.chat_history) - 1 and st.session_state.get("show_recommendation", False):
                    rec_response = generate_recommendation_buttons(st.session_state.current_topic)
                    if rec_response:
                        st.session_state.chat_history.append(("Study recommendation request", rec_response))
                        st.session_state.show_recommendation = False
                        st.rerun()
                
                st.markdown("---")
            st.markdown('</div>', unsafe_allow_html=True)
            auto_scroll_script()

def process_input():
    user_message = st.session_state.get("user_input", "")
    if user_message:
        try:
            # Check if question is AI/ML related with spelling correction
            if not is_ai_ml_related(user_message):
                st.warning("⚠️ This chatbot specializes in AI/ML topics. While I can still answer, I recommend asking about AI, Machine Learning, or Data Science.")
            
            headers = {"Authorization": f"Bearer {st.session_state.user_token}"}
            response = requests.post(API_URL, json={"user_message": user_message}, headers=headers, verify=False)
            
            if response.status_code == 200:
                bot_response = response.json().get("response", "No response available.")
                formatted_response = bot_response.replace('\n', '\n\n')
                animate_response(formatted_response)
                st.session_state.chat_history.append((user_message, formatted_response))
                
                # Check if we should offer study recommendations
                if any(keyword in user_message.lower() for keyword in ["what is", "explain", "how to learn", "about"]):
                    st.session_state.show_recommendation = True
                    st.session_state.current_topic = user_message.split("what is")[-1].split("explain")[-1].strip(" ?")
                
                save_chat_to_firebase(st.session_state.user_email, st.session_state.chat_history)
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
        except Exception as e:
            logging.error("Chatbot request failed", exc_info=True)
            st.error("❌ Failed to connect to the chatbot service.")
        finally:
            st.session_state.process_input = False

# Main App Logic
st.title("🎓 AI Tutor Chatbot")

# Initialize session state variables if they don't exist
if "user_token" not in st.session_state:
    st.session_state.update({
        "chat_history": [],
        "show_recommendation": False,
        "current_topic": ""
    })

# Handle Google Sign-In response
if 'credential' in st.session_state:
    handle_google_sign_in(st.session_state.credential)

if "user_token" not in st.session_state:
    with st.sidebar:
        st.header("Authentication")
        
        # Separate buttons for each auth option
        col1, col2 = st.columns(2)
        with col1:
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
        
        with col2:
            auth_option = st.radio("Email Auth", ["Login", "Sign Up"], horizontal=True)
        
        # Email-based authentication
        if auth_option:
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
                            "current_topic": ""
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