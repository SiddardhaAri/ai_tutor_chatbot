import streamlit as st
import requests
import pyrebase
import json
import time
import pandas as pd
import logging
from streamlit.components.v1 import html
from fuzzywuzzy import fuzz

# Configure logging
logging.basicConfig(level=logging.ERROR)

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
        
        /* Recommendations styling */
        .recommendations-container {
            margin: 1rem;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
        }
        
        .recommendations-container button {
            width: 100%;
            margin: 5px 0;
            text-align: left;
            padding: 8px;
            border: 1px solid #e0e0e0;
            background: white;
            border-radius: 5px;
            cursor: pointer;
            white-space: normal;
            word-wrap: break-word;
        }
        
        .recommendations-container button:hover {
            background: #f0f0f0;
        }
        
        /* Mobile optimization */
        @media (max-width: 768px) {
            .chat-history-container {
                margin-top: 80px;
                max-height: calc(100vh - 130px);
            }
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

AI_ML_KEYWORDS = [
    # Core concepts
    "ai", "ml", "machine learning", "deep learning", "artificial intelligence", "neural networks", 
    "computer vision", "natural language processing", "reinforcement learning",
    "supervised learning", "unsupervised learning", "semi-supervised learning",
    "transfer learning", "ensemble learning", "active learning", "online learning",
    "feature engineering", "model training", "hyperparameter tuning", "overfitting",
    "underfitting", "bias-variance", "regularization", "optimization", "gradient descent",
    
    # Algorithms & Techniques
    "linear regression", "logistic regression", "decision trees", "random forest",
    "svm", "k-means", "knn", "naive bayes", "xgboost", "lightgbm", "catboost",
    "cnn", "rnn", "lstm", "transformer", "gan", "autoencoder", "attention mechanism",
    
    # Applications
    "predictive modeling", "pattern recognition", "anomaly detection", "recommendation systems",
    "time series analysis", "sentiment analysis", "object detection", "speech recognition",
    "text generation", "image generation", "data mining", "predictive analytics",
    "fraud detection", "chatbot development", "autonomous vehicles", "robotics",
    
    # Tools & Frameworks
    "python", "scikit-learn", "tensorflow", "pytorch", "keras", "opencv", "nltk",
    "spacy", "huggingface", "pandas", "numpy", "matplotlib", "seaborn", "jupyter",
    "colab", "mlflow", "kubeflow", "airflow", "docker", "fastapi",
    
    # Data Concepts
    "data science", "data engineering", "data preprocessing", "data cleaning",
    "feature selection", "dimensionality reduction", "pca", "eda", "data augmentation",
    "cross-validation", "train-test split", "data pipeline", "big data",
    
    # Advanced Topics
    "generative ai", "llm", "gpt", "bert", "stable diffusion", "graph neural networks",
    "meta learning", "few-shot learning", "self-supervised learning", "quantum machine learning",
    "explainable ai", "ai ethics", "mlops", "model deployment", "model monitoring",
    
    # Mathematical Foundations
    "linear algebra", "calculus", "statistics", "probability", "bayesian inference",
    "information theory", "algorithm complexity", "numerical methods", "activation function",
    
    # Industry Terms
    "ai model", "ml pipeline", "model inference", "model serving", "feature store",
    "model registry", "hyperparameter optimization", "neural architecture search"
]

def is_ai_ml_related(question: str) -> bool:
    question_lower = question.lower()
    for keyword in AI_ML_KEYWORDS:
        # Check using fuzzy partial ratio for approximate matches
        if fuzz.partial_ratio(keyword, question_lower) >= 80:
            return True
    return False

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
            "last_activity": time.time()
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
    
def get_recommendations():
    """Get last 3 unique previous questions (excluding current one)"""
    if len(st.session_state.chat_history) < 2:
        return []
    
    # Exclude the most recent question
    user_messages = [msg[0] for msg in st.session_state.chat_history[:-1]]
    
    seen = set()
    unique_messages = []
    for msg in reversed(user_messages):
        if msg not in seen:
            seen.add(msg)
            unique_messages.append(msg)
        if len(unique_messages) >= 3:
            break
    return list(reversed(unique_messages[-3:]))

def main_chat_interface():
    st.write(f"👋 Welcome, {st.session_state.user_email}!")
    
    # Fixed input at top
    with st.container():
        st.markdown('<div class="fixed-input-container">', unsafe_allow_html=True)
        user_message = st.text_input(
            "Ask me anything:", 
            key="user_input",
            value=st.session_state.pop("recommended_question", ""),
            on_change=lambda: st.session_state.update(process_input=True)
        )
        
        if st.button("Get Answer") or st.session_state.get("process_input"):
            process_input()
        st.markdown('</div>', unsafe_allow_html=True)

    # Chat history and recommendations
    with st.container():
        if st.session_state.chat_history:
            st.markdown('<div class="chat-history-container">', unsafe_allow_html=True)
            
            # Display all messages
            for idx, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
                st.markdown(f"**👤 You:** {user_msg}")
                st.markdown(f"**🤖 AI Tutor:**  \n{bot_msg}", unsafe_allow_html=True)
                
                # Show recommendations after latest response
                if idx == len(st.session_state.chat_history) - 1:
                    recommendations = get_recommendations()
                    if recommendations:
                        st.markdown('<div class="recommendations-container">', unsafe_allow_html=True)
                        st.markdown("**🔍 Recommended follow-up questions:**")
                        for rec_idx, question in enumerate(recommendations):
                            if st.button(question, key=f"rec_{rec_idx}_{idx}"):
                                st.session_state.recommended_question = question
                                st.session_state.process_input = True
                                st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("---")

            st.markdown('</div>', unsafe_allow_html=True)
            auto_scroll_script()

def process_input():
    user_message = st.session_state.get("user_input", "")
    if user_message:
        try:
            if not is_ai_ml_related(user_message):
                st.warning("⚠️ This chatbot specializes in AI/ML topics. While I can still answer, I recommend asking about AI, Machine Learning, or Data Science.")
            
            headers = {"Authorization": f"Bearer {st.session_state.user_token}"}
            response = requests.post(API_URL, json={"user_message": user_message}, headers=headers, verify=False)
            
            if response.status_code == 200:
                bot_response = response.json().get("response", "No response available.")
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

# Main App Logic
st.title("🎓 AI Tutor Chatbot")

# Handle Google Sign-In response
if 'credential' in st.session_state:
    handle_google_sign_in(st.session_state.credential)

if "user_token" not in st.session_state:
    with st.sidebar:
        st.header("Authentication")
        choice = st.selectbox("Choose Action", ["Login", "Sign Up", "Google Sign-In"])
        
        if choice == "Google Sign-In":
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
        
        elif choice in ["Login", "Sign Up"]:
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            
            if choice == "Sign Up":
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
            
            elif choice == "Login":
                if st.button("Login"):
                    try:
                        user = auth.sign_in_with_email_and_password(email, password)
                        st.session_state.update({
                            "user_token": user["idToken"],
                            "user_email": user["email"],
                            "chat_history": [],
                            "last_activity": time.time()
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