import streamlit as st
import requests
import pyrebase
import json
import time
import pandas as pd
import logging
from streamlit.components.v1 import html

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Force CSS with !important overrides
st.markdown("""
    <style>
        /* Fixed input container */
        div[data-testid="stVerticalBlock"] > div:has(> .element-container > .stTextInput) {
            position: fixed !important;
            bottom: 0 !important;
            left: 0 !important;
            right: 0 !important;
            background: white !important;
            padding: 1rem !important;
            z-index: 9999 !important;
            border-top: 1px solid #e0e0e0 !important;
            box-shadow: 0 -4px 6px -1px rgba(0,0,0,0.1) !important;
        }

        /* Chat history container */
        .chat-history {
            margin-bottom: 150px !important;
            height: calc(100vh - 200px) !important;
            overflow-y: auto !important;
            padding: 1rem !important;
        }

        /* Message styling */
        .message {
            margin: 1rem 0 !important;
            padding: 1rem !important;
            border-radius: 15px !important;
            max-width: 80% !important;
        }

        .user-message {
            background: #f0f2f6 !important;
            margin-left: auto !important;
        }

        .bot-message {
            background: #e3f2fd !important;
            margin-right: auto !important;
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px !important;
        }

        ::-webkit-scrollbar-track {
            background: #f1f1f1 !important;
        }

        ::-webkit-scrollbar-thumb {
            background: #888 !important;
            border-radius: 4px !important;
        }

        /* Input field styling */
        .stTextInput>div>div>input {
            padding: 12px 16px !important;
            border-radius: 25px !important;
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
    "machine learning", "deep learning", "neural networks", "NLP",
    "computer vision", "reinforcement learning", "AI", "ML", "Python",
    "scikit-learn", "PyTorch", "TensorFlow", "data science", "chatbot",
    "OpenAI", "LLM", "artificial intelligence", "data engineering",
    "feature engineering", "predictive modeling", "generative AI"
]

def is_ai_ml_related(question: str) -> bool:
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
            placeholder.markdown(f'<div class="bot-message message">🤖 AI Tutor: {animated_text}▌</div>', 
                               unsafe_allow_html=True)
            time.sleep(0.05)
        animated_text += "\n\n"
    placeholder.markdown(f'<div class="bot-message message">🤖 AI Tutor: {animated_text}</div>', 
                       unsafe_allow_html=True)

def save_chat_to_firebase(user_email, chat_history):
    try:
        db.child("chats").child(user_email.replace(".", "_")).set(chat_history)
    except Exception as e:
        logging.error(f"Error saving chat history: {e}")

def main_chat_interface():
    st.write(f"👋 Welcome, {st.session_state.user_email}!")
    
    # Chat History Container
    with st.container():
        st.markdown('<div class="chat-history">', unsafe_allow_html=True)
        if st.session_state.chat_history:
            for user_msg, bot_msg in st.session_state.chat_history:
                st.markdown(f'<div class="user-message message">👤 You: {user_msg}</div>', 
                           unsafe_allow_html=True)
                st.markdown(f'<div class="bot-message message">🤖 AI Tutor: {bot_msg}</div>', 
                           unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align: center; color: #666; margin-top: 2rem;'>No messages yet. Start chatting below!</div>", 
                      unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Fixed Input Container
    with st.container():
        cols = st.columns([5, 1])
        with cols[0]:
            user_message = st.text_input("Ask me anything:", 
                                       key="user_input",
                                       label_visibility="collapsed",
                                       placeholder="Type your AI/ML question...")
        with cols[1]:
            if st.button("Send", use_container_width=True, type="primary"):
                st.session_state.process_input = True

    if st.session_state.get("process_input"):
        process_input()

def process_input():
    user_message = st.session_state.get("user_input", "")
    if user_message:
        try:
            if not is_ai_ml_related(user_message):
                st.warning("⚠️ This chatbot specializes in AI/ML topics. For best results, ask about:")
                st.markdown("- Machine Learning algorithms  \n- Neural Networks  \n- Natural Language Processing  \n- Data Science concepts")
            
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
            st.session_state.user_input = ""
            st.rerun()

# Main App
st.title("🎓 AI Tutor Chatbot")

# Handle Google Sign-In
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

# Session timeout (30 minutes)
SESSION_TIMEOUT = 1800
if "last_activity" in st.session_state and (time.time() - st.session_state.last_activity) > SESSION_TIMEOUT:
    st.session_state.clear()
    st.sidebar.warning("Session expired. Please log in again.")

# Chat history download
if "user_token" in st.session_state and st.sidebar.button("Download Chat History"):
    chat_df = pd.DataFrame(st.session_state.chat_history, columns=["User", "AI Tutor"])
    csv = chat_df.to_csv(index=False)
    st.sidebar.download_button(
        label="📥 Download",
        data=csv,
        file_name="ai_tutor_chat_history.csv",
        mime="text/csv"
    )