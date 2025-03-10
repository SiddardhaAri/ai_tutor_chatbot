import streamlit as st
import requests
import pyrebase
import json
import time
import pandas as pd
import logging
from streamlit.components.v1 import html

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Firebase Config
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

# AI/ML-related keywords
AI_ML_KEYWORDS = [
    "machine learning", "deep learning", "neural networks", "NLP",
    "computer vision", "reinforcement learning", "AI", "ML", "Python",
    "scikit-learn", "PyTorch", "TensorFlow", "data science", "chatbot",
    "OpenAI", "LLM", "artificial intelligence", "data engineering",
    "feature engineering", "predictive modeling", "generative AI"
]

# Define function BEFORE it's used
def is_ai_ml_related(question: str) -> bool:
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in AI_ML_KEYWORDS)

# Parse Firebase Authentication Errors
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

# Google Sign-In Button
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
        window.parent.postMessage({type: 'google_signin', credential: response.credential}, '*');
    }
    </script>
    """
    html(google_sign_in_html)

# Handle Google Sign-In Response
def handle_google_sign_in(response):
    try:
        credential = response['credential']
        user = auth.sign_in_with_google(credential)
        st.session_state["user_token"] = user["idToken"]
        st.session_state["user_email"] = user["email"]
        st.session_state.setdefault("chat_history", [])
        st.session_state["last_activity"] = time.time()
        st.sidebar.success(f"✅ Logged in as {st.session_state['user_email']}")
    except Exception as e:
        st.sidebar.error(f"❌ Error: {parse_firebase_error(e)}")

# Typing Animation with Proper Formatting
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

# Save chat history to Firebase
def save_chat_to_firebase(user_email, chat_history):
    try:
        db.child("chats").child(user_email.replace(".", "_"))\
            .set(chat_history)
    except Exception as e:
        logging.error(f"Error saving chat history: {e}")

# Main Chat Interface
def main_chat_interface():
    st.write(f"👋 Welcome, {st.session_state['user_email']}!")
    
    # Chat History at the Top
    if st.session_state["chat_history"]:
        st.subheader("Chat History")
        for user_msg, bot_msg in st.session_state["chat_history"]:
            st.markdown(f"**👤 You:** {user_msg}")
            st.markdown(f"**🤖 AI Tutor:**  \n{bot_msg}", unsafe_allow_html=True)
            st.markdown("---")
    
    # Input at the Bottom
    user_message = st.text_input("Ask me anything:", key="user_input", on_change=process_input)
    
    if st.button("Get Answer") or st.session_state.get("process_input", False):
        process_input()

def process_input():
    user_message = st.session_state.get("user_input", "")
    if user_message:
        try:
            if not is_ai_ml_related(user_message):
                st.warning("⚠️ This chatbot specializes in AI/ML topics.")

            headers = {"Authorization": f"Bearer {st.session_state['user_token']}"}
            response = requests.post(API_URL, json={"user_message": user_message}, headers=headers, verify=False)

            if response.status_code == 200:
                bot_response = response.json().get("response", "No response available.")
                formatted_response = bot_response.replace('\n', '\n\n')
                animate_response(formatted_response)
                st.session_state["chat_history"].append((user_message, formatted_response))
                save_chat_to_firebase(st.session_state["user_email"], st.session_state["chat_history"])
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
        except Exception as e:
            logging.error("Chatbot request failed", exc_info=True)
            st.error("❌ Failed to connect to the chatbot service.")
    st.session_state["process_input"] = False

# App Layout
st.title("🎓 AI Tutor Chatbot")

if "user_token" not in st.session_state:
    choice = st.sidebar.selectbox("Login / Sign Up", ["Login", "Sign Up", "Google Sign-In"])
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")

    if choice == "Sign Up":
        if st.sidebar.button("Create Account"):
            try:
                auth.create_user_with_email_and_password(email, password)
                st.sidebar.success("✅ Account created! Please log in.")
                db.child("users").child(email.replace(".", "_"))\
                    .set({"email": email, "created_at": time.ctime()})
            except Exception as e:
                st.sidebar.error(f"❌ Error: {parse_firebase_error(e)}")

    elif choice == "Login":
        if st.sidebar.button("Login"):
            try:
                user = auth.sign_in_with_email_and_password(email, password)
                st.session_state["user_token"] = user["idToken"]
                st.session_state["user_email"] = user["email"]
                st.session_state.setdefault("chat_history", [])
                st.session_state["last_activity"] = time.time()
                st.sidebar.success(f"✅ Logged in as {st.session_state['user_email']}")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {parse_firebase_error(e)}")

    elif choice == "Google Sign-In":
        google_sign_in()

else:
    main_chat_interface()
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.sidebar.success("👋 Logged out!")

# Session Timeout (30 minutes)
SESSION_TIMEOUT = 1800
if "last_activity" in st.session_state and time.time() - st.session_state["last_activity"] > SESSION_TIMEOUT:
    st.session_state.clear()
    st.sidebar.warning("Session expired. Please log in again.")