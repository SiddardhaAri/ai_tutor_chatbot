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

# AI/ML-related keywords
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

def main_chat_interface():
    st.write(f"👋 Welcome, {st.session_state.user_email}!")
    
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for user_msg, bot_msg in st.session_state.chat_history:
            st.markdown(f"**👤 You:** {user_msg}")
            st.markdown(f"**🤖 AI Tutor:**  \n{bot_msg}", unsafe_allow_html=True)
            st.markdown("---")
    
    user_message = st.text_input("Ask me anything:", key="user_input", 
                               on_change=lambda: st.session_state.update(process_input=True))
    
    if st.button("Get Answer") or st.session_state.get("process_input"):
        process_input()

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

st.markdown("""
    <style>
        .fixed-input {
            position: fixed;
            bottom: 20px;
            left: 2rem;
            right: 2rem;
            z-index: 999;
            background: white;
            padding: 1rem;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        }
        .chat-history {
            margin-bottom: 150px;
            overflow-y: auto;
            max-height: calc(100vh - 250px);
        }
    </style>
""", unsafe_allow_html=True)

# Modified main chat interface function
def main_chat_interface():
    st.write(f"👋 Welcome, {st.session_state.user_email}!")
    
    # Chat History Container
    with st.container():
        if st.session_state.chat_history:
            st.subheader("Chat History")
            with st.container():
                st.markdown('<div class="chat-history">', unsafe_allow_html=True)
                for user_msg, bot_msg in st.session_state.chat_history:
                    st.markdown(f"**👤 You:** {user_msg}")
                    st.markdown(f"**🤖 AI Tutor:**  \n{bot_msg}", unsafe_allow_html=True)
                    st.markdown("---")
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Fixed Input Container at Bottom
    with st.container():
        st.markdown('<div class="fixed-input">', unsafe_allow_html=True)
        user_message = st.text_input("Ask me anything:", key="user_input", 
                                   on_change=lambda: st.session_state.update(process_input=True))
        
        if st.button("Get Answer") or st.session_state.get("process_input"):
            process_input()
        
        st.markdown('</div>', unsafe_allow_html=True)
