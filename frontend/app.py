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

# Custom CSS
st.markdown("""
    <style>
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
        .recommendation-btn {
            background-color: #f0f2f6;
            color: #1e88e5;
            border: 1px solid #1e88e5;
            border-radius: 4px;
            padding: 8px 12px;
            margin: 5px 0;
            cursor: pointer;
        }
        .follow-up-question {
            font-style: italic;
            color: #666;
            margin: 5px 0;
            cursor: pointer;
            text-decoration: underline;
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

def correct_spelling(text):
    """Correct spelling internally without showing suggestions"""
    blob = TextBlob(text)
    return str(blob.correct())

def process_input():
    user_message = st.session_state.get("user_input", "")
    if user_message:
        try:
            # Correct spelling internally
            corrected_message = correct_spelling(user_message)
            if corrected_message.lower() != user_message.lower():
                user_message = corrected_message  # Use corrected version for processing
            
            headers = {"Authorization": f"Bearer {st.session_state.user_token}"}
            response = requests.post(
                API_URL, 
                json={
                    "user_message": user_message,
                    "ask_for_study_plan": st.session_state.get("ask_for_study_plan", False)
                }, 
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                bot_response = data.get("response", "No response available.")
                follow_up_questions = data.get("follow_up_questions", [])
                
                st.session_state.chat_history.append((user_message, bot_response))
                st.session_state.follow_up_questions = follow_up_questions
                st.session_state.ask_for_study_plan = False  # Reset flag
                
                # Check if we should ask about study plan
                if "study plan" in bot_response.lower():
                    st.session_state.ask_for_study_plan = True
                
                save_chat_to_firebase(st.session_state.user_email, st.session_state.chat_history)
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
        except Exception as e:
            logging.error("Chatbot request failed", exc_info=True)
            st.error("❌ Failed to connect to the chatbot service.")
        finally:
            st.session_state.process_input = False

def handle_follow_up_question(question):
    """Handle when a follow-up question is clicked"""
    st.session_state.user_input = question
    st.session_state.process_input = True

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
        const credential = response.credential;
        window.parent.postMessage({
            type: 'streamlit:setComponentValue',
            data: credential
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
            "follow_up_questions": [],
            "ask_for_study_plan": False
        })
        st.rerun()
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

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

    # Chat history and follow-up questions
    with st.container():
        st.markdown('<div class="chat-history-container">', unsafe_allow_html=True)
        
        for user_msg, bot_msg in st.session_state.chat_history:
            st.markdown(f"**👤 You:** {user_msg}")
            st.markdown(f"**🤖 AI Tutor:**  \n{bot_msg}", unsafe_allow_html=True)
            st.markdown("---")
        
        # Show follow-up questions if available
        if st.session_state.get("follow_up_questions"):
            st.markdown("**Suggested follow-up questions:**")
            for question in st.session_state.follow_up_questions:
                st.markdown(
                    f'<div class="follow-up-question" onclick="parent.postMessage({{type: \'streamlit:setComponentValue\', data: \'{question}\'}}, \'*\')">'
                    f'{question}'
                    '</div>', 
                    unsafe_allow_html=True
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # JavaScript for handling follow-up questions
        html("""
        <script>
        window.addEventListener('message', (event) => {
            if (event.data.type === 'streamlit:setComponentValue') {
                Streamlit.setComponentValue(event.data.data);
            }
        });
        </script>
        """, height=0)

# Initialize session state
if "user_token" not in st.session_state:
    st.session_state.update({
        "chat_history": [],
        "follow_up_questions": [],
        "ask_for_study_plan": False
    })

# Main App Logic
st.title("🎓 AI Tutor Chatbot")

if "user_token" not in st.session_state:
    with st.sidebar:
        st.header("Authentication")
        
        # Google Sign-In
        if st.button("Sign in with Google", key="google_signin_btn"):
            google_sign_in()
            google_credential = html(
                """
                <script>
                window.addEventListener('message', (event) => {
                    if (event.data.type === 'streamlit:setComponentValue') {
                        Streamlit.setComponentValue(event.data);
                    }
                });
                </script>
                """, 
                height=0
            )
            if google_credential:
                handle_google_sign_in(google_credential)
        
        # Email Auth
        auth_option = st.radio("Email Auth", ["Login", "Sign Up"], horizontal=True)
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        
        if auth_option == "Sign Up":
            if st.button("Create Account"):
                try:
                    user = auth.create_user_with_email_and_password(email, password)
                    db.child("users").child(email.replace(".", "_")).set({
                        "email": email, 
                        "created_at": time.ctime()
                    })
                    st.success("✅ Account created! Please log in.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        elif auth_option == "Login":
            if st.button("Login"):
                try:
                    user = auth.sign_in_with_email_and_password(email, password)
                    st.session_state.update({
                        "user_token": user["idToken"],
                        "user_email": user["email"],
                        "chat_history": [],
                        "follow_up_questions": [],
                        "ask_for_study_plan": False
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

else:
    main_chat_interface()
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

# Download Chat History
if "user_token" in st.session_state and st.sidebar.button("Download Chat History"):
    chat_df = pd.DataFrame(st.session_state.chat_history, columns=["User", "AI Tutor"])
    st.sidebar.download_button(
        "📥 Download Chat", 
        chat_df.to_csv(index=False), 
        "chat_history.csv", 
        "text/csv"
    )