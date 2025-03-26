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
        
        /* Follow-up question styling */
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

def correct_spelling(text):
    """Correct spelling in the given text without showing the correction message"""
    blob = TextBlob(text)
    return str(blob.correct())

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
        const responsePayload = {
            credential: response.credential
        };
        window.parent.postMessage({
            type: 'streamlit:setComponentValue',
            data: responsePayload
        }, '*');
    }
    </script>
    """
    html(google_sign_in_html, height=50)

def handle_google_sign_in(credential):
    try:
        # Use the credential directly for authentication
        user = auth.sign_in_with_google(credential)
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
    const chatContainer = document.querySelector('.chat-history-container');
    if (chatContainer) {
        observer.observe(chatContainer, {
            childList: true,
            subtree: true
        });
    }
    </script>
    """
    html(scroll_js, height=0)

def show_follow_up_questions(question):
    """Show follow-up questions as clickable buttons"""
    st.markdown("**Suggested follow-up:**")
    st.markdown(f"""
    <div class="follow-up-question" onclick="Streamlit.setComponentValue('{question}')">
        {question}
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
    
    if button_value:
        return button_value
    return None

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
                
                # Check if we should ask for study plan (only once per session)
                if (i == len(st.session_state.chat_history) - 1 and 
                    not st.session_state.asked_for_study_plan and
                    "study plan" not in user_msg.lower() and
                    "learning path" not in user_msg.lower()):
                    
                    if st.button("Would you like me to create a study plan for this topic?"):
                        st.session_state.asked_for_study_plan = True
                        process_study_plan_request(user_msg)
                
                # Show follow-up questions for the last message
                if i == len(st.session_state.chat_history) - 1:
                    follow_up = show_follow_up_questions(f"What are the key concepts I should learn about {user_msg.split()[-1]}?")
                    if follow_up:
                        st.session_state.user_input = follow_up
                        st.session_state.process_input = True
                        st.rerun()
                
                st.markdown("---")
            st.markdown('</div>', unsafe_allow_html=True)
            auto_scroll_script()

def process_study_plan_request(topic):
    """Process request for study plan"""
    try:
        headers = {"Authorization": f"Bearer {st.session_state.user_token}"}
        response = requests.post(API_URL, 
                               json={"user_message": f"Create a comprehensive study plan for: {topic}"}, 
                               headers=headers, 
                               verify=False)
        
        if response.status_code == 200:
            study_plan = response.json().get("response", "No response available.")
            formatted_response = study_plan.replace('\n', '\n\n')
            st.session_state.chat_history.append(("Study plan request", formatted_response))
            save_chat_to_firebase(st.session_state.user_email, st.session_state.chat_history)
        else:
            st.error(f"❌ API Error {response.status_code}: {response.text}")
    except Exception as e:
        logging.error("Study plan request failed", exc_info=True)
        st.error("❌ Failed to generate study plan.")

def process_input():
    user_message = st.session_state.get("user_input", "")
    if user_message:
        try:
            # Correct spelling internally without showing message
            corrected_message = correct_spelling(user_message)
            if corrected_message != user_message:
                user_message = corrected_message
            
            headers = {"Authorization": f"Bearer {st.session_state.user_token}"}
            response = requests.post(API_URL, 
                                    json={"user_message": user_message}, 
                                    headers=headers, 
                                    verify=False)
            
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