import streamlit as st
import requests
import pyrebase
import json

# 🔹 Firebase Config (Replace with actual Firebase config)
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

# 🔹 Initialize Firebase
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

def parse_firebase_error(e):
    try:
        error_json = json.loads(e.args[1])
        error_message = error_json['error']['message']
        errors = {
            "EMAIL_NOT_FOUND": "Email not found. Please sign up first.",
            "INVALID_PASSWORD": "Incorrect password. Please try again.",
            "EMAIL_EXISTS": "This email is already registered. Please log in.",
            "TOO_MANY_ATTEMPTS_TRY_LATER": "Too many failed attempts. Try again later."
        }
        return errors.get(error_message, "Authentication error. Please try again.")
    except:
        return "An unexpected error occurred. Please try again."

# 🔹 Backend API URL
API_URL = "https://ai-tutor-chatbot-fkjr.onrender.com/chat"

# Custom styling for better UI
st.markdown("""
    <style>
        .chat-container {
            max-height: 400px;
            overflow-y: scroll;
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .chat-input {
            width: 100%;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ccc;
            margin-top: 10px;
        }
        .header {
            background-color: #1E90FF;
            color: white;
            padding: 10px;
            border-radius: 5px;
        }
        .user-msg {
            background-color: #f1f1f1;
            padding: 5px;
            border-radius: 5px;
            margin-bottom: 5px;
        }
        .bot-msg {
            background-color: #e0f7fa;
            padding: 5px;
            border-radius: 5px;
            margin-bottom: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# 🔹 UI Header
st.title("🎓 AI Tutor Chatbot")

# Sidebar for Login/Signup
choice = st.sidebar.selectbox("Login / Sign Up", ["Login", "Sign Up"])
email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")

if choice == "Sign Up":
    if st.sidebar.button("Create Account"):
        try:
            user = auth.create_user_with_email_and_password(email, password)
            st.sidebar.success("✅ Account created! Please log in.")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {parse_firebase_error(e)}")

if choice == "Login":
    if st.sidebar.button("Login"):
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            st.session_state["user_token"] = user["idToken"]
            st.session_state["user_email"] = user["email"]
            st.session_state["chat_history"] = []
            st.sidebar.success(f"✅ Logged in as {st.session_state['user_email']}")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {parse_firebase_error(e)}")

# Logout Button
if "user_token" in st.session_state:
    if st.sidebar.button("Logout"):
        del st.session_state["user_token"]
        del st.session_state["user_email"]
        del st.session_state["chat_history"]
        st.sidebar.success("👋 Logged out!")

# Chat Section
if "user_token" in st.session_state:
    st.write(f"👋 Welcome, {st.session_state['user_email']}!")

    # Display Chat History in a scrollable container
    if "chat_history" in st.session_state and st.session_state["chat_history"]:
        with st.container():
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for user_msg, bot_msg in reversed(st.session_state["chat_history"]):
                st.markdown(f'<div class="user-msg">👤 You: {user_msg}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="bot-msg">🤖 AI Tutor: {bot_msg}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # User message input
    user_message = st.text_input("Ask me about AI/ML:", key="user_message", placeholder="Type your question here...")

    # Send button to get a response
    if st.button("Get Answer"):
        if user_message.strip():
            headers = {"Authorization": f"Bearer {st.session_state['user_token']}"}
            try:
                response = requests.post(API_URL, json={"user_message": user_message}, headers=headers, verify=False)
                if response.status_code == 200:
                    bot_response = response.json().get("response", "No response available.")
                    st.session_state["chat_history"].append((user_message, bot_response))
                    st.write("🤖 AI Tutor:", bot_response)
                else:
                    st.error(f"❌ API Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error("❌ Failed to connect to the chatbot service.")
        else:
            st.warning("❗ Please enter a message to get a response.")
else:
    st.warning("🔒 Please log in to access the chatbot.")
