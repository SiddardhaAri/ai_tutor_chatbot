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

st.title("🎓 AI Tutor Chatbot")
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
            st.session_state["username"] = user["email"].split('@')[0]  # Get the username part of the email
            st.sidebar.success(f"✅ Logged in as {st.session_state['username']}")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {parse_firebase_error(e)}")

if "user_token" in st.session_state:
    if st.sidebar.button("Logout"):
        del st.session_state["user_token"]
        del st.session_state["user_email"]
        del st.session_state["chat_history"]
        del st.session_state["username"]
        st.sidebar.success("👋 Logged out!")

if "user_token" in st.session_state:
    st.write(f"👋 Welcome, {st.session_state['username']}!")  # Display only the username
    
    # Chat history is now above the input field
    if "chat_history" in st.session_state and st.session_state["chat_history"]:
        for user_msg, bot_msg in reversed(st.session_state["chat_history"]):
            st.write(f"👤 {st.session_state['username']}: {user_msg}")  # Show username instead of email
            st.write(f"🤖 AI Tutor: {bot_msg}")
            st.markdown("---")
    
    user_message = st.text_input("Ask me about AI/ML:")
    
    if st.button("Get Answer"):
        headers = {"Authorization": f"Bearer {st.session_state['user_token']}"}
        try:
            response = requests.post(API_URL, json={"user_message": user_message}, headers=headers, verify=False)
            if response.status_code == 200:
                bot_response = response.json().get("response", "No response available.")
                st.session_state["chat_history"].append((user_message, bot_response))
                st.write("🤖 AI Tutor:", bot_response)
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
        except Exception:
            st.error("❌ Failed to connect to the chatbot service.")
else:
    st.warning("🔒 Please log in to access the chatbot.")
