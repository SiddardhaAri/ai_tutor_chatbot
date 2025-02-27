import streamlit as st
import requests
import pyrebase
import json

# 🔹 Firebase Config
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

st.title("🎓 AI Tutor Chatbot")

choice = st.sidebar.selectbox("Login / Sign Up", ["Login", "Sign Up"])
email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")

if choice == "Sign Up":
    if st.sidebar.button("Create Account"):
        try:
            auth.create_user_with_email_and_password(email, password)
            st.sidebar.success("✅ Account created! Please log in.")
        except Exception as e:
            st.sidebar.error("❌ Signup Failed")

if choice == "Login":
    if st.sidebar.button("Login"):
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            st.session_state["user_token"] = user["idToken"]
            st.session_state["user_email"] = user["email"]
            st.session_state["chat_history"] = []
            st.sidebar.success("✅ Logged in successfully!")
        except Exception:
            st.sidebar.error("❌ Login Failed")

if "user_token" in st.session_state:
    if st.sidebar.button("Logout"):
        del st.session_state["user_token"]
        del st.session_state["user_email"]
        del st.session_state["chat_history"]
        st.sidebar.success("👋 Logged out!")

if "user_token" in st.session_state:
    username = st.session_state["user_email"].split("@")[0]
    st.write(f"👋 Welcome, {username}!")

    # Move chat history ABOVE the input field
    if "chat_history" in st.session_state and st.session_state["chat_history"]:
        for user_msg, bot_msg in st.session_state["chat_history"]:
            st.markdown(f"<div class='chat-container'><span class='user-message'>👤 {user_msg}</span><br><span class='bot-message'>🤖 {bot_msg}</span></div>", unsafe_allow_html=True)

    # Input Field (Press Enter to Submit)
    user_message = st.text_input("Ask me about AI/ML:", key="chat_input")

    if user_message:  # Automatically submits on Enter
        headers = {"Authorization": f"Bearer {st.session_state['user_token']}"}
        try:
            response = requests.post("https://ai-tutor-chatbot-fkjr.onrender.com/chat", json={"user_message": user_message}, headers=headers, verify=False)
            if response.status_code == 200:
                bot_response = response.json().get("response", "No response available.")
                st.session_state["chat_history"].append((user_message, bot_response))
                st.experimental_rerun()  # Refresh the UI
            else:
                st.error(f"❌ API Error {response.status_code}")
        except Exception:
            st.error("❌ Failed to connect.")

else:
    st.warning("🔒 Please log in to access the chatbot.")
