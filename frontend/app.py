import streamlit as st
import requests
import pyrebase
import json
import time
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR)

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

API_URL = "https://ai-tutor-chatbot-fkjr.onrender.com/chat"

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
    except:
        return "An unexpected error occurred. Please try again."

st.title("🎓 AI Tutor Chatbot")

choice = st.sidebar.selectbox("Login / Sign Up", ["Login", "Sign Up"])
email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")

# Sign Up Functionality
if choice == "Sign Up":
    if st.sidebar.button("Create Account"):
        try:
            auth.create_user_with_email_and_password(email, password)
            st.sidebar.success("✅ Account created! Please log in.")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {parse_firebase_error(e)}")

# Login Functionality
if choice == "Login":
    if st.sidebar.button("Login"):
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            st.session_state["user_token"] = user["idToken"]
            st.session_state["user_email"] = user["email"]
            st.session_state["chat_history"] = []
            st.session_state["last_activity"] = time.time()
            st.sidebar.success(f"✅ Logged in as {st.session_state['user_email']}")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {parse_firebase_error(e)}")

# Logout
if "user_token" in st.session_state and st.sidebar.button("Logout"):
    st.session_state.clear()
    st.sidebar.success("👋 Logged out!")

# Session Timeout (30 minutes)
SESSION_TIMEOUT = 1800
if "last_activity" in st.session_state and time.time() - st.session_state["last_activity"] > SESSION_TIMEOUT:
    st.session_state.clear()
    st.sidebar.warning("Session expired. Please log in again.")

# AI/ML Topic Recommendations
def recommend_topics(user_message):
    recommendations = {
        "machine learning": ["Supervised Learning", "Unsupervised Learning"],
        "deep learning": ["Neural Networks", "CNNs", "RNNs"],
        "nlp": ["Transformers", "Sentiment Analysis"],
    }
    for keyword, topics in recommendations.items():
        if keyword in user_message.lower():
            return topics
    return ["Explore AI Ethics", "Model Interpretability"]

# Typing Animation
def animate_response(response):
    placeholder = st.empty()
    animated_text = ""
    for word in response.split():
        animated_text += word + " "
        placeholder.write(animated_text)
        time.sleep(0.1)

# Main Chat Interface
if "user_token" in st.session_state:
    st.write(f"👋 Welcome, {st.session_state['user_email']}!")
    user_message = st.text_input("Ask me about AI/ML:")

    if st.button("Get Answer") and user_message:
        try:
            headers = {"Authorization": f"Bearer {st.session_state['user_token']}"}
            response = requests.post(API_URL, json={"user_message": user_message}, headers=headers, verify=False)
            if response.status_code == 200:
                bot_response = response.json().get("response", "No response available.")
                st.session_state["chat_history"].append((user_message, bot_response))

                animate_response(bot_response)

                suggestions = recommend_topics(user_message)
                st.write("📌 Recommended Topics:", ", ".join(suggestions))
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
        except Exception as e:
            logging.error("Chatbot request failed", exc_info=True)
            st.error("❌ Failed to connect to the chatbot service.")

    if st.session_state["chat_history"]:
        st.subheader("Chat History")
        for user_msg, bot_msg in st.session_state["chat_history"]:
            st.markdown(f"**👤 You:** {user_msg}")
            st.markdown(f"**🤖 AI Tutor:** {bot_msg}")
            st.markdown("---")

    # Download Chat History
    if st.sidebar.button("Download Chat History"):
        chat_df = pd.DataFrame(st.session_state["chat_history"], columns=["User", "AI Tutor"])
        st.sidebar.download_button("📥 Download Chat", chat_df.to_csv(index=False), "chat_history.csv", "text/csv")

else:
    st.warning("🔒 Please log in to access the chatbot.")
