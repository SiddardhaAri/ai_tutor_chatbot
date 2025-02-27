import streamlit as st
import requests
import pyrebase
import json

# 🔹 Firebase Config (Replace with your actual Firebase config)
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

# 🔹 Backend API URL
API_URL = "https://ai-tutor-chatbot-fkjr.onrender.com/chat"

# 🔹 Streamlit UI
st.title("🎓 AI Tutor Chatbot")

# 🔹 Login / Signup
choice = st.sidebar.selectbox("Login / Sign Up", ["Login", "Sign Up"])
email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")

# 🔹 Helper function to extract clean error messages
def parse_firebase_error(e):
    try:
        error_json = json.loads(e.args[1])  # Extract the JSON part of the error
        error_message = error_json['error']['message']
        if error_message == "EMAIL_NOT_FOUND":
            return "Email not found. Please sign up first."
        elif error_message == "INVALID_PASSWORD":
            return "Incorrect password. Please try again."
        elif error_message == "EMAIL_EXISTS":
            return "This email is already registered. Please log in."
        elif error_message == "TOO_MANY_ATTEMPTS_TRY_LATER":
            return "Too many failed attempts. Try again later."
        else:
            return "Authentication error. Please try again."
    except:
        return "An unexpected error occurred. Please try again."

if choice == "Sign Up":
    if st.sidebar.button("Create Account"):
        try:
            user = auth.create_user_with_email_and_password(email, password)
            st.sidebar.success("✅ Account created! Please log in.")
        except Exception as e:
            error_msg = parse_firebase_error(e)
            st.sidebar.error(f"❌ Error: {error_msg}")

if choice == "Login":
    if st.sidebar.button("Login"):
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            st.session_state["user_token"] = user["idToken"]
            st.session_state["user_email"] = user["email"]
            st.session_state["chat_history"] = []  # Initialize chat history
            st.sidebar.success(f"✅ Logged in as {user['email']}")
        except Exception as e:
            error_msg = parse_firebase_error(e)
            st.sidebar.error(f"❌ Error: {error_msg}")

# 🔹 Logout
if "user_token" in st.session_state:
    if st.sidebar.button("Logout"):
        del st.session_state["user_token"]
        del st.session_state["user_email"]
        del st.session_state["chat_history"]
        st.sidebar.success("👋 Logged out!")

# 🔹 Chatbot Access (Only if logged in)
if "user_token" in st.session_state:
    st.write(f"👋 Welcome, {st.session_state['user_email']}!")

    user_message = st.text_input("Ask me about AI/ML:")
    
    if st.button("Get Answer"):
        headers = {"Authorization": f"Bearer {st.session_state['user_token']}"}
        try:
            response = requests.post(API_URL, json={"user_message": user_message}, headers=headers, verify=False)
            
            if response.status_code == 200:
                bot_response = response.json().get("response", "No response available.")
                st.session_state["chat_history"].append((user_message, bot_response))  # Save to chat history
                st.write("🤖 AI Tutor:", bot_response)
            else:
                st.error(f"❌ API Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error("❌ Failed to connect to the chatbot service.")
    
    # 🔹 Display chat history
    if "chat_history" in st.session_state and st.session_state["chat_history"]:
        st.subheader("📜 Chat History")
        for user_msg, bot_msg in st.session_state["chat_history"]:
            st.write(f"👤 You: {user_msg}")
            st.write(f"🤖 AI Tutor: {bot_msg}")
            st.markdown("---")
else:
    st.warning("🔒 Please log in to access the chatbot.")
