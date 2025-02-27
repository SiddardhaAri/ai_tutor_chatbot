import streamlit as st
import requests
import pyrebase

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
st.title("🎓 AI Tutor Chatbot with Firebase Authentication")

# 🔹 Login / Signup
choice = st.sidebar.selectbox("Login / Sign Up", ["Login", "Sign Up"])
email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")

if choice == "Sign Up":
    if st.sidebar.button("Create Account"):
        try:
            user = auth.create_user_with_email_and_password(email, password)
            st.sidebar.success("✅ Account created! Please log in.")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {e}")

if choice == "Login":
    if st.sidebar.button("Login"):
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            st.session_state["user_token"] = user["idToken"]
            st.session_state["user_email"] = user["email"]
            st.sidebar.success(f"✅ Logged in as {user['email']}")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {e}")

# 🔹 Logout
if "user_token" in st.session_state:
    if st.sidebar.button("Logout"):
        del st.session_state["user_token"]
        del st.session_state["user_email"]
        st.sidebar.success("👋 Logged out!")

# 🔹 Chatbot Access (Only if logged in)
if "user_token" in st.session_state:
    st.write(f"👋 Welcome, {st.session_state['user_email']}!")

    user_message = st.text_input("Ask me about AI/ML:")
    
    if st.button("Get Answer"):
        headers = {"Authorization": f"Bearer {st.session_state['user_token']}"}
        response = requests.post(API_URL, json={"user_message": user_message}, headers=headers, verify=False)

        if response.status_code == 200:
            st.write("🤖 AI Tutor:", response.json()["response"])
        else:
            st.write(f"❌ Error {response.status_code}: {response.text}")

else:
    st.warning("🔒 Please log in to access the chatbot.")
