import streamlit as st
import requests

API_URL = "https://ai-tutor-backend-qtkg.onrender.com/chat/"

st.title("🎓 AI Tutor Chatbot")

user_message = st.text_input("Ask me about AI/ML:")

if st.button("Get Answer"):
    response = requests.post(API_URL, json={"user_message": user_message}, verify=False)

    if response.status_code == 200:
        st.write("🤖 AI Tutor:", response.json()["response"])
    else:
        st.write(f"❌ Error {response.status_code}: {response.text}")
