import streamlit as st
import requests

# FastAPI Backend URL (Make sure your FastAPI server is running)
API_URL = "https://ai-tutor-backend-qtkg.onrender.com"

st.title("🎓 AI Tutor Chatbot")

# User Input
user_message = st.text_input("Ask me about AI/ML:")

if st.button("Get Answer"):
    response = requests.post(API_URL, json={"user_message": user_message})
    if response.status_code == 200:
        st.write("🤖 AI Tutor:", response.json()["response"])
    else:
        st.write("❌ Error fetching response")
