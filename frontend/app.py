import streamlit as st
import requests

# FastAPI Backend URL
API_URL = "http://127.0.0.1:8000/chat/"

st.title("🎓 AI Tutor Chatbot")

# User Input
user_message = st.text_input("Ask me about AI/ML:")

if st.button("Get Answer"):
    response = requests.post(API_URL, json={"user_message": user_message})
    if response.status_code == 200:
        st.write("🤖 AI Tutor:", response.json()["response"])
    else:
        st.write("❌ Error fetching response")
