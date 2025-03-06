import streamlit as st
import requests
import pyrebase
import json
import time
import pandas as pd
import logging
from scholarly import scholarly
from arxiv import Search
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.ERROR)

# 🔹 Firebase Config
firebase_config = {"apiKey": "AIzaSy...", "authDomain": "aitutorbot-bb549.firebaseapp.com"}
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

API_URL = "https://ai-tutor-chatbot-fkjr.onrender.com/chat"
summary_pipeline = pipeline("summarization")

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
            st.sidebar.error("❌ Authentication error.")

if choice == "Login":
    if st.sidebar.button("Login"):
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            st.session_state["user_token"] = user["idToken"]
            st.session_state["user_email"] = user["email"]
            st.session_state["chat_history"] = []
            st.session_state["learned_topics"] = set()
            st.sidebar.success(f"✅ Logged in as {st.session_state['user_email']}")
        except Exception as e:
            st.sidebar.error("❌ Authentication error.")

# Auto-Complete for AI Terms
def auto_complete_terms():
    terms = ["Neural Networks", "Transformers", "GANs", "RL", "NLP"]
    return st.selectbox("Suggested AI Topics", terms, key="autocomplete")

# Fetch Research Papers
def fetch_research_papers(query):
    scholar_results = [paper.bib["title"] for paper in scholarly.search_pubs(query)][:3]
    arxiv_results = [result.title for result in Search(query).results()][:3]
    return scholar_results + arxiv_results

# Summarization Feature
def summarize_text(text):
    return summary_pipeline(text[:1000])[0]['summary_text'] if len(text) > 1000 else text

# Main Chat Interface
if "user_token" in st.session_state:
    st.write(f"👋 Welcome, {st.session_state['user_email']}!")
    suggested_topic = auto_complete_terms()
    user_message = st.text_input("Ask me about AI/ML:", value=suggested_topic)

    if st.button("Get Answer") and user_message:
        try:
            headers = {"Authorization": f"Bearer {st.session_state['user_token']}"}
            response = requests.post(API_URL, json={"user_message": user_message}, headers=headers)
            if response.status_code == 200:
                bot_response = response.json().get("response", "No response available.")
                st.session_state["chat_history"].append((user_message, bot_response))
                
                st.write("🤖 AI Tutor:", bot_response)
                
                summary = summarize_text(bot_response)
                st.write("📌 Summary:", summary)
                
                st.session_state["learned_topics"].add(user_message)
                next_topics = fetch_research_papers(user_message)
                if next_topics:
                    st.write("📖 Suggested Research Papers:", next_topics)
            else:
                st.error("❌ API Error")
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
