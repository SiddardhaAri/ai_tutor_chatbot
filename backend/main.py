import os
import sqlite3
import json
import requests
import firebase_admin
from firebase_admin import auth, credentials
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import psycopg2
from datetime import datetime

# RAG Imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Load Firebase credentials from JSON string in environment variable
firebase_credentials_json = os.getenv("FIREBASE_CREDENTIALS")
if not firebase_credentials_json:
    raise ValueError("Firebase credentials not set in environment variables.")
cred_dict = json.loads(firebase_credentials_json)
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)

# Load OpenRouter key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Load ChromaDB path
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

# Load Postgres URL
DATABASE_URL = os.getenv("DATABASE_URL")

# FastAPI App
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Chroma on startup
@app.on_event("startup")
def load_vector_db():
    global vectordb, embedding_model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)

# Auth token verification
def verify_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization token")
    try:
        token = authorization.split(" ")[1]
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

@app.get("/")
async def root():
    return {"message": "AI Tutor Chatbot Backend is running!"}

class ChatRequest(BaseModel):
    user_message: str

# Save chat to PostgreSQL
def save_chat_to_db(user_email, user_message, ai_response):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chats (user_email, user_message, ai_response, created_at)
            VALUES (%s, %s, %s, %s)
        """, (user_email, user_message, ai_response, datetime.utcnow()))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print("❌ Failed to save chat to DB:", e)

@app.post("/chat/")
async def chat(request: ChatRequest, user=Depends(verify_token)):
    user_question = request.user_message

    # Step 1: Retrieve from Chroma
    relevant_docs = vectordb.similarity_search(user_question, k=3)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # ✅ Print RAG context for verification
    print("📚 RAG Context:\n", context)

    # Step 2: Construct prompt
    prompt = f"""Use the following context to answer the user's question.

Context:
{context}

Question: {user_question}
Answer:"""

    payload = {
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [{"role": "user", "content": prompt}]
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)

    if response.status_code == 200:
        final_response = response.json()["choices"][0]["message"]["content"]
        save_chat_to_db(user['email'], user_question, final_response)
        return {"response": final_response}
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

@app.get("/protected/")
async def protected_route(user=Depends(verify_token)):
    return {"message": f"Hello, {user['email']}!"}
