import os
import json
import psycopg2
import requests
import firebase_admin
import chromadb
from firebase_admin import auth, credentials
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from dotenv import load_dotenv
from urllib.parse import urlparse
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# 🔹 OpenRouter API Key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("❌ OPENROUTER_API_KEY is missing in .env file")

# 🔹 Firebase Authentication Setup
firebase_credentials_json = os.getenv("FIREBASE_CREDENTIALS")
if not firebase_credentials_json:
    raise ValueError("❌ Firebase credentials not set in environment variables.")

cred_dict = json.loads(firebase_credentials_json)
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)

# 🔹 PostgreSQL Database Connection (External DB)
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL is missing in .env file")

db_url = urlparse(DATABASE_URL)
conn = psycopg2.connect(
    dbname=db_url.path[1:],
    user=db_url.username,
    password=db_url.password,
    host=db_url.hostname,
    port=db_url.port,
    sslmode="require"  # Ensures secure connection
)
cur = conn.cursor()

# 🔹 ChromaDB Initialization
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="ai_tutor_knowledge")

# 🔹 FastAPI App
app = FastAPI()

# 🔹 Enable CORS (for frontend compatibility)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

# 🔹 Firebase Token Verification
async def verify_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization token")
    
    try:
        token = authorization.split(" ")[1]  # Extract token from "Bearer <token>"
        decoded_token = auth.verify_id_token(token)
        return decoded_token  # Returns user info
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# 🔹 Root Route
@app.get("/")
async def root():
    return {"message": "✅ AI Tutor Chatbot Backend is running!"}

# 🔹 Protected Route (Test Authentication)
@app.get("/protected/")
async def protected_route(user=Depends(verify_token)):
    return {"message": f"Hello, {user['email']}!"}

# 🔹 Chat Request Model
class ChatRequest(BaseModel):
    student_id: int
    user_message: str

# 🔹 Chat Route (Handles AI Responses + ChromaDB + PostgreSQL)
@app.post("/chat/")
async def chat(request: ChatRequest, user=Depends(verify_token)):
    query = request.user_message

    # 🔹 Step 1: Check if response exists in ChromaDB
    results = collection.query(query_texts=[query], n_results=1)

    if results["documents"]:
        response = results["documents"][0]
    else:
        # 🔹 Step 2: Query OpenRouter API (Mistral 7B)
        api_url = "https://openrouter.ai/api/v1/chat/completions"
        payload = {
            "model": "mistralai/mistral-7b-instruct:free",
            "messages": [{"role": "user", "content": query}]
        }
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        api_response = requests.post(api_url, json=payload, headers=headers, verify=True)
        
        if api_response.status_code == 200:
            response = api_response.json()["choices"][0]["message"]["content"]
        elif api_response.status_code == 402:
            raise HTTPException(status_code=402, detail="Insufficient credits. Add more at https://openrouter.ai/credits")
        elif api_response.status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid API key. Check and update it.")
        elif api_response.status_code == 400:
            raise HTTPException(status_code=400, detail="Invalid model ID. Ensure you're using 'mistralai/mistral-7b-instruct:free'.")
        else:
            raise HTTPException(status_code=api_response.status_code, detail=api_response.text)

        # 🔹 Step 3: Store conversation in PostgreSQL
        cur.execute(
            "INSERT INTO conversations (student_id, message, response) VALUES (%s, %s, %s)",
            (request.student_id, query, response)
        )
        conn.commit()

        # 🔹 Step 4: Store response in ChromaDB for future retrieval
        collection.add(documents=[response], metadatas=[{"message": query}])

    return {"response": response}
