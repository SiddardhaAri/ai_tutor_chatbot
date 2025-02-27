import os
import sqlite3
import json
import requests
import firebase_admin
from firebase_admin import auth, credentials
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

OPENROUTER_API_KEY = "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"

# Load Firebase credentials from an environment variable
firebase_credentials_json = os.getenv("FIREBASE_CREDENTIALS")
if not firebase_credentials_json:
    raise ValueError("Firebase credentials not set in environment variables.")

cred_dict = json.loads(firebase_credentials_json)
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

async def verify_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization token")
    
    try:
        token = authorization.split(" ")[1]  # Extract token from "Bearer <token>"
        decoded_token = auth.verify_id_token(token)
        return decoded_token  # Returns user info
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
@app.get("/")  # Add this route to avoid 404
async def root():
    return {"message": "AI Tutor Chatbot Backend is running!"}

def get_db_connection():
    conn = sqlite3.connect("chatbot.db")
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/protected/")
async def protected_route(user=Depends(verify_token)):
    return {"message": f"Hello, {user['email']}!"}

class ChatRequest(BaseModel):
    user_message: str

@app.post("/chat/")
async def chat(request: ChatRequest):
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [{"role": "user", "content": request.user_message}]
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(api_url, json=payload, headers=headers, verify=True)
    if response.status_code == 200:
        return {"response": response.json()["choices"][0]["message"]["content"]}
    elif response.status_code == 402:
        raise HTTPException(status_code=402, detail="Insufficient credits. Add more at https://openrouter.ai/credits")
    elif response.status_code == 401:
        raise HTTPException(status_code=401, detail="Invalid API key. Check and update it.")
    elif response.status_code == 400:
        raise HTTPException(status_code=400, detail="Invalid model ID. Ensure you're using 'mistralai/mistral-7b-instruct:free'.")
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)
