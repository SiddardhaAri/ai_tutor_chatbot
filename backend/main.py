import os
import json
import requests
import firebase_admin
from firebase_admin import auth, credentials
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardcoded OpenRouter API Key
OPENROUTER_API_KEY = "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"

# Initialize Firebase
def initialize_firebase():
    try:
        firebase_credentials_json = os.getenv("FIREBASE_CREDENTIALS")
        if not firebase_credentials_json:
            raise ValueError("Firebase credentials not set")
        cred = credentials.Certificate(json.loads(firebase_credentials_json))
        firebase_admin.initialize_app(cred)
    except Exception as e:
        logger.error(f"Firebase initialization failed: {str(e)}")
        raise

initialize_firebase()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatRequest(BaseModel):
    student_id: str
    user_message: str

async def verify_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(401, "Missing authorization token")
    try:
        token = authorization.split(" ")[1]
        return auth.verify_id_token(token)
    except Exception as e:
        logger.error(f"Token verification failed: {str(e)}")
        raise HTTPException(401, "Invalid or expired token")

@app.post("/chat/")
async def chat(request: ChatRequest, user=Depends(verify_token)):
    try:
        # Direct API call without any database operations
        api_response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={
                "model": "mistralai/mistral-7b-instruct:free",
                "messages": [{"role": "user", "content": request.user_message}]
            },
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=10
        )
        api_response.raise_for_status()
        return {"response": api_response.json()["choices"][0]["message"]["content"]}
        
    except requests.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        raise HTTPException(502, "AI service unavailable")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(500, "Internal server error")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
def root():
    return {"message": "AI Tutor API is running"}