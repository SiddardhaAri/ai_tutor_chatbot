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
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardcoded OpenRouter API Key (from your working version)
OPENROUTER_API_KEY = "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"

# Load environment variables
load_dotenv()

# Initialize services
def initialize_services():
    try:
        # Firebase Setup
        firebase_credentials_json = os.getenv("FIREBASE_CREDENTIALS")
        if not firebase_credentials_json:
            raise ValueError("Firebase credentials not set")
        
        cred = credentials.Certificate(json.loads(firebase_credentials_json))
        firebase_admin.initialize_app(cred)
        
        # Database Setup
        DATABASE_URL = os.getenv("DATABASE_URL")
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL missing")
            
        db_url = urlparse(DATABASE_URL)
        conn = psycopg2.connect(
            dbname=db_url.path[1:],
            user=db_url.username,
            password=db_url.password,
            host=db_url.hostname,
            port=db_url.port,
            sslmode="require"
        )
        conn.autocommit = False
        
        # ChromaDB Setup
        CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = chroma_client.get_or_create_collection(name="ai_tutor_knowledge")
        
        return conn, collection
    
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise

# Initialize all services
conn, collection = initialize_services()
cur = conn.cursor()

# FastAPI App
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

# Endpoints
@app.get("/health")
async def health_check():
    try:
        cur.execute("SELECT 1")
        return {
            "status": "healthy",
            "database": "connected",
            "chromadb": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(500, detail=f"Service unhealthy: {str(e)}")

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
        # Validate input
        if 'email' not in user:
            raise HTTPException(400, "User email missing")
        
        student_id = str(request.student_id)  # Ensure string type
        
        # Insert/update student
        cur.execute(
            """INSERT INTO students (id, email)
            VALUES (%s, %s)
            ON CONFLICT (id) DO UPDATE
            SET email = EXCLUDED.email""",
            (student_id, user['email'])
        )
        
        # Check ChromaDB first
        results = collection.query(query_texts=[request.user_message], n_results=1)
        if results["documents"]:
            return {"response": results["documents"][0]}
        
        # Call OpenRouter API with hardcoded key
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
            timeout=30
        )
        api_response.raise_for_status()
        response = api_response.json()["choices"][0]["message"]["content"]
        
        # Store conversation
        cur.execute(
            """INSERT INTO conversations (student_id, message, response)
            VALUES (%s, %s, %s)""",
            (student_id, request.user_message, response)
        )
        collection.add(documents=[response], metadatas=[{"message": request.user_message}])
        
        conn.commit()
        return {"response": response}
        
    except requests.RequestException as e:
        conn.rollback()
        logger.error(f"API request failed: {str(e)}")
        raise HTTPException(502, "AI service unavailable")
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(500, "Database operation failed")
    except Exception as e:
        conn.rollback()
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(500, "Internal server error")

@app.get("/")
def root():
    return {"message": "AI Tutor API is running"}

# Close connections on shutdown
@app.on_event("shutdown")
def shutdown_event():
    cur.close()
    conn.close()