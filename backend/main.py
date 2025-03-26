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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# 🔹 OpenRouter API Key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("❌ OPENROUTER_API_KEY is missing in .env file")

# 🔹 Firebase Authentication Setup
try:
    firebase_credentials_json = os.getenv("FIREBASE_CREDENTIALS")
    if not firebase_credentials_json:
        raise ValueError("Firebase credentials not set in environment variables")

    cred_dict = json.loads(firebase_credentials_json)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
except Exception as e:
    logger.error(f"🔥 Firebase initialization failed: {str(e)}")
    raise

# 🔹 PostgreSQL Database Connection (External DB)
try:
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is missing in .env file")

    db_url = urlparse(DATABASE_URL)
    conn = psycopg2.connect(
        dbname=db_url.path[1:],
        user=db_url.username,
        password=db_url.password,
        host=db_url.hostname,
        port=db_url.port,
        sslmode="require"
    )
    conn.autocommit = False  # Enable transactions
    cur = conn.cursor()
    logger.info("✅ Successfully connected to PostgreSQL database")
except Exception as e:
    logger.error(f"🔥 Database connection failed: {str(e)}")
    raise

# 🔹 ChromaDB Initialization
try:
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(name="ai_tutor_knowledge")
    logger.info("✅ ChromaDB collection initialized")
except Exception as e:
    logger.error(f"🔥 ChromaDB initialization failed: {str(e)}")
    raise

# 🔹 FastAPI App
app = FastAPI()

# 🔹 Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Health Check Endpoint
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

# 🔹 Firebase Token Verification
async def verify_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(401, "Missing authorization token")

    try:
        token = authorization.split(" ")[1]
        return auth.verify_id_token(token)
    except Exception as e:
        logger.error(f"Token verification failed: {str(e)}")
        raise HTTPException(401, "Invalid or expired token")

# 🔹 Chat Request Model
class ChatRequest(BaseModel):
    student_id: str  # Ensure student_id is always a string
    user_message: str

# 🔹 Enhanced Chat Route
@app.post("/chat/")
async def chat(request: ChatRequest, user=Depends(verify_token)):
    try:
        # Validate email exists in Firebase token
        if 'email' not in user:
            raise HTTPException(400, "User email missing in authentication token")

        # 🔹 Ensure student_id is always a string
        student_id_str = str(request.student_id)

        # Debugging log
        logger.info(f"🛠 DEBUG: Inserting student_id={student_id_str} (Type: {type(student_id_str)})")

        # Database operations with transaction management
        try:
            cur.execute(
                """INSERT INTO students (id, email)
                VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE
                SET email = EXCLUDED.email""",
                (student_id_str, user['email'])
            )
            conn.commit()
            logger.info(f"✅ Updated student record for {user['email']}")
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"🔥 Database error: {str(e)}")
            raise HTTPException(500, f"Student record update failed: {str(e)}")

        # 🔹 Check ChromaDB for existing answers
        results = collection.query(query_texts=[request.user_message], n_results=1)

        if results["documents"]:
            return {"response": results["documents"][0]}

        # 🔹 OpenRouter API call
        try:
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
        except requests.RequestException as e:
            logger.error(f"🔥 OpenRouter API failed: {str(e)}")
            raise HTTPException(502, f"AI service unavailable: {str(e)}")

        # 🔹 Store conversation in PostgreSQL
        try:
            cur.execute(
                """INSERT INTO conversations (student_id, message, response)
                VALUES (%s, %s, %s)""",
                (student_id_str, request.user_message, response)
            )
            conn.commit()
            collection.add(documents=[response], metadatas=[{"message": request.user_message}])
        except Exception as e:
            conn.rollback()
            logger.error(f"🔥 Storage failed: {str(e)}")
            raise HTTPException(500, "Failed to save conversation")

        return {"response": response}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🔥 Unexpected error: {str(e)}")
        raise HTTPException(500, "Internal server error")

@app.get("/")
def read_root():
    return {"message": "API is running!"}
