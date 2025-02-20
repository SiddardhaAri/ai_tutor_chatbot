import os
import sqlite3
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware

OPENROUTER_API_KEY = "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"


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ai-tutor-backend-qtkg.onrender.com"],  # You can specify allowed origins instead of "*"
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


def get_db_connection():
    conn = sqlite3.connect("chatbot.db")
    conn.row_factory = sqlite3.Row
    return conn


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
@app.get("/")
@app.head("/")
async def read_root():
    return {"message": "Hello, World!"}
