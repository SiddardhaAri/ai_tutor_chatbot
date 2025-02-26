import os
import sqlite3
import requests
import urllib3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

urllib3.disable_warnings()
app = FastAPI()

# ✅ Fix: Allow all origins (for debugging)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Fix: Store API key securely
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

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

    # ✅ Fix: Remove unnecessary GET request
    response = requests.post(api_url, json=payload, headers=headers, verify=True)

    # ✅ Fix: Log response for debugging
    import json
    print(json.dumps(response.json(), indent=4))

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
