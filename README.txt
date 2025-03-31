AI Tutor Chatbot - Project Documentation
Overview
The AI Tutor Chatbot is an intelligent learning assistant designed to support students and professionals in understanding Artificial Intelligence, Machine Learning, and Data Science topics. It leverages a combination of RAG (Retrieval-Augmented Generation), Firebase Authentication, and Mistral 7B LLM to provide accurate, personalized responses with traceable source context.

This chatbot is deployed entirely on Render using:
- A FastAPI backend with ChromaDB and PostgreSQL
- A Streamlit frontend with Firebase authentication
Key Features
- Interactive AI Tutor UI built with Streamlit
- Secure authentication via Firebase (Email & Google Sign-In)
- Context-aware answering powered by Retrieval-Augmented Generation
- Embedded vector database using ChromaDB
- Live follow-up recommendations
- Persistent chat logging to PostgreSQL
- Responsive light/dark mode

Architecture Overview
            User (Student) <--> Streamlit Frontend (app.py)
                                           |
                                           v
               Firebase Auth <-------- Authentication
                                           |
                                           v
                User Question -------> FastAPI Backend (main.py)
                                            |
                           +----------------+----------------+
                           |                                 |
                           v                                 v
                        ChromaDB                         OpenRouter API
                    (Vector Search)                     (Mistral 7B LLM)
                           |                                 |
                           +----------------+----------------+
                                            v
                          Final Answer + Contextual Suggestions
                                            |
                                            v
                                 PostgreSQL Logging
Folder Structure
AI_TUTOR_CHATBOT/
+-- backend/
    +-- db_setup.py          # Creates PostgreSQL table
    +-- ingest.py            # Ingests PDFs, text, URLs into ChromaDB
    +-- main.py              # FastAPI RAG backend
    +-- chroma_db/           # Persisted vector DB
+-- frontend/
    +-- app.py               # Streamlit chatbot frontend
+-- docs/
    +-- quantum.txt
    +-- sample.txt
    +-- sample2.txt
+-- pdfs/
    +-- AI.pdf
+-- urls.txt
+-- .env
+-- requirements.txt
+-- aitutorbot-bb549-...json
+-- venv/


Setup Instructions
1. Clone Repository
   git clone https://github.com/your-username/ai-tutor-chatbot.git
   cd ai-tutor-chatbot

2. Create and Configure .env
   OPENROUTER_API_KEY=your_openrouter_api_key
   DATABASE_URL=postgresql://... # from Render PostgreSQL dashboard
   CHROMA_DB_PATH=./chroma_db
   FIREBASE_CREDENTIALS=./aitutorbot-bb549-firebase-adminsdk.json

3. Install Requirements
   pip install -r requirements.txt

4. Set Up PostgreSQL Table
   python backend/db_setup.py

5. Ingest Learning Material
   python backend/ingest.py

6. Run FastAPI Backend
   uvicorn backend.main:app --reload --port 8000

7. Run Streamlit Frontend
   streamlit run frontend/app.py


Backend API
Endpoint: POST /chat/
- Auth required: Bearer Firebase Token
- Request: { "user_message": "What is overfitting in ML?" }
- Response: { "response": "Overfitting is..." }

RAG Flow:
1. Uses LangChain ChromaDB to find 3 most relevant context chunks
2. Constructs prompt with these chunks
3. Sends prompt to OpenRouter API (Mistral 7B)
4. Logs question and answer in PostgreSQL


Firebase & Authentication
- Uses Firebase for:
  - Email/password signup/login
  - Google Sign-In
- Uses Pyrebase (frontend) and Firebase Admin SDK (backend)
- Tokens are passed to backend as Authorization: Bearer <token>

Data Ingestion (Docs, PDFs, URLs)
- Supported:
  - .txt and .pdf files from /docs/ and /pdfs/
  - URLs listed in urls.txt
- Chunked into 500-character segments with 100 overlap
- Stored into local ChromaDB with HuggingFace embeddings

Example Sources
- sample.txt: Machine learning intro
- quantum.txt: Fictional AI concept
- AI.pdf: Academic material
- urls.txt: Live ML content from trusted sites

Deployment (on Render)
Services:
- FastAPI Backend:
  - Build Command: pip install -r requirements.txt
  - Start Command: uvicorn backend.main:app --host 0.0.0.0 --port 10000
- Streamlit Frontend:
  - Start Command: streamlit run frontend/app.py
- PostgreSQL:
  - Managed via Render Dashboard
  - Connection string placed in .env

Future Improvements
- Add learning path suggestions for user level (beginner/intermediate/expert)
- Add LLM response scoring & feedback
- Enable chat export in PDF
- Use SQLite fallback for offline dev
- Improve UI with chat avatars and markdown formatting

Team & Contribution
- Backend API & RAG Logic: Siddardha Ari
- Firebase Integration: Keyur
- UI/UX with Streamlit: Ankur
- Deployment & Infra: Yaswanth Sathya Sai

Special Thanks
Behnaz Merikhi
Sergiy Dybskiy


