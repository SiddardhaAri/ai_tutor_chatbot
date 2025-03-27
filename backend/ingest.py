import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
DOCS_FOLDER = "./docs"
PDF_FOLDER = "./pdfs"
URL_LIST_FILE = "./urls.txt"

# Optional: set user-agent for web scraping
if "USER_AGENT" not in os.environ:
    os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36"

def load_documents():
    documents = []

    # Load .txt files
    if os.path.exists(DOCS_FOLDER):
        for filename in os.listdir(DOCS_FOLDER):
            if filename.endswith(".txt"):
                path = os.path.join(DOCS_FOLDER, filename)
                documents.extend(TextLoader(path).load())

    # Load PDF files
    if os.path.exists(PDF_FOLDER):
        for filename in os.listdir(PDF_FOLDER):
            if filename.endswith(".pdf"):
                path = os.path.join(PDF_FOLDER, filename)
                documents.extend(PyPDFLoader(path).load())

    # Load URLs from urls.txt
    if os.path.exists(URL_LIST_FILE):
        with open(URL_LIST_FILE, "r") as f:
            urls = [line.strip() for line in f.readlines() if line.strip()]
            for url in urls:
                loader = WebBaseLoader(url)
                documents.extend(loader.load())

    return documents

def chunk_and_store():
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_documents(documents)

    # ✅ Strict filter to ensure valid plain text strings only
    texts_cleaned = [
        doc.page_content.strip()
        for doc in texts
        if hasattr(doc, 'page_content') and isinstance(doc.page_content, str) and doc.page_content.strip()
    ]

    print(f"🧪 Previewing first 3 chunks to embed:\n")
    for i, t in enumerate(texts_cleaned[:3]):
        print(f"Chunk {i+1}: {repr(t[:200])}...\n")

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(texts=texts_cleaned, embedding=embedding_model, persist_directory=CHROMA_DB_PATH)
    vectordb.persist()
    print("✅ Embedded and stored PDFs, URLs, and .txt files in ChromaDB")

if __name__ == "__main__":
    chunk_and_store()