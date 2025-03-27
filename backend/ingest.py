from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

# Setup
CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
DOCS_FOLDER = "./docs"  # You can place your PDFs, text files here

def load_documents():
    documents = []
    for filename in os.listdir(DOCS_FOLDER):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(DOCS_FOLDER, filename))
            documents.extend(loader.load())
    return documents

def chunk_and_store():
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(docs, embedding_model, persist_directory=CHROMA_PATH)
    vectordb.persist()
    print("✅ Documents embedded and stored in ChromaDB")

if __name__ == "__main__":
    chunk_and_store()
