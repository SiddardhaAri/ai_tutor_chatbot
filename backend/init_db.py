import psycopg2
import os
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

# Get DATABASE_URL from .env
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL is not set in .env file")

def initialize_database():
    # Parse the DATABASE_URL
    db_url = urlparse(DATABASE_URL)
    
    # Extract database connection details
    DB_NAME = db_url.path[1:]  # Remove leading '/'
    DB_USER = db_url.username
    DB_PASS = db_url.password
    DB_HOST = db_url.hostname
    DB_PORT = db_url.port

    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            host=DB_HOST,
            port=DB_PORT,
            sslmode="require"
        )
        conn.autocommit = True  # For schema changes
        cur = conn.cursor()

        # Drop existing tables if they exist (careful with this in production)
        cur.execute("DROP TABLE IF EXISTS conversations CASCADE")
        cur.execute("DROP TABLE IF EXISTS students CASCADE")

        # Create Students Table (UPDATED SCHEMA)
        cur.execute("""
        CREATE TABLE students (
            id TEXT PRIMARY KEY,         -- Firebase UID
            email TEXT UNIQUE NOT NULL,
            name TEXT,                   -- Made optional
            knowledge_level TEXT DEFAULT 'beginner',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Create Conversations Table (UPDATED SCHEMA)
        cur.execute("""
        CREATE TABLE conversations (
            id SERIAL PRIMARY KEY,
            student_id TEXT REFERENCES students(id) ON DELETE CASCADE,
            message TEXT NOT NULL,
            response TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        print("✅ Database initialized successfully!")
        return True

    except Exception as e:
        print("❌ Database initialization failed:", e)
        return False
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    initialize_database()