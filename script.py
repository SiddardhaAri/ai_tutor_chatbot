import psycopg2

# Render PostgreSQL credentials
DB_HOST = "dpg-cvhjj8lds78s7398kn90-a.oregon-postgres.render.com"
DB_NAME = "aitutorchatbotdb"
DB_USER = "aitutorchatbotdb_user"
DB_PASS = "6ic6Raxg18PQQPYmtsDJfuGqzm4FP138"

try:
    # Connect to PostgreSQL with SSL
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=5432,
        sslmode="require"
    )
    cur = conn.cursor()

    # Execute SQL commands
    sql_commands = """
    ALTER TABLE conversations DROP CONSTRAINT IF EXISTS conversations_student_id_fkey;
    ALTER TABLE students ALTER COLUMN id TYPE TEXT USING id::text;
    ALTER TABLE conversations 
    ADD CONSTRAINT conversations_student_id_fkey FOREIGN KEY (student_id) 
    REFERENCES students(id) ON DELETE CASCADE;
    """
    cur.execute(sql_commands)
    conn.commit()

    print("SQL Commands Executed Successfully")

    cur.close()
    conn.close()
except Exception as e:
    print(f"Error: {e}")
