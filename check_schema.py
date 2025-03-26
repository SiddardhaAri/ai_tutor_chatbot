import psycopg2
import os

DATABASE_URL = "postgresql://aitutorchatbotdb_user:6ic6Raxg18PQQPYmtsDJfuGqzm4FP138@dpg-cvhjj8lds78s7398kn90-a.oregon-postgres.render.com/aitutorchatbotdb"

conn = psycopg2.connect(DATABASE_URL, sslmode="require")
cur = conn.cursor()

# Check students table
cur.execute("""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'students'
""")
print("Students Table Schema:")
for row in cur.fetchall():
    print(f"- {row[0]}: {row[1]}")

cur.close()
conn.close()