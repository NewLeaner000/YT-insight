import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and "sslmode" not in DATABASE_URL:
    DATABASE_URL += "?sslmode=require"

engine = create_engine(DATABASE_URL)

def fix_db():
    print("Fixing vector dimensions in DB...")
    with engine.connect() as conn:
        try:
            conn.execute(text("DELETE FROM session_videos;"))
            conn.execute(text("DELETE FROM youtube_comments;"))
            conn.execute(text("ALTER TABLE youtube_comments ALTER COLUMN embedding TYPE vector(3072);"))
            conn.commit()
            print("Successfully cleared old comments and altered embedding column to 3072 dimensions.")
        except Exception as e:
            print(f"Error altering column: {e}")

if __name__ == "__main__":
    fix_db()
