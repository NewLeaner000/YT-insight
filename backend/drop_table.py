import os
import sys
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.database.connection import engine

def drop_and_recreate():
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS youtube_comments;"))
        conn.commit()
    print("Table dropped.")

if __name__ == "__main__":
    drop_and_recreate()
