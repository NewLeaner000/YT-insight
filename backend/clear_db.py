from app.database.connection import engine
from sqlalchemy import text

def clear_all():
    print("Clearing all data from database...")
    with engine.connect() as conn:
        try:
            conn.execute(text("TRUNCATE TABLE messages CASCADE;"))
            conn.execute(text("TRUNCATE TABLE session_videos CASCADE;"))
            conn.execute(text("TRUNCATE TABLE youtube_comments CASCADE;"))
            conn.execute(text("TRUNCATE TABLE sessions CASCADE;"))
            conn.commit()
            print("Successfully cleared all database tables!")
        except Exception as e:
            print(f"Error: {e}")
            conn.rollback()

if __name__ == "__main__":
    clear_all()
