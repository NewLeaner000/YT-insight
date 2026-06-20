from app.database.connection import engine, Base
from app.database.models import YouTubeComment

print("Dropping youtube_comments table...")
YouTubeComment.__table__.drop(engine, checkfirst=True)

print("Recreating tables...")
Base.metadata.create_all(bind=engine)
print("Done!")
