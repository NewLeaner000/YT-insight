from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="YouTube Live Insight Engine API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://yt-insight-lime.vercel.app",  # Vercel production frontend
        "https://yt-insight.vercel.app",        # Vercel alternate
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Backend is running"}

from app.api import ingest, chat, sessions, health
app.include_router(ingest.router, prefix="/api")
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(sessions.router, prefix="/api", tags=["Sessions"])
app.include_router(health.router, prefix="/api", tags=["Health"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
