from dotenv import load_dotenv
import os

load_dotenv()
from app.services.embeddings import embeddings_manager

def test():
    model = embeddings_manager.get_model()
    vec = model.embed_query("Hello World")
    print(f"Embedding dimensions: {len(vec)}")

if __name__ == "__main__":
    test()
