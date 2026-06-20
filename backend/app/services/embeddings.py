import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class EmbeddingsManager:
    def __init__(self):
        self._model = None

    def get_model(self):
        if self._model is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("WARNING: GEMINI_API_KEY is not set.")
                return None
            print("Initializing Google Generative AI Embeddings Model...")
            self._model = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001", 
                google_api_key=api_key
            )
        return self._model

embeddings_manager = EmbeddingsManager()
