import os
import pickle
from deep_translator import GoogleTranslator

class StackingSentimentModel:
    def __init__(self):
        # Load the production models exported from the Offline Notebook Training
        service_dir = os.path.dirname(__file__)
        ngram_path = os.path.join(service_dir, "sentiment_model_ngram.pkl")
        stacking_path = os.path.join(service_dir, "sentiment_model_stacking.pkl")
        
        self.is_loaded = False
        self.tfidf = None
        self.stacking_model = None
        
        if os.path.exists(ngram_path) and os.path.exists(stacking_path):
            try:
                print("Loading Stacking ML Pipeline...")
                with open(ngram_path, 'rb') as f:
                    ngram_pipeline = pickle.load(f)
                    self.tfidf = ngram_pipeline.named_steps['tfidf']
                    
                with open(stacking_path, 'rb') as f:
                    self.stacking_model = pickle.load(f)
                    
                self.is_loaded = True
                print("Stacking ML Pipeline loaded successfully.")
            except Exception as e:
                print(f"Error loading models: {e}")
        else:
            print("WARNING: Required .pkl files not found! Please run the notebook first.")
            
        self.translator = GoogleTranslator(source='auto', target='en')

    def translate_batch(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        
        # Replace empty strings to avoid translation errors
        clean_texts = [t if t and t.strip() else " " for t in texts]
        try:
            results = self.translator.translate_batch(clean_texts)
            return [res if res is not None else clean_texts[i] for i, res in enumerate(results)]
        except Exception as e:
            print(f"Translation failed: {e}")
            return clean_texts

    def predict(self, texts: list[str]) -> list[str]:
        if not texts or not self.is_loaded:
            # Fallback if model isn't trained yet
            return ["NEUTRAL"] * len(texts) if texts else []
            
        tfidf_matrix = self.tfidf.transform(texts)
        return self.stacking_model.predict(tfidf_matrix)
        
    def predict_proba(self, texts: list[str]) -> list[float]:
        if not texts or not self.is_loaded:
            # Fallback if model isn't trained yet
            return [0.0] * len(texts) if texts else []
            
        tfidf_matrix = self.tfidf.transform(texts)
        probs = self.stacking_model.predict_proba(tfidf_matrix)
        # Return the max probability score as confidence
        return [float(max(p)) for p in probs]

# Export singleton instance
sentiment_model = StackingSentimentModel()
