import os
import pickle
from deep_translator import GoogleTranslator

class BaselineSentimentModel:
    def __init__(self):
        # Load the production models exported from the Offline Notebook Training
        # Note: We use the 12MB Baseline model instead of the 917MB Stacking model
        # because Render Free Tier only provides 512MB RAM total, and GitHub limits files to 100MB.
        service_dir = os.path.dirname(__file__)
        baseline_path = os.path.join(service_dir, "sentiment_model_baseline.pkl")
        
        self.is_loaded = False
        self.model_pipeline = None
        
        if os.path.exists(baseline_path):
            try:
                print("Loading Baseline ML Pipeline (12MB) for Free-Tier Deployment...")
                with open(baseline_path, 'rb') as f:
                    self.model_pipeline = pickle.load(f)
                    
                self.is_loaded = True
                print("Baseline ML Pipeline loaded successfully.")
            except Exception as e:
                print(f"Error loading models: {e}")
        else:
            print("WARNING: Required baseline .pkl file not found! Please run the notebook first.")
            
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
            
        return self.model_pipeline.predict(texts)
        
    def predict_proba(self, texts: list[str]) -> list[float]:
        if not texts or not self.is_loaded:
            # Fallback if model isn't trained yet
            return [0.0] * len(texts) if texts else []
            
        probs = self.model_pipeline.predict_proba(texts)
        # Return the max probability score as confidence
        return [float(max(p)) for p in probs]

# Export singleton instance
sentiment_model = BaselineSentimentModel()
