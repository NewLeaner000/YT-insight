import pickle
with open('app/services/sentiment_model_ngram.pkl', 'rb') as f:
    p = pickle.load(f)
print(p.named_steps['tfidf'])
