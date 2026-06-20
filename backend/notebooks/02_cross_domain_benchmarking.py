# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
# ---

# %% [markdown]
# # Phase 0.5: Cross-Domain Benchmarking
# Instead of retraining, we will load our 4 pre-trained `.pkl` models and evaluate their true generalization power on unseen domains.

# %%
import os
import pickle
import pandas as pd
import numpy as np
import re
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("CROSS-DOMAIN BENCHMARKING WITH PRE-TRAINED MODELS")
print("="*60)

# %% [markdown]
# ## 1. Define Meta-Feature Function & Load Models
# We must define `extract_meta_features` at the top level BEFORE loading the pickle file, because Python's `pickle` saves functions by reference, not by code.

# %%
analyzer = SentimentIntensityAnalyzer()

def extract_meta_features(text_series):
    """Extract handcrafted features that TF-IDF cannot capture."""
    meta = pd.DataFrame(index=text_series.index)
    
    # VADER Sentiment Scores
    vader_scores = text_series.apply(lambda x: analyzer.polarity_scores(str(x)))
    meta['vader_compound'] = vader_scores.apply(lambda x: x['compound'])
    meta['vader_pos'] = vader_scores.apply(lambda x: x['pos'])
    meta['vader_neg'] = vader_scores.apply(lambda x: x['neg'])
    meta['vader_neu'] = vader_scores.apply(lambda x: x['neu'])
    
    # Punctuation
    meta['exclamation_count'] = text_series.apply(lambda x: str(x).count('!'))
    meta['question_count'] = text_series.apply(lambda x: str(x).count('?'))
    
    # Capitalization ratio
    meta['caps_ratio'] = text_series.apply(lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1))
    
    # Length & Word count
    meta['text_length'] = text_series.apply(lambda x: len(str(x)))
    meta['word_count'] = text_series.apply(lambda x: len(str(x).split()))
    
    # Negation words
    negation_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing',
                      "don't", "doesn't", "didn't", "won't", "wouldn't",
                      "can't", "couldn't", "shouldn't", "isn't", "aren't",
                      "wasn't", "weren't", "haven't", "hasn't", "hadn't"}
    meta['negation_count'] = text_series.apply(lambda x: sum(1 for w in str(x).lower().split() if w in negation_words))
    
    # Elongation & Emoticons
    meta['elongation_count'] = text_series.apply(lambda x: len(re.findall(r'(.)\1{2,}', str(x))))
    meta['emoticon_count'] = text_series.apply(lambda x: len(re.findall(r'[:;=][-~]?[)D(P/\\|]|<3|\bxD\b|\bXD\b', str(x))))
    
    return meta

# Go up one level from 'notebooks' to 'backend', then into 'app/services'
model_dir = os.path.join(os.path.dirname(os.path.abspath('')), 'app', 'services')

print(f"Loading Models from {model_dir} ...")
try:
    with open(os.path.join(model_dir, 'sentiment_model_baseline.pkl'), 'rb') as f:
        baseline_model = pickle.load(f)
    with open(os.path.join(model_dir, 'sentiment_model_ngram.pkl'), 'rb') as f:
        ngram_model = pickle.load(f)
    with open(os.path.join(model_dir, 'sentiment_model_hybrid.pkl'), 'rb') as f:
        hybrid_bundle = pickle.load(f)
    with open(os.path.join(model_dir, 'sentiment_model_stacking.pkl'), 'rb') as f:
        stacking_model = pickle.load(f)
    print("All 4 models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")

# %% [markdown]
# ## 2. Feature Engineering Helper
# This function applies the saved TF-IDF vectorizer and Scaler to the new datasets.

# %%
def prepare_hybrid_features(df, text_col):
    print("  [Processing] Extracting Meta Features & VADER scores...")
    tfidf_matrix = hybrid_bundle['tfidf'].transform(df[text_col].astype(str))
    meta_features = extract_meta_features(df[text_col])
    meta_scaled = hybrid_bundle['scaler'].transform(meta_features)
    X_hybrid = hstack([tfidf_matrix, meta_scaled])
    return tfidf_matrix, X_hybrid

# %% [markdown]
# ## 3. Benchmarking Core Function

# %%
def run_benchmark(dataset_name, df, text_col, target_col):
    print(f"\n" + "="*60)
    print(f"BENCHMARKING ON: {dataset_name.upper()}")
    print(f"Dataset Size: {len(df):,} rows")
    print("="*60)
    
    df = df.dropna(subset=[text_col, target_col])
    X_raw = df[text_col].astype(str)
    y_true = df[target_col]
    
    # 1. Evaluate Baseline
    y_pred_base = baseline_model.predict(X_raw)
    acc_base = accuracy_score(y_true, y_pred_base)
    print(f"[1] Baseline (LR) Accuracy:  {acc_base * 100:.2f}%")
    
    # 2. Evaluate N-Grams
    y_pred_ngram = ngram_model.predict(X_raw)
    acc_ngram = accuracy_score(y_true, y_pred_ngram)
    print(f"[2] N-Grams Accuracy:        {acc_ngram * 100:.2f}%")
    
    # Prepare Matrices
    tfidf_matrix, X_hybrid = prepare_hybrid_features(df, text_col)
    
    # 3. Evaluate Hybrid
    y_pred_hybrid = hybrid_bundle['model'].predict(X_hybrid)
    acc_hybrid = accuracy_score(y_true, y_pred_hybrid)
    print(f"[3] Hybrid Model Accuracy:   {acc_hybrid * 100:.2f}%")
    
    # 4. Evaluate Stacking
    y_pred_stack = stacking_model.predict(tfidf_matrix)
    acc_stack = accuracy_score(y_true, y_pred_stack)
    print(f"[4] Stacking Model Accuracy: {acc_stack * 100:.2f}%")

# %% [markdown]
# ## 4. Run on Stanford SST-2 & TweetEval

# %%
print("\nLoading SST-2 Dataset (Movie Reviews)...")
try:
    sst2 = load_dataset("stanfordnlp/sst2")
    df_sst2 = pd.DataFrame(sst2['validation'])
    df_sst2['label_text'] = df_sst2['label'].map({0: 'Negative', 1: 'Positive'})
    run_benchmark("Stanford SST-2 (Movie Reviews)", df_sst2, text_col='sentence', target_col='label_text')
except Exception as e:
    print(f"Error evaluating SST-2: {e}")

# %%
print("\nLoading TweetEval Dataset (Twitter Sentiment)...")
try:
    tweet_eval = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    df_tweet = pd.DataFrame(tweet_eval['test'])
    df_tweet['label_text'] = df_tweet['label'].map({0: 'Negative', 1: 'Neutral', 2: 'Positive'})
    run_benchmark("TweetEval (Twitter Sentiment)", df_tweet, text_col='text', target_col='label_text')
except Exception as e:
    print(f"Error evaluating TweetEval: {e}")

# %% [markdown]
# ## 📊 Cross-Domain Evaluation & Insights
#
# We evaluated our 4 pre-trained models on two distinct datasets to test their generalization capabilities. The results yielded fascinating insights into the difference between frequency-based learning (TF-IDF) and structural learning (Hybrid Meta-features).
#
# ### 1. Stanford SST-2 (Movie Reviews)
# * **Characteristics:** Formal, structured, grammatically correct, binary sentiment (Positive/Negative).
# * **Winner:** **Stacking Ensemble (59.63%)**
# * **Insight:** Because movie reviews are highly dependent on formal vocabulary rather than internet slang, our Stacking model (which mastered over 10 million N-Gram features) generalized the best. *(Note: The absolute accuracy is capped because our model predicts 3 classes, while SST-2 strictly evaluates on 2 classes, penalizing any valid 'Neutral' predictions).*
#
# ### 2. TweetEval (Twitter Sentiment)
# * **Characteristics:** Short-form, messy, heavily reliant on slang, capitalization, and punctuation (highly similar to YouTube).
# * **Winner:** **Hybrid Model (61.77%)**
# * **Insight:** This is our most critical finding. The Stacking model experienced a performance drop here because specific vocabulary trends on YouTube do not perfectly overlap with Twitter. However, the **Hybrid Model** triumphed by relying on underlying structural features (VADER lexicons, exclamation counts, capitalization ratios). It proves that **meta-feature engineering is highly domain-agnostic** and transfers exceptionally well across different social media platforms.
#
# ### Final Conclusion for Production
# While the **Stacking Model** is the ultimate champion on pure YouTube data (73.30%), the **Hybrid Model** exhibits superior robustness and adaptability when analyzing text from unverified or mixed social media domains.
