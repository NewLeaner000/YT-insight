# %% [markdown]
# # Phase 0: Kaggle Optimized ML Pipeline
# This notebook is adapted to run on Kaggle. It will export the final model to `/kaggle/working/` so you can download it easily.

# %%
# Run this cell if you are on Kaggle to install missing dependencies
# !pip install vaderSentiment datasets

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import learning_curve
import pickle
import os

# %% [markdown]
# ## 1. Load Dataset
# We use the HuggingFace `AmaanP314/youtube-comment-sentiment` dataset as our primary training data.

# %%
print("Loading dataset from HuggingFace...")
dataset = load_dataset("AmaanP314/youtube-comment-sentiment")
df = pd.DataFrame(dataset['train'])
print(f"Original dataset shape: {df.shape}")
display(df.head())

# %% [markdown]
# ## 2. Advanced EDA & Data Cleaning
# ### 2.1. Handling Nulls and Missing Values

# %%
print("Null values before cleaning:")
print(df.isnull().sum())

# Drop any rows where CommentText or Sentiment is null
df = df.dropna(subset=['CommentText', 'Sentiment'])
print(f"Shape after dropping nulls: {df.shape}")

# %% [markdown]
# ### 2.2. Outlier Detection (Text Length)
# Extremely short comments (e.g., ".") or extremely long spam comments add noise to the TF-IDF matrix. We will analyze the text length distribution to find a reasonable cut-off.

# %%
# Calculate text length (number of characters)
df['text_length'] = df['CommentText'].astype(str).apply(len)

plt.figure(figsize=(10, 5))
sns.histplot(df['text_length'], bins=100, kde=True, color='blue')
plt.title('Distribution of Comment Lengths')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
plt.xlim(0, 1000)
plt.show()

print("Text Length Percentiles:")
print(df['text_length'].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))

# %% [markdown]
# **Decision:** We will remove outliers: comments shorter than 3 characters, and longer than 500 characters (which covers >95% of normal comments).

# %%
# Filter outliers
min_length = 3
max_length = 500
df_cleaned = df[(df['text_length'] >= min_length) & (df['text_length'] <= max_length)]
print(f"Shape after removing outliers: {df_cleaned.shape}")
print(f"Removed {len(df) - len(df_cleaned)} outliers.")

# %% [markdown]
# ### 2.3. Class Balance
# Let's check if our classes (Positive vs Negative vs Neutral) are balanced.

# %%
plt.figure(figsize=(6, 4))
sns.countplot(data=df_cleaned, x='Sentiment', palette='Set2')
plt.title('Class Distribution')
plt.show()

print(df_cleaned['Sentiment'].value_counts(normalize=True))

# %% [markdown]
# ### 2.4. Text Normalization
# Although `TfidfVectorizer` automatically tokenizes and ignores newlines, explicitly removing `\n`, `\t`, and multiple spaces makes the text cleaner for visual inspection and debugging.

# %%
import re
print("Normalizing text (removing \\n, \\t, and extra spaces)...")
df_cleaned['CommentText'] = df_cleaned['CommentText'].astype(str).apply(lambda x: re.sub(r'\s+', ' ', x).strip())

# %% [markdown]
# ### 2.5. Word Frequency Analysis (Top Words & Special Terms)
# Let's visualize the most frequent words (unigrams) and phrases (bigrams) in the dataset to see what vocabulary YouTube users commonly use.

# %%
from sklearn.feature_extraction.text import CountVectorizer

def plot_top_words(text_series, title, n=20, ngram_range=(1,1)):
    # Use CountVectorizer to count word frequencies
    vec = CountVectorizer(stop_words='english', ngram_range=ngram_range).fit(text_series)
    bag_of_words = vec.transform(text_series)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]
    
    df_freq = pd.DataFrame(words_freq, columns=['Word', 'Frequency'])
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_freq, y='Word', x='Frequency', palette='magma')
    plt.title(title)
    plt.xlabel('Frequency')
    plt.ylabel('Word / Phrase')
    plt.show()

print("Plotting Top 20 Unigrams (Single Words)...")
# Sample 100k rows for faster plotting to save memory
sample_plot = df_cleaned['CommentText'].sample(n=min(100000, len(df_cleaned)), random_state=42)
plot_top_words(sample_plot, "Top 20 Most Frequent Words in YouTube Comments", n=20, ngram_range=(1,1))

print("Plotting Top 20 Bigrams (Two-Word Phrases)...")
plot_top_words(sample_plot, "Top 20 Most Frequent Bigrams in YouTube Comments", n=20, ngram_range=(2,2))

# %% [markdown]
# ## 3. Baseline Modeling (The Arena)
# We will compare 3 models using `GridSearchCV` on a 50,000 row sample.

# %%
df_sample = df_cleaned.sample(n=50000, random_state=42)
X_sample = df_sample['CommentText']
y_sample = df_sample['Sentiment']

X_train_samp, X_test_samp, y_train_samp, y_test_samp = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample)
print(f"Sample Training set: {X_train_samp.shape[0]} samples")
print(f"Sample Testing set: {X_test_samp.shape[0]} samples")

# %% [markdown]
# ### 3.1. Define Pipelines and Run Grid Search

# %%
models = {
    "Naive Bayes": (Pipeline([('tfidf', TfidfVectorizer(stop_words='english')), ('clf', MultinomialNB())]), {'tfidf__max_df': [0.8, 0.9], 'tfidf__ngram_range': [(1, 1), (1, 2)], 'clf__alpha': [0.1, 0.5, 1.0]}),
    "Logistic Regression": (Pipeline([('tfidf', TfidfVectorizer(stop_words='english')), ('clf', LogisticRegression(max_iter=1000))]), {'tfidf__max_df': [0.8, 0.9], 'clf__C': [0.1, 1.0, 10.0]}),
    "Linear SVC": (Pipeline([('tfidf', TfidfVectorizer(stop_words='english')), ('clf', LinearSVC())]), {'tfidf__max_df': [0.8, 0.9], 'clf__C': [0.1, 1.0, 10.0]})
}

best_models = {}
test_results = []

for name, (pipeline, param_grid) in models.items():
    print(f"--- Training {name} ---")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1, scoring='accuracy')
    grid_search.fit(X_train_samp, y_train_samp)
    
    best_models[name] = grid_search.best_estimator_
    
    # Evaluate on Unseen Test Data
    y_pred = grid_search.best_estimator_.predict(X_test_samp)
    acc = accuracy_score(y_test_samp, y_pred)
    test_results.append({'Model': name, 'Test Accuracy': acc})
    print(f"Best CV Score: {grid_search.best_score_:.4f} | Test Accuracy: {acc:.4f}\n")

# %% [markdown]
# ## 4. Advanced Experiments (N-Grams)
# We will now test the N-Gram model. First, we will evaluate it on the same 50k sample for a direct comparison with the baselines, and then scale it up to the full dataset.

# %% [markdown]
# ### 4.1. Apples-to-Apples Comparison (50k Sample)

# %%
print("\n--- Experiment 1: N-Grams on 50k Sample ---")
safe_stopwords = ['the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 'and', 'or', 'is', 'am', 'are']

ngram_model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=safe_stopwords, max_df=0.8, ngram_range=(1, 3))),
    ('clf', LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1))
])

ngram_model.fit(X_train_samp, y_train_samp)
y_pred_samp_ngram = ngram_model.predict(X_test_samp)
acc_samp_ngram = accuracy_score(y_test_samp, y_pred_samp_ngram)
print(f"Accuracy (N-Grams on 50k): {acc_samp_ngram * 100:.2f}%\n")

# %% [markdown]
# ### 4.2. Scaling Up (Full Dataset)
# Now let's train this advanced model on the full 800k dataset to see its maximum potential.

# %%
print("--- Preparing Full Dataset ---")
X_full = df_cleaned['CommentText']
y_full = df_cleaned['Sentiment']
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)

print(f"Full Training set: {X_train_full.shape[0]:,} samples")
print(f"Full Testing set: {X_test_full.shape[0]:,} samples")

print("\n--- Experiment 2: N-Grams on Full Dataset ---")
ngram_model.fit(X_train_full, y_train_full)
y_pred_full_ngram = ngram_model.predict(X_test_full)
acc_full_ngram = accuracy_score(y_test_full, y_pred_full_ngram)
print(f"Accuracy (N-Grams on Full Data): {acc_full_ngram * 100:.2f}%")

# Checkpoint Export: Save immediately to prevent loss on later OOM errors
import os
import pickle
output_dir = '/kaggle/working'
os.makedirs(output_dir, exist_ok=True)
ngram_filename = os.path.join(output_dir, 'sentiment_model_ngram_full.pkl')
print(f"Checkpoint: Exporting N-Grams model to {ngram_filename}...")
with open(ngram_filename, 'wb') as f:
    pickle.dump(ngram_model, f)

# %% [markdown]
# ### 4.3. Learning Curve Analysis
# Let's prove if adding more data significantly improves the model's accuracy, or if the TF-IDF approach has reached its theoretical ceiling (plateau).

# %%
print("\nGenerating Learning Curve...")
sample_size = min(500000, len(df_cleaned))
df_learning = df_cleaned.sample(n=sample_size, random_state=42)

train_sizes_arr = np.array([10000, 50000, 100000, 200000])
train_sizes_res, train_scores, test_scores = learning_curve(
    Pipeline([('tfidf', TfidfVectorizer(stop_words='english', max_df=0.8)), ('clf', LogisticRegression(max_iter=1000))]),
    df_learning['CommentText'], df_learning['Sentiment'], 
    train_sizes=train_sizes_arr, cv=3, scoring='accuracy', n_jobs=-1
)

test_scores_mean = np.mean(test_scores, axis=1)
for size, score in zip(train_sizes_res, test_scores_mean):
    print(f"Data Size: {size:,} rows --> Accuracy: {score*100:.2f}%")

# %% [markdown]
# ## 5. Performance Optimization: Hybrid Feature Engineering
# TF-IDF only counts words. It cannot understand context, punctuation emphasis, or capitalization.
# We will combine TF-IDF with handcrafted meta-features to give the model richer signals.
#
# **Bottleneck identified:** TF-IDF treats "NOT good" the same as "good" (after stopword removal).
# VADER understands negation, punctuation emphasis, and slang natively.

# %% [markdown]
# ### 5.1. Quick Test on 50k Sample (Measure Before Scaling)

# %%
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler

analyzer = SentimentIntensityAnalyzer()

def extract_meta_features(text_series):
    """Extract handcrafted features that TF-IDF cannot capture."""
    meta = pd.DataFrame(index=text_series.index)
    
    # VADER Sentiment Scores (understands negation, slang, punctuation)
    vader_scores = text_series.apply(lambda x: analyzer.polarity_scores(str(x)))
    meta['vader_compound'] = vader_scores.apply(lambda x: x['compound'])
    meta['vader_pos'] = vader_scores.apply(lambda x: x['pos'])
    meta['vader_neg'] = vader_scores.apply(lambda x: x['neg'])
    meta['vader_neu'] = vader_scores.apply(lambda x: x['neu'])
    
    # Punctuation features
    meta['exclamation_count'] = text_series.apply(lambda x: str(x).count('!'))
    meta['question_count'] = text_series.apply(lambda x: str(x).count('?'))
    
    # Capitalization ratio ("THIS IS AMAZING" vs "this is amazing")
    meta['caps_ratio'] = text_series.apply(lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1))
    
    # Text length (short = emotional, long = neutral/explanatory)
    meta['text_length'] = text_series.apply(lambda x: len(str(x)))
    
    # Word count
    meta['word_count'] = text_series.apply(lambda x: len(str(x).split()))
    
    # Negation word count ("not", "never", "no", "don't", "won't", etc.)
    negation_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing',
                      "don't", "doesn't", "didn't", "won't", "wouldn't",
                      "can't", "couldn't", "shouldn't", "isn't", "aren't",
                      "wasn't", "weren't", "haven't", "hasn't", "hadn't"}
    meta['negation_count'] = text_series.apply(lambda x: sum(1 for w in str(x).lower().split() if w in negation_words))
    
    # Word elongation count ("sooooo", "goood", "amazzzzing")
    meta['elongation_count'] = text_series.apply(lambda x: len(re.findall(r'(.)\1{2,}', str(x))))
    
    # Emoji/emoticon count (common text emoticons)
    meta['emoticon_count'] = text_series.apply(lambda x: len(re.findall(r'[:;=][-~]?[)D(P/\\|]|<3|\bxD\b|\bXD\b', str(x))))
    
    return meta

print("--- Hybrid Features: Quick Test on 50k Sample ---")
df_50k = df_cleaned.sample(n=50000, random_state=42)
X_50k = df_50k['CommentText']
y_50k = df_50k['Sentiment']
X_train_50k, X_test_50k, y_train_50k, y_test_50k = train_test_split(X_50k, y_50k, test_size=0.2, random_state=42, stratify=y_50k)

print(f"50k Training set: {X_train_50k.shape[0]} samples")
print(f"50k Testing set: {X_test_50k.shape[0]} samples")

# Step 1: TF-IDF features (sublinear_tf compresses term frequencies logarithmically)
tfidf_vec = TfidfVectorizer(stop_words=safe_stopwords, max_df=0.8, ngram_range=(1, 3), sublinear_tf=True)
X_train_tfidf = tfidf_vec.fit_transform(X_train_50k)
X_test_tfidf = tfidf_vec.transform(X_test_50k)

# Step 2: Meta features
print("Extracting VADER + meta features (this may take 1-2 minutes)...")
meta_train = extract_meta_features(X_train_50k)
meta_test = extract_meta_features(X_test_50k)

# Scale meta features to match TF-IDF range
scaler = StandardScaler()
meta_train_scaled = scaler.fit_transform(meta_train)
meta_test_scaled = scaler.transform(meta_test)

# Step 3: Combine TF-IDF + Meta features
X_train_hybrid = hstack([X_train_tfidf, csr_matrix(meta_train_scaled)])
X_test_hybrid = hstack([X_test_tfidf, csr_matrix(meta_test_scaled)])

# Step 4: Train and evaluate
hybrid_model = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
hybrid_model.fit(X_train_hybrid, y_train_50k)
y_pred_hybrid_50k = hybrid_model.predict(X_test_hybrid)
acc_hybrid_50k = accuracy_score(y_test_50k, y_pred_hybrid_50k)

print(f"\nAccuracy (Hybrid Features on 50k): {acc_hybrid_50k * 100:.2f}%")
print(f"Improvement over pure N-Grams on 50k: +{(acc_hybrid_50k - acc_samp_ngram) * 100:.2f}%")

# %% [markdown]
# ### 5.2. Scale Hybrid Features to Full Dataset

# %%
print("\n--- Hybrid Features: Scaling to Full Dataset ---")

# TF-IDF on full data (with sublinear_tf)
tfidf_vec_full = TfidfVectorizer(stop_words=safe_stopwords, max_df=0.8, ngram_range=(1, 3), sublinear_tf=True)
X_train_tfidf_full = tfidf_vec_full.fit_transform(X_train_full)
X_test_tfidf_full = tfidf_vec_full.transform(X_test_full)

# Meta features on full data
print("Extracting VADER + meta features on full dataset (this may take 5-10 minutes)...")
meta_train_full = extract_meta_features(X_train_full)
meta_test_full = extract_meta_features(X_test_full)

scaler_full = StandardScaler()
meta_train_full_scaled = scaler_full.fit_transform(meta_train_full)
meta_test_full_scaled = scaler_full.transform(meta_test_full)

# Combine
X_train_hybrid_full = hstack([X_train_tfidf_full, csr_matrix(meta_train_full_scaled)])
X_test_hybrid_full = hstack([X_test_tfidf_full, csr_matrix(meta_test_full_scaled)])

# Train and evaluate
hybrid_model_full = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
hybrid_model_full.fit(X_train_hybrid_full, y_train_full)
y_pred_hybrid_full = hybrid_model_full.predict(X_test_hybrid_full)
acc_hybrid_full = accuracy_score(y_test_full, y_pred_hybrid_full)

print(f"\nAccuracy (Hybrid Features on Full Data): {acc_hybrid_full * 100:.2f}%")
print(f"Improvement over pure N-Grams on Full Data: +{(acc_hybrid_full - acc_full_ngram) * 100:.2f}%")

# Checkpoint Export
hybrid_bundle = {
    'tfidf': tfidf_vec_full,
    'scaler': scaler_full,
    'model': hybrid_model_full,
    'extract_meta_features': extract_meta_features
}
hybrid_filename = os.path.join(output_dir, 'sentiment_model_hybrid_full.pkl')
print(f"Checkpoint: Exporting Hybrid model bundle to {hybrid_filename}...")
with open(hybrid_filename, 'wb') as f:
    pickle.dump(hybrid_bundle, f)

# %% [markdown]
# ## 6. Performance Optimization: Stacking Ensemble
# Instead of relying on a single model, we stack multiple diverse classifiers.
# Each base model "sees" the data differently, and a meta-learner combines their strengths.

# %% [markdown]
# ### 6.1. Quick Test on 50k Sample

# %%
from sklearn.ensemble import StackingClassifier

print("\n--- Stacking Ensemble: Quick Test on 50k Sample ---")

# Base estimators (diverse algorithms)
base_estimators = [
    ('lr', LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)),
    ('svc', LinearSVC(max_iter=2000)),
    ('nb', MultinomialNB(alpha=0.1))
]

# We must use a non-negative input for NB, so we use Hybrid features only for LR/SVC
# and build a stacking on the TF-IDF matrix (which is non-negative)
stacking_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=3,
    n_jobs=-1,
    passthrough=False
)

stacking_model.fit(X_train_tfidf, y_train_50k)
y_pred_stack_50k = stacking_model.predict(X_test_tfidf)
acc_stack_50k = accuracy_score(y_test_50k, y_pred_stack_50k)

print(f"Accuracy (Stacking Ensemble on 50k): {acc_stack_50k * 100:.2f}%")
print(f"Improvement over pure N-Grams on 50k: +{(acc_stack_50k - acc_samp_ngram) * 100:.2f}%")

# %% [markdown]
# ### 6.2. Scale Stacking to Full Dataset

# %%
print("\n--- Stacking Ensemble: Scaling to Full Dataset ---")
print("Note: This will take several minutes due to 3-fold CV on 800k rows.")

stacking_model_full = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=3,
    n_jobs=-1,
    passthrough=False
)

stacking_model_full.fit(X_train_tfidf_full, y_train_full)
y_pred_stack_full = stacking_model_full.predict(X_test_tfidf_full)
acc_stack_full = accuracy_score(y_test_full, y_pred_stack_full)

print(f"Accuracy (Stacking Ensemble on Full Data): {acc_stack_full * 100:.2f}%")
print(f"Improvement over pure N-Grams on Full Data: +{(acc_stack_full - acc_full_ngram) * 100:.2f}%")

# %% [markdown]
# ## 7. Final Results Summary

# %%
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(f"{'Method':<40} {'50k':>8} {'Full':>8}")
print("-"*60)
print(f"{'Baseline (Logistic Regression)':<40} {test_results[1]['Test Accuracy']*100:>7.2f}%")
print(f"{'N-Grams (1,3) + Safe Stopwords':<40} {acc_samp_ngram*100:>7.2f}% {acc_full_ngram*100:>7.2f}%")
print(f"{'Hybrid (TF-IDF + VADER + Meta)':<40} {acc_hybrid_50k*100:>7.2f}% {acc_hybrid_full*100:>7.2f}%")
print(f"{'Stacking (LR + SVC + NB)':<40} {acc_stack_50k*100:>7.2f}% {acc_stack_full*100:>7.2f}%")

# --- Ultimate Combo: Hybrid features fed into Stacking ---
print("\n--- Ultimate Combo: Hybrid + Stacking on 50k ---")

from sklearn.ensemble import StackingClassifier as SC2

# We cannot use MultinomialNB with negative values from StandardScaler,
# so we use only LR + SVC as base estimators for hybrid input
combo_base = [
    ('lr', LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)),
    ('svc', LinearSVC(max_iter=2000))
]

combo_model_50k = SC2(
    estimators=combo_base,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=3, n_jobs=-1, passthrough=False
)
combo_model_50k.fit(X_train_hybrid, y_train_50k)
y_pred_combo_50k = combo_model_50k.predict(X_test_hybrid)
acc_combo_50k = accuracy_score(y_test_50k, y_pred_combo_50k)
print(f"Accuracy (Ultimate Combo on 50k): {acc_combo_50k * 100:.2f}%")

print("\n--- Ultimate Combo: Scaling to Full Dataset ---")
combo_model_full = SC2(
    estimators=combo_base,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=3, n_jobs=-1, passthrough=False
)
combo_model_full.fit(X_train_hybrid_full, y_train_full)
y_pred_combo_full = combo_model_full.predict(X_test_hybrid_full)
acc_combo_full = accuracy_score(y_test_full, y_pred_combo_full)
print(f"Accuracy (Ultimate Combo on Full Data): {acc_combo_full * 100:.2f}%")

print(f"\n{'Ultimate Combo (Hybrid + Stacking)':<40} {acc_combo_50k*100:>7.2f}% {acc_combo_full*100:>7.2f}%")
print("="*60)

# Determine the best model
full_results = {
    'N-Grams': (acc_full_ngram, ngram_model),
    'Hybrid': (acc_hybrid_full, hybrid_model_full),
    'Stacking': (acc_stack_full, stacking_model_full),
    'Ultimate Combo': (acc_combo_full, combo_model_full)
}
best_name = max(full_results, key=lambda k: full_results[k][0])
best_acc, best_model_obj = full_results[best_name]
print(f"\nBest Model: {best_name} with {best_acc * 100:.2f}% accuracy")

# %% [markdown]
# ## 8. Export the Best Model
# We will save the best performing model into# %%
# Define the target directory for Kaggle output
output_dir = '/kaggle/working'
os.makedirs(output_dir, exist_ok=True)

# For Hybrid or Ultimate Combo, we need to save the vectorizer, scaler, and model separately
if best_name in ('Hybrid', 'Ultimate Combo'):
    hybrid_bundle = {
        'tfidf': tfidf_vec_full,
        'scaler': scaler_full,
        'model': best_model_obj,
        'extract_meta_features': extract_meta_features
    }
    bundle_filename = os.path.join(output_dir, 'sentiment_model_hybrid.pkl')
    print(f"\nExporting {best_name} model bundle to {bundle_filename} ...")
    with open(bundle_filename, 'wb') as f:
        pickle.dump(hybrid_bundle, f)
else:
    model_filename = os.path.join(output_dir, 'sentiment_model_ngram.pkl')
    print(f"\nExporting {best_name} model to {model_filename} ...")
    with open(model_filename, 'wb') as f:
        pickle.dump(best_model_obj, f)

print("Export complete! The backend is now equipped with the best model.")
