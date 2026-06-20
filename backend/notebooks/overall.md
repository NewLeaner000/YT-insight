## Final Results & Model Benchmarks

The training process followed a Scale-up strategy: Rapid prototyping and evaluation on a 50,000-row sample to identify the best techniques, followed by full-scale training on the complete 800,000-row dataset to maximize performance.

| Sequence | Architecture | Accuracy (50k Sample) | Accuracy (800k Full Data) | Notes & Optimizations |
|:---|:---|:---:|:---:|:---|
| 1 | **Baseline (Logistic Regression)** | ~64.00% | 68.20% | Basic TF-IDF implementation. Performance is bottlenecked by the inability to capture context beyond isolated words. |
| 2 | **N-Grams (1,3) + Sublinear TF** | ~63.99% | 71.54% | Optimized by compressing term frequencies logarithmically (sublinear_tf) and utilizing trigrams to capture local context and negations. |
| 3 | **Hybrid (TF-IDF + VADER + Meta)** | 66.11% | ~72.50% | Engineered custom metadata features (punctuation counts, capitalization ratios, slang detection) and integrated VADER lexicon scores to understand sentiment nuances and sarcasm. |
| 4 | **Stacking Ensemble (LR + SVC + NB)** | 65.23% | **73.30%** | Implemented the SAGA solver to optimize RAM allocation for massive sparse matrices. Combined predictions from 3 diverse linear models to offset individual weaknesses. Achieved peak performance for CPU-based ML. |

**Conclusion:** The Stacking Ensemble combined with Hybrid Features successfully broke the 72% default ceiling typical of traditional Bag-of-Words text analysis. Reaching 73.30% accuracy is highly impressive for a 3-class classification problem (Positive/Neutral/Negative) on heavily noisy data such as YouTube comments containing severe misspellings and internet slang.

**Selected Model for Deployment:** sentiment_model_stacking.pkl
