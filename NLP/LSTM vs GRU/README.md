# LSTM vs GRU for Quora Insincere Question Detection

This project explores a comparison between **Bidirectional LSTM and GRU architectures** for detecting insincere questions using the **Quora Insincere Questions Dataset (1.3M samples)**.

The goal was not just to train models, but to analyze **how data engineering decisions impact NLP performance**.

---

# Project Highlights

• Dataset size: **1.3M questions**  
• Architecture comparison: **BiLSTM vs GRU**  
• Pretrained embeddings: **GloVe + FastText (600 dimensions)**  
• Vocabulary engineering using **Zipf’s Law**  
• Focus on **efficient training pipelines**

---

# Key Engineering Insight

Choosing the correct vocabulary size is critical for large NLP datasets.

Instead of selecting an arbitrary limit, the vocabulary was determined by analyzing the **word frequency distribution**.

Observations:

```
Total raw vocabulary: ~247,077 tokens
Words appearing ≥5 times: 65,073
Vocabulary after preprocessing: ~50K tokens
```

Most words in the dataset appeared **only once or twice**, contributing mostly noise.

Using a **frequency cutoff ≥5** significantly reduced vocabulary size while preserving useful signal.

---

# Embedding Strategy

Two pretrained embeddings were combined:

• **GloVe (840B tokens, 300d)**  
• **FastText Wiki News (300d)**

These were concatenated to create **600-dimensional word embeddings**.

The embedding layer was **frozen during training** to improve stability and reduce training time.

---

# Model Architectures

## Bidirectional LSTM

Captures long-range sequential dependencies by processing text in both directions.

Architecture:

```
Embedding (600d pretrained)
↓
Bidirectional LSTM
↓
GlobalMaxPooling
↓
Dense Layer
↓
Sigmoid Output
```

---

## GRU

GRU is a lighter alternative to LSTM with fewer parameters.

It generally trains **faster**, but sometimes captures sequence structure less effectively.

---

# Results

| Model | F1 Score | Recall | Precision |
|------|------|------|------|
| **Bidirectional LSTM** | **0.8472** | **0.9085** | 0.7937 |
| GRU | 0.6751 | 0.7210 | 0.6347 |

### Key Observation

The **GRU trained faster**, but the **Bidirectional LSTM produced significantly stronger classification performance**.

This highlights the classic trade-off between **training speed and predictive accuracy**.

---

# Visualizations

The notebook includes several analysis visualizations:

• Zipf's Law word frequency distribution  
• Model training curves (Validation Loss & AUC)  
• Final model performance comparison  

These visualizations helped guide key design decisions.

---

# Lessons Learned

1. **Data preprocessing strongly affects embedding coverage**
2. Vocabulary engineering can dramatically improve efficiency
3. Faster architectures (GRU) do not always outperform deeper sequence models
4. Careful dataset analysis often matters more than architecture complexity

---

# Tech Stack

Python  
TensorFlow / Keras  
NumPy  
Pandas  
Matplotlib  

---

# Future Work

Potential improvements:

• Transformer-based models (BERT / RoBERTa)  
• Attention mechanisms on RNN architectures  
• Hyperparameter optimization  
• Better handling of rare tokens and OOV words

---

# Notebook

The full experiment is available in:

```
lstm-vs-gru.ipynb
```

---

# Author

**Hassaan Waheed**

Machine Learning enthusiast interested in **NLP systems, efficient ML pipelines, and large-scale data experimentation**.
