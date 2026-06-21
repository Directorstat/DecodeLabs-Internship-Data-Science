# NLP Sentiment Analysis: Amazon Fine Food Reviews

A Natural Language Processing project that builds a complete text classification
pipeline to mathematically categorise unstructured Amazon product reviews as
Positive or Negative sentiment. Using NLTK for text preprocessing, TF-IDF for
mathematical text representation, and Linear SVM for classification, this project
demonstrates end-to-end NLP engineering from raw text to production-ready predictions.

---

## Problem Statement

Businesses receive thousands of unstructured customer reviews daily but lack the
capacity to read and categorise them manually. A single product on Amazon can
accumulate tens of thousands of reviews. This project builds a machine that reads
human text mathematically and predicts whether each review expresses positive or
negative sentiment, enabling automated review monitoring, product quality tracking,
and customer feedback analysis at scale.

---

## Objectives

- Build a strict text preprocessing pipeline covering tokenization, stop-word
  removal, and lemmatization
- Convert raw text into mathematical feature arrays using TF-IDF vectorization
- Train and compare Naive Bayes and Linear SVM classifiers on 420,644 reviews
- Evaluate models using Macro F1-Score as the primary metric to handle class
  imbalance fairly
- Deliver a production-ready prediction function that classifies new reviews in
  real time

---

## Dataset

- Source: Kaggle / Stanford SNAP
- Name: Amazon Fine Food Reviews
- Link: kaggle.com/datasets/snap/amazon-fine-food-reviews
- Size: 568,454 reviews across 10 columns
- Coverage: October 1999 to October 2012
- Final dataset after cleaning: 525,806 reviews

### Raw Columns Used

| Column | Description |
|---|---|
| Score | Star rating from 1 to 5. Used to engineer sentiment label |
| Text | Full review text. Primary input feature for NLP pipeline |

---

## Methodology

### 1. Label Engineering
- Dropped 42,640 neutral three-star reviews (7.5% of data)
- Reviews with 4 to 5 stars labelled Positive (1)
- Reviews with 1 to 2 stars labelled Negative (0)
- Final distribution: 443,769 Positive (84.4%) and 82,037 Negative (15.6%)

### 2. Text Preprocessing Pipeline
Applied seven sequential cleaning steps to every review:

| Step | Operation | Purpose |
|---|---|---|
| 1 | Lowercase conversion | Treats "Good" and "good" as identical |
| 2 | HTML tag removal | Removes Amazon formatting artifacts |
| 3 | URL removal | Strips irrelevant web links |
| 4 | Punctuation removal | Keeps only alphabetic content |
| 5 | Tokenization | Splits text into individual word units |
| 6 | Stop-word removal | Removes words like "the", "is", "at" |
| 7 | Lemmatization | Reduces words to base forms |

Example transformation:
- Original: "I bought this product and it was absolutely terrible! Worst purchase ever."
- Processed: "bought product absolutely terrible worst purchase ever"

### 3. Train/Test Split
- Split: 80% training (420,644 reviews), 20% testing (105,162 reviews)
- Stratified split preserves 84.40% positive ratio in both sets

### 4. TF-IDF Vectorization
- max_features: 50,000 (top words by frequency)
- ngram_range: (1, 2) captures unigrams and bigrams
- min_df: 5 (ignores words in fewer than 5 documents)
- sublinear_tf: True (log normalization of term frequency)
- Fitted on training data only. Applied to test data without refitting.
- Final matrix: 420,644 rows x 50,000 features

### 5. Model Training
Trained two classifiers and compared performance:
- Naive Bayes: MultinomialNB with alpha=0.1
- Linear SVM: LinearSVC with class_weight='balanced', C=1.0

---

## Results

### Model Comparison

| Metric | Naive Bayes | Linear SVM |
|---|---|---|
| Overall Accuracy | 92% | 94% |
| Negative Precision | 0.90 | 0.76 |
| Negative Recall | 0.55 | 0.90 |
| Negative F1 | 0.68 | 0.83 |
| Positive Precision | 0.92 | 0.98 |
| Positive Recall | 0.99 | 0.95 |
| Positive F1 | 0.95 | 0.96 |
| Macro F1 Score | 0.8191 | 0.8959 |
| Training Time | 0.24 seconds | 43.28 seconds |

### Winner: Linear SVM

Linear SVM outperforms Naive Bayes on every performance metric except training
speed. Most critically, SVM achieves a Negative recall of 0.90 compared to Naive
Bayes's 0.55, correctly identifying 14,804 out of 16,408 negative reviews versus
only 9,038 for Naive Bayes. This 78% improvement in catching negative sentiment
makes SVM the clear production choice. Naive Bayes wins only on speed, which is
irrelevant after training is complete.

---

## Live Prediction Examples

| Review | Predicted Sentiment |
|---|---|
| "This product is absolutely amazing. Best purchase I have ever made." | Positive |
| "Terrible quality. Complete waste of money. Never buying again." | Negative |
| "The flavor was okay but nothing special. Expected better for the price." | Negative |
| "My dog loves this food. Will definitely reorder every month." | Positive |

---

## Key Findings

- Naive Bayes trained in 0.24 seconds but missed 45% of negative reviews due to
  its context-blind word independence assumption
- Linear SVM took 43 seconds to train but reduced missed negative reviews by 78%
  through optimised decision boundary learning across 50,000 dimensions
- Bigram features (ngram_range=(1,2)) captured critical negative phrases like
  "not good", "nothing special", and "expected better" that unigram models miss
- class_weight='balanced' in SVM was essential for improving minority class recall
  from 0.55 to 0.90 without additional data collection
- The word "good" appeared in both positive and negative top feature lists for
  Naive Bayes, confirming context-blindness as its core weakness on sentiment tasks
- Processing 525,806 reviews through the full NLTK pipeline took 265.98 seconds

---

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python |
| Text Preprocessing | NLTK (punkt, stopwords, wordnet, lemmatizer) |
| Text Vectorization | Scikit-learn TfidfVectorizer |
| Classification | Scikit-learn MultinomialNB, LinearSVC |
| Evaluation | Scikit-learn classification_report, confusion_matrix |
| Visualisation | Matplotlib, Seaborn |
| Model Saving | Joblib |
| Development | Google Colab |

---

## Project Structure
---

## How to Run Locally

1. Clone the repository

```bash
git clone https://github.com/Directorstat/DecodeLabs-Internship.git
cd DecodeLabs-Internship
```

2. Install dependencies

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn joblib
```

3. Download NLTK data

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
```

4. Download the dataset from Kaggle
kaggle.com/datasets/snap/amazon-fine-food-reviews

5. Open and run the notebook

```bash
jupyter notebook sentiment_analysis.ipynb
```

6. Predict sentiment on new reviews

```python
import joblib
svm_model = joblib.load('svm_sentiment_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

def predict_sentiment(review_text):
    cleaned = preprocess_text(review_text)
    vectorized = tfidf.transform([cleaned])
    prediction = svm_model.predict(vectorized)[0]
    return "Positive" if prediction == 1 else "Negative"

print(predict_sentiment("This product is outstanding value for money."))
```

---

## Author

Abdulrouf, Waliyullahi Abiodun
Data Scientist
GitHub: github.com/Directorstat
Contact: abiodunajaniabdulrouf@gmail.com

---

## Acknowledgement

This project was developed as part of the Decode Labs Data Science Internship program
under the Technical Mastery Phase: NLP and Sentiment Analysis track. Dataset sourced
from the Stanford Network Analysis Project via Kaggle.

---

## Disclaimer

All review data used in this project is publicly available on Kaggle. No personally
identifiable information beyond publicly visible Amazon usernames was used or retained
at any stage of this analysis.