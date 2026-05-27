# Task-04 : Twitter Sentiment Analysis

## Objective
Analyze and visualize sentiment patterns in social media data to understand public opinion and attitudes towards different brands and topics.

---

## Dataset Used
Twitter Sentiment Dataset

Files Used:
- twitter_training.csv
- twitter_validation.csv

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn
- WordCloud
- Google Colab

---

## Project Features

### Data Cleaning
- Removed missing values
- Cleaned tweet text
- Removed special characters
- Removed stopwords
- Performed stemming

### Exploratory Data Analysis
- Sentiment distribution
- Brand analysis
- Tweet visualization
- Brand-wise sentiment comparison

### NLP Techniques
- Tokenization
- Stopword Removal
- Stemming
- TF-IDF Vectorization

### Machine Learning
- Naive Bayes Classifier
- Sentiment Prediction
- Model Evaluation

---

## Visualizations Included
- Sentiment Countplot
- Brand Distribution Graph
- Positive Word Cloud
- Negative Word Cloud
- Confusion Matrix
- Brand-wise Sentiment Analysis

---

## Machine Learning Workflow

1. Data Loading
2. Data Cleaning
3. Text Preprocessing
4. Feature Extraction using TF-IDF
5. Model Training
6. Prediction
7. Evaluation

---

## Libraries Used

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
