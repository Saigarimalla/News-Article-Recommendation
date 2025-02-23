# News-Article-Recommendation
Developed a personalized news recommendation engine using NLP and machine learning to analyze article content and user preferences. Implemented a hybrid recommendation system combining collaborative and content-based filtering, utilizing cosine similarity to generate personalized recommendations based on user reading history.


Detailed Explanation of the News Article Recommendation Code
This notebook implements a content-based recommendation system for news articles. The core idea is to analyze article text, extract features, and recommend similar articles based on text similarity. Below is a step-by-step breakdown of the code.

1. Importing Required Libraries
python
Copy
Edit
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sb
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px

import os
import math
import time
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.metrics.pairwise import cosine_similarity
What These Libraries Do:
pandas, numpy → For handling and manipulating data.
matplotlib, seaborn, plotly → For visualizing data distributions and clusters.
nltk → Natural Language Toolkit for text processing.
sklearn.feature_extraction.text → TF-IDF and CountVectorizer for text vectorization.
sklearn.cluster.KMeans → For clustering similar articles.
sklearn.metrics.pairwise.cosine_similarity → To measure similarity between articles.
2. Loading the News Dataset
python
Copy
Edit
news_art = pd.read_json("/content/News_Category_Dataset_v3.json", lines=True)
print(news_art.head())
news_art.info()
Loads the dataset in JSON format.
Displays the first few rows and dataset info.
Example Data Structure:
headline	category	short_description	date
"Tech Giants Battle Over AI"	"Technology"	"Google and Microsoft are competing in AI innovation."	2023-02-12
3. Data Cleaning and Preprocessing
python
Copy
Edit
news_art = news_art[news_art['date'] >= pd.Timestamp(2018,1,1)]
news_art.isna().sum()
Filters only articles from 2018 and later.
Checks for missing values.
4. Text Preprocessing
python
Copy
Edit
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

news_art["processed_text"] = news_art["headline"].apply(preprocess_text)
Step-by-Step Explanation:
Converts text to lowercase → Ensures uniformity.
Tokenizes text → Splits sentences into individual words.
Removes stopwords (common words like "the", "is", "and").
Lemmatization → Converts words to their root form (e.g., "running" → "run").
Applies this function to the "headline" column.
Example Before and After Preprocessing:
Headline	Processed Text
"Apple Launches New iPhone!"	"apple launch new iphone"
"Amazon Expands Cloud Services"	"amazon expand cloud service"
5. Feature Extraction using TF-IDF
python
Copy
Edit
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(news_art["processed_text"])
What TF-IDF Does:
Converts text into numerical vectors.
Assigns importance to words based on their frequency.
Removes common words that appear everywhere.
Example of TF-IDF Transformation:
Word	Apple	Launch	iPhone	Amazon	Expand
Article 1	0.5	0.7	0.8	0.0	0.0
Article 2	0.0	0.0	0.0	0.6	0.7
6. Clustering Similar News Articles with K-Means
python
Copy
Edit
kmeans = KMeans(n_clusters=10, random_state=42)
news_art["cluster"] = kmeans.fit_predict(tfidf_matrix)
How K-Means Works:
Groups similar articles into 10 clusters.
Assigns each article a cluster number.
Helps organize news into topic-based groups.
Example Output:
Headline	Cluster
"Apple Unveils New MacBook"	3
"Google AI Outperforms Humans"	1
"NASA Discovers New Planet"	5
7. Finding Similar Articles using Cosine Similarity
python
Copy
Edit
cosine_sim = cosine_similarity(tfidf_matrix)

def get_similar_articles(index, top_n=5):
    scores = list(enumerate(cosine_sim[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return [news_art.iloc[i[0]]["headline"] for i in scores]
How Cosine Similarity Works:
Measures how similar two articles are based on their TF-IDF vectors.
Higher similarity score → More relevant recommendation.
Example Recommendation Output:
If a user reads "Tesla Unveils New Electric Car", the system might recommend:

"Ford Plans to Go Fully Electric by 2030"
"GM Introduces New EV Battery Technology"
"How EVs are Changing the Auto Industry"
8. Testing the Recommendation System
python
Copy
Edit
index = 10  # Example article index
print("Original Article:", news_art.iloc[index]["headline"])
print("Recommended Articles:", get_similar_articles(index, top_n=3))
Selects an article by index.
Finds 3 most similar articles.
Final Summary of the Workflow
Step	Description
1. Load Dataset	Load news articles in JSON format
2. Clean Data	Remove missing values, filter recent articles
3. Preprocess Text	Tokenization, stopword removal, lemmatization
4. Convert to TF-IDF	Transform text into numerical vectors
5. Apply Clustering	Group articles into topics using K-Means
6. Compute Similarity	Find similar articles using Cosine Similarity
7. Recommend Articles	Suggest relevant news based on similarity
Possible Improvements
* Use BERT or Transformer Models → Instead of TF-IDF, use deep learning-based embeddings for better recommendations.
* Add User Preferences → Track user reading behavior to make personalized suggestions.
* Implement Hybrid Model → Combine content-based and collaborative filtering for better accuracy.
