# News-Article-Recommendation
Developed a personalized news recommendation engine using NLP and machine learning to analyze article content and user preferences. Implemented a hybrid recommendation system combining collaborative and content-based filtering, utilizing cosine similarity to generate personalized recommendations based on user reading history.

Explanation of the News Article Recommendation Code
The provided Jupyter Notebook includes the following major steps:

1. Importing Libraries
The code imports necessary Python libraries for data processing, visualization, and machine learning:

Pandas, NumPy: For handling datasets.
Matplotlib, Seaborn, Plotly: For data visualization.
NLTK: For text preprocessing (tokenization, stopword removal, and lemmatization).
Scikit-learn: For text vectorization (TF-IDF), clustering (K-Means), and similarity measurement (cosine similarity).
2. Loading the Dataset
python
Copy
Edit
news_art = pd.read_json("/content/News_Category_Dataset_v3.json", lines=True)
news_art.info()
The dataset is loaded from a JSON file.
info() is used to check dataset structure.
3. Data Cleaning and Preprocessing
python
Copy
Edit
news_art = news_art[news_art['date'] >= pd.Timestamp(2018,1,1)]
news_art.isna().sum()
Filters articles from 2018 onwards.
Checks for missing values.
4. Text Preprocessing
python
Copy
Edit
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())  
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

news_art["processed_text"] = news_art["headline"].apply(preprocess_text)
Converts text to lowercase.
Removes stopwords and non-alphanumeric characters.
Lemmatizes words (reduces words to their base form).
5. Feature Extraction using TF-IDF
python
Copy
Edit
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(news_art["processed_text"])
Converts processed text into numerical feature vectors using TF-IDF.
6. Clustering with K-Means
python
Copy
Edit
kmeans = KMeans(n_clusters=10, random_state=42)
news_art["cluster"] = kmeans.fit_predict(tfidf_matrix)
Groups similar articles into 10 clusters using K-Means.
7. Finding Similar Articles using Cosine Similarity
python
Copy
Edit
cosine_sim = cosine_similarity(tfidf_matrix)

def get_similar_articles(index, top_n=5):
    scores = list(enumerate(cosine_sim[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return [news_art.iloc[i[0]]["headline"] for i in scores]
Computes cosine similarity between articles.
Recommends top N similar articles based on similarity scores.
Summary
This notebook builds a news article recommendation system using:

Text preprocessing (tokenization, stopword removal, lemmatization)
TF-IDF vectorization (converting text to numerical representation)
K-Means clustering (grouping similar articles)
Cosine similarity (finding most similar articles)
Would you like me to modify or improve any part of the recommendation model? ​​









