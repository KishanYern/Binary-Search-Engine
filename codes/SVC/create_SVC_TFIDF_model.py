import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV
import requests
import joblib
import nltk
import time

# Load positive and negative words
def load_negative_words(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if not line.startswith(';') and line.strip() != ""]

def load_positive_words(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if not line.startswith(';') and line.strip() != ""]

negative_words = load_negative_words("words/negative-words.txt")
positive_words = load_positive_words("words/positive-words.txt")

# Initialize filters
def initialize_filter_lists():
    punctuation_words = set(punctuation)
    extra_words = ["''", "``", "\n\n"]
    punctuation_words.update(extra_words)

    stopwords_list = requests.get(
        "https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt"
    ).content
    stop_words = set(stopwords_list.decode().splitlines())

    return punctuation_words, stop_words

# Preprocessing
def preprocess_text(text, punctuation_words, stop_words):
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    return [
        lemmatizer.lemmatize(word) for word in tokens if word not in punctuation_words and word not in stop_words
    ]

def preprocess_reviews(review_df):
    punctuation_words, stop_words = initialize_filter_lists()
    review_df["preproccessed_text"] = review_df["review_text"].apply(
        lambda x: preprocess_text(x, punctuation_words, stop_words)
    )
    return review_df

# Sentiment Label Creation
def create_label(review_df, min_score, max_score):
    def sentiment_value(row):
        sentiment_score = sum([1 if word in positive_words else -1 if word in negative_words else 0 for word in row["preproccessed_text"].split(' ')])
        normalized_sentiment_score = 5 * (sentiment_score - min_score) / (max_score - min_score)
        return int(row['customer_review_rating']) - 3 + normalized_sentiment_score - 3

    return review_df.apply(lambda row: 1 if sentiment_value(row) > 0 else 0, axis=1)

# Save Model and Vectorizer
def save_model_and_vectorizer(model, vectorizer, model_path="models/Tfidf/svc_model.joblib", vectorizer_path="models/Tfidf/vectorizer.joblib"):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

# Model Building
def build_svc_model_with_tuning(review_df):
    sentiment_scores = review_df["preproccessed_text"].apply(
        lambda row: sum([1 if word in positive_words else -1 if word in negative_words else 0 for word in row.split(' ')])
    )
    min_score, max_score = sentiment_scores.min(), sentiment_scores.max()
    review_df['label'] = create_label(review_df, min_score, max_score)

    X = review_df['preproccessed_text']
    y = review_df['label']

    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    print("Unique classes in y_train:", set(y_train))
    print("Class distribution in y_train:", pd.Series(y_train).value_counts())

    param_dist = {'C': np.logspace(-4, 4, 20), 'max_iter': [1000, 2000, 3000], 'loss': ['hinge', 'squared_hinge']}
    linear_svc = LinearSVC(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=linear_svc, param_distributions=param_dist, n_iter=10, scoring='accuracy', cv=5, verbose=2, n_jobs=-1, random_state=42
    )
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))
    print("Classification Report on Test Set:\n", classification_report(y_test, y_pred))

    # Save the model and vectorizer
    save_model_and_vectorizer(best_model, vectorizer)

    # Generate graphs
    generate_graphs(review_df, vectorizer, best_model, X_test, y_test)
    return best_model, vectorizer

# Visualization
def generate_graphs(review_df, vectorizer, best_model, X_test, y_test):
    # Sentiment Scores Distribution
    sentiment_scores = review_df["preproccessed_text"].apply(
        lambda row: sum([1 if word in positive_words else -1 if word in negative_words else 0 for word in row])
    )
    plt.figure(figsize=(10, 6))
    plt.hist(sentiment_scores, bins=30, color='skyblue', edgecolor='black')
    plt.title("Distribution of Sentiment Scores", fontsize=16)
    plt.xlabel("Sentiment Score", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.show()

    # Class Distribution
    class_counts = review_df['label'].value_counts()
    plt.figure(figsize=(8, 6))
    class_counts.plot(kind='bar', color=['salmon', 'lightgreen'])
    plt.title("Distribution of Labels (Positive vs Negative)", fontsize=16)
    plt.xticks(ticks=[0, 1], labels=['Negative', 'Positive'], rotation=0, fontsize=12)
    plt.ylabel("Count", fontsize=14)
    plt.show()

    # TF-IDF Feature Importance
    feature_names = vectorizer.get_feature_names_out()
    coefs = best_model.coef_.flatten()
    top_features = np.argsort(coefs)[-10:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(10), coefs[top_features], color='lightblue')
    plt.yticks(range(10), [feature_names[i] for i in top_features], fontsize=12)
    plt.title("Top 10 Features Contributing to Positive Sentiment", fontsize=16)
    plt.xlabel("Coefficient Value", fontsize=14)
    plt.show()

    # Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, cmap='Blues', values_format='d')
    plt.title("Confusion Matrix", fontsize=16)
    plt.show()

def main():
    review_df = pd.read_pickle("../bigrams_with_BERT/preprocessed_with_sentiment.pkl")
    svm_model, vectorizer = build_svc_model_with_tuning(review_df)
    
if __name__ == "__main__":
    main()
