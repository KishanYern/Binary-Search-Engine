import pandas as pd
from nltk.tokenize import word_tokenize  # word_tokenize - https://www.nltk.org/api/nltk.tokenize.word_tokenize.html
from nltk.stem import WordNetLemmatizer
from string import punctuation  # For the common punctuation words - https://docs.python.org/3/library/string.html#string.punctuation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import requests
import nltk
import time
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import joblib
from gensim.models import Word2Vec

def load_negative_words(file_path):
    negative_words = []
    with open(file_path, 'r') as file:
        for line in file:
            # Ignore comment lines and blank lines
            if line.startswith(';') or line.strip() == '':
                continue
            # Add the word to the list
            negative_words.append(line.strip())
    return negative_words

def load_positive_words(file_path):
    positive_words = []
    with open(file_path, 'r') as file:
        for line in file:
            # Ignore comment lines and blank lines
            if line.startswith(';') or line.strip() == '':
                continue
            # Add the word to the list
            positive_words.append(line.strip())
    return positive_words

# Getting positive and negative words from http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
negative_words = load_negative_words("words/negative-words.txt")
positive_words = load_positive_words("words/positive-words.txt")

def initialize_filter_lists():
    # Set of punctuation symbols to be filtered out
    punctuation_words = set(punctuation)
    extra_words = ["''", "``", "\n\n"]
    for word in extra_words:
        punctuation_words.add(word)

    # Set of stop words to be filtered out
    stopwords_list = requests.get(
        "https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt"
    ).content
    stop_words = set(stopwords_list.decode().splitlines())

    return punctuation_words, stop_words

def initialize():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")

def preprocess_text(text, punctuation_words, stop_words):
    # Tokenize and convert to lowercase
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()

    # Filter and lemmatize tokens
    filtered_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in punctuation_words and word not in stop_words
    ]

    return filtered_tokens

def preprocess_reviews(review_df):
    punctuation_words, stop_words = initialize_filter_lists()

    # Preprocess each review and save the results in a new column
    review_df["preprocessed_text"] = review_df["review_text"].apply(
        lambda x: preprocess_text(x, punctuation_words, stop_words)
    )

    return review_df

def min_max_normalize(sentiment_score, min_score, max_score):
    # Scale sentiment score between -1 and 1
    return 5 * (sentiment_score - min_score) / (max_score - min_score)

def sentiment_value(row, min_score, max_score):
    rating = int(row['customer_review_rating'])
    normalized_rating = rating - 3  # Mean rating is assumed to be 3
    sentiment_score = 0

    for word in row["preprocessed_text"]:
        if word in negative_words:
            sentiment_score -= 1
        elif word in positive_words:
            sentiment_score += 1

    # Apply min-max normalization to sentiment score
    normalized_sentiment_score = min_max_normalize(sentiment_score, min_score, max_score) - 3
    return normalized_rating + normalized_sentiment_score

def create_label(review_df, min_score, max_score):
    # Apply sentiment_value and create label column
    return review_df.apply(
        lambda row: 1 if sentiment_value(row, min_score, max_score) > 0 else 0, axis=1
    )

def train_word2vec(reviews):
    tokenized_reviews = [review.split() for review in reviews]
    word2vec_model = Word2Vec(sentences=tokenized_reviews, vector_size=100, window=5, min_count=1, workers=4)
    return word2vec_model

def get_review_embedding(review, model):
    # Access word vectors from the `wv` (KeyedVectors) attribute
    words = review.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)  # Average word vectors for the review
    else:
        return np.zeros(model.vector_size)  # Return zero vector if no words match

def save_model_and_vectorizer(model, vectorizer, model_path="models/Word2Vec/svc_model.joblib", vectorizer_path="models/Word2Vec/vectorizer.joblib"):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

def build_svc_model_with_word2vec(review_df, word2vec_model):
    # Preprocess the review texts
    review_df = preprocess_reviews(review_df)

    # Create label based on sentiment score and rating, use min-max scaling to scale the sentiment
    sentiment_scores = review_df["preprocessed_text"].apply(
        lambda row: sum([1 if word in positive_words else -1 if word in negative_words else 0 for word in row])
    )
    min_score = sentiment_scores.min()
    max_score = sentiment_scores.max()

    # Create label based on sentiment score and rating
    review_df['label'] = create_label(review_df, min_score, max_score)

    # Generate review embeddings using Word2Vec
    X = review_df['preprocessed_text'].apply(lambda text_list: ' '.join(text_list))
    review_embeddings = np.array([get_review_embedding(review, word2vec_model) for review in X])
    y = review_df['label']

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(review_embeddings, y, test_size=0.2, random_state=42)

    # Set up parameter grid for RandomizedSearchCV
    param_dist = {
        'C': np.logspace(-4, 4, 20),  # Test a wide range of C values
        'max_iter': [1000, 2000, 3000],  # Max number of iterations
        'loss': ['hinge', 'squared_hinge'],  # Loss functions for LinearSVC
    }

    # Initialize LinearSVC
    linear_svc = LinearSVC(random_state=42)

    # Perform RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=linear_svc,
        param_distributions=param_dist,
        n_iter=10,  # Number of random samples
        scoring='accuracy',
        cv=5,  # 5-fold cross-validation
        verbose=2,
        n_jobs=-1,  # Use all available CPU cores
        random_state=42
    )

    # Fit the model to the training data
    random_search.fit(X_train, y_train)

    # Best model from RandomizedSearchCV
    best_model = random_search.best_estimator_
    print("Best Parameters:", random_search.best_params_)
    print("Best Cross-Validation Accuracy:", random_search.best_score_)

    # Evaluate on the test set
    y_pred = best_model.predict(X_test)
    print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))
    print("Classification Report on Test Set:\n", classification_report(y_test, y_pred))

    return best_model, word2vec_model

def main():
    review_df = pd.read_pickle("../reviews_segment.pkl")  # Load the dataset

    # Train Word2Vec model
    start_time = time.time()
    word2vec_model = train_word2vec(review_df['review_text'])  # Train Word2Vec on your reviews

    # Train SVM model using Word2Vec embeddings
    svm_model, word2vec_model = build_svc_model_with_word2vec(review_df, word2vec_model)
    end_time = time.time()

    # Print the elapsed time
    print(f"Time to train SVC model with 210000 reviews: {end_time - start_time:.2f} seconds")

    # Save the trained model and Word2Vec embeddings
    save_model_and_vectorizer(svm_model, word2vec_model, model_path="models/svc_model_word2vec.joblib", vectorizer_path="models/word2vec_model.joblib")

if __name__ == "__main__":
    main()