import pandas as pd
import pickle
from transformers import pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter

def get_lists():
    with open('./postings_list.pkl', 'rb') as file:
        postings = pickle.load(file)
    
    with open('./review_id_list.pkl', 'rb') as file:
        review_id = pickle.load(file)
    
    return postings, review_id

def determine_opinion_sentiment(opinion_term):
    """
    Determine whether the opinion term is positive or negative using a pre-trained sentiment analysis model.

    Parameters:
        opinion_term (str): The opinion term to classify.

    Returns:
        str: Sentiment of the opinion term ("positive" or "negative").
        float: Confidence score for the predicted sentiment.
    """
    # Load sentiment analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Analyze sentiment
    result = sentiment_pipeline(opinion_term)[0]
    sentiment = result["label"].lower()  # Convert to lowercase ("POSITIVE" or "NEGATIVE")
    
    return sentiment

def calculate_metrics(method4_reviews, relevant_reviews, review_df):
    """
    Calculate precision, recall, and F-score for method4_reviews against relevant_reviews.

    Parameters:
        method4_reviews (list): List of reviews retrieved by method 4.
        relevant_reviews (list): List of relevant reviews based on sentiment and criteria.
        review_df (pd.DataFrame): Entire dataset with all reviews.

    Returns:
        dict: Dictionary containing precision, recall, and F-score.
    """
    # Convert review IDs into binary vectors for evaluation
    review_df['review_id'] = review_df['review_id'].str.replace("'", "")
    all_reviews = review_df['review_id'].tolist()
    method4_vector = [1 if review_id in method4_reviews else 0 for review_id in all_reviews]
    relevant_vector = [1 if review_id in relevant_reviews else 0 for review_id in all_reviews]

    # Calculate metrics
    precision = precision_score(relevant_vector, method4_vector)
    recall = recall_score(relevant_vector, method4_vector)
    f1 = f1_score(relevant_vector, method4_vector)

    return {
        "Precision": precision,
        "Recall": recall,
        "F-Score": f1
    }

def compare_method4(postings, review_id, review_df, aspect1, aspect2, opinion, filename):
    """
    Compare method4_reviews with relevant_reviews and calculate evaluation metrics.

    Parameters:
        postings (dict): Postings list for terms.
        review_id (list): List of review IDs.
        review_df (pd.DataFrame): Entire dataset with reviews and sentiment.
        aspect1 (str): First aspect term.
        aspect2 (str): Second aspect term.
        opinion (str): Opinion term.
        filename (str): Filename containing method4_reviews.

    Returns:
        None
    """
    # Load method4_reviews from file
    with open(filename, 'rb') as file:
        method4_reviews = pickle.load(file)

    # Retrieve the posting lists
    aspect1_list = postings.get(aspect1, [])
    aspect2_list = postings.get(aspect2, [])

    # Boolean retrieval: Perform AND operation on the aspect1 and aspect2 posting lists
    combined_indexes = [index for index in aspect1_list if index in aspect2_list]
    retrieved_reviews = [review_id[index] for index in combined_indexes]
    retrieved_reviews = set(retrieved_reviews)

    # Determine sentiment of opinion
    opinion_sentiment = determine_opinion_sentiment(opinion)

    review_df['review_id'] = review_df['review_id'].str.replace("'", "")

    # Filter relevant reviews based on sentiment
    if opinion_sentiment == "positive":
        relevant_reviews = review_df[
            (review_df['review_id'].isin(retrieved_reviews)) & (review_df['sentiment'] >= "3 stars")
        ]['review_id'].tolist()
    else:  # "negative"
        relevant_reviews = review_df[
            (review_df['review_id'].isin(retrieved_reviews)) & (review_df['sentiment'] < "3 stars")
        ]['review_id'].tolist()

    # Calculate metrics
    metrics = calculate_metrics(method4_reviews['review_index'].tolist(), relevant_reviews, review_df)

    # Print metrics as a table
    metrics_table = pd.DataFrame([metrics])
    print(metrics_table)

def compare_method5(postings, review_id, review_df, aspect1, aspect2, opinion, filename):
    """
    Compare method5_reviews with relevant_reviews and calculate evaluation metrics.

    Parameters:
        postings (dict): Postings list for terms.
        review_id (list): List of review IDs.
        review_df (pd.DataFrame): Entire dataset with reviews and sentiment.
        aspect1 (str): First aspect term.
        aspect2 (str): Second aspect term.
        opinion (str): Opinion term.
        filename (str): Filename containing method4_reviews.

    Returns:
        None
    """
    # Load method4_reviews from file
    with open(filename, 'rb') as file:
        method5_reviews = pickle.load(file)

    # Retrieve the posting lists
    aspect1_list = postings.get(aspect1, [])
    aspect2_list = postings.get(aspect2, [])

    opinion_terms = opinion.split(' ')

    # Perform Boolean retrieval for all expanded terms
    retrieved_indexes = set()
    retrieved_indexes.update(aspect1_list)
    retrieved_indexes.update(aspect2_list)
    for term in opinion_terms:
        if term in postings:
            retrieved_indexes.update(postings[term])
    retrieved_reviews = [review_id[index] for index in list(retrieved_indexes)]

    # Determine sentiment of opinion
    opinion_sentiment = determine_opinion_sentiment(opinion)

    review_df['review_id'] = review_df['review_id'].str.replace("'", "")

    # Filter relevant reviews based on sentiment
    if opinion_sentiment == "positive":
        relevant_reviews = review_df[
            (review_df['review_id'].isin(retrieved_reviews)) & (review_df['sentiment'] >= "3 stars")
        ]['review_id'].tolist()
    else:  # "negative"
        relevant_reviews = review_df[
            (review_df['review_id'].isin(retrieved_reviews)) & (review_df['sentiment'] < "3 stars")
        ]['review_id'].tolist()

    # Calculate metrics
    metrics = calculate_metrics(method5_reviews['review_index'].tolist(), relevant_reviews, review_df)

    # Print metrics as a table
    metrics_table = pd.DataFrame([metrics])
    print(metrics_table)


if __name__ == '__main__':
    postings, review_id = get_lists()

    review_df = pd.read_pickle("../codes/bigrams_with_BERT/preprocessed_with_sentiment.pkl")

    compare_method5(postings, review_id, review_df, "image", "quality", "sharp", "./image_quality_sharp_method5.pkl")
