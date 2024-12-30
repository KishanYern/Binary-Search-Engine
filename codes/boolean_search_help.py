import pandas as pd # To output a dataframe
import numpy as np
import pickle # To convert the dataframe to a pickle file
import argparse # Parsing the input
# from string import punctuation  # For the common punctuation words - https://docs.python.org/3/library/string.html#string.punctuation
# import requests
# from nltk.tokenize import word_tokenize  # word_tokenize - https://www.nltk.org/api/nltk.tokenize.word_tokenize.html
# from nltk.stem import WordNetLemmatizer
import joblib
import faiss
import spacy
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from sentence_transformers import SentenceTransformer

def get_lists():
    with open('../outputs/postings_list.pkl', 'rb') as file:
        postings = pickle.load(file)
    
    with open('../outputs/review_id_list.pkl', 'rb') as file:
        review_id = pickle.load(file)

    return postings, review_id

def load_negative_words(file_path="SVC/words/negative-words.txt"):
    negative_words = []
    with open(file_path, 'r') as file:
        for line in file:
            # Ignore comment lines and blank lines
            if line.startswith(';') or line.strip() == '':
                continue
            # Add the word to the list
            negative_words.append(line.strip())
    return negative_words

def load_positive_words(file_path="SVC/words/positive-words.txt"):
    positive_words = []
    with open(file_path, 'r') as file:
        for line in file:
            # Ignore comment lines and blank lines
            if line.startswith(';') or line.strip() == '':
                continue
            # Add the word to the list
            positive_words.append(line.strip())
    return positive_words

# def initialize_filter_lists():
#     # Set of punctuation symbols to be filtered out
#     punctuation_words = set(punctuation)
#     extra_words = ["''", "``", "\n\n"]
#     for word in extra_words:
#         punctuation_words.add(word)

#     # Set of stop words to be filtered out
#     stopwords_list = requests.get(
#         "https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt"
#     ).content
#     stop_words = set(stopwords_list.decode().splitlines())

#     return punctuation_words, stop_words

# def preprocess_text(text, punctuation_words, stop_words):
#     # Tokenize and convert to lowercase
#     tokens = word_tokenize(text.lower())
#     lemmatizer = WordNetLemmatizer()

#     # Filter and lemmatize tokens
#     filtered_tokens = [
#         lemmatizer.lemmatize(word)
#         for word in tokens
#         if word not in punctuation_words and word not in stop_words
#     ]

#     return filtered_tokens

# def preprocess_reviews(review_df):
#     punctuation_words, stop_words = initialize_filter_lists()

#     # Preprocess each review and save the results in a new column
#     review_df["preprocessed_text"] = review_df["review_text"].apply(
#         lambda x: preprocess_text(x, punctuation_words, stop_words)
#     )

#     return review_df

def load_model_and_vectorizer(model_path="SVC/models/Tfidf/svc_model.joblib", vectorizer_path="SVC/models/Tfidf/vectorizer.joblib"):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def embed_texts(texts, model):
    """Generate embeddings for a list of texts using a SentenceTransformer model."""
    return model.encode(texts, convert_to_tensor=True)

def expand_query(aspect1, aspect2, opinion, bigrams_with_scores, top_n=15, score_threshold=0.1):
    """Expand the query using collocations, filtering for relevant POS tags and scores."""
    nlp = spacy.load("en_core_web_sm")
    
    # Define relevant POS tags
    relevant_pos = {"NOUN", "VERB", "ADJ", "ADV"}
    
    # Filter bigrams based on score threshold
    high_score_bigrams = [(bigram, score) for bigram, score in bigrams_with_scores if score >= score_threshold]
    
    # Filter bigrams based on presence of aspects/opinion and POS relevance
    related_bigrams = []
    for bigram, score in high_score_bigrams:
        # POS tagging the terms in the bigram
        doc = nlp(" ".join(bigram))
        bigram_pos_tags = [token.pos_ for token in doc]
        
        # Check if bigram contains aspects/opinion and relevant POS
        if (aspect1 in bigram or aspect2 in bigram or opinion in bigram) and any(
            pos in relevant_pos for pos in bigram_pos_tags
        ):
            related_bigrams.append((bigram, score))
    
    # Select top N bigrams based on score
    top_bigrams = sorted(related_bigrams, key=lambda x: x[1], reverse=True)

    if len(top_bigrams) > top_n:
        top_bigrams = top_bigrams[:top_n]
    
    # Extract unique terms from the top bigrams
    expanded_terms = set(term for bigram, _ in top_bigrams for term in bigram)
    
    return expanded_terms

def extract_collocations(review_texts, min_freq=5):
    """Extract collocations (bigrams) from review texts."""
    tokens = [word for review in review_texts for word in review.split()]
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    bigram_finder.apply_freq_filter(min_freq)
    bigrams_with_scores = bigram_finder.score_ngrams(BigramAssocMeasures.likelihood_ratio)
    return bigrams_with_scores

def method1(postings, review_id, aspect1, aspect2, opinion):
    """
    The first method performs an OR operation between aspect1 OR aspect2 
    and an AND operation on all terms in the opinion list.
    """
    
    # Retrieve the posting lists
    aspect1_list = postings.get(aspect1, [])
    aspect2_list = postings.get(aspect2, [])
    opinion_list = opinion.split(" ")  # Split the opinion into individual words
    
    # Combine aspect1 and aspect2 lists using an OR operation (union)
    aspect_union = set(aspect1_list) | set(aspect2_list)
    
    # Start with the union of aspect1 and aspect2
    union_list = aspect_union
    
    # For each word in the opinion list, perform an AND operation with the current union list
    for word in opinion_list:
        opinion_posting = postings.get(word, [])  # Get the posting list for each word in the opinion
        union_list &= set(opinion_posting)  # Perform AND operation to ensure all opinion terms are present
    
    # Retrieve the review_ids based on the final union of aspects and opinion terms
    result = [review_id[index] for index in union_list]
    
    return result

def method2(postings, review_id, aspect1, aspect2, opinion):
    """
    The second method performs an AND operation between aspect1 AND aspect2 AND all terms in the opinion list.
    """
    
    # Retrieve the posting lists
    aspect1_list = postings.get(aspect1, [])
    aspect2_list = postings.get(aspect2, [])
    opinion_list = opinion.split(" ")  # Split the opinion into individual words
    
    # Find the indexes in common between aspect1 and aspect2
    aspect1_aspect2_intersection = set(aspect1_list) & set(aspect2_list)
    
    # Start with the intersection of aspect1 and aspect2 lists
    intersect_list = aspect1_aspect2_intersection

    # For each word in the opinion list, intersect with the current list
    for word in opinion_list:
        opinion_posting = postings.get(word, [])  # Get the posting list for each word in the opinion
        intersect_list &= set(opinion_posting)  # Perform AND operation with current intersected list
    
    # Retrieve the review_ids based on the intersection of all terms
    result = [review_id[index] for index in intersect_list]
    
    return result


def method3(postings, review_id, aspect1, aspect2, opinion):
    """
    The third method performs an AND operation between aspect1 OR aspect2 and all terms in the opinion list.
    """
    
    # Retrieve the posting lists
    aspect1_list = postings.get(aspect1, [])
    aspect2_list = postings.get(aspect2, [])
    opinion_list = opinion.split(" ")  # Splitting the opinion into individual words
    
    # Find the indexes in both the aspect1 and aspect2 lists
    aspect1_aspect2_union = list(set(aspect1_list + aspect2_list))
    
    # Start with the union of aspect1 and aspect2 lists
    intersect_list = set(aspect1_aspect2_union)

    # For each word in the opinion list, intersect with the current list
    for word in opinion_list:
        opinion_posting = postings.get(word, [])  # Get posting list for each word in opinion
        intersect_list &= set(opinion_posting)  # Perform AND operation with current intersected list
    
    # Retrieve the review_ids based on the intersection of all terms
    result = [review_id[index] for index in intersect_list]
    
    return result

def method4(review_df, postings, review_id, aspect1, aspect2, opinion_terms, model, vectorizer):
    """
    Perform Boolean retrieval and filter reviews based on sentiment using the SVM model.
    """

    # Retrieve the posting lists
    aspect1_list = postings.get(aspect1, [])
    aspect2_list = postings.get(aspect2, [])

    # Boolean retrieval: Perform AND operation on the aspect1 and aspect2 posting lists
    combined_indexes = [index for index in aspect1_list if index in aspect2_list]
    retrieved_reviews = [review_id[index] for index in combined_indexes]

    # Filter the reviews DataFrame
    review_df['review_id'] = review_df['review_id'].str.replace("'", "")
    processed_reviews = review_df[review_df['review_id'].isin(retrieved_reviews)]

    positive_words = load_positive_words()
    negative_words = load_negative_words()

    # Transform reviews using the loaded vectorizer
    reviews_tfidf = vectorizer.transform(processed_reviews['preproccessed_text'])

    # Predict sentiment using the SVM model
    sentiments = model.predict(reviews_tfidf)

    opinion_terms_list = opinion_terms.split(" ") # if opinion term is multiple
    opinion_sentiment = 0

    for opinion_word in opinion_terms_list: # use lexicon to find opinion sentiment value
        if opinion_word in positive_words:
            opinion_sentiment += 1
        if opinion_word in negative_words:
            opinion_sentiment -= 1

    # Filter reviews with positive or negative sentiment based on the opinion sentiment
    if opinion_sentiment >= 0:
        return [
            retrieved_reviews[i] for i in range(len(sentiments)) if sentiments[i] == 1
        ]
    else:
        return [
            retrieved_reviews[i] for i in range(len(sentiments)) if sentiments[i] == 0
        ]

def method5(
    reviews, postings, review_id, aspect1, aspect2, opinion, bigrams_with_scores, model, review_embeddings
):
    """Retrieve and rank reviews based on expanded query, sentiment, and top 25% similarity."""

    percentile_score = np.percentile([score for _, score in bigrams_with_scores], 85)
    expanded_terms = expand_query(aspect1, aspect2, opinion, bigrams_with_scores, score_threshold=percentile_score)
    expanded_terms.add(aspect1)
    expanded_terms.add(aspect2)
    expanded_terms.add(opinion)

    # Perform Boolean retrieval for all expanded terms
    retrieved_indexes = set()
    for term in expanded_terms:
        if term in postings:
            retrieved_indexes.update(postings[term])
    retrieved_reviews = [review_id[index] for index in list(retrieved_indexes)]

    # Filter the reviews DataFrame
    reviews['review_id'] = reviews['review_id'].str.replace("'", "")
    sub_reviews = reviews[reviews['review_id'].isin(retrieved_reviews)]

    # Filter reviews with matching sentiment
    query_sentiment = reviews.loc[reviews['review_text'].str.contains(opinion, case=False, na=False), 'sentiment'].iloc[0]
    if query_sentiment > "3 stars":
        sub_reviews = sub_reviews[sub_reviews['sentiment'] >= query_sentiment]
    else:
        sub_reviews = sub_reviews[sub_reviews['sentiment'] <= query_sentiment]

    # Compute semantic similarity with FAISS
    sub_review_embeddings = review_embeddings[sub_reviews.index]
    query_embedding = embed_texts([" ".join(expanded_terms)], model)
    dimension = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dimension)
    index.add(sub_review_embeddings)

    # Perform similarity search
    distances, indexes = index.search(query_embedding, len(sub_reviews))
    similarities = 1 - (distances / distances.max())  # Convert distances to similarity scores

    # Calculate the 75th percentile similarity score
    top_25_threshold = np.percentile(similarities[0], 75)

    # Get indices of reviews in the top 25% of similarity scores
    valid_indexes = np.where(similarities[0] >= top_25_threshold)[0]

    # Filter reviews based on valid indexes
    filtered_reviews = sub_reviews.iloc[indexes.flatten()[valid_indexes]]

    return filtered_reviews['review_id'].tolist()

def main():

    parser = argparse.ArgumentParser(description="Perform the boolean search.")
    
    parser.add_argument("-a1", "--aspect1", type=str, required=True, default=None, help="First word of the aspect")
    parser.add_argument("-a2", "--aspect2", type=str, required=True, default=None, help="Second word of the aspect")
    parser.add_argument("-o", "--opinion", type=str, required=True, default=None, help="Only word of the opinion")
    parser.add_argument("-m", "--method", type=str, required=True, default=None, help="The method of boolean operation. Methods\
                        can be method1, method2 or method3")

    # Parse the arguments
    args = parser.parse_args()

    # Retrieve the posting and review_id lists
    postings, review_id = get_lists()

    # Extract the arguments
    aspect1, aspect2, opinion = args.aspect1, args.aspect2, args.opinion

    if args.method.lower() == "method1":
        result = method1(postings, review_id, aspect1, aspect2, opinion)
    elif args.method.lower() == "method2":
        result = method2(postings, review_id, aspect1, aspect2, opinion)
    elif args.method.lower() == "method3":
        result = method3(postings, review_id, aspect1, aspect2, opinion)
    elif args.method.lower() == "method4":
        # Load the SVM model and vectorizer
        model, vectorizer = load_model_and_vectorizer()
        review_df = pd.read_pickle("./bigrams_with_BERT/preprocessed_with_sentiment.pkl")

        result = method4(review_df, postings, review_id, aspect1, aspect2, opinion, model, vectorizer)
    elif args.method.lower() == "method5":
        review_df = pd.read_pickle("./bigrams_with_BERT/preprocessed_with_sentiment.pkl")
        review_embeddings = np.load("./bigrams_with_BERT/review_embeddings.npy")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        bigrams_with_scores = extract_collocations(review_df['preproccessed_text'])

        result = method5(review_df, postings, review_id, aspect1, aspect2, opinion, bigrams_with_scores, model, review_embeddings)
    else:
        print("\n!! The method is not supported !!\n")
        return

    revs = pd.DataFrame()
    revs["review_index"] = result
    revs.to_pickle("../outputs/" + args.aspect1 + "_" + args.aspect2 + "_" + args.opinion + "_" + args.method + ".pkl")


if __name__ == "__main__":
    main()
