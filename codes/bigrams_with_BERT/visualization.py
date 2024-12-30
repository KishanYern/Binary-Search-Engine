import pandas as pd
import numpy as np
import pickle
import faiss
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from wordcloud import WordCloud
from matplotlib_venn import venn2
import spacy

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

def get_lists():
    """Load the postings and review ID lists."""
    with open('../../outputs/postings_list.pkl', 'rb') as file:
        postings = pickle.load(file)
    with open('../../outputs/review_id_list.pkl', 'rb') as file:
        review_id = pickle.load(file)
    return postings, review_id


def extract_collocations(review_texts, min_freq=5):
    """Extract collocations (bigrams) from review texts."""
    tokens = [word for review in review_texts for word in review.split()]
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    bigram_finder.apply_freq_filter(min_freq)
    bigrams_with_scores = bigram_finder.score_ngrams(BigramAssocMeasures.likelihood_ratio)
    return bigrams_with_scores


def get_sentiment_score(text, sentiment_analyzer, max_length=512):
    """Get sentiment score, handling text longer than max token length."""
    encoded = tokenizer.encode(text, add_special_tokens=True)  # Tokenize and encode
    if len(encoded) > max_length:
        # Truncate to max_length
        truncated_text = tokenizer.decode(encoded[:max_length - 1], skip_special_tokens=True)
        sentiment = sentiment_analyzer(truncated_text)[0]['label']
    else:
        sentiment = sentiment_analyzer(text)[0]['label']
    return sentiment


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


def embed_texts(texts, model):
    """Generate embeddings for a list of texts using a SentenceTransformer model."""
    return model.encode(texts, convert_to_tensor=False)


def method5(
    reviews_df, postings, review_id, aspect1, aspect2, opinion, bigrams_with_scores, model, review_embeddings
):
    """Retrieve and rank reviews based on expanded query, sentiment, and top 25% similarity."""
    percentile_score = np.percentile([score for _, score in bigrams_with_scores], 85)
    expanded_terms = expand_query(aspect1, aspect2, opinion, bigrams_with_scores, score_threshold=percentile_score)
    expanded_terms.add(aspect1)
    expanded_terms.add(aspect2)
    expanded_terms.add(opinion)

    method1_results = set(postings[aspect1] + postings[aspect2] + postings[opinion])

    # Perform Boolean retrieval for all expanded terms
    retrieved_indexes = set()
    for term in expanded_terms:
        if term in postings:
            retrieved_indexes.update(postings[term])
    retrieved_reviews = [review_id[index] for index in list(retrieved_indexes)]

    # Filter the reviews DataFrame
    reviews_df['review_id'] = reviews_df['review_id'].str.replace("'", "")
    sub_reviews = reviews_df[reviews_df['review_id'].isin(retrieved_reviews)]

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

    return filtered_reviews['review_id'].tolist(), expanded_terms, similarities, top_25_threshold, [review_id[index] for index in method1_results]


def visualize_method5(expanded_terms, bigrams_with_scores, sub_reviews, similarities, top_25_threshold, method1_reviews):
    """Create visualizations for Method 5 results with POS tagging and score filtering."""
    nlp = spacy.load("en_core_web_sm")
    
    # Define relevant POS tags
    relevant_pos = {"NOUN", "VERB", "ADJ", "ADV"}

    # Filter bigrams based on score threshold
    percentile_score = np.percentile([score for _, score in bigrams_with_scores], 85)
    high_score_bigrams = [(bigram, score) for bigram, score in bigrams_with_scores if score >= percentile_score]

    # Filter bigrams based on presence of aspects/opinion and POS relevance
    filtered_bigrams = []
    for bigram, score in high_score_bigrams:
        # POS tagging the terms in the bigram
        doc = nlp(" ".join(bigram))
        bigram_pos_tags = [token.pos_ for token in doc]

        # Check if bigram contains aspects/opinion and relevant POS
        if any(pos in relevant_pos for pos in bigram_pos_tags):
            filtered_bigrams.append((bigram, score))

    top_bigrams = sorted(filtered_bigrams, key=lambda x: x[1], reverse=True)[:15]
    # Visualization 1: Word Cloud of Expanded Terms
    plt.figure(figsize=(10, 6))
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(expanded_terms))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Expanded Query Terms")
    plt.show()

    # Visualization 2: Venn Diagram of Reviews (Method 1 vs Method 5)
    plt.figure(figsize=(8, 8))
    venn2(
        [set(method1_reviews), set(sub_reviews)],
        set_labels=("Method 1 Reviews", "Method 5 Reviews"),
    )
    plt.title("Venn Diagram of Overlapping Reviews")
    plt.show()

    # Visualization 3: Distribution of Similarity Scores in Method 5
    plt.figure(figsize=(10, 6))
    sns.histplot(similarities[0], bins=20, kde=True, color="blue")
    plt.axvline(x=top_25_threshold, color='red', linestyle='--', label='Top 25% Threshold')
    plt.xlabel("Similarity Scores")
    plt.ylabel("Frequency")
    plt.title("Distribution of Similarity Scores in Method 5")
    plt.legend()
    plt.show()

    # Visualization 4: Top Bigrams with Scores
    plt.figure(figsize=(10, 6))
    bigram_labels = [" ".join(bigram) for bigram, _ in top_bigrams]
    bigram_scores = [score for _, score in top_bigrams]
    sns.barplot(x=bigram_scores, y=bigram_labels, palette="viridis")
    plt.xlabel("Likelihood Ratio Score")
    plt.title("Top 15 Bigrams Related to Query (Filtered by POS and Score)")
    plt.show()



def main():

    # Load data
    postings, review_id = get_lists()
    reviews_df = pd.read_pickle("preprocessed_with_sentiment.pkl")
    review_embeddings = np.load("review_embeddings.npy")

    # Load SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Extract collocations
    bigrams_with_scores = extract_collocations(reviews_df['preproccessed_text'])

    # Method 5 inputs
    aspect1 = "audio"
    aspect2 = "quality"
    opinion = "poor"

    # Method 5 execution
    method5_reviews, expanded_terms, similarities, top_25_threshold, method1_reviews = method5(
        reviews_df, postings, review_id, aspect1, aspect2, opinion, bigrams_with_scores, model, review_embeddings
    )

    # Visualizations
    visualize_method5(expanded_terms, bigrams_with_scores, method5_reviews, similarities, top_25_threshold, method1_reviews)


if __name__ == "__main__":
    main()
