import pandas as pd
import pickle
from collections import defaultdict
import nltk # For finding and installing dependencies
from nltk.tokenize import word_tokenize # word_tokenize - https://www.nltk.org/api/nltk.tokenize.word_tokenize.html
from nltk.stem import WordNetLemmatizer
from string import punctuation # For the common punctuation words - https://docs.python.org/3/library/string.html#string.punctuation
import requests # https://gist.github.com/sebleier/554280 comments section compiled all the stop words


def initialize_filter_lists():
    # Set of punctuation symbols to be filtered out
    global punctuation_words
    punctuation_words = set(punctuation) 
    extra_words = ["''", "``", "\n\n"]

    for word in extra_words:
        punctuation_words.add(word)

    # Set of stop words to be filtered out
    global stop_words
    stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
    stop_words = set(stopwords_list.decode().splitlines()) 

def initialize():
    # Download the dependencies if they are not installed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    # Initialize the filter lists
    initialize_filter_lists()

def create_filtered_postings_list(review_df, min_frequency=5):
    # Temporary dictionary to count occurrences
    word_count = defaultdict(int)

    # Create the lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Count the occurrences of each word
    for index, row in review_df.iterrows():
        words = word_tokenize(row['review_text'].lower())
        for word in words:
            # Lemmatize the word and filter stop words
            lemmatized_word = lemmatizer.lemmatize(word)
            if lemmatized_word not in stop_words:
                word_count[lemmatized_word] += 1

    # Build the postings list with filtering based on stop words, punctuation, and frequencies
    postings = defaultdict(list)
    for index, row in review_df.iterrows():
        words = word_tokenize(row['review_text'].lower())
        for word in words:
            # Lemmatize the word before adding to postings
            lemmatized_word = lemmatizer.lemmatize(word)
            if (
                lemmatized_word not in punctuation_words and
                lemmatized_word not in stop_words and
                word_count[lemmatized_word] >= min_frequency
            ):
                postings[lemmatized_word].append(index)

    return postings


def main():
    # Initialize the filter lists and install dependencies
    initialize()

    # Load the input DataFrame
    review_df = pd.read_pickle("codes/reviews_segment.pkl")
    
    # Create and filter postings list
    postings = create_filtered_postings_list(review_df)
    
    # Save the filtered postings list
    with open("outputs/postings_list.pkl", "wb") as file:
        pickle.dump(postings, file)

if __name__ == "__main__":
    main()
