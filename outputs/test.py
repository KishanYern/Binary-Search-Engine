import pandas as pd
import pickle
from nltk.stem import WordNetLemmatizer
from directory_tree import display_tree

def test_review_id_list():
    with open('outputs/review_id_list.pkl', 'rb') as file:
        data = pickle.load(file)

    review_df = pd.read_pickle("../codes/reviews_segment.pkl")
    print(len(data))
    # print(len(review_df))
    
def test_postings_list():
    with open('outputs/postings_list.pkl', 'rb') as file:
        postings = pickle.load(file)

    with open('outputs/review_id_list.pkl', 'rb') as file:
        review = pickle.load(file)
    
    print(postings)
    
def test_outputs(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    
    print(len(data))
    # display_tree('.')


def test_lemmatizer():
    lemmatizer = WordNetLemmatizer()
    print(lemmatizer.lemmatize("broken", pos='a'))

if __name__ == "__main__":
    test_outputs("./wifi_signal_strong_method5.pkl")
