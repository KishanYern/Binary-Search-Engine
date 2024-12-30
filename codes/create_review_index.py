import pandas as pd
import pickle

def create_review_index(review_df):
    # Build the postings list with filtering based on the stop words and frequencies
    review_index = {}
    for index, row in review_df.iterrows():
        review_index[index] = row['review_id'][1:-1]

    return review_index
    

def main():
    # Load the data from pickle file
    review_df = pd.read_pickle("codes/reviews_segment.pkl")
    
    # Create the review id list
    review_index = create_review_index(review_df)
    
    # Save the review id list to a pickle file
    with open('outputs/review_id_list.pkl', 'wb') as file:
        pickle.dump(review_index, file)

if __name__ == "__main__":
    main()
