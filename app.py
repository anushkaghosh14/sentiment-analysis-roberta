import streamlit as st
from sentiment_analysis import preprocess_text, classify_sentiment, polarity_scores_roberta
import pandas as pd

# Load the data
file_path = "data/reviews.csv"
data = pd.read_csv(file_path)

# Streamlit app
def main():
    st.title("Sentiment Analysis App")

    # Text input for user to enter a review
    user_review = st.text_area("Enter a review:", height=200)

    if st.button("Analyze"):
        if user_review:
            # Preprocess the input review
            processed_review = preprocess_text(user_review)
            # Get sentiment
            sentiment = classify_sentiment(polarity_scores_roberta(processed_review))
            # Display the result
            st.write(f"Sentiment: **{sentiment}**")
        else:
            st.error("Please enter a review for analysis.")

if __name__ == "__main__":
    main()
