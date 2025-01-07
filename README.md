This project uses a pre-trained model called RoBERTa to perform sentiment analysis on Zomato users' reviews. The goal is to predict the sentiment of a given review (positive, negative, or neutral) based on the text.


The Project Structure is as follows:


sentiment-analysis/
├── data/
│ └── reviews.csv # Your dataset
├── app.py # Streamlit app
├── sentiment_analysis.py # Sentiment analysis code
├── requirements.txt # List of dependencies
├── results/


Uses the **RoBERTa** pre-trained model for text classification.
Includes a user-friendly **Streamlit app** for interactive testing of review sentiments.
Processes and predicts sentiment on Zomato user reviews dataset.
