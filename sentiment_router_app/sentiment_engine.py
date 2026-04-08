from transformers import pipeline
import streamlit as st

@st.cache_resource
def get_sentiment_pipeline():
    """
    Loads and caches the sentiment analysis pipeline.
    Using a robust 3-label model: 'cardiffnlp/twitter-roberta-base-sentiment'
    Labels: 0 -> Negative, 1 -> Neutral, 2 -> Positive
    """
    model_path = "cardiffnlp/twitter-roberta-base-sentiment"
    return pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

def analyze_query(query: str):
    """
    Analyzes the sentiment of a query and returns a normalized label.
    """
    pipe = get_sentiment_pipeline()
    result = pipe(query)[0]
    
    label_map = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive"
    }
    
    sentiment = label_map.get(result["label"], "neutral")
    score = result["score"]
    
    return sentiment, score
