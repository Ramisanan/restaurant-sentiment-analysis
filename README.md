# Restaurant Sentiment Analysis 🍽️

This project analyzes Google restaurant reviews using transformer-based NLP models.  
It extracts sentiment labels, scores, and keywords, aggregates results per restaurant,  
and identifies positive and negative drivers across 49 restaurants.

## Features
- PostgreSQL data integration for restaurant and review tables
- HuggingFace Transformers for sentiment & summarization
- Keyword extraction with KeyBERT
- Aggregation of sentiment and keyword trends per restaurant
- Visualization of positive vs. negative keyword drivers

## Visualization Output
- Bar charts for positive and negative keywords
- Restaurant-wise sentiment comparison table

## Run
```bash
python3 nlp_analysis.py
