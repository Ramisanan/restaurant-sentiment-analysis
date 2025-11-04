# Restaurant Sentiment Analysis  

This project analyzes Google restaurant reviews using **Transformer-based NLP models** and **Large Language Models (LLMs)**.  
It extracts sentiment labels, scores, and keywords to understand customer opinions and identify improvement areas across 49 restaurants.  

---

## Features  
- **PostgreSQL Integration:** Structured storage for restaurant and review data  
- **Transformer Models:** Hugging Face pipelines for sentiment analysis & summarization  
- **Keyword Extraction:** Using KeyBERT (`all-MiniLM-L6-v2`) for top phrases  
- **LLM Integration (Ollama):** Local inference using LLaMA for comparison with transformers  
- **Comparative Visualization:** Evaluate both NLP and LLM model performance  
- **Restaurant-Level Insights:** Identify positive/negative sentiment drivers  

---

## Visualization Outputs  
- Bar charts comparing NLP vs LLM sentiment alignment  
- Scatter plots showing correlation between sentiment scores  
- Word clouds and bar charts for positive & negative keyword frequencies  
- Restaurant-wise sentiment summaries for decision insights  

---

## Tools & Technologies  
- **Python 3.10+**  
- **PostgreSQL 15+**  
- **Transformers (Hugging Face)**  
- **KeyBERT**  
- **Ollama (for local LLM inference)**  
- **Pandas, Seaborn, Matplotlib, WordCloud**  

---

## Project Workflow  
1. **Data Extraction:**  
   Reviews fetched via the Apify Google Places API.  

2. **Data Processing:**  
   - Normalize and flatten JSON review data.  
   - Generate unique restaurant IDs using hash functions.  
   - Store in PostgreSQL tables:  
     - `restaurant_details`  
     - `restaurant_reviews`  

3. **NLP Model Analysis:**  
   - Sentiment analysis via `cardiffnlp/twitter-roberta-base-sentiment`.  
   - Text summarization using `facebook/bart-large-cnn`.  
   - Keyword extraction with `KeyBERT`.  
   - Results stored in `review_nlp_results`.  

4. **LLM Analysis (Ollama):**  
   - LLaMA (`llama3`) used to analyze reviews with structured JSON output.  
   - Results stored in `review_llm_results`.  

5. **Visualization & Comparison:**  
   - Compare LLM and Transformer sentiment predictions.  
   - Aggregate scores by restaurant.  
   - Identify improvement opportunities using negative keyword patterns.  

---

## Run the Project  
Clone the repository and run the analysis scripts sequentially:  
```bash
# Step 1: NLP model analysis
python3 nlp_analysis.py

# Step 2: LLM-based analysis
python3 llm_analysis.py

# Step 3: Visualization
python3 visualization.ipynb
