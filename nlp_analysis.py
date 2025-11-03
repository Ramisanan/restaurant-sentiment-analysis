import requests
import pandas as pd
from sqlalchemy import create_engine, text
import hashlib
import json
from transformers import pipeline
import pandas as pd
import os
from keybert import KeyBERT
kw_model = KeyBERT('all-MiniLM-L6-v2')
# It sets up a PostgreSQL connection with the engine
pg_engine = create_engine("postgresql+psycopg2://postgres:1234@localhost:5432/postgres")

def upload_to_db(df, table_name, schema_name='google_review'):
    """
    Uploads a DataFrame to PostgreSQL using a temp table + MERGE (PostgreSQL 15+)
    """
    temp_table = f"temp_{table_name}"

    with pg_engine.connect() as conn:
        trans = conn.begin()
        try:
            # Upload to temporary table first
            df.to_sql(
                name=temp_table,
                con=conn,
                schema=schema_name,
                if_exists='replace',
                index=False
            )

            # Identify primary key columns
            pk_query = f"""
                SELECT a.attname AS column_name
                FROM pg_index i
                JOIN pg_attribute a
                  ON a.attrelid = i.indrelid
                 AND a.attnum = ANY(i.indkey)
                WHERE i.indrelid = '{schema_name}.{table_name}'::regclass
                  AND i.indisprimary;
            """
            res = conn.execute(text(pk_query))
            primary_keys = [r[0] for r in res]

            if not primary_keys:
                raise Exception(f"No primary key found for {schema_name}.{table_name}")

            # Build MERGE statement
            update_keys_condition = " AND ".join([f"target.{pk} = source.{pk}" for pk in primary_keys])
            update_columns = ", ".join([f"{col} = source.{col}" for col in df.columns if col not in primary_keys])

            upsert_sql = f"""
                MERGE INTO {schema_name}.{table_name} AS target
                USING {schema_name}.{temp_table} AS source
                ON {update_keys_condition}
                WHEN MATCHED THEN
                    UPDATE SET {update_columns}
                WHEN NOT MATCHED THEN
                    INSERT ({', '.join(df.columns)})
                    VALUES ({', '.join([f'source.{c}' for c in df.columns])});
            """

            # Execute MERGE
            conn.execute(text(upsert_sql))
            conn.execute(text(f"DROP TABLE {schema_name}.{temp_table}"))
            trans.commit()
            print(f"Upsert completed successfully into {schema_name}.{table_name}")
        except Exception as e:
            trans.rollback()
            print("Upsert failed:", e)
            raise




# Load sentiment and summarization pipelines
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# --- Load review data ---
reviews_df = pd.read_sql("""
    SELECT review_id, text 
    FROM google_review.restaurant_reviews
    WHERE text IS NOT NULL
""", pg_engine)


# inside your NLP loop

# --- Run NLP ---
results = []
for _, row in reviews_df.iterrows():
    review_text = str(row['text'])[:512]  # limit long reviews for model safety
    try:
        sentiment = sentiment_analyzer(review_text)[0]
        summary = summarizer(review_text, max_length=40, min_length=10, do_sample=False)[0]['summary_text']
        keywords = kw_model.extract_keywords(review_text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
        keyword_list = [kw[0] for kw in keywords]

        results.append({
            'review_id': row['review_id'],
            'sentiment_label': sentiment['label'],
            'sentiment_score': float(sentiment['score']),
            'summary': summary,
            'keywords': ', '.join(keyword_list)
        })
    except Exception:
        continue

results_df = pd.DataFrame(results)
print(results_df.head())
upload_to_db(results_df, 'review_nlp_results')