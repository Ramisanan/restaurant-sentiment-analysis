import requests
import pandas as pd
from sqlalchemy import create_engine, text
import hashlib
import json
from transformers import pipeline
import pandas as pd
import ollama
import re

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
            print(f"✅ Upsert completed successfully into {schema_name}.{table_name}")
        except Exception as e:
            trans.rollback()
            print("❌ Upsert failed:", e)
            raise

reviews_df = pd.read_sql("""
    SELECT review_id, text 
    FROM google_review.restaurant_reviews
    WHERE text IS NOT NULL
""", pg_engine)




def extract_json_block(text):
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    return {"raw_output": text.strip()}

def analyze_review_ollama(review_text, model_name="llama3"):
    """
    Run inference locally via Ollama and return structured JSON.
    """
    prompt = f"""
    Analyze this customer review.
    1. Determine sentiment (Positive, Neutral, or Negative)
    2. Provide a sentiment_score between 0 and 1
    3. Summarize the review in one sentence
    4. Extract 5 keywords
    Return valid JSON with keys:
    sentiment_label, sentiment_score, summary, keywords (list).

    Review: {review_text}
    """

    try:
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        text_output = response["message"]["content"]
        result = extract_json_block(text_output)
        result["model_name"] = model_name
        return result
    except Exception as e:
        print(f"❌ Error processing review (Ollama): {e}")
        return None
results = []

for _, row in reviews_df.iterrows():
    review_text = str(row['text'])[:512]
    analysis = analyze_review_ollama(review_text, model_name="llama3")

    if not analysis:
        continue

    results.append({
        'review_id': row['review_id'],
        'sentiment_label': analysis.get('sentiment_label'),
        'sentiment_score': float(analysis.get('sentiment_score', 0.0)),
        'summary': analysis.get('summary'),
        'keywords': ', '.join(analysis.get('keywords', []))
                    if isinstance(analysis.get('keywords'), list)
                    else analysis.get('keywords'),
        'model_name': analysis.get('model_name')
    })

if results:
    results_df = pd.DataFrame(results)
    upload_to_db(results_df, 'review_llm_results')