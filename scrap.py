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
            print(f"✅ Upsert completed successfully into {schema_name}.{table_name}")
        except Exception as e:
            trans.rollback()
            print("❌ Upsert failed:", e)
            raise


new_token = os.getenv("APIFY_API_TOKEN")
url = f"https://api.apify.com/v2/acts/compass~crawler-google-places/runs/last/dataset/items?token={new_token}"

# --- FETCH DATA FROM API ---
try:
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
    else:
        print(f"Error: Received status code {response.status_code}")
        print(f"Response: {response.text}")
        data = []
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
    data = []

# --- PROCESS DATA ---
if data:
    # Columns to keep from the API response
    columns_to_keep = [
        'title', 'description', 'categoryName', 'address',
        'street', 'city', 'postalCode', 'state', 'countryCode', 'phone',
        'totalScore', 'categories', 'reviewsCount', 
        'reviewsDistribution', 'reviews'
    ]

    # Step 1 — Create the base DataFrame
    df = pd.DataFrame(data)
    df = df[[col for col in columns_to_keep if col in df.columns]]

    # Step 2 — Define hash ID generator
    def generate_hash_id(street, postal_code, title):
        """Generate deterministic unique hash from address and name"""
        base_string = f"{street or ''}_{postal_code or ''}_{title or ''}".lower().strip()
        return hashlib.sha256(base_string.encode()).hexdigest()[:16]  # 16-char hash

    # Step 3 — Create restaurant_id column (hash-based)
    df.insert(
        0,
        'restaurant_id',
        df.apply(lambda x: generate_hash_id(x.get('street'), x.get('postalCode'), x.get('title')), axis=1)
    )

    # Step 4 — Combine address fields into a single full_address column
    def combine_address(row):
        parts = [
            str(row.get('street') or ''),
            str(row.get('city') or ''),
            str(row.get('state') or ''),
            str(row.get('postalCode') or ''),
            str(row.get('countryCode') or '')
        ]
        # Join non-empty parts with commas and strip trailing spaces
        return ', '.join([p.strip() for p in parts if p.strip()])

    df['full_address'] = df.apply(combine_address, axis=1)
    df.rename(columns={
    'categoryName': 'categoryname',
    'totalScore': 'totalscore',
    'reviewsCount': 'reviewscount',
    'reviewsDistribution': 'reviewsdistribution'
}, inplace=True)

    # Step 5 — Build the reviews table
    def expand_reviews(df):
        records = []  # store flattened reviews

        for _, row in df.iterrows():
            restaurant_id = row['restaurant_id']
            restaurant_name = row.get('title')
            total_score = row.get('totalscore')  # renamed earlier

            # reviews column contains list of dicts
            reviews = row.get('reviews', [])
            if isinstance(reviews, list):
                for review in reviews:
                    records.append({
                        'restaurant_id': restaurant_id,
                        'restaurant_name': restaurant_name,
                        'total_score': total_score,
                        'review_id': review.get('reviewId'),
                        'reviewer_id': review.get('reviewerId'),
                        'reviewer_name': review.get('name'),
                        'reviewer_number_of_reviews': review.get('reviewerNumberOfReviews'),
                        'is_local_guide': review.get('isLocalGuide'),
                        'stars': review.get('stars'),
                        'text': review.get('text'),
                        'text_translated': review.get('textTranslated'),
                        'publish_at': review.get('publishAt'),
                        'published_at': review.get('publishedAtDate'),
                        'likes_count': review.get('likesCount'),
                        'rating': review.get('rating'),
                        'response_from_owner_date': review.get('responseFromOwnerDate'),
                        'response_from_owner_text': review.get('responseFromOwnerText'),
                        'visited_in': review.get('visitedIn'),
                        'original_language': review.get('originalLanguage'),
                        'translated_language': review.get('translatedLanguage'),
                        'review_detailed_rating': review.get('reviewDetailedRating'),
                        'review_context': review.get('reviewContext'),
                        'review_image_urls': review.get('reviewImageUrls')
                    })

        return pd.DataFrame(records)


    # --- Flatten reviews ---
    reviews_df = expand_reviews(df)
    json_cols = ['review_detailed_rating', 'review_context', 'review_image_urls']

    for col in json_cols:
        if col in reviews_df.columns:
            reviews_df[col] = reviews_df[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x) if x is not None else None
            )
    if 'published_at' in reviews_df.columns:
        reviews_df['published_at'] = pd.to_datetime(
            reviews_df['published_at'], errors='coerce'
        )

    if 'rating' in reviews_df.columns:
        reviews_df['rating'] = pd.to_numeric(reviews_df['rating'], errors='coerce')
    
    if 'response_from_owner_date' in reviews_df.columns:
        reviews_df['response_from_owner_date'] = pd.to_datetime(
            reviews_df['response_from_owner_date'], errors='coerce'
        )
    # --- Drop reviews column from main table ---
    df.drop(columns=['reviews'], inplace=True, errors='ignore')


    def convert_json_columns(df):
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
        return df


    restaurant_df = df[['restaurant_id', 'title', 'description', 'categoryname', 'full_address', 'phone', 'totalscore', 'reviewscount', 'reviewsdistribution']]
    print(restaurant_df)
    if 'reviewsdistribution' in restaurant_df.columns:
        restaurant_df['reviewsdistribution'] = restaurant_df['reviewsdistribution'].apply(
            lambda x: json.dumps(x) if isinstance(x, (dict, list)) else str(x) if x is not None else None
        )

    upload_to_db(restaurant_df, 'restaurant_details')
    upload_to_db(reviews_df, 'restaurant_reviews')




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
upload_to_db(results_df, 'review_llm_results')
hf_token = os.getenv("HF_API_TOKEN")
model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
url = f"https://api-inference.huggingface.co/models/{model}"
headers = {"Authorization": f"Bearer {hf_token}"}


