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

if data:
    columns_to_keep = [
        'title', 'description', 'categoryName', 'address',
        'street', 'city', 'postalCode', 'state', 'countryCode', 'phone',
        'totalScore', 'categories', 'reviewsCount', 
        'reviewsDistribution', 'reviews'
    ]

    df = pd.DataFrame(data)
    df = df[[col for col in columns_to_keep if col in df.columns]]

    def generate_hash_id(street, postal_code, title):
        """Generate deterministic unique hash from address and name"""
        base_string = f"{street or ''}_{postal_code or ''}_{title or ''}".lower().strip()
        return hashlib.sha256(base_string.encode()).hexdigest()[:16]  # 16-char hash

    df.insert(
        0,
        'restaurant_id',
        df.apply(lambda x: generate_hash_id(x.get('street'), x.get('postalCode'), x.get('title')), axis=1)
    )

    def combine_address(row):
        parts = [
            str(row.get('street') or ''),
            str(row.get('city') or ''),
            str(row.get('state') or ''),
            str(row.get('postalCode') or ''),
            str(row.get('countryCode') or '')
        ]
        return ', '.join([p.strip() for p in parts if p.strip()])

    df['full_address'] = df.apply(combine_address, axis=1)
    df.rename(columns={
    'categoryName': 'categoryname',
    'totalScore': 'totalscore',
    'reviewsCount': 'reviewscount',
    'reviewsDistribution': 'reviewsdistribution'
}, inplace=True)

    def expand_reviews(df):
        records = []  

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


