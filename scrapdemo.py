# important Libraries 
import asyncio
from playwright.async_api import async_playwright
import pandas as pd
from sqlalchemy import create_engine, text
import pandas as pd

# It sets up a PostgreSQL connection with the engine
pg_engine = create_engine("postgresql+psycopg2://postgres:1234@localhost:5432/postgres")


def upload_to_db(data):
    df = pd.DataFrame(data)
    df = df.drop_duplicates('product_id', keep='last')
    df.to_sql(name='temp_product_details', con=pg_engine, schema='products', if_exists='replace', index=False)

    with pg_engine.connect() as conn:
        trans = conn.begin()
        try:
            pk_query = """
                SELECT a.attname as column_name
                FROM pg_index i
                JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                WHERE i.indrelid = 'products.product_details'::regclass AND i.indisprimary;
            """
            res = conn.execute(text(pk_query))
            primary_keys = [row['column_name'] for row in res]

            update_keys_condition = " AND ".join([f"target.{pk} = source.{pk}" for pk in primary_keys])
            update_columns = ", ".join([f"{col} = source.{col}" for col in df.columns])
            upsert_sql = f"""
                MERGE INTO products.product_details AS target
                USING products.temp_product_details AS source
                ON {update_keys_condition}
                WHEN MATCHED THEN
                    UPDATE SET {update_columns}
                WHEN NOT MATCHED THEN
                    INSERT ({', '.join(df.columns)}) 
                    VALUES ({', '.join([f'source.{col}' for col in df.columns])});
            """
            conn.execute(text(upsert_sql))
            conn.execute(text("DROP TABLE products.temp_product_details"))
            trans.commit()
            print("Upsert completed successfully.")
        except Exception as e:
            trans.rollback()
            print("Upsert failed:", e)
            raise e


# Initialize empty list to store product data


async def main():
    async with async_playwright() as p:
        product_data = []
        browser = await p.chromium.launch(headless=False)  # set True later for background runs
        page = await browser.new_page()
        
        url = 'https://www.walmart.com/browse/wearable-technology/3944_1229723?povid=Electronics_CP_web_6up_slot4_wearabletech'
        await page.goto(url, wait_until='domcontentloaded')
        await page.wait_for_timeout(3000)  # let dynamic content render

        # Grab all product anchors
        product_anchors = await page.locator("a[link-identifier]").element_handles()

        print(f"Found {len(product_anchors)} products")

        for anchor in product_anchors:
            # Get ID
            product_id = await anchor.get_attribute("link-identifier")
            # Only proceed if ID is an integer
            if not (product_id and product_id.isdigit()):
                continue
            # Get href (relative)
            href = await anchor.get_attribute("href")
            full_url = f"https://www.walmart.com{href}" if href else "N/A"

            # Get name (inner text)
            name = (await anchor.inner_text()).strip()

            # Get price using the closest product container
            price_element = await anchor.evaluate_handle('el => el.parentElement.querySelector("span.f2")')
            price = "N/A"
            if price_element:
                price_handle = price_element.as_element()
                if price_handle:
                    price = await price_handle.inner_text()
            print(f"ID: {product_id}")
            print(f"Name: {name}")
            print(f"URL: {full_url}")
            print(f"Price: ${price}")
            print("-" * 50)

            product_data.append({
    "product_id": product_id,
    "product_name": name.strip(),  # Clean up any newlines
    "category": "wearable-technology",
    "price": str(price).strip().replace('\n', '')
})
        print(product_data)
        upload_to_db(product_data)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
