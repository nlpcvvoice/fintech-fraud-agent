import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError

def fetch_ulb_fraud_data() -> pd.DataFrame:
    """
    Fetch the ULB credit card fraud sample dataset from BigQuery
    using the Google Cloud Client Library.

    Returns:
        pd.DataFrame with fraud features, amount, and Class label
    """
    # 1. Initialize the BigQuery client (uses managed identity in a GCP notebook)
    client = bigquery.Client()
    
    # 2. Write a standard SQL query
    # Note: backtick-wrap dataset and table names for safety and correctness
    query = """
        SELECT * 
        FROM `fraud_detection.ulb_credit_card_fraud`
        # LIMIT 1000
    """
    
    print("⏳ Connecting to BigQuery and fetching ULB fraud data...")
    
    try:
        # 3. Run the query and convert to a Pandas DataFrame with .to_dataframe()
        query_job = client.query(query)
        df = query_job.to_dataframe()
        
        print(f"✅ Fetched {len(df)} fraud samples successfully.")
        return df
        
    except GoogleCloudError as e:
        print(f"❌ GCP connection or query failed: {e}")
        raise
    except Exception as e:
        print(f"❌ Unexpected local conversion error: {e}")
        raise

if __name__ == "__main__":
    # Self-test logic when running this script locally
    fraud_df = fetch_ulb_fraud_data()
    print("\n🔍 Preview first 2 rows of ULB fraud feature data:")
    print(fraud_df.head(2))
