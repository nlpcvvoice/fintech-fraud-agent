from google.cloud import bigquery

# Initialize the BigQuery client
client = bigquery.Client()

# SQL query targeting the ULB credit card fraud dataset in BQ public data
query = """
SELECT * 
FROM `bigquery-public-data.ml_datasets.ulb_fraud_detection`
LIMIT 1000
"""

# Run the query and convert to a Pandas DataFrame
df = client.query(query).to_dataframe()

# Preview your data
print(df.head())
