from opensearchpy import OpenSearch, helpers
import random
from faker import Faker

# Configuration for AWS OpenSearch
host = 'YOUR_OPENSEARCH_DOMAIN_ENDPOINT'  # example: 'search-mydomain.us-west-1.es.amazonaws.com'
port = 443
auth = ('YOUR_ACCESS_KEY', 'YOUR_SECRET_KEY')

# Creating an instance of OpenSearch class
opensearch_client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)

# Define index name and settings
index_name = 'sample_data_index'
settings = {
    "settings": {
        "analysis": {
            "analyzer": {
                "ngram_analyzer": {
                    "type": "custom",
                    "tokenizer": "ngram_tokenizer"
                }
            },
            "tokenizer": {
                "ngram_tokenizer": {
                    "type": "ngram",
                    "min_gram": 3,
                    "max_gram": 3,
                    "token_chars": ["letter", "digit"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            # Define your properties (fields) here
            "title": {"type": "text", "analyzer": "ngram_analyzer"},
            # Add other fields
        }
    }
}

# Delete the index if it exists
if opensearch_client.indices.exists(index_name):
    opensearch_client.indices.delete(index=index_name)

# Create index with the settings and mappings
opensearch_client.indices.create(index=index_name, body=settings)

fake = Faker()

# Generate multiple documents for OpenSearch index
documents = [{
    "_index": index_name,
    "_source": {
        "title": fake.sentence(),
        # Include other fields here
    }
} for _ in range(100)]

# Bulk insert documents into OpenSearch index
helpers.bulk(opensearch_client, documents)

print("Documents indexed successfully.")
