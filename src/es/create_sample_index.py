from elasticsearch import Elasticsearch
import json

# Initialize Elasticsearch client
es = Elasticsearch()

# Define index name
index_name = 'sample_data_index'

# Define settings and mappings
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
            "title": {
                "type": "text",
                "analyzer": "ngram_analyzer"
            },
            "description": {
                "type": "text",
                "analyzer": "ngram_analyzer"
            },
            "author": {"type": "text"},
            "publish_date": {"type": "date"},
            "isbn": {"type": "text"},
            "category": {"type": "keyword"},
            "price": {"type": "float"},
            "pages": {"type": "integer"},
            "language": {"type": "keyword"},
            "publisher": {"type": "text"},
            "availability": {"type": "boolean"},
            "format": {"type": "keyword"},
            "weight": {"type": "float"},
            "dimensions": {"type": "text"},
            "rating": {"type": "float"}
        }
    }
}

# Delete the index if it already exists
if es.indices.exists(index_name):
    es.indices.delete(index=index_name)

# Create the index with specified settings and mappings
es.indices.create(index=index_name, body=settings)
