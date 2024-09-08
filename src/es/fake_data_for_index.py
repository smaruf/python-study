import random
from faker import Faker

fake = Faker()

def generate_document():
    return {
        "title": fake.sentence(),
        "description": fake.text(),
        "author": fake.name(),
        "publish_date": fake.date(),
        "isbn": fake.isbn13(),
        "category": fake.word(),
        "price": round(random.uniform(10, 100), 2),
        "pages": random.randint(100, 1000),
        "language": fake.language_code(),
        "publisher": fake.company(),
        "availability": fake.boolean(),
        "format": random.choice(["Hardcover", "Paperback", "eBook"]),
        "weight": round(random.uniform(0.1, 2.5), 2),
        "dimensions": f'{random.randint(5, 10)}x{random.randint(5, 10)}x{random.randint(1, 3)}',
        "rating": round(random.uniform(1, 5), 1)
    }

# Bulk index the sample documents
actions = [
    {
        "_index": index_name,
        "_source": generate_document()
    }
    for _ in range(100)
]

results = es.bulk(index=index_name, body=actions, refresh=True)

# Print results to check if documents are indexed successfully
print(json.dumps(results, indent=4))
