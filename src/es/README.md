# Elasticsearch N-gram Index Project

## Overview

This project demonstrates the implementation of an Elasticsearch index using an n-gram tokenizer for enhanced text search. Utilizing n-grams allows for more inclusive and versatile search queries by breaking down words into their constituent n-grams. The project features a Python script that configures the Elasticsearch index with custom settings and populates it with 100 synthetic yet realistic documents to simulate a real-world scenario such as a content library or product catalog.

## Features

- **Custom N-gram Analyzer**: Optimizes search capabilities by utilizing n-grams for text analysis in Elasticsearch.
- **Rich Document Structure**: Introduces a complex document structure with 15 unique fields, showcasing a practical example suitable for bibliographic or catalog data.
- **Automated Data Generation**: Leverages the `Faker` library to automatically generate and populate the index with data, ready for search queries.
- **Python & Elasticsearch Integration**: Uses the official Elasticsearch Python client for streamlined operations on the Elasticsearch cluster.

## Pre-requisites

The following requirements are necessary to run this project:

- **Elasticsearch 7.x/8.x**: Ensure Elasticsearch is installed and operational on your system.
- **Python 3.x**
- **Required Python Packages**:
  - `elasticsearch`
  - `Faker`

### Installing Dependencies

- **Elasticsearch**: Follow the detailed [Elasticsearch Installation Guide](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html).
- **Python Packages**: Install the necessary Python packages by running:
  ```bash
  pip install elasticsearch Faker
