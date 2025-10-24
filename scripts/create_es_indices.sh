#!/bin/bash

# Create Elasticsearch indices
echo "Creating Elasticsearch indices..."

# News dataset
python main.py create_index --dataset=news --index_type=boolean --index_name=es_news_boolean
python main.py create_index --dataset=news --index_type=wordcount --index_name=es_news_wordcount
python main.py create_index --dataset=news --index_type=tfidf --index_name=es_news_tfidf

# Wiki dataset
python main.py create_index --dataset=wiki --index_type=boolean --index_name=es_wiki_boolean
python main.py create_index --dataset=wiki --index_type=wordcount --index_name=es_wiki_wordcount
python main.py create_index --dataset=wiki --index_type=tfidf --index_name=es_wiki_tfidf

echo "Elasticsearch indices created!"