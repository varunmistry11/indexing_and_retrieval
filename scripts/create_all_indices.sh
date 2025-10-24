#!/bin/bash

# Function to create indices for a dataset
create_indices_for_dataset() {
    DATASET=$1
    PREFIX=$2
    
    echo "Creating indices for $DATASET dataset..."
    
    # Index types (x=1,2,3)
    python main.py create_index --dataset=$DATASET --index_type=self_boolean --index_name=${PREFIX}_boolean
    python main.py create_index --dataset=$DATASET --index_type=self_wordcount --index_name=${PREFIX}_wordcount
    python main.py create_index --dataset=$DATASET --index_type=self_tfidf --index_name=${PREFIX}_tfidf
    
    # Optimizations
    python main.py create_index --dataset=$DATASET --index_type=self_tfidf --index_name=${PREFIX}_tfidf_skip
    python main.py create_index --dataset=$DATASET --index_type=self_tfidf_threshold --index_name=${PREFIX}_threshold
    python main.py create_index --dataset=$DATASET --index_type=self_tfidf_earlystop --index_name=${PREFIX}_earlystop
    
    # Query processors
    python main.py create_index --dataset=$DATASET --index_type=self_tfidf_optimized --index_name=${PREFIX}_docatat
    
    # Compression
    python main.py create_index --dataset=$DATASET --index_type=self_tfidf_compressed --index_name=${PREFIX}_compressed
    python main.py create_index --dataset=$DATASET --index_type=self_tfidf_gzip --index_name=${PREFIX}_gzip
    
    echo "Completed $DATASET dataset!"
}

# Create for both datasets
create_indices_for_dataset "news" "news"
create_indices_for_dataset "wiki" "wiki"

echo "All indices created!"