import os
import json
from datasets import load_dataset

# --- Configuration ---
DATASET_NAME = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.en"
OUTPUT_DIR = "huggingface_data"
OUTPUT_FILENAME = "wikipedia_subset.jsonl"

# --- Parameters ---
# Set the number of articles you want to download.
# This is a more direct way to limit data than targeting an exact file size.
# 10,000 articles is a good starting point for a sizable but manageable dataset.
NUM_ARTICLES_TO_DOWNLOAD = 10000

def download_and_save_wikipedia_subset():
    """
    Downloads a subset of the Wikipedia dataset from Hugging Face and saves it
    to a JSONL file.
    """
    print(f"Starting download of the first {NUM_ARTICLES_TO_DOWNLOAD} articles from '{DATASET_NAME}'...")

    try:
        # The `split` parameter allows slicing the dataset.
        # 'train[:{NUM_ARTICLES_TO_DOWNLOAD}]' tells the library to only get the first N items.
        # The library is smart enough to only download the required data.
        dataset_subset = load_dataset(
            DATASET_NAME,
            DATASET_CONFIG,
            split=f'train[:{NUM_ARTICLES_TO_DOWNLOAD}]',
            trust_remote_code=True # Required for this dataset
        )

        print("Dataset subset loaded successfully. Now saving to file...")

        # Create the output directory if it doesn't exist
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"Created directory: {OUTPUT_DIR}")

        filepath = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
        
        # Save the data to a JSONL file (one JSON object per line)
        with open(filepath, 'w', encoding='utf-8') as f:
            for article in dataset_subset:
                # Create a dictionary for the current article
                article_data = {
                    "id": article["id"],
                    "url": article["url"],
                    "title": article["title"],
                    "text": article["text"]
                }
                # Write the dictionary as a JSON string on a new line
                f.write(json.dumps(article_data, ensure_ascii=False) + '\n')

        print(f"Successfully saved {NUM_ARTICLES_TO_DOWNLOAD} articles to '{filepath}'")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have the 'datasets' and 'pyarrow' libraries installed:")
        print("pip install datasets pyarrow")

if __name__ == "__main__":
    download_and_save_wikipedia_subset()
