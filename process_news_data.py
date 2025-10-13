import os
import json
import zipfile
from pathlib import Path
import time
from typing import List, Dict, Any
import glob

# --- Configuration ---
LOCAL_REPO_PATH = "free-news-datasets"  # Path to cloned repository
NEWS_DATASETS_DIR = os.path.join(LOCAL_REPO_PATH, "News_Datasets")
OUTPUT_DIR = "news_data"
OUTPUT_FILENAME = "news_subset.jsonl"

# --- Parameters ---
MAX_ARTICLES_TO_DOWNLOAD = None   # Set to None to process all articles, or set a number to limit
MAX_ZIP_FILES_TO_PROCESS = None   # Set to None to process all available ZIP files, or set a number to limit

def find_zip_files() -> List[str]:
    """Find all ZIP files in the local repository."""
    if not os.path.exists(NEWS_DATASETS_DIR):
        print(f"Error: Repository directory '{NEWS_DATASETS_DIR}' not found!")
        print("Please make sure you have cloned the repository:")
        print("  git clone https://github.com/Webhose/free-news-datasets.git")
        return []
    
    # Find all ZIP files in the News_Datasets directory
    zip_pattern = os.path.join(NEWS_DATASETS_DIR, "*.zip")
    zip_files = glob.glob(zip_pattern)
    
    if not zip_files:
        print(f"No ZIP files found in {NEWS_DATASETS_DIR}")
        return []
    
    # Sort files for consistent processing order
    zip_files.sort()
    
    print(f"Found {len(zip_files)} ZIP files:")
    for zip_file in zip_files:
        file_size = os.path.getsize(zip_file) / (1024 * 1024)  # Size in MB
        print(f"  {os.path.basename(zip_file)} ({file_size:.1f} MB)")
    
    return zip_files

def extract_and_process_zip(zip_path: str) -> List[Dict[str, Any]]:
    """Extract JSON files from zip and return processed articles."""
    articles = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            json_files = [f for f in zip_ref.namelist() if f.endswith('.json')]
            print(f"Found {len(json_files)} JSON files in {zip_path}")
            
            for json_file in json_files:
                try:
                    with zip_ref.open(json_file) as f:
                        content = f.read().decode('utf-8')
                        article_data = json.loads(content)
                        
                        # Extract relevant fields for indexing
                        processed_article = {
                            "id": article_data.get("uuid", ""),
                            "url": article_data.get("url", ""),
                            "title": article_data.get("title", ""),
                            "text": article_data.get("text", ""),
                            "author": article_data.get("author", ""),
                            "published": article_data.get("published", ""),
                            "site": article_data.get("thread", {}).get("site", ""),
                            "categories": article_data.get("categories", [])
                        }
                        
                        # Only add articles with meaningful text content
                        if processed_article["text"] and len(processed_article["text"].strip()) > 100:
                            articles.append(processed_article)
                            
                except Exception as e:
                    print(f"Error processing {json_file}: {e}")
                    continue
                    
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
    
    return articles

def process_news_data_from_local_repo():
    """Process news data from locally cloned repository."""
    print(f"Processing news data from local repository...")
    if MAX_ARTICLES_TO_DOWNLOAD is None:
        print("Processing ALL available articles (no limit)")
    else:
        print(f"Target: {MAX_ARTICLES_TO_DOWNLOAD} articles")
    
    # Find available ZIP files
    zip_files = find_zip_files()
    if not zip_files:
        return
    
    # Limit number of files to process if specified
    if MAX_ZIP_FILES_TO_PROCESS is not None:
        zip_files = zip_files[:MAX_ZIP_FILES_TO_PROCESS]
        print(f"Processing first {len(zip_files)} ZIP files...")
    else:
        print(f"Processing all {len(zip_files)} ZIP files...")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_articles = []
    zip_files_processed = 0
    
    for zip_path in zip_files:
        # Check if we should stop due to article limit
        if MAX_ARTICLES_TO_DOWNLOAD is not None and len(all_articles) >= MAX_ARTICLES_TO_DOWNLOAD:
            print(f"Reached target of {MAX_ARTICLES_TO_DOWNLOAD} articles, stopping...")
            break
        
        print(f"\nProcessing {os.path.basename(zip_path)}...")
        
        # Extract and process articles
        articles = extract_and_process_zip(zip_path)
        all_articles.extend(articles)
        zip_files_processed += 1
        
        print(f"Extracted {len(articles)} articles from {os.path.basename(zip_path)}")
        print(f"Total articles so far: {len(all_articles)}")
        
        # Stop if we have enough articles (only if limit is set)
        if MAX_ARTICLES_TO_DOWNLOAD is not None and len(all_articles) >= MAX_ARTICLES_TO_DOWNLOAD:
            all_articles = all_articles[:MAX_ARTICLES_TO_DOWNLOAD]
            print(f"Trimmed to target of {MAX_ARTICLES_TO_DOWNLOAD} articles")
            break
    
    # Save processed articles to JSONL file
    if all_articles:
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
        
        print(f"\nSaving {len(all_articles)} articles to '{output_path}'...")
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, article in enumerate(all_articles):
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
                # Progress indicator for large datasets
                if (i + 1) % 10000 == 0:
                    print(f"  Saved {i + 1:,} articles...")
        
        print(f"\nSuccessfully saved {len(all_articles):,} articles to '{output_path}'")
        
        # Print comprehensive statistics
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        print(f"Total articles processed: {len(all_articles):,}")
        print(f"ZIP files processed: {zip_files_processed}")
        
        # Count articles by site
        sites = {}
        for article in all_articles:
            site = article.get('site', 'unknown')
            sites[site] = sites.get(site, 0) + 1
        
        print(f"\nUnique news sites: {len(sites):,}")
        print("Top 15 sites:")
        for i, (site, count) in enumerate(sorted(sites.items(), key=lambda x: x[1], reverse=True)[:15], 1):
            print(f"  {i:2d}. {site}: {count:,} articles")
        
        # Calculate text length statistics
        text_lengths = [len(article['text']) for article in all_articles if article['text']]
        if text_lengths:
            avg_length = sum(text_lengths) / len(text_lengths)
            min_length = min(text_lengths)
            max_length = max(text_lengths)
            # Calculate median
            sorted_lengths = sorted(text_lengths)
            median_length = sorted_lengths[len(sorted_lengths) // 2]
            
            print(f"\nText length statistics:")
            print(f"  Average: {avg_length:,.0f} characters")
            print(f"  Median:  {median_length:,} characters")
            print(f"  Min:     {min_length:,} characters")
            print(f"  Max:     {max_length:,} characters")
        
        # Count articles by categories
        all_categories = []
        articles_with_categories = 0
        for article in all_articles:
            if article.get('categories'):
                articles_with_categories += 1
                all_categories.extend(article['categories'])
        
        if all_categories:
            category_counts = {}
            for cat in all_categories:
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            print(f"\nCategory statistics:")
            print(f"  Articles with categories: {articles_with_categories:,} ({articles_with_categories/len(all_articles)*100:.1f}%)")
            print(f"  Unique categories: {len(category_counts):,}")
            print("  Top 15 categories:")
            for i, (cat, count) in enumerate(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:15], 1):
                print(f"    {i:2d}. {cat}: {count:,} articles")
        
        # Calculate storage size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\nOutput file size: {file_size_mb:.1f} MB")
    
    print(f"\nNews data processing complete!")
    print(f"Processed {zip_files_processed} ZIP files with all available articles")

def create_combined_dataset():
    """Combine Wikipedia and News data into a single dataset for indexing."""
    wiki_path = os.path.join("huggingface_data", "wikipedia_subset.jsonl")
    news_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    combined_path = os.path.join("combined_data", "combined_dataset.jsonl")
    
    os.makedirs("combined_data", exist_ok=True)
    
    combined_count = 0
    
    with open(combined_path, 'w', encoding='utf-8') as out_f:
        # Add Wikipedia articles
        if os.path.exists(wiki_path):
            print("Adding Wikipedia articles...")
            with open(wiki_path, 'r', encoding='utf-8') as wiki_f:
                for line in wiki_f:
                    article = json.loads(line.strip())
                    # Add source identifier
                    article['source'] = 'wikipedia'
                    out_f.write(json.dumps(article, ensure_ascii=False) + '\n')
                    combined_count += 1
        
        # Add News articles
        if os.path.exists(news_path):
            print("Adding news articles...")
            with open(news_path, 'r', encoding='utf-8') as news_f:
                for line in news_f:
                    article = json.loads(line.strip())
                    # Add source identifier
                    article['source'] = 'news'
                    out_f.write(json.dumps(article, ensure_ascii=False) + '\n')
                    combined_count += 1
    
    print(f"Created combined dataset with {combined_count} articles at '{combined_path}'")
    return combined_path

if __name__ == "__main__":
    # Check if repository exists
    if not os.path.exists(LOCAL_REPO_PATH):
        print(f"Repository '{LOCAL_REPO_PATH}' not found!")
        print("Please clone the repository first:")
        print("  git clone https://github.com/Webhose/free-news-datasets.git")
        exit(1)
    
    # Process news data from local repository
    process_news_data_from_local_repo()
    
    # Create combined dataset
    # print("\n" + "="*50)
    # print("Creating combined dataset...")
    # combined_path = create_combined_dataset()
    
    # print(f"\nDataset preparation complete!")
    # print(f"Combined dataset available at: {combined_path}")
    # print("\nNext steps:")
    # print("1. Run preprocessing (stemming, stop word removal)")
    # print("2. Index data into Elasticsearch (ESIndex-v1.0)")
    # print("3. Implement your own indexing system (SelfIndex-v1.0)")