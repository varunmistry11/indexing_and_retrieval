# Indexing and Retrieval System 

A production-ready, plug-and-play information retrieval system built with Elasticsearch, featuring comprehensive benchmarking, dataset-aware query generation, and advanced boolean query parsing.

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/varunmistry11/indexing_and_retrieval
cd indexing_and_retrieval
bash setup.sh

# 2. Activate environment
source venv/bin/activate

# 3. Start Elasticsearch
sudo systemctl start elasticsearch

# 4. Download datasets
python download_huggingface_data.py
python process_news_data.py

# 5. Verify setup
python main.py setup

# 6. Create your first index
python main.py create_index --dataset=wiki --index_type=boolean

# 7. Query the index
python main.py query_index "machine learning"

# 8. Run complete experiment
python main.py run_experiment --dataset=wiki --index_type=boolean
```

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Datasets](#datasets)
- [Index Types](#index-types)
- [Query System](#query-system)
- [Benchmarking](#benchmarking)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)

## ✨ Features

### Core Functionality
- ✅ **Three Index Types**: Boolean (x=1), WordCount (x=2), TF-IDF (x=3)
- ✅ **Multiple Datasets**: Wikipedia, News, Combined
- ✅ **Text Preprocessing**: Stemming, stopword removal, punctuation handling
- ✅ **Boolean Query Parser**: Full support for AND, OR, NOT, phrases, parentheses
- ✅ **Performance Benchmarking**: Latency (P95, P99), throughput, memory usage
- ✅ **Visualization**: Word frequency plots, latency distributions

### Advanced Features
- 🎯 **Dataset-Aware Query Generation**: Generates relevant queries for wiki/news datasets
- 🔄 **Reproducible Benchmarks**: Cached queries ensure fair comparisons
- 🔌 **Plug-and-Play Configuration**: Switch datasets/indices via CLI without code changes
- 📊 **Comprehensive Metrics**: All required artifacts (A, B, C, D) automatically generated
- 🛡️ **Security**: Environment variables for sensitive credentials

### Architecture
- **Modular Design**: Clean separation of concerns
- **Extensible**: Easy to add new index types, datastores, optimizations
- **Type-Safe**: Comprehensive type hints throughout
- **Well-Documented**: Inline documentation and separate guides

## 📁 Project Structure

```
indexing_and_retrieval/
├── conf/                           # Hydra configurations
│   ├── config.yaml                # Main config
│   ├── dataset/                   # Dataset configs (wiki, news, combined)
│   ├── index/                     # Index type configs (boolean, wordcount, tfidf)
│   └── datastore/                 # Datastore configs (elasticsearch, etc.)
├── src/                           # Source code
│   ├── index_base.py             # Abstract base class
│   ├── indices/
│   │   └── elasticsearch_index.py # ESIndex-v1.0 implementation
│   ├── data/
│   │   └── data_loader.py        # Dataset loading
│   ├── preprocessing/
│   │   └── text_preprocessor.py  # Text preprocessing
│   └── utils/
│       ├── benchmark.py          # Performance benchmarking
│       ├── plotting.py           # Visualization
│       ├── query_generator.py    # Dataset-aware query generation
│       └── query_parser.py       # Boolean query parser
├── data/                          # Data directory
├── indices/                       # Index storage
├── results/                       # Benchmark results
├── plots/                         # Generated plots
├── main.py                        # CLI entry point
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables
└── setup.sh                      # Automated setup script
```

## 🔧 Installation

### Prerequisites

- Python 3.8+
- Elasticsearch 7.x or 8.x
- 4GB+ RAM
- 10GB+ disk space

### Step 1: Clone Repository

```bash
git clone https://github.com/varunmistry11/indexing_and_retrieval
cd indexing_and_retrieval
```

### Step 2: Automated Setup

```bash
# Run setup script
bash setup.sh
```

This will:
- Create virtual environment
- Install Python dependencies
- Download NLTK data
- Create directory structure
- Check Elasticsearch connection

### Step 3: Manual Setup (Alternative)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Create directories
mkdir -p {data,indices,results,plots,huggingface_data,news_data,combined_data}
```

### Step 4: Elasticsearch Setup

#### Option A: Disable Security (Recommended for Development)

```bash
# Edit Elasticsearch config
sudo nano /etc/elasticsearch/elasticsearch.yml

# Add at the end:
xpack.security.enabled: false
xpack.security.enrollment.enabled: false
xpack.security.http.ssl.enabled: false

# Restart Elasticsearch
sudo systemctl restart elasticsearch

# Verify
curl -X GET "localhost:9200/"
```

#### Option B: Enable Security (Production)

```bash
# Reset password
sudo /usr/share/elasticsearch/bin/elasticsearch-reset-password -u elastic -i

# Configure .env file
cat > .env << 'EOF'
ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200
ELASTICSEARCH_SCHEME=https
ELASTICSEARCH_USE_AUTH=true
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=your_password_here
ELASTICSEARCH_VERIFY_CERTS=false
EOF
```

### Step 5: Download Datasets

```bash
# Download Wikipedia dataset (10,000 articles)
python download_huggingface_data.py

# Download and process News dataset
git clone https://github.com/Webhose/free-news-datasets.git
python process_news_data.py
```

### Step 6: Verify Installation

```bash
python main.py setup
```

Expected output:
```
✓ Connected to Elasticsearch
✓ Found wiki dataset
✓ Found news dataset
Setup completed successfully!
```

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Elasticsearch Configuration
ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200
ELASTICSEARCH_SCHEME=http
ELASTICSEARCH_USE_AUTH=false
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=changeme
ELASTICSEARCH_VERIFY_CERTS=false
```

### Main Configuration (conf/config.yaml)

```yaml
# Text preprocessing
preprocessing:
  lowercase: true
  remove_punctuation: true
  remove_stopwords: true
  stemming: true
  min_word_length: 2
  max_word_length: 50

# Benchmarking
benchmark:
  num_queries: 1000
  warmup_queries: 10
  measure_memory: true
  measure_latency: true
```

### Switching Configurations

```bash
# Switch dataset
python main.py run_experiment --dataset=wiki
python main.py run_experiment --dataset=news

# Switch index type
python main.py run_experiment --index_type=boolean
python main.py run_experiment --index_type=wordcount
python main.py run_experiment --index_type=tfidf

# Combine options
python main.py run_experiment --dataset=combined --index_type=tfidf
```

## 💻 Usage

### Basic Commands

```bash
# Setup and verification
python main.py setup

# Create an index
python main.py create_index --dataset=wiki --index_type=boolean

# Query an index
python main.py query_index "artificial intelligence"

# List all indices
python main.py list_indices

# Delete an index
python main.py delete_index --index_name=wiki_boolean_20251012
```

### Query Examples

```bash
# Simple term query
python main.py query_index "election"

# Boolean AND
python main.py query_index "election AND government"

# Boolean OR
python main.py query_index "climate OR environment"

# Boolean NOT
python main.py query_index "technology AND NOT software"

# Phrase query
python main.py query_index '"machine learning"'

# Complex query with parentheses
python main.py query_index "(election OR vote) AND government"

# Limit results
python main.py query_index "python" --max_results=20
```

### Running Experiments

```bash
# Complete experiment (index + benchmark + plots)
python main.py run_experiment --dataset=wiki --index_type=boolean

# This will:
# 1. Create index
# 2. Generate/load test queries
# 3. Run benchmarks
# 4. Generate performance plots
# 5. Save all results
```

### Generate Word Frequency Plots

```bash
# Generate plots for dataset analysis
python main.py generate_word_plots --dataset=wiki --sample_size=10000
python main.py generate_word_plots --dataset=news --sample_size=10000
```

### Generate Test Queries

```bash
# Generate 1000 queries for wiki dataset
python main.py generate_queries --num_queries=1000 --dataset=wiki

# View sample queries
python main.py generate_queries --num_queries=100 --dataset=news

# Save to custom location
python main.py generate_queries --num_queries=5000 --output=my_queries.json
```

### Test Query Parsing

```bash
# Test how queries are parsed
python main.py test_query_parsing

# Test specific query
python main.py test_query_parsing --query="(climate OR environment) AND energy"
```

### Benchmark Existing Index

```bash
# Benchmark a specific index
python main.py benchmark --index_name=wiki_boolean_20251012

# With custom experiment name
python main.py benchmark --experiment_name=my_benchmark
```

## 📊 Datasets

### Wikipedia Dataset
- **Source**: HuggingFace (wikimedia/wikipedia)
- **Version**: 20231101.en
- **Size**: 10,000 articles (configurable)
- **Fields**: id, title, text, url
- **Topics**: Geography, history, science, arts, politics, technology

### News Dataset
- **Source**: webz.io free-news-datasets
- **Size**: 104,038 articles
- **Fields**: id, title, text, author, published, site, categories
- **Topics**: Politics, economy, international, health, environment, sports

### Dataset Configuration

Edit `conf/dataset/{dataset}.yaml` to customize:

```yaml
name: wikipedia
source_file: huggingface_data/wikipedia_subset.jsonl
sample_size: null  # null for all, or specify number
fields:
  id_field: id
  text_field: text
  title_field: title
```

## 📑 Index Types

### 1. Boolean Index (x=1)
- **Description**: Basic inverted index with document IDs and positions
- **Use Case**: Existence queries, basic search
- **Features**: Term presence, position information
- **Ranking**: Basic (document frequency)

```bash
python main.py create_index --index_type=boolean
```

### 2. WordCount Index (x=2)
- **Description**: Index with term frequency information
- **Use Case**: Frequency-based ranking
- **Features**: Term counts, better relevance
- **Ranking**: Word count based

```bash
python main.py create_index --index_type=wordcount
```

### 3. TF-IDF Index (x=3)
- **Description**: Full TF-IDF scoring with BM25
- **Use Case**: Production-quality relevance ranking
- **Features**: TF-IDF scores, BM25 algorithm
- **Ranking**: Advanced relevance scoring

```bash
python main.py create_index --index_type=tfidf
```

### Index Comparison

| Feature | Boolean | WordCount | TF-IDF |
|---------|---------|-----------|---------|
| Document IDs | ✓ | ✓ | ✓ |
| Positions | ✓ | ✓ | ✓ |
| Term Frequency | ✗ | ✓ | ✓ |
| IDF Scoring | ✗ | ✗ | ✓ |
| BM25 | ✗ | ✗ | ✓ |
| Best For | Existence | Frequency | Relevance |

## 🔍 Query System

### Supported Query Syntax

```python
# Simple term
"election"

# Phrase query (exact match)
"machine learning"

# Boolean AND (all terms required)
"python AND programming"

# Boolean OR (any term)
"climate OR environment"

# Boolean NOT (exclude term)
"technology AND NOT software"

# Complex with parentheses
"(election OR vote) AND government"
```

### Query Generation

The system generates **dataset-aware queries**:

**Wikipedia Queries**: geography, history, science, arts, politics
**News Queries**: politics, economy, health, environment, crime

```bash
# Generate 1000 relevant queries
python main.py generate_queries --num_queries=1000 --dataset=wiki
```

**Query Types Distribution**:
- 20% Single term
- 15% Two-term AND
- 10% Two-term OR
- 10% Three-term AND
- 10% Three-term OR
- 15% Phrase queries
- 10% NOT queries
- 10% Complex boolean

### Query Parsing

All queries are automatically converted to proper Elasticsearch Query DSL:

```python
Input:  "(climate OR environment) AND energy"

Output: {
  "bool": {
    "must": [
      {
        "bool": {
          "should": [
            {"multi_match": {"query": "climate"}},
            {"multi_match": {"query": "environment"}}
          ],
          "minimum_should_match": 1
        }
      },
      {"multi_match": {"query": "energy"}}
    ]
  }
}
```

### Preprocessing Pipeline

```
Raw Query: "Machine Learning AND Python"
    ↓
Lowercase: "machine learning and python"
    ↓
Stemming: "machin learn and python"
    ↓
Parse: AND(PHRASE("machin learn"), TERM("python"))
    ↓
Elasticsearch Query DSL
```

## 📈 Benchmarking

### Automatic Metrics

The system automatically measures and reports:

#### Artifact A: Latency Metrics
- Mean latency
- Median latency
- **P95 latency** (95th percentile)
- **P99 latency** (99th percentile)
- Min/Max latency
- Standard deviation

#### Artifact B: Throughput
- **Queries per second** (QPS)
- Indexing throughput (docs/sec)

#### Artifact C: Memory Footprint
- Memory before/after indexing
- Peak memory during queries
- **Memory increase** from baseline

#### Artifact D: Functional Metrics
- Number of results per query
- Average results per query
- Result relevance scores

### Running Benchmarks

```bash
# Full experiment with all metrics
python main.py run_experiment --dataset=wiki --index_type=boolean

# Benchmark existing index
python main.py benchmark --index_name=wiki_boolean_20251012

# Custom number of queries
python main.py benchmark --num_queries=5000
```

### Results Output

```
results/
├── wiki_boolean_20251012_detailed.json  # All query results
├── wiki_boolean_20251012_stats.json     # Statistical summary
└── wiki_boolean_20251012_indexing.json  # Indexing metrics

plots/
├── wiki_boolean_20251012_latency.png    # Latency distribution
├── wiki_freq_before.png                  # Before preprocessing
├── wiki_freq_after.png                   # After preprocessing
└── wiki_freq_comparison.png              # Comparison plot
```

### Example Output

```
==================================================
BENCHMARK RESULTS
==================================================
Experiment: wiki_boolean_20251012
Total Queries: 1000
Mean Latency: 12.34 ms
P95 Latency: 45.67 ms
P99 Latency: 78.90 ms
Throughput: 81.23 queries/sec
Max Memory: 256.78 MB
==================================================
```

### Reproducibility

All benchmarks are reproducible:
- **Fixed seed (42)**: Same queries generated every time
- **Cached queries**: Automatically saved and reused
- **Fair comparisons**: Same queries across all index types

```bash
# These use identical queries → fair comparison
python main.py run_experiment --dataset=wiki --index_type=boolean
python main.py run_experiment --dataset=wiki --index_type=wordcount
python main.py run_experiment --dataset=wiki --index_type=tfidf
```

## 📚 Documentation

- **README.md** (this file) - Main documentation
- **README_USAGE.md** - Detailed usage guide
- **QUICK_REFERENCE.md** - Command cheatsheet
- **QUERY_GENERATION_GUIDE.md** - Query generation system
- **QUERY_PARSING.md** - Boolean query parsing
- **IMPLEMENTATION_SUMMARY.md** - Complete implementation overview

## 🐛 Troubleshooting

### Elasticsearch Connection Issues

```bash
# Check if Elasticsearch is running
curl -X GET "localhost:9200/"

# If not running
sudo systemctl start elasticsearch

# Check status
sudo systemctl status elasticsearch

# View logs
sudo tail -f /var/log/elasticsearch/elasticsearch.log
```

### NLTK Data Missing

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Dataset Not Found

```bash
# Check if datasets exist
ls huggingface_data/wikipedia_subset.jsonl
ls news_data/news_subset.jsonl

# If missing, download
python download_huggingface_data.py
python process_news_data.py
```

### Import Errors

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# Check installation
pip list | grep elasticsearch
pip list | grep hydra
```

### Permission Issues

```bash
# If Elasticsearch config is read-only
sudo chmod 666 /etc/elasticsearch/elasticsearch.yml

# If index storage has permission issues
chmod -R 755 indices/
```

### Memory Issues

If you encounter memory issues with large datasets:

```bash
# Edit config to use smaller sample
nano conf/dataset/wiki.yaml
# Set: sample_size: 1000

# Or use command line
python main.py create_index --dataset=wiki --sample_size=1000
```

### Query Cache Issues

```bash
# Clear query cache
rm -rf data/query_cache/

# Regenerate queries
python main.py generate_queries --num_queries=1000 --dataset=wiki
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

See LICENSE file for details.

---

**Ready to get started?** Run `bash setup.sh` and follow the quick start guide above!