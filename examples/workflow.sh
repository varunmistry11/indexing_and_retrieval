#!/bin/bash
# Complete workflow example for ESIndex-v1.0

set -e

echo "=========================================="
echo "Complete Workflow: ESIndex-v1.0"
echo "=========================================="

# Activate virtual environment
source venv/bin/activate

# Step 1: Setup and verification
echo -e "\n[Step 1] Setup and verification..."
python main.py setup

# Step 2: Generate word frequency plots for preprocessing analysis
echo -e "\n[Step 2] Generating word frequency plots..."
echo "  - Wikipedia dataset..."
python main.py generate_word_plots --dataset=wiki --sample_size=10000

echo "  - News dataset..."
python main.py generate_word_plots --dataset=news --sample_size=10000

# Step 3: Create indices with different information types

# Boolean Index (x=1)
echo -e "\n[Step 3.1] Creating Boolean Index (Activity 1.x=1)..."
python main.py create_index --dataset=combined --index_type=boolean

# WordCount Index (x=2) 
echo -e "\n[Step 3.2] Creating WordCount Index (Activity 1.x=2)..."
python main.py create_index --dataset=combined --index_type=wordcount

# TF-IDF Index (x=3)
echo -e "\n[Step 3.3] Creating TF-IDF Index (Activity 1.x=3)..."
python main.py create_index --dataset=combined --index_type=tfidf

# Step 4: Run complete experiments with benchmarking
echo -e "\n[Step 4] Running experiments with benchmarking..."

echo "  - Boolean experiment..."
python main.py run_experiment --dataset=combined --index_type=boolean

echo "  - WordCount experiment..."
python main.py run_experiment --dataset=combined --index_type=wordcount

echo "  - TF-IDF experiment..."
python main.py run_experiment --dataset=combined --index_type=tfidf

# Step 5: Test queries on different index types
echo -e "\n[Step 5] Testing queries on different indices..."

# Get latest indices
BOOL_INDEX=$(python main.py list_indices | grep boolean | tail -1 | awk '{print $2}')
WC_INDEX=$(python main.py list_indices | grep wordcount | tail -1 | awk '{print $2}')
TFIDF_INDEX=$(python main.py list_indices | grep tfidf | tail -1 | awk '{print $2}')

# Test queries
TEST_QUERIES=(
    "machine learning"
    "artificial intelligence"
    "python programming"
    '"natural language processing"'
    "AI AND robotics"
)

echo "  Testing Boolean Index..."
for query in "${TEST_QUERIES[@]}"; do
    echo "    Query: $query"
    python main.py query_index "$query" --index_name="$BOOL_INDEX" --max_results=5 | head -20
done

echo "  Testing TF-IDF Index (for comparison)..."
python main.py query_index "machine learning" --index_name="$TFIDF_INDEX" --max_results=5 | head -20

# Step 6: List all indices
echo -e "\n[Step 6] Listing all created indices..."
python main.py list_indices

# Step 7: Show results
echo -e "\n=========================================="
echo "Workflow completed successfully!"
echo "=========================================="
echo ""
echo "Results locations:"
echo "  - Benchmark results: results/"
echo "  - Performance plots: plots/"
echo "  - Word frequency plots: plots/"
echo ""
echo "To view results:"
echo "  ls -lh results/"
echo "  ls -lh plots/"
echo ""
echo "To analyze results:"
echo "  cat results/combined_boolean_*_stats.json | jq ."
echo "  cat results/combined_wordcount_*_stats.json | jq ."
echo "  cat results/combined_tfidf_*_stats.json | jq ."
echo ""
echo "Next steps for Activity 2 (SelfIndex-v1.0):"
echo "  1. Implement src/indices/self_index.py"
echo "  2. Add configuration files"
echo "  3. Run comparative experiments"