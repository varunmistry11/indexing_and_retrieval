#!/bin/bash
# Setup script for Indexing and Retrieval Assignment

set -e  # Exit on error

echo "=========================================="
echo "Setting up Indexing and Retrieval Project"
echo "=========================================="

# Check Python version
echo -e "\n[1/8] Checking Python version..."
python3 --version || { echo "Python 3 not found. Please install Python 3.8+"; exit 1; }

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "\n[2/8] Creating virtual environment..."
    python3 -m venv venv
else
    echo -e "\n[2/8] Virtual environment already exists"
fi

# Activate virtual environment
echo -e "\n[3/8] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo -e "\n[4/8] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo -e "\n[5/8] Installing Python dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo -e "\n[6/8] Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# Check Elasticsearch
echo -e "\n[7/8] Checking Elasticsearch..."
if curl -s -X GET "localhost:9200/" > /dev/null 2>&1; then
    echo "✓ Elasticsearch is running"
else
    echo "✗ Elasticsearch is not running"
    echo "Please start Elasticsearch with:"
    echo "  sudo systemctl start elasticsearch"
fi

# Create necessary directories
echo -e "\n[8/8] Creating directories..."
mkdir -p data
mkdir -p indices
mkdir -p results
mkdir -p plots
mkdir -p huggingface_data
mkdir -p news_data
mkdir -p combined_data

echo -e "\n=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo -e "\nNext steps:"
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Download datasets:"
echo "   python download_huggingface_data.py"
echo "   git clone https://github.com/Webhose/free-news-datasets.git"
echo "   python process_news_data.py"
echo ""
echo "3. Verify setup:"
echo "   python main.py setup"
echo ""
echo "4. Create your first index:"
echo "   python main.py create_index --dataset=wiki --index_type=boolean"
echo ""
echo "5. Run a complete experiment:"
echo "   python main.py run_experiment --dataset=wiki --index_type=boolean"
echo ""
echo "For more information, see README_USAGE.md"