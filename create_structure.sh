#!/bin/bash
# Script to create the complete directory structure

echo "Creating directory structure for Indexing and Retrieval Assignment..."

# Create main directories
mkdir -p conf/dataset
mkdir -p conf/index
mkdir -p conf/datastore

mkdir -p src/indices
mkdir -p src/data
mkdir -p src/preprocessing
mkdir -p src/utils

mkdir -p examples
mkdir -p data
mkdir -p indices
mkdir -p results
mkdir -p plots
mkdir -p huggingface_data
mkdir -p news_data
mkdir -p combined_data

# Create __init__.py files
touch src/__init__.py
touch src/indices/__init__.py
touch src/data/__init__.py
touch src/preprocessing/__init__.py
touch src/utils/__init__.py

# Create .gitkeep files for empty directories
touch indices/.gitkeep
touch results/.gitkeep
touch plots/.gitkeep

echo "✓ Directory structure created successfully!"
echo ""
echo "Directory tree:"
tree -L 2 -a || ls -R

echo ""
echo "Next steps:"
echo "1. Copy all configuration files to conf/"
echo "2. Copy all source files to src/"
echo "3. Copy main.py to project root"
echo "4. Copy requirements.txt to project root"
echo "5. Run: bash setup.sh"