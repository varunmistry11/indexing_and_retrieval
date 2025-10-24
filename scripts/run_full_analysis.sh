#!/bin/bash

echo "=========================================="
echo "FULL ANALYSIS PIPELINE"
echo "=========================================="

# Step 1: Create all indices
echo -e "\n[1/6] Creating indices..."
./scripts/create_all_indices.sh

# Step 2: Create ES indices (if available)
echo -e "\n[2/6] Creating Elasticsearch indices..."
./scripts/create_es_indices.sh || echo "  ⚠️  Elasticsearch unavailable, skipping"

# Step 3: Validate correctness
echo -e "\n[3/6] Validating correctness..."
python scripts/validate_correctness.py

# Step 4: Run benchmarks
echo -e "\n[4/6] Running benchmarks..."
python scripts/benchmark_all.py

# Step 5: Generate plots
echo -e "\n[5/6] Generating plots..."
python scripts/generate_plots.py
python scripts/generate_comprehensive_plots.py

# Step 6: Run unit tests
echo -e "\n[6/6] Running unit tests..."
pytest tests/ -v

echo -e "\n=========================================="
echo "ANALYSIS COMPLETE!"
echo "=========================================="
echo "Results: results/"
echo "Plots: plots/"
echo "=========================================="