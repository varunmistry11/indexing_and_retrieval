#!/bin/bash

echo "======================================================================"
echo "COMPLETE ANALYSIS PIPELINE"
echo "======================================================================"
echo "Start time: $(date)"
echo ""
echo "This script will:"
echo "  1. Create all index variants (54 indices)"
echo "  2. Benchmark all indices with queries"
echo "  3. Generate all assignment plots"
echo "======================================================================"
echo ""

# Step 1: Create indices
echo ""
echo "======================================================================"
echo "STEP 1: Creating all indices with metrics collection"
echo "======================================================================"
python scripts/create_all_indices.py

if [ $? -ne 0 ]; then
    echo "âŒ Index creation failed!"
    exit 1
fi

# Step 2: Benchmark queries
echo ""
echo "======================================================================"
echo "STEP 2: Benchmarking all indices"
echo "======================================================================"
python scripts/benchmark_all_queries.py

if [ $? -ne 0 ]; then
    echo "âŒ Benchmarking failed!"
    exit 1
fi

# Step 3: Generate plots
echo ""
echo "======================================================================"
echo "STEP 3: Generating assignment plots"
echo "======================================================================"
python scripts/generate_assignment_plots.py

if [ $? -ne 0 ]; then
    echo "âŒ Plot generation failed!"
    exit 1
fi

# Summary
echo ""
echo "======================================================================"
echo "âœ… COMPLETE ANALYSIS FINISHED!"
echo "======================================================================"
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  - Indexing metrics: results/indexing/"
echo "  - Query metrics: results/queries/"
echo "  - Plots: plots/"
echo ""
echo "======================================================================"
