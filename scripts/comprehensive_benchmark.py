#!/usr/bin/env python
"""
Comprehensive benchmark across datasets and document sizes.
Tests: News (1K, 10K, 100K), Wiki (1K, 10K, 100K)
Implementations: SelfIndex (all configs), Elasticsearch
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import json
import subprocess
from datetime import datetime

# Configuration matrix
DATASETS = ['news', 'wiki']
DOC_SIZES = [1000, 10000, 100000]  # Adjust based on your data availability
INDEX_TYPES_SELF = ['self_boolean', 'self_wordcount', 'self_tfidf', 
                    'self_tfidf_compressed', 'self_tfidf_gzip']
INDEX_TYPES_ES = ['boolean', 'wordcount', 'tfidf']

def update_dataset_sample_size(dataset, size):
    """Update dataset config with sample size."""
    config_path = Path(f"conf/dataset/{dataset}.yaml")
    
    # Read config
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    # Update sample_size line
    updated = []
    for line in lines:
        if 'sample_size:' in line:
            if size is None:
                updated.append('sample_size: null  # Use all documents\n')
            else:
                updated.append(f'sample_size: {size}  # Limited for testing\n')
        else:
            updated.append(line)
    
    # Write back
    with open(config_path, 'w') as f:
        f.writelines(updated)
    
    print(f"  Updated {dataset} dataset: sample_size={size}")

def create_index(dataset, index_type, size, implementation='selfindex'):
    """Create a single index."""
    index_name = f"{implementation}_{dataset}_{index_type.replace('self_', '')}_{size}"
    
    print(f"  Creating: {index_name}")
    
    cmd = [
        'python', 'main.py', 'create_index',
        f'--dataset={dataset}',
        f'--index_type={index_type}',
        f'--index_name={index_name}'
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return index_name
    except subprocess.CalledProcessError as e:
        print(f"    ❌ Failed: {e}")
        return None

def benchmark_configuration(dataset, size):
    """Benchmark all indices for a dataset/size combination."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {dataset.upper()} - {size:,} documents")
    print(f"{'='*60}")
    
    # Update dataset size
    update_dataset_sample_size(dataset, size)
    
    created_indices = []
    
    # Create SelfIndex variants
    print("\nCreating SelfIndex variants...")
    for index_type in INDEX_TYPES_SELF:
        idx_name = create_index(dataset, index_type, size, 'selfindex')
        if idx_name:
            created_indices.append(idx_name)
    
    # Create Elasticsearch variants (optional)
    print("\nCreating Elasticsearch variants...")
    for index_type in INDEX_TYPES_ES:
        idx_name = create_index(dataset, index_type, size, 'es')
        if idx_name:
            created_indices.append(idx_name)
    
    print(f"\n✅ Created {len(created_indices)} indices")
    return created_indices

def run_comprehensive_benchmark():
    """Run complete benchmark suite."""
    print("="*60)
    print("COMPREHENSIVE BENCHMARK SUITE")
    print("="*60)
    print(f"Start time: {datetime.now()}")
    
    all_results = {}
    
    for dataset in DATASETS:
        for size in DOC_SIZES:
            config_key = f"{dataset}_{size}"
            
            # Create indices
            indices = benchmark_configuration(dataset, size)
            
            # Benchmark them
            print(f"\nBenchmarking {len(indices)} indices...")
            # Use existing benchmark_all.py logic here
            # Or call it as subprocess
            
            all_results[config_key] = {
                'dataset': dataset,
                'size': size,
                'indices': indices,
                'timestamp': datetime.now().isoformat()
            }
    
    # Save comprehensive results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "comprehensive_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"End time: {datetime.now()}")
    print(f"Results saved to: results/comprehensive_results.json")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_comprehensive_benchmark()