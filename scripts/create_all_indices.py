#!/usr/bin/env python
"""
Create all index variants with metrics collection.
Handles 9 variants × 2 datasets × 3 sizes = 54 indices
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf
import json
import time
import psutil
from datetime import datetime
import os
from dotenv import load_dotenv

from src.indices.self_index import SelfIndex
from src.indices.rocksdb_index import RocksDBIndex
from src.data.data_loader import DataLoader

# Load environment variables
load_dotenv()
OmegaConf.register_new_resolver("env", os.getenv)

# Index variant configurations
INDEX_VARIANTS = [
    # (name, config_overrides, description)
    ('boolean_custom', ['index=self_boolean'], 'Boolean (x=1, y=1)'),
    ('wordcount_custom', ['index=self_wordcount'], 'WordCount (x=2, y=1)'),
    ('tfidf_custom', ['index=self_tfidf'], 'TF-IDF (x=3, y=1)'),
    ('tfidf_rocksdb', ['index=rocksdb_tfidf', 'datastore=rocksdb'], 'TF-IDF RocksDB (x=3, y=2)'),
    ('tfidf_skip', ['index=self_tfidf_optimized'], 'TF-IDF + Skip (x=3, i=1)'),
    ('tfidf_gap', ['index=self_tfidf_compressed'], 'TF-IDF + Gap (x=3, z=1)'),
    ('tfidf_gzip', ['index=self_tfidf_gzip'], 'TF-IDF + Gzip (x=3, z=2)'),
    ('tfidf_threshold', ['index=self_tfidf_threshold'], 'TF-IDF + Threshold (x=3, o=th)'),
    ('tfidf_earlystop', ['index=self_tfidf_earlystop'], 'TF-IDF + EarlyStop (x=3, o=es)'),
]

# Elasticsearch variants (if available)
ES_VARIANTS = [
    ('es_boolean', ['index=boolean'], 'ES Boolean (x=1)'),
    ('es_wordcount', ['index=wordcount'], 'ES WordCount (x=2)'),
    ('es_tfidf', ['index=tfidf'], 'ES TF-IDF (x=3)'),
]

DATASETS = ['news', 'wiki']
#DOC_SIZES = [1000, 10000, 100000]
DOC_SIZES = [1000, 5000, 10000]


def get_index_size(index_path):
    """Calculate total size of index on disk in MB."""
    total_size = 0
    index_path = Path(index_path)
    
    if not index_path.exists():
        return 0.0
    
    for file in index_path.rglob('*'):
        if file.is_file():
            total_size += file.stat().st_size
    
    return total_size / (1024 * 1024)  # MB


def get_elasticsearch_size(index_name, es_client):
    """Get Elasticsearch index size."""
    try:
        stats = es_client.indices.stats(index=index_name)
        size_bytes = stats['indices'][index_name]['total']['store']['size_in_bytes']
        return size_bytes / (1024 * 1024)
    except:
        return 0.0


def create_index_with_metrics(dataset, size, variant_name, config_overrides, description):
    """Create a single index and collect metrics."""
    
    print(f"\n{'='*70}")
    print(f"Creating: {dataset}_{size}_{variant_name}")
    print(f"Description: {description}")
    print(f"{'='*70}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    index_id = f"{dataset}_{size}_{variant_name}"
    
    # Check if index already exists
    index_base_path = Path('indices')
    if 'rocksdb' in variant_name:
        index_path = index_base_path / 'rocksdb' / index_id
    else:
        index_path = index_base_path / 'selfindex' / index_id
    
    if index_path.exists():
        print(f"âš ï¸  Index already exists, skipping creation")
        print(f"   Path: {index_path}")
        
        # Still collect size metrics
        index_size = get_index_size(index_path)
        print(f"   Existing index size: {index_size:.2f} MB")
        return None
    
    # Load configuration
    overrides = [
        f'dataset={dataset}',
        f'dataset.sample_size={size}',
    ] + config_overrides
    
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(config_name="config_self", overrides=overrides)
    
    # Initialize appropriate index
    if 'rocksdb' in variant_name:
        index = RocksDBIndex(cfg)
    else:
        index = SelfIndex(cfg)
    
    # Load data
    print(f"Loading {size} documents from {dataset} dataset...")
    loader = DataLoader(cfg)
    documents = list(loader.load_dataset())
    
    print(f"Loaded {len(documents)} documents")
    
    # Measure memory before
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Create index and measure time
    print("Creating index...")
    start_time = time.perf_counter()
    
    try:
        index.create_index(index_id, documents)
        
        end_time = time.perf_counter()
        
        # Measure memory after
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Calculate metrics
        indexing_time = end_time - start_time
        throughput = len(documents) / indexing_time if indexing_time > 0 else 0
        memory_used = mem_after - mem_before
        index_size = get_index_size(index_path)
        
        metrics = {
            'index_id': index_id,
            'dataset': dataset,
            'doc_count': size,
            'variant': variant_name,
            'description': description,
            'timestamp': timestamp,
            'indexing_time_seconds': round(indexing_time, 2),
            'throughput_docs_per_sec': round(throughput, 2),
            'memory_before_mb': round(mem_before, 2),
            'memory_after_mb': round(mem_after, 2),
            'memory_used_mb': round(memory_used, 2),
            'index_size_mb': round(index_size, 2),
            'config_overrides': config_overrides,
            'index_path': str(index_path)
        }
        
        # Print summary
        print(f"\nâœ… Index created successfully!")
        print(f"   Time: {metrics['indexing_time_seconds']} seconds")
        print(f"   Throughput: {metrics['throughput_docs_per_sec']} docs/sec")
        print(f"   Memory used: {metrics['memory_used_mb']} MB")
        print(f"   Index size: {metrics['index_size_mb']} MB")
        
        # Save metrics
        results_dir = Path('results') / 'indexing'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_file = results_dir / f"{index_id}_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"   Metrics saved: {metrics_file}")
        
        return metrics
        
    except Exception as e:
        print(f"\n❌ Error creating index: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_es_index_with_metrics(dataset, size, variant_name, config_overrides, description):
    """Create an Elasticsearch index and collect metrics."""
    
    print(f"\n{'='*70}")
    print(f"Creating ES: {dataset}_{size}_{variant_name}")
    print(f"Description: {description}")
    print(f"{'='*70}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    index_id = f"{dataset}_{size}_{variant_name}"
    
    # Check if ES is available
    try:
        from src.indices.elasticsearch_index import ElasticsearchIndex
    except ImportError:
        print("⚠️  Elasticsearch not available, skipping")
        return None
    
    # Load configuration
    overrides = [
        f'dataset={dataset}',
        f'dataset.sample_size={size}',
        'datastore=elasticsearch'
    ] + config_overrides
    
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    
    # Initialize ES index
    try:
        index = ElasticsearchIndex(cfg)
        
        # Check if index already exists
        if index.es.indices.exists(index=index_id):
            print(f"⚠️  Index already exists, skipping creation")
            index_size = get_elasticsearch_size(index_id, index.es)
            print(f"   Existing index size: {index_size:.2f} MB")
            return None
        
    except Exception as e:
        print(f"❌ Elasticsearch connection failed: {e}")
        return None
    
    # Load data
    print(f"Loading {size} documents from {dataset} dataset...")
    loader = DataLoader(cfg)
    documents = list(loader.load_dataset())
    print(f"Loaded {len(documents)} documents")
    
    # Measure memory before
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)
    
    # Create index and measure time
    print("Creating ES index...")
    start_time = time.perf_counter()
    
    try:
        index.create_index(index_id, documents)
        end_time = time.perf_counter()
        
        # Measure memory after
        mem_after = process.memory_info().rss / (1024 * 1024)
        
        # Get index size from ES
        index_size = get_elasticsearch_size(index_id, index.es)
        
        # Calculate metrics
        indexing_time = end_time - start_time
        throughput = len(documents) / indexing_time if indexing_time > 0 else 0
        memory_used = mem_after - mem_before
        
        metrics = {
            'index_id': index_id,
            'dataset': dataset,
            'doc_count': size,
            'variant': variant_name,
            'description': description,
            'implementation': 'elasticsearch',
            'timestamp': timestamp,
            'indexing_time_seconds': round(indexing_time, 2),
            'throughput_docs_per_sec': round(throughput, 2),
            'memory_before_mb': round(mem_before, 2),
            'memory_after_mb': round(mem_after, 2),
            'memory_used_mb': round(memory_used, 2),
            'index_size_mb': round(index_size, 2),
            'config_overrides': config_overrides
        }
        
        # Print summary
        print(f"\n✅ ES Index created successfully!")
        print(f"   Time: {metrics['indexing_time_seconds']} seconds")
        print(f"   Throughput: {metrics['throughput_docs_per_sec']} docs/sec")
        print(f"   Memory used: {metrics['memory_used_mb']} MB")
        print(f"   Index size: {metrics['index_size_mb']} MB")
        
        # Save metrics
        results_dir = Path('results') / 'indexing'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_file = results_dir / f"{index_id}_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"   Metrics saved: {metrics_file}")
        
        return metrics
        
    except Exception as e:
        print(f"\n❌ Error creating ES index: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    total_self_variants = len(INDEX_VARIANTS)
    total_es_variants = len(ES_VARIANTS)
    total_all_variants = total_self_variants + total_es_variants
    
    print("="*70)
    print("INDEX CREATION WITH METRICS COLLECTION")
    print("="*70)
    print(f"Start time: {datetime.now()}")
    print(f"\nConfiguration:")
    print(f"  Datasets: {DATASETS}")
    print(f"  Document sizes: {DOC_SIZES}")
    print(f"  SelfIndex/RocksDB variants: {total_self_variants}")
    print(f"  Elasticsearch variants: {total_es_variants}")
    print(f"  Total variants per dataset/size: {total_all_variants}")
    print(f"  Total indices: {len(DATASETS) * len(DOC_SIZES) * total_all_variants}")
    print("="*70)
    
    all_metrics = []
    successful = 0
    skipped = 0
    failed = 0
    
    for dataset in DATASETS:
        for size in DOC_SIZES:
            print(f"\n\n{'#'*70}")
            print(f"# DATASET: {dataset.upper()} | SIZE: {size:,} documents")
            print(f"{'#'*70}")
            
            # Create SelfIndex and RocksDB variants
            print(f"\n{'='*70}")
            print(f"Creating SelfIndex/RocksDB variants...")
            print(f"{'='*70}")
            
            for variant_name, config_overrides, description in INDEX_VARIANTS:
                metrics = create_index_with_metrics(
                    dataset, size, variant_name, config_overrides, description
                )
                
                if metrics:
                    all_metrics.append(metrics)
                    successful += 1
                elif metrics is None:
                    skipped += 1
                else:
                    failed += 1
            
            # Create Elasticsearch variants (if available)
            print(f"\n{'='*70}")
            print(f"Creating Elasticsearch variants...")
            print(f"{'='*70}")
            
            for variant_name, config_overrides, description in ES_VARIANTS:
                metrics = create_es_index_with_metrics(
                    dataset, size, variant_name, config_overrides, description
                )
                
                if metrics:
                    all_metrics.append(metrics)
                    successful += 1
                elif metrics is None:
                    skipped += 1
                else:
                    failed += 1
    
    # Save consolidated metrics
    results_dir = Path('results') / 'indexing'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    consolidated_file = results_dir / f"all_indices_{timestamp}.json"
    with open(consolidated_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Create summary
    summary = {
        'timestamp': timestamp,
        'total_attempted': len(DATASETS) * len(DOC_SIZES) * total_all_variants,
        'successful': successful,
        'skipped': skipped,
        'failed': failed,
        'datasets': DATASETS,
        'doc_sizes': DOC_SIZES,
        'self_variants': total_self_variants,
        'es_variants': total_es_variants,
        'total_variants': total_all_variants
    }
    
    summary_file = results_dir / f"summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total indices attempted: {summary['total_attempted']}")
    print(f"  - SelfIndex/RocksDB: {len(DATASETS) * len(DOC_SIZES) * total_self_variants}")
    print(f"  - Elasticsearch: {len(DATASETS) * len(DOC_SIZES) * total_es_variants}")
    print(f"Successfully created: {successful}")
    print(f"Skipped (already exist): {skipped}")
    print(f"Failed: {failed}")
    print(f"\nResults saved to: {results_dir}")
    print(f"Consolidated metrics: {consolidated_file}")
    print(f"Summary: {summary_file}")
    print(f"\nEnd time: {datetime.now()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()