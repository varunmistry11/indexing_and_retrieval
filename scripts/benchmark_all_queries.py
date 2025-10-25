#!/usr/bin/env python
"""
Benchmark all created indices with query performance metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import OmegaConf
import json
import time
import psutil
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv

from src.indices.self_index import SelfIndex
from src.indices.rocksdb_index import RocksDBIndex
from src.utils.query_generator import QueryGenerator

# Load environment variables
load_dotenv()
OmegaConf.register_new_resolver("env", os.getenv)

# Same variants as creation script
INDEX_VARIANTS = [
    'boolean_custom', 'wordcount_custom', 'tfidf_custom', 'tfidf_rocksdb',
    'tfidf_skip', 'tfidf_gap', 'tfidf_gzip', 'tfidf_threshold', 'tfidf_earlystop'
]

ES_VARIANTS = ['es_boolean', 'es_wordcount', 'es_tfidf']

DATASETS = ['news', 'wiki']
DOC_SIZES = [1000, 5000, 10000]
#DOC_SIZES = [1000, 10000, 100000]
NUM_QUERIES = 100
WARMUP_QUERIES = 5


def load_or_generate_queries(dataset, size, num_queries=100):
    """Load cached queries or generate new ones."""
    cache_dir = Path('data') / 'query_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"queries_{dataset}_{size}_{num_queries}.json"
    
    # Try to load from cache
    if cache_file.exists():
        print(f"  Loading cached queries from {cache_file}")
        with open(cache_file, 'r') as f:
            data = json.load(f)
            return data['queries']
    
    # Generate new queries
    print(f"  Generating {num_queries} queries for {dataset}...")
    
    # Load config for query generation
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(config_name="config_self", overrides=[f'dataset={dataset}'])
    
    query_gen = QueryGenerator(cfg, seed=42)
    queries = query_gen.generate_queries(num_queries)
    
    # Save to cache
    with open(cache_file, 'w') as f:
        json.dump({
            'dataset': dataset,
            'size': size,
            'num_queries': num_queries,
            'seed': 42,
            'timestamp': datetime.now().isoformat(),
            'queries': queries
        }, f, indent=2)
    
    print(f"  Saved queries to {cache_file}")
    return queries


def benchmark_index(dataset, size, variant_name, queries):
    """Benchmark a single index with queries."""
    
    print(f"\n{'='*70}")
    print(f"Benchmarking: {dataset}_{size}_{variant_name}")
    print(f"{'='*70}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    index_id = f"{dataset}_{size}_{variant_name}"
    
    # Check if index exists
    index_base_path = Path('indices')
    if 'rocksdb' in variant_name:
        index_path = index_base_path / 'rocksdb' / index_id
        config_overrides = ['index=rocksdb_tfidf', 'datastore=rocksdb']
    else:
        index_path = index_base_path / 'selfindex' / index_id
        
        # Map variant to config
        config_map = {
            'boolean_custom': 'self_boolean',
            'wordcount_custom': 'self_wordcount',
            'tfidf_custom': 'self_tfidf',
            'tfidf_skip': 'self_tfidf_optimized',
            'tfidf_gap': 'self_tfidf_compressed',
            'tfidf_gzip': 'self_tfidf_gzip',
            'tfidf_threshold': 'self_tfidf_threshold',
            'tfidf_earlystop': 'self_tfidf_earlystop',
        }
        config_type = config_map.get(variant_name, 'self_tfidf')
        config_overrides = [f'index={config_type}']
    
    if not index_path.exists():
        print(f"âŒ Index not found: {index_path}")
        return None
    
    print(f"âœ… Index found: {index_path}")
    
    # Load configuration
    overrides = [f'dataset={dataset}'] + config_overrides
    
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(config_name="config_self", overrides=overrides)
    
    # Load index
    try:
        if 'rocksdb' in variant_name:
            index = RocksDBIndex(cfg)
        else:
            index = SelfIndex(cfg)
        
        print("Loading index...")
        index.load_index(str(index_path))
        print("âœ… Index loaded")
        
    except Exception as e:
        print(f"âŒ Failed to load index: {e}")
        return None
    
    # Warmup queries
    print(f"\nRunning {WARMUP_QUERIES} warmup queries...")
    for i in range(min(WARMUP_QUERIES, len(queries))):
        try:
            index.query(queries[i], index_id)
        except Exception as e:
            print(f"  Warmup query {i} failed: {e}")
    
    # Benchmark queries
    print(f"\nBenchmarking {len(queries)} queries...")
    
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB
    
    latencies = []
    successful_queries = 0
    failed_queries = 0
    
    for i, query in enumerate(queries):
        try:
            start_time = time.perf_counter()
            result_json = index.query(query, index_id)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            successful_queries += 1
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(queries)} queries...")
                
        except Exception as e:
            failed_queries += 1
            if failed_queries <= 3:  # Only print first 3 failures
                print(f"  Query {i} failed: {e}")
    
    mem_after = process.memory_info().rss / (1024 * 1024)  # MB
    peak_mem = mem_after
    
    # Calculate metrics
    if latencies:
        metrics = {
            'index_id': index_id,
            'dataset': dataset,
            'doc_count': size,
            'variant': variant_name,
            'timestamp': timestamp,
            'total_queries': len(queries),
            'successful_queries': successful_queries,
            'failed_queries': failed_queries,
            'latency_ms': {
                'mean': round(float(np.mean(latencies)), 3),
                'median': round(float(np.median(latencies)), 3),
                'std': round(float(np.std(latencies)), 3),
                'min': round(float(np.min(latencies)), 3),
                'max': round(float(np.max(latencies)), 3),
                'p50': round(float(np.percentile(latencies, 50)), 3),
                'p95': round(float(np.percentile(latencies, 95)), 3),
                'p99': round(float(np.percentile(latencies, 99)), 3),
            },
            'throughput_qps': round(successful_queries / (sum(latencies) / 1000), 2),
            'memory_mb': {
                'before': round(mem_before, 2),
                'after': round(mem_after, 2),
                'peak': round(peak_mem, 2)
            },
            'index_path': str(index_path)
        }
        
        # Print summary
        print(f"\nâœ… Benchmark complete!")
        print(f"   Successful queries: {successful_queries}/{len(queries)}")
        print(f"   Mean latency: {metrics['latency_ms']['mean']:.2f} ms")
        print(f"   P95 latency: {metrics['latency_ms']['p95']:.2f} ms")
        print(f"   P99 latency: {metrics['latency_ms']['p99']:.2f} ms")
        print(f"   Throughput: {metrics['throughput_qps']:.2f} qps")
        print(f"   Peak memory: {metrics['memory_mb']['peak']:.2f} MB")
        
        # Save metrics
        results_dir = Path('results') / 'queries'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_file = results_dir / f"{index_id}_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"   Metrics saved: {metrics_file}")
        
        return metrics
        
    else:
        print(f"\n❌ No successful queries")
        return None


def benchmark_es_index(dataset, size, variant_name, queries):
    """Benchmark an Elasticsearch index."""
    
    print(f"\n{'='*70}")
    print(f"Benchmarking ES: {dataset}_{size}_{variant_name}")
    print(f"{'='*70}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    index_id = f"{dataset}_{size}_{variant_name}"
    
    # Check if ES is available
    try:
        from src.indices.elasticsearch_index import ElasticsearchIndex
    except ImportError:
        print("⚠️  Elasticsearch not available, skipping")
        return None
    
    # Map variant to config
    config_map = {
        'es_boolean': 'boolean',
        'es_wordcount': 'wordcount',
        'es_tfidf': 'tfidf',
    }
    config_type = config_map.get(variant_name, 'tfidf')
    
    # Load configuration
    overrides = [
        f'dataset={dataset}',
        f'index={config_type}',
        'datastore=elasticsearch'
    ]
    
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    
    # Load index
    try:
        index = ElasticsearchIndex(cfg)
        
        if not index.es.indices.exists(index=index_id):
            print(f"❌ ES Index not found: {index_id}")
            return None
        
        print(f"✅ ES Index found: {index_id}")
        
    except Exception as e:
        print(f"❌ Elasticsearch connection failed: {e}")
        return None
    
    # Warmup queries
    print(f"\nRunning {WARMUP_QUERIES} warmup queries...")
    for i in range(min(WARMUP_QUERIES, len(queries))):
        try:
            index.query(queries[i], index_id)
        except Exception as e:
            print(f"  Warmup query {i} failed: {e}")
    
    # Benchmark queries
    print(f"\nBenchmarking {len(queries)} queries...")
    
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)
    
    latencies = []
    successful_queries = 0
    failed_queries = 0
    
    for i, query in enumerate(queries):
        try:
            start_time = time.perf_counter()
            result_json = index.query(query, index_id)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            successful_queries += 1
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(queries)} queries...")
                
        except Exception as e:
            failed_queries += 1
            if failed_queries <= 3:
                print(f"  Query {i} failed: {e}")
    
    mem_after = process.memory_info().rss / (1024 * 1024)
    
    # Calculate metrics
    if latencies:
        metrics = {
            'index_id': index_id,
            'dataset': dataset,
            'doc_count': size,
            'variant': variant_name,
            'implementation': 'elasticsearch',
            'timestamp': timestamp,
            'total_queries': len(queries),
            'successful_queries': successful_queries,
            'failed_queries': failed_queries,
            'latency_ms': {
                'mean': round(float(np.mean(latencies)), 3),
                'median': round(float(np.median(latencies)), 3),
                'std': round(float(np.std(latencies)), 3),
                'min': round(float(np.min(latencies)), 3),
                'max': round(float(np.max(latencies)), 3),
                'p50': round(float(np.percentile(latencies, 50)), 3),
                'p95': round(float(np.percentile(latencies, 95)), 3),
                'p99': round(float(np.percentile(latencies, 99)), 3),
            },
            'throughput_qps': round(successful_queries / (sum(latencies) / 1000), 2),
            'memory_mb': {
                'before': round(mem_before, 2),
                'after': round(mem_after, 2),
                'peak': round(mem_after, 2)
            }
        }
        
        # Print summary
        print(f"\n✅ ES Benchmark complete!")
        print(f"   Successful queries: {successful_queries}/{len(queries)}")
        print(f"   Mean latency: {metrics['latency_ms']['mean']:.2f} ms")
        print(f"   P95 latency: {metrics['latency_ms']['p95']:.2f} ms")
        print(f"   Throughput: {metrics['throughput_qps']:.2f} qps")
        
        # Save metrics
        results_dir = Path('results') / 'queries'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_file = results_dir / f"{index_id}_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"   Metrics saved: {metrics_file}")
        return metrics
        
    else:
        print(f"\n❌ No successful queries")
        return None


def main():
    total_self_variants = len(INDEX_VARIANTS)
    total_es_variants = len(ES_VARIANTS)
    total_all_variants = total_self_variants + total_es_variants
    
    print("="*70)
    print("QUERY BENCHMARK WITH METRICS COLLECTION")
    print("="*70)
    print(f"Start time: {datetime.now()}")
    print(f"\nConfiguration:")
    print(f"  Datasets: {DATASETS}")
    print(f"  Document sizes: {DOC_SIZES}")
    print(f"  SelfIndex/RocksDB variants: {total_self_variants}")
    print(f"  Elasticsearch variants: {total_es_variants}")
    print(f"  Total variants per dataset/size: {total_all_variants}")
    print(f"  Queries per benchmark: {NUM_QUERIES}")
    print(f"  Total benchmarks: {len(DATASETS) * len(DOC_SIZES) * total_all_variants}")
    print("="*70)
    
    all_metrics = []
    successful = 0
    failed = 0
    
    for dataset in DATASETS:
        for size in DOC_SIZES:
            print(f"\n\n{'#'*70}")
            print(f"# DATASET: {dataset.upper()} | SIZE: {size:,} documents")
            print(f"{'#'*70}")
            
            # Load/generate queries once per dataset+size
            queries = load_or_generate_queries(dataset, size, NUM_QUERIES)
            
            # Benchmark SelfIndex and RocksDB variants
            print(f"\n{'='*70}")
            print(f"Benchmarking SelfIndex/RocksDB variants...")
            print(f"{'='*70}")
            
            for variant_name in INDEX_VARIANTS:
                metrics = benchmark_index(dataset, size, variant_name, queries)
                
                if metrics:
                    all_metrics.append(metrics)
                    successful += 1
                else:
                    failed += 1
            
            # Benchmark Elasticsearch variants
            print(f"\n{'='*70}")
            print(f"Benchmarking Elasticsearch variants...")
            print(f"{'='*70}")
            
            for variant_name in ES_VARIANTS:
                metrics = benchmark_es_index(dataset, size, variant_name, queries)
                
                if metrics:
                    all_metrics.append(metrics)
                    successful += 1
                else:
                    failed += 1
    
    # Save consolidated metrics
    results_dir = Path('results') / 'queries'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    consolidated_file = results_dir / f"all_benchmarks_{timestamp}.json"
    with open(consolidated_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Create summary
    summary = {
        'timestamp': timestamp,
        'total_benchmarks': len(DATASETS) * len(DOC_SIZES) * total_all_variants,
        'successful': successful,
        'failed': failed,
        'num_queries': NUM_QUERIES,
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
    print(f"Total benchmarks: {summary['total_benchmarks']}")
    print(f"  - SelfIndex/RocksDB: {len(DATASETS) * len(DOC_SIZES) * total_self_variants}")
    print(f"  - Elasticsearch: {len(DATASETS) * len(DOC_SIZES) * total_es_variants}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults saved to: {results_dir}")
    print(f"Consolidated metrics: {consolidated_file}")
    print(f"Summary: {summary_file}")
    print(f"\nEnd time: {datetime.now()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()