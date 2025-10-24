#!/usr/bin/env python
"""
Benchmark all index configurations and collect metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indices.self_index import SelfIndex
from src.utils.benchmark import Benchmarker
from src.utils.query_generator import QueryGenerator
import hydra
import json
import time
import psutil
import os
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

# Load .env variables and register resolver
load_dotenv()
OmegaConf.register_new_resolver("env", os.getenv)

INDICES = {
    # Format: name: (config_type, description)
    "idx_boolean": ("self_boolean", "Boolean Index (x=1)"),
    "idx_wordcount": ("self_wordcount", "WordCount Index (x=2)"),
    "idx_tfidf": ("self_tfidf", "TF-IDF Index (x=3)"),
    "idx_tfidf_skip": ("self_tfidf", "TF-IDF + Skip Pointers (i=1)"),
    "idx_threshold": ("self_tfidf_threshold", "TF-IDF + Thresholding (o=th)"),
    "idx_earlystop": ("self_tfidf_earlystop", "TF-IDF + Early Stop (o=es)"),
    "idx_docatat": ("self_tfidf_optimized", "TF-IDF + DocAtATime (q=D)"),
    "idx_compressed": ("self_tfidf_compressed", "TF-IDF + Gap Encoding (z=1)"),
    "idx_gzip": ("self_tfidf_gzip", "TF-IDF + Gzip (z=2)"),
}

ES_INDICES = {
    # Format: name: (implementation, config_type, description)
    "es_news_boolean": ("elasticsearch", "boolean", "ES Boolean (News)"),
    "es_news_wordcount": ("elasticsearch", "wordcount", "ES WordCount (News)"),
    "es_news_tfidf": ("elasticsearch", "tfidf", "ES TF-IDF (News)"),
    "es_wiki_boolean": ("elasticsearch", "boolean", "ES Boolean (Wiki)"),
    "es_wiki_wordcount": ("elasticsearch", "wordcount", "ES WordCount (Wiki)"),
    "es_wiki_tfidf": ("elasticsearch", "tfidf", "ES TF-IDF (Wiki)"),
}

def benchmark_index(index_name, config_type, queries):
    """Benchmark a single index."""
    print(f"\nBenchmarking: {index_name}")
    
    # Load config
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(config_name="config_self", overrides=[f"index={config_type}"])
    
    # Load index
    index = SelfIndex(cfg)
    index_path = Path(f"indices/selfindex/{index_name}")
    
    if not index_path.exists():
        print(f"  ❌ Not found, skipping")
        return None
    
    index.load_index(str(index_path))
    
    # Measure index size
    index_size = sum(f.stat().st_size for f in index_path.rglob('*') if f.is_file())
    
    # Measure query performance
    latencies = []
    memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    for query in queries[:50]:  # Use 50 queries for speed
        start = time.perf_counter()
        try:
            index.query(query)
            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)
        except:
            pass
    
    memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    import numpy as np
    
    metrics = {
        'index_name': index_name,
        'index_size_mb': index_size / (1024 * 1024),
        'num_queries': len(latencies),
        'latency_mean': np.mean(latencies),
        'latency_p50': np.percentile(latencies, 50),
        'latency_p95': np.percentile(latencies, 95),
        'latency_p99': np.percentile(latencies, 99),
        'throughput_qps': len(latencies) / (sum(latencies) / 1000),
        'memory_mb': memory_after,
    }
    
    print(f"  Index Size: {metrics['index_size_mb']:.2f} MB")
    print(f"  Mean Latency: {metrics['latency_mean']:.2f} ms")
    print(f"  P95 Latency: {metrics['latency_p95']:.2f} ms")
    print(f"  Throughput: {metrics['throughput_qps']:.2f} qps")
    
    return metrics

def benchmark_es_index(index_name, config_type, queries):
    """Benchmark an Elasticsearch index."""
    print(f"\nBenchmarking ES: {index_name}")
    
    # Load config with elasticsearch
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(config_name="config", overrides=[f"index={config_type}"])
    
    from src.indices.elasticsearch_index import ElasticsearchIndex
    
    try:
        index = ElasticsearchIndex(cfg)
        
        # Check if index exists
        if not index.es.indices.exists(index=index_name):
            print(f"  ❌ Index not found, skipping")
            return None
        
        # Measure query performance
        latencies = []
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        for query in queries[:50]:
            start = time.perf_counter()
            try:
                index.query(query, index_name)
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)
            except:
                pass
        
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Get index size from ES
        stats = index.es.indices.stats(index=index_name)
        index_size_bytes = stats['indices'][index_name]['total']['store']['size_in_bytes']
        
        metrics = {
            'index_name': index_name,
            'implementation': 'elasticsearch',
            'index_size_mb': index_size_bytes / (1024 * 1024),
            'num_queries': len(latencies),
            'latency_mean': np.mean(latencies),
            'latency_p50': np.percentile(latencies, 50),
            'latency_p95': np.percentile(latencies, 95),
            'latency_p99': np.percentile(latencies, 99),
            'throughput_qps': len(latencies) / (sum(latencies) / 1000),
            'memory_mb': memory_after,
        }
        
        print(f"  Index Size: {metrics['index_size_mb']:.2f} MB")
        print(f"  Mean Latency: {metrics['latency_mean']:.2f} ms")
        
        return metrics
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def main():
    print("="*60)
    print("BENCHMARKING ALL CONFIGURATIONS")
    print("="*60)
    
    # Generate test queries
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(config_name="config_self")
    
    query_gen = QueryGenerator(cfg, seed=42)
    queries = query_gen.generate_queries(100)
    
    # Benchmark SelfIndex
    results = []
    for index_name, (config_type, description) in INDICES.items():
        metrics = benchmark_index(index_name, config_type, queries)
        if metrics:
            metrics['description'] = description
            metrics['implementation'] = 'selfindex'
            results.append(metrics)
    
    # Benchmark Elasticsearch 
    for index_name, (impl, config_type, description) in ES_INDICES.items():
        metrics = benchmark_es_index(index_name, config_type, queries)
        if metrics:
            results.append(metrics)

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to results/benchmark_results.json")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        print(f"{r['index_name']:20s} | Size: {r['index_size_mb']:6.2f}MB | "
              f"Latency: {r['latency_mean']:6.2f}ms | "
              f"P95: {r['latency_p95']:6.2f}ms")

if __name__ == "__main__":
    main()