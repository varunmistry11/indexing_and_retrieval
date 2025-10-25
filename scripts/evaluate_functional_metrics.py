#!/usr/bin/env python
"""
Evaluate functional metrics (precision, recall, F1) using ES as baseline.
File: scripts/evaluate_functional_metrics.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import OmegaConf
import json
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv
from collections import defaultdict

from src.indices.self_index import SelfIndex
from src.indices.rocksdb_index import RocksDBIndex
from src.indices.elasticsearch_index import ElasticsearchIndex
from src.utils.query_generator import QueryGenerator

# Load environment variables
load_dotenv()
OmegaConf.register_new_resolver("env", os.getenv)

# Variants to evaluate (excluding ES itself)
EVAL_VARIANTS = [
    'boolean_custom', 'wordcount_custom', 'tfidf_custom', 'tfidf_rocksdb',
    'tfidf_skip', 'tfidf_gap', 'tfidf_gzip', 'tfidf_threshold', 'tfidf_earlystop'
]

DATASETS = ['news', 'wiki']
DOC_SIZES = [1000, 5000, 10000]
NUM_QUERIES = 100


def load_or_generate_queries(dataset, size, num_queries=100):
    """Load cached queries or generate new ones."""
    cache_dir = Path('data') / 'query_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"queries_{dataset}_{size}_{num_queries}.json"
    
    if cache_file.exists():
        print(f"  Loading cached queries from {cache_file}")
        with open(cache_file, 'r') as f:
            data = json.load(f)
            return data['queries']
    
    print(f"  Generating {num_queries} queries for {dataset}...")
    
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(config_name="config_self", overrides=[f'dataset={dataset}'])
    
    query_gen = QueryGenerator(cfg, seed=42)
    queries = query_gen.generate_queries(num_queries)
    
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


def load_index(dataset, size, variant_name):
    """Load an index based on variant name."""
    
    index_id = f"{dataset}_{size}_{variant_name}"
    
    # Determine index type and load appropriate index
    if 'rocksdb' in variant_name:
        index_path = Path('indices') / 'rocksdb' / index_id
        config_overrides = ['index=rocksdb_tfidf', 'datastore=rocksdb']
    else:
        index_path = Path('indices') / 'selfindex' / index_id
        
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
        print(f"❌ Index not found: {index_path}")
        return None
    
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
        
        index.load_index(str(index_path))
        return index
        
    except Exception as e:
        print(f"❌ Failed to load index: {e}")
        return None


def load_es_index(dataset, size):
    """Load Elasticsearch index."""
    
    index_id = f"{dataset}_{size}_es_tfidf"
    
    try:
        from src.indices.elasticsearch_index import ElasticsearchIndex
    except ImportError:
        print("⚠️  Elasticsearch not available")
        return None
    
    # Load configuration
    overrides = [
        f'dataset={dataset}',
        f'index=tfidf',
        'datastore=elasticsearch'
    ]
    
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    
    try:
        index = ElasticsearchIndex(cfg)
        
        if not index.es.indices.exists(index=index_id):
            print(f"❌ ES Index not found: {index_id}")
            return None
        
        return index
        
    except Exception as e:
        print(f"❌ Elasticsearch connection failed: {e}")
        return None


def get_top_k_results(index, query, index_id, k=20):
    """Get top-K document IDs from index."""
    try:
        result_json = index.query(query, index_id)
        
        # Handle string JSON response
        if isinstance(result_json, str):
            results = json.loads(result_json)
        else:
            results = result_json
        
        # Extract doc IDs - handle different result formats
        doc_ids = []
        
        # Format 1: {'results': [...]} (SelfIndex)
        if 'results' in results:
            for result in results['results'][:k]:
                # Try different doc ID keys
                if 'doc_id' in result:
                    doc_ids.append(result['doc_id'])
                elif 'id' in result:
                    doc_ids.append(result['id'])
                elif 'document_id' in result:
                    doc_ids.append(result['document_id'])
        
        # Format 2: {'documents': [...]} (Elasticsearch)
        elif 'documents' in results:
            for result in results['documents'][:k]:
                # Try different doc ID keys
                if 'id' in result:
                    doc_ids.append(result['id'])
                elif 'doc_id' in result:
                    doc_ids.append(result['doc_id'])
                elif '_id' in result:
                    doc_ids.append(result['_id'])
                # Also check in 'source' field (ES sometimes nests it)
                elif 'source' in result and 'id' in result['source']:
                    doc_ids.append(result['source']['id'])
        
        # Format 3: Direct list
        elif isinstance(results, list):
            for result in results[:k]:
                if isinstance(result, dict):
                    if 'doc_id' in result:
                        doc_ids.append(result['doc_id'])
                    elif 'id' in result:
                        doc_ids.append(result['id'])
                else:
                    doc_ids.append(result)
        
        # Format 4: {'hits': {'hits': [...]}} (native ES format)
        elif 'hits' in results and 'hits' in results['hits']:
            for hit in results['hits']['hits'][:k]:
                if '_id' in hit:
                    doc_ids.append(hit['_id'])
                elif '_source' in hit and 'doc_id' in hit['_source']:
                    doc_ids.append(hit['_source']['doc_id'])
        
        return doc_ids
        
    except Exception as e:
        import traceback
        print(f"  Query failed with error: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        return []


def compute_precision_recall_f1(retrieved_docs, relevant_docs, k):
    """
    Compute precision, recall, F1 at rank K.
    
    Args:
        retrieved_docs: List of retrieved document IDs (top-K from current index)
        relevant_docs: List of relevant document IDs (top-K from ES baseline)
        k: Cutoff rank
    
    Returns:
        Dict with precision, recall, F1
    """
    
    retrieved_set = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs[:k])
    
    # True positives: docs in both sets
    true_positives = len(retrieved_set & relevant_set)
    
    # Precision: TP / retrieved
    precision = true_positives / k if k > 0 else 0
    
    # Recall: TP / relevant
    recall = true_positives / len(relevant_set) if len(relevant_set) > 0 else 0
    
    # F1 score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives
    }


def cache_es_results(dataset, size, queries):
    """Cache ES results for all queries."""
    
    print(f"\n{'='*70}")
    print(f"Caching ES baseline results: {dataset}_{size}")
    print(f"{'='*70}")
    
    cache_dir = Path('data') / 'es_baseline_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"es_results_{dataset}_{size}.json"
    
    # Check if already cached
    if cache_file.exists():
        print(f"✅ ES baseline already cached: {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    # Load ES index
    es_index = load_es_index(dataset, size)
    if not es_index:
        print("❌ Could not load ES index")
        return None
    
    index_id = f"{dataset}_{size}_es_tfidf"
    
    # Query ES for all queries
    es_results = {}
    
    for i, query in enumerate(queries):
        try:
            doc_ids = get_top_k_results(es_index, query, index_id, k=20)
            es_results[query] = doc_ids
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(queries)} queries...")
        
        except Exception as e:
            print(f"  Query {i} failed: {e}")
            es_results[query] = []
    
    # Save cache
    with open(cache_file, 'w') as f:
        json.dump(es_results, f, indent=2)
    
    print(f"✅ Cached ES results: {cache_file}")
    return es_results


def evaluate_variant(dataset, size, variant_name, queries, es_baseline):
    """Evaluate a single variant against ES baseline."""
    
    print(f"\n{'='*70}")
    print(f"Evaluating: {dataset}_{size}_{variant_name}")
    print(f"{'='*70}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    index_id = f"{dataset}_{size}_{variant_name}"
    
    # Load index
    index = load_index(dataset, size, variant_name)
    if not index:
        return None
    
    print(f"✅ Index loaded successfully")
    
    # Metrics storage
    metrics_by_k = {5: [], 10: [], 20: []}
    
    successful_queries = 0
    failed_queries = 0
    
    # Test first query with detailed output
    if queries:
        test_query = queries[0]
        print(f"\n🔍 Testing first query: '{test_query}'")
        
        # Test ES baseline
        es_docs = es_baseline.get(test_query, [])
        print(f"   ES baseline returned {len(es_docs)} docs")
        if es_docs:
            print(f"   ES top-3: {es_docs[:3]}")
        
        # Test current index
        try:
            test_docs = get_top_k_results(index, test_query, index_id, k=20)
            print(f"   Current index returned {len(test_docs)} docs")
            if test_docs:
                print(f"   Current top-3: {test_docs[:3]}")
            else:
                print(f"   ⚠️  No results from current index - checking query method...")
                # Try to see what query returns
                raw_result = index.query(test_query, index_id)
                print(f"   Raw result type: {type(raw_result)}")
                print(f"   Raw result (first 200 chars): {str(raw_result)[:200]}")
        except Exception as e:
            print(f"   ❌ Test query failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 Processing all {len(queries)} queries...")
    
    for i, query in enumerate(queries):
        # Get ES baseline results
        es_docs = es_baseline.get(query, [])
        if not es_docs:
            continue
        
        # Get current index results
        try:
            current_docs = get_top_k_results(index, query, index_id, k=20)
            
            if not current_docs:
                failed_queries += 1
                if failed_queries <= 3:
                    print(f"  Query {i}: No results returned")
                continue
            
            successful_queries += 1
            
            # Compute metrics at different K values
            for k in [5, 10, 20]:
                metrics = compute_precision_recall_f1(current_docs, es_docs, k)
                metrics_by_k[k].append(metrics)
        
        except Exception as e:
            failed_queries += 1
            if failed_queries <= 3:
                print(f"  Query {i} failed: {e}")
        
        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(queries)} queries... (Success: {successful_queries}, Failed: {failed_queries})")
    
    # Aggregate metrics
    results = {
        'index_id': index_id,
        'dataset': dataset,
        'doc_count': size,
        'variant': variant_name,
        'timestamp': timestamp,
        'total_queries': len(queries),
        'successful_queries': successful_queries,
        'failed_queries': failed_queries,
    }
    
    # Average metrics for each K
    for k in [5, 10, 20]:
        if metrics_by_k[k]:
            results[f'precision@{k}'] = round(np.mean([m['precision'] for m in metrics_by_k[k]]), 4)
            results[f'recall@{k}'] = round(np.mean([m['recall'] for m in metrics_by_k[k]]), 4)
            results[f'f1@{k}'] = round(np.mean([m['f1'] for m in metrics_by_k[k]]), 4)
        else:
            results[f'precision@{k}'] = 0
            results[f'recall@{k}'] = 0
            results[f'f1@{k}'] = 0
    
    # Print summary
    print(f"\n✅ Evaluation complete!")
    print(f"   Successful queries: {successful_queries}/{len(queries)}")
    print(f"   Failed queries: {failed_queries}")
    if successful_queries > 0:
        print(f"   Precision@10: {results['precision@10']:.4f}")
        print(f"   Recall@10: {results['recall@10']:.4f}")
        print(f"   F1@10: {results['f1@10']:.4f}")
    else:
        print(f"   ⚠️  All queries failed - check index query method")
    
    # Save metrics
    results_dir = Path('results') / 'functional'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = results_dir / f"{index_id}_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   Metrics saved: {metrics_file}")
    
    return results


def main():
    print("="*70)
    print("FUNCTIONAL METRICS EVALUATION (using ES as baseline)")
    print("="*70)
    print(f"Start time: {datetime.now()}")
    print(f"\nConfiguration:")
    print(f"  Datasets: {DATASETS}")
    print(f"  Document sizes: {DOC_SIZES}")
    print(f"  Variants to evaluate: {len(EVAL_VARIANTS)}")
    print(f"  Queries per evaluation: {NUM_QUERIES}")
    print(f"  Baseline: Elasticsearch TF-IDF")
    print("="*70)
    
    all_metrics = []
    successful = 0
    failed = 0
    
    for dataset in DATASETS:
        for size in DOC_SIZES:
            print(f"\n\n{'#'*70}")
            print(f"# DATASET: {dataset.upper()} | SIZE: {size:,} documents")
            print(f"{'#'*70}")
            
            # Load queries
            queries = load_or_generate_queries(dataset, size, NUM_QUERIES)
            
            # Cache ES baseline results
            es_baseline = cache_es_results(dataset, size, queries)
            
            if not es_baseline:
                print(f"⚠️  Skipping {dataset}_{size} - ES baseline not available")
                continue
            
            # Evaluate each variant
            for variant_name in EVAL_VARIANTS:
                metrics = evaluate_variant(dataset, size, variant_name, queries, es_baseline)
                
                if metrics:
                    all_metrics.append(metrics)
                    successful += 1
                else:
                    failed += 1
    
    # Save consolidated metrics
    results_dir = Path('results') / 'functional'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    consolidated_file = results_dir / f"all_functional_metrics_{timestamp}.json"
    with open(consolidated_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Create summary
    summary = {
        'timestamp': timestamp,
        'total_evaluations': len(DATASETS) * len(DOC_SIZES) * len(EVAL_VARIANTS),
        'successful': successful,
        'failed': failed,
        'baseline': 'elasticsearch_tfidf',
        'metrics': ['precision@k', 'recall@k', 'f1@k'],
        'k_values': [5, 10, 20]
    }
    
    summary_file = results_dir / f"summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total evaluations: {summary['total_evaluations']}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults saved to: {results_dir}")
    print(f"Consolidated metrics: {consolidated_file}")
    print(f"Summary: {summary_file}")
    print(f"\nEnd time: {datetime.now()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()