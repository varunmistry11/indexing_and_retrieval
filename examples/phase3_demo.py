"""
Phase 3 Demo and Comparison Scripts
File: examples/phase3_demo.py

Demonstrates SelfIndex functionality and compares with Elasticsearch.
"""

import sys
import time
from pathlib import Path
from typing import List, Dict
import tempfile
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indices.self_index import SelfIndex


def create_mock_config(index_type='TFIDF', query_proc='TERMatat', optimization='Null'):
    """Helper to create mock config."""
    config = MagicMock()
    config.index.type = index_type
    config.index.query_proc = query_proc
    config.index.optimization = optimization
    config.index.get = MagicMock(side_effect=lambda k, d=None: {
        'optimization': optimization,
        'query_proc': query_proc
    }.get(k, d))
    config.datastore.format = 'json'
    config.compression.type = 'NONE'
    config.preprocessing = MagicMock()
    config.preprocessing.lowercase = True
    config.preprocessing.remove_punctuation = True
    config.preprocessing.remove_stopwords = True
    config.preprocessing.stemming = True
    config.preprocessing.stemmer = 'porter'
    config.preprocessing.min_word_length = 1
    config.preprocessing.max_word_length = 50
    config.paths.index_storage = tempfile.mkdtemp()
    return config


def demo_basic_indexing():
    """Demo 1: Basic indexing and querying."""
    print("="*60)
    print("DEMO 1: Basic Indexing and Querying")
    print("="*60)
    
    # Sample documents
    documents = [
        {'id': 'doc1', 'text': 'Python programming language for data science'},
        {'id': 'doc2', 'text': 'Machine learning with Python and TensorFlow'},
        {'id': 'doc3', 'text': 'Java programming for enterprise applications'},
        {'id': 'doc4', 'text': 'Deep learning and neural networks in Python'},
        {'id': 'doc5', 'text': 'Natural language processing with transformers'},
    ]
    
    # Create index
    config = create_mock_config()
    index = SelfIndex(config)
    
    print("\nCreating index...")
    stats = index.create_index_from_documents('demo_dataset', documents)
    
    print(f"\nIndex Statistics:")
    print(f"  Documents indexed: {stats['documents_processed']}")
    print(f"  Vocabulary size: {stats['vocabulary_size']}")
    print(f"  Indexing time: {stats['indexing_time_seconds']:.3f}s")
    print(f"  Index size: {stats['index_size_bytes']:,} bytes")
    
    # Query examples
    queries = [
        'python',
        'machine learning',
        '"python" AND "learning"',
        '"java" OR "python"',
    ]
    
    print("\n" + "="*60)
    print("Query Examples:")
    print("="*60)
    
    for query in queries:
        print(f"\nQuery: {query}")
        start = time.time()
        results = index.query_with_params(query, k=3)
        duration = time.time() - start
        
        print(f"  Time: {duration*1000:.2f}ms")
        print(f"  Results: {len(results)}")
        for r in results:
            print(f"    {r['rank']}. {r['doc_id']} (score: {r['score']:.4f})")
    
    print("\n" + "="*60)


def demo_index_types():
    """Demo 2: Compare different index types (x=1,2,3)."""
    print("\n" + "="*60)
    print("DEMO 2: Index Type Comparison (x=1, x=2, x=3)")
    print("="*60)
    
    documents = [
        {'id': 'doc1', 'text': 'Python Python Python programming'},
        {'id': 'doc2', 'text': 'Python programming language'},
        {'id': 'doc3', 'text': 'Java programming language'},
        {'id': 'doc4', 'text': 'Ruby programming'},
    ]
    
    index_types = ['BOOLEAN', 'WORDCOUNT', 'TFIDF']
    query = 'python'
    
    for idx_type in index_types:
        print(f"\n--- Index Type: {idx_type} ---")
        
        config = create_mock_config(index_type=idx_type)
        config.preprocessing.remove_stopwords = False
        config.preprocessing.stemming = False
        
        index = SelfIndex(config)
        index.create_index_from_documents(f'demo_{idx_type.lower()}', documents)
        results = index.query_with_params(query, k=10)
        
        print(f"Query: '{query}'")
        for r in results[:3]:
            print(f"  {r['rank']}. {r['doc_id']} - Score: {r['score']:.4f}")


def demo_query_processors():
    """Demo 3: Compare query processors (q=T vs q=D)."""
    print("\n" + "="*60)
    print("DEMO 3: Query Processor Comparison (q=T vs q=D)")
    print("="*60)
    
    # Create larger dataset
    documents = []
    for i in range(100):
        text = f"Document {i} about "
        if i % 3 == 0:
            text += "machine learning"
        if i % 5 == 0:
            text += " deep learning"
        if i % 7 == 0:
            text += " natural language processing"
        documents.append({'id': f'doc{i}', 'text': text})
    
    processors = ['TERMatat', 'DOCatat']
    query = 'machine learning'
    
    for proc in processors:
        print(f"\n--- Processor: {proc} ---")
        
        config = create_mock_config(query_proc=proc)
        index = SelfIndex(config)
        index.create_index_from_documents(f'demo_{proc.lower()}', documents)
        
        start = time.time()
        results = index.query_with_params(query, k=10)
        duration = time.time() - start
        
        print(f"Query: '{query}'")
        print(f"Time: {duration*1000:.2f}ms")
        print(f"Results: {len(results)}")


def demo_skip_pointers():
    """Demo 4: Compare with and without skip pointers (i=0 vs i=1)."""
    print("\n" + "="*60)
    print("DEMO 4: Skip Pointers Comparison (i=0 vs i=1)")
    print("="*60)
    
    # Create dataset with many documents
    documents = []
    for i in range(500):
        text = f"Document {i} "
        if i % 2 == 0:
            text += "python programming"
        if i % 3 == 0:
            text += " java programming"
        documents.append({'id': f'doc{i}', 'text': text})
    
    optimizations = ['Null', 'Skipping']
    query = '"python" AND "programming"'
    
    for opt in optimizations:
        print(f"\n--- Optimization: {opt} ---")
        
        config = create_mock_config(optimization=opt)
        index = SelfIndex(config)
        
        create_start = time.time()
        index.create_index_from_documents(f'demo_skip_{opt.lower()}', documents)
        create_time = time.time() - create_start
        
        query_times = []
        for _ in range(10):
            start = time.time()
            results = index.query_with_params(query, k=10)
            query_times.append(time.time() - start)
        
        avg_query_time = sum(query_times) / len(query_times)
        
        print(f"Index creation time: {create_time:.3f}s")
        print(f"Average query time: {avg_query_time*1000:.2f}ms")
        print(f"Results: {len(results)}")


def comparison_plot_memory():
    """Generate Plot.C: Memory footprint across index types."""
    print("\n" + "="*60)
    print("PLOT.C: Memory Footprint Comparison")
    print("="*60)
    
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("matplotlib and pandas required for plots. Skipping...")
        return
    
    documents = []
    for i in range(1000):
        documents.append({
            'id': f'doc{i}',
            'text': f'Sample document {i} with various terms for indexing test'
        })
    
    index_types = ['BOOLEAN', 'WORDCOUNT', 'TFIDF']
    memory_data = []
    
    for idx_type in index_types:
        config = create_mock_config(index_type=idx_type)
        index = SelfIndex(config)
        stats = index.create_index_from_documents(f'memory_test_{idx_type.lower()}', documents)
        
        memory_data.append({
            'Index Type': idx_type,
            'Size (MB)': stats['index_size_bytes'] / (1024 * 1024)
        })
        
        print(f"{idx_type}: {stats['index_size_bytes'] / (1024 * 1024):.2f} MB")
    
    # Plot
    df = pd.DataFrame(memory_data)
    plt.figure(figsize=(10, 6))
    plt.bar(df['Index Type'], df['Size (MB)'])
    plt.xlabel('Index Type (x)')
    plt.ylabel('Memory Footprint (MB)')
    plt.title('Plot.C: Memory Footprint vs Index Type')
    plt.savefig('plot_c_memory.png')
    print("\nPlot saved as 'plot_c_memory.png'")
    plt.close()


def comparison_plot_latency():
    """Generate Plot.A: Latency comparison."""
    print("\n" + "="*60)
    print("PLOT.A: Latency Comparison")
    print("="*60)
    
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("matplotlib and pandas required for plots. Skipping...")
        return
    
    documents = []
    for i in range(500):
        documents.append({
            'id': f'doc{i}',
            'text': f'Sample document {i} about machine learning and data science'
        })
    
    queries = [
        'machine', 
        'machine learning',
        '"machine" AND "learning"',
        'machine OR data',
    ]
    
    config = create_mock_config()
    index = SelfIndex(config)
    index.create_index_from_documents('latency_test', documents)
    
    latency_data = []
    
    for query in queries:
        times = []
        for _ in range(20):  # Run each query 20 times
            start = time.time()
            results = index.query_with_params(query, k=10)
            times.append((time.time() - start) * 1000)  # Convert to ms
        
        times.sort()
        p50 = times[len(times) // 2]
        p95 = times[int(len(times) * 0.95)]
        p99 = times[int(len(times) * 0.99)]
        
        latency_data.append({
            'Query': query[:20] + '...' if len(query) > 20 else query,
            'P50': p50,
            'P95': p95,
            'P99': p99
        })
        
        print(f"\nQuery: {query}")
        print(f"  P50: {p50:.2f}ms, P95: {p95:.2f}ms, P99: {p99:.2f}ms")
    
    # Plot
    df = pd.DataFrame(latency_data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(df))
    width = 0.25
    
    ax.bar([i - width for i in x], df['P50'], width, label='P50')
    ax.bar(x, df['P95'], width, label='P95')
    ax.bar([i + width for i in x], df['P99'], width, label='P99')
    
    ax.set_xlabel('Query')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Plot.A: Query Latency (P50, P95, P99)')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Query'], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('plot_a_latency.png')
    print("\nPlot saved as 'plot_a_latency.png'")
    plt.close()


def comparison_benchmark():
    """Comprehensive benchmark of SelfIndex."""
    print("\n" + "="*60)
    print("COMPREHENSIVE BENCHMARK")
    print("="*60)
    
    try:
        import pandas as pd
    except ImportError:
        print("pandas required for benchmark. Skipping...")
        return
    
    # Test with different document sizes
    doc_counts = [100, 500, 1000]
    
    benchmark_results = []
    
    for doc_count in doc_counts:
        print(f"\n--- Testing with {doc_count} documents ---")
        
        documents = []
        for i in range(doc_count):
            documents.append({
                'id': f'doc{i}',
                'text': f'Sample document {i} about machine learning, data science, and artificial intelligence'
            })
        
        config = create_mock_config(optimization='Skipping')
        index = SelfIndex(config)
        
        # Measure indexing time
        start = time.time()
        stats = index.create_index_from_documents(f'benchmark_{doc_count}', documents)
        index_time = time.time() - start
        
        # Measure query time
        query_times = []
        for _ in range(10):
            start = time.time()
            results = index.query_with_params('machine learning', k=10)
            query_times.append(time.time() - start)
        
        avg_query_time = sum(query_times) / len(query_times)
        
        benchmark_results.append({
            'Documents': doc_count,
            'Index Time (s)': index_time,
            'Avg Query Time (ms)': avg_query_time * 1000,
            'Index Size (MB)': stats['index_size_bytes'] / (1024 * 1024),
            'Docs/sec': doc_count / index_time
        })
        
        print(f"  Index time: {index_time:.2f}s")
        print(f"  Query time: {avg_query_time*1000:.2f}ms")
        print(f"  Index size: {stats['index_size_bytes'] / (1024 * 1024):.2f} MB")
        print(f"  Throughput: {doc_count / index_time:.2f} docs/sec")
    
    # Summary table
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    df = pd.DataFrame(benchmark_results)
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv('benchmark_results.csv', index=False)
    print("\nResults saved to 'benchmark_results.csv'")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print(" "*15 + "PHASE 3: SELFINDEX DEMO")
    print("="*70)
    
    try:
        # Basic demos
        demo_basic_indexing()
        demo_index_types()
        demo_query_processors()
        demo_skip_pointers()
        
        # Comparison plots
        comparison_plot_memory()
        comparison_plot_latency()
        
        # Comprehensive benchmark
        comparison_benchmark()
        
        print("\n" + "="*70)
        print(" "*20 + "ALL DEMOS COMPLETED!")
        print("="*70)
        
    except Exception as e:
        print(f"\nError running demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()