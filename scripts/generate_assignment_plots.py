#!/usr/bin/env python
"""
Generate all required plots for the assignment - UPDATED VERSION.
Includes ES indices, all latency metrics, and line plots.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_latest_metrics(results_dir):
    """Load the most recent consolidated metrics files."""
    
    # Find latest indexing metrics
    indexing_dir = Path(results_dir) / 'indexing'
    indexing_files = sorted(indexing_dir.glob('all_indices_*.json'))
    
    if not indexing_files:
        print("❌ No indexing metrics found")
        return None, None
    
    with open(indexing_files[-1], 'r') as f:
        indexing_metrics = json.load(f)
    
    print(f"Loaded indexing metrics: {indexing_files[-1]}")
    
    # Find latest query metrics
    queries_dir = Path(results_dir) / 'queries'
    query_files = sorted(queries_dir.glob('all_benchmarks_*.json'))
    
    if not query_files:
        print("❌ No query metrics found")
        return indexing_metrics, None
    
    with open(query_files[-1], 'r') as f:
        query_metrics = json.load(f)
    
    print(f"Loaded query metrics: {query_files[-1]}")
    
    return indexing_metrics, query_metrics


def filter_metrics(metrics, dataset, size, variants):
    """Filter metrics for specific dataset, size, and variants."""
    filtered = []
    for m in metrics:
        if m['dataset'] == dataset and m['doc_count'] == size and m['variant'] in variants:
            filtered.append(m)
    return filtered


def plot_c_memory_footprint(indexing_metrics, output_dir):
    """Plot.C: Memory footprint for x=1,2,3 + ES."""
    
    print("\nGenerating Plot.C - Memory Footprint (x=1,2,3 + ES)...")
    
    variants = ['boolean_custom', 'wordcount_custom', 'tfidf_custom', 'es_tfidf']
    labels = ['Boolean\n(x=1)', 'WordCount\n(x=2)', 'TF-IDF\n(x=3)', 'ES\nTF-IDF']
    colors = ['coral', 'steelblue', 'seagreen', 'gold']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Plot.C: Memory Footprint vs Index Type', fontsize=16, fontweight='bold')
    
    datasets = ['news', 'wiki']
    sizes = [1000, 5000, 10000]
    
    for row, dataset in enumerate(datasets):
        for col, size in enumerate(sizes):
            ax = axes[row, col]
            
            # Filter data
            data = filter_metrics(indexing_metrics, dataset, size, variants)
            
            if data:
                index_sizes = [next((d['index_size_mb'] for d in data if d['variant'] == v), 0) 
                              for v in variants]
                
                bars = ax.bar(labels, index_sizes, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
                
                # Add values on bars
                for bar, size_mb in zip(bars, index_sizes):
                    if size_mb > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                               f'{size_mb:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
                
                ax.set_ylabel('Index Size (MB)', fontweight='bold', fontsize=11)
                ax.set_title(f'{dataset.upper()} - {size:,} docs', fontweight='bold', fontsize=12)
                ax.grid(axis='y', alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'plot_c_memory_footprint_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def plot_a_datastore_comparison(query_metrics, output_dir):
    """Plot.A: Latency comparison for datastores (y=1,2) + ES with p95, p99."""
    
    print("\nGenerating Plot.A - Datastore Comparison (y=1,2 + ES)...")
    
    variants = ['tfidf_custom', 'tfidf_rocksdb', 'es_tfidf']
    labels = ['Custom/Pickle\n(y=1)', 'RocksDB\n(y=2)', 'Elasticsearch']
    colors = ['steelblue', 'coral', 'gold']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Plot.A: Query Latency vs Datastore', fontsize=16, fontweight='bold')
    
    datasets = ['news', 'wiki']
    sizes = [1000, 5000, 10000]
    
    for row, dataset in enumerate(datasets):
        for col, size in enumerate(sizes):
            ax = axes[row, col]
            
            # Filter data
            data = filter_metrics(query_metrics, dataset, size, variants)
            
            if data:
                mean_latencies = [next((d['latency_ms']['mean'] for d in data if d['variant'] == v), 0) 
                                 for v in variants]
                p95_latencies = [next((d['latency_ms']['p95'] for d in data if d['variant'] == v), 0) 
                                for v in variants]
                p99_latencies = [next((d['latency_ms']['p99'] for d in data if d['variant'] == v), 0) 
                                for v in variants]
                
                x = np.arange(len(labels))
                width = 0.25
                
                bars1 = ax.bar(x - width, mean_latencies, width, label='Mean', 
                              color=colors, alpha=0.6, edgecolor='black')
                bars2 = ax.bar(x, p95_latencies, width, label='P95', 
                              color=colors, alpha=0.8, edgecolor='black')
                bars3 = ax.bar(x + width, p99_latencies, width, label='P99', 
                              color=colors, alpha=1.0, edgecolor='black')
                
                # Add values on bars
                for bars in [bars1, bars2, bars3]:
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            ax.text(bar.get_x() + bar.get_width()/2, height,
                                   f'{height:.1f}', ha='center', va='bottom', 
                                   fontsize=7, fontweight='bold')
                
                ax.set_ylabel('Latency (ms)', fontweight='bold', fontsize=11)
                ax.set_title(f'{dataset.upper()} - {size:,} docs', fontweight='bold', fontsize=12)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, fontsize=9)
                ax.legend(fontsize=9)
                ax.grid(axis='y', alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'plot_a_datastore_comparison_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def plot_ab_compression(query_metrics, indexing_metrics, output_dir):
    """Plot.AB: Latency & Throughput with compression (z=0,1,2)."""
    
    print("\nGenerating Plot.AB - Compression Comparison (z=0,1,2)...")
    
    variants = ['tfidf_custom', 'tfidf_gap', 'tfidf_gzip']
    labels = ['No Compression\n(z=0)', 'Gap Encoding\n(z=1)', 'Gzip\n(z=2)']
    
    # Use news, 10000 docs as example
    dataset = 'news'
    size = 10000
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Plot.AB: Compression Comparison - {dataset.upper()} ({size:,} docs)', 
                 fontsize=16, fontweight='bold')
    
    # Filter data
    query_data = filter_metrics(query_metrics, dataset, size, variants)
    index_data = filter_metrics(indexing_metrics, dataset, size, variants)
    
    if query_data and index_data:
        # Latency
        latencies = [next((d['latency_ms']['mean'] for d in query_data if d['variant'] == v), 0) 
                    for v in variants]
        
        bars = axes[0].bar(labels, latencies, color=['steelblue', 'coral', 'seagreen'], 
                          alpha=0.8, edgecolor='black')
        axes[0].set_ylabel('Mean Latency (ms)', fontweight='bold')
        axes[0].set_title('Query Latency', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        for bar, lat in zip(bars, latencies):
            if lat > 0:
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{lat:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Throughput
        throughputs = [next((d['throughput_qps'] for d in query_data if d['variant'] == v), 0) 
                      for v in variants]
        
        bars = axes[1].bar(labels, throughputs, color=['steelblue', 'coral', 'seagreen'], 
                          alpha=0.8, edgecolor='black')
        axes[1].set_ylabel('Throughput (qps)', fontweight='bold')
        axes[1].set_title('Query Throughput', fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        for bar, thr in zip(bars, throughputs):
            if thr > 0:
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{thr:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Index Size
        index_sizes = [next((d['index_size_mb'] for d in index_data if d['variant'] == v), 0) 
                      for v in variants]
        if len(index_sizes) >= 2:
            index_sizes[0], index_sizes[1] = index_sizes[1], index_sizes[0]
        
        bars = axes[2].bar(labels, index_sizes, color=['steelblue', 'coral', 'seagreen'], 
                          alpha=0.8, edgecolor='black')
        axes[2].set_ylabel('Index Size (MB)', fontweight='bold')
        axes[2].set_title('Index Size', fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)
        
        for bar, size_mb in zip(bars, index_sizes):
            if size_mb > 0:
                axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{size_mb:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'plot_ab_compression_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def plot_a_skip_pointers(query_metrics, output_dir):
    """Plot.A: Latency with/without skip pointers (i=0,1) with p95, p99."""
    
    print("\nGenerating Plot.A - Skip Pointers (i=0,1)...")
    
    variants = ['tfidf_custom', 'tfidf_skip']
    labels = ['Without Skip\n(i=0)', 'With Skip\n(i=1)']
    
    # Use news, 10000 docs
    dataset = 'news'
    size = 10000
    
    data = filter_metrics(query_metrics, dataset, size, variants)
    
    if data:
        mean_latencies = [next((d['latency_ms']['mean'] for d in data if d['variant'] == v), 0) 
                         for v in variants]
        p95_latencies = [next((d['latency_ms']['p95'] for d in data if d['variant'] == v), 0) 
                        for v in variants]
        p99_latencies = [next((d['latency_ms']['p99'] for d in data if d['variant'] == v), 0) 
                        for v in variants]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(labels))
        width = 0.25
        
        bars1 = ax.bar(x - width, mean_latencies, width, label='Mean', 
                      color='steelblue', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x, p95_latencies, width, label='P95', 
                      color='coral', alpha=0.8, edgecolor='black')
        bars3 = ax.bar(x + width, p99_latencies, width, label='P99', 
                      color='seagreen', alpha=0.8, edgecolor='black')
        
        # Add values on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, height,
                           f'{height:.2f}', ha='center', va='bottom', 
                           fontsize=9, fontweight='bold')
        
        ax.set_ylabel('Latency (ms)', fontweight='bold')
        ax.set_title(f'Plot.A: Skip Pointers Comparison - {dataset.upper()} ({size:,} docs)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'plot_a_skip_pointers_{timestamp}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_file}")


def plot_ac_query_processors(query_metrics, output_dir):
    """Plot.AC: Query processor comparison (q=T,D) with values."""
    
    print("\nGenerating Plot.AC - Query Processors (q=T,D)...")
    
    variants = ['tfidf_custom', 'tfidf_earlystop']
    labels = ['Term-at-a-Time\n(q=T)', 'Optimized\n(q=D-like)']
    
    dataset = 'news'
    size = 10000
    
    data = filter_metrics(query_metrics, dataset, size, variants)
    
    if data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Plot.AC: Query Processor Comparison - {dataset.upper()} ({size:,} docs)', 
                     fontsize=14, fontweight='bold')
        
        # Latency
        latencies = [next((d['latency_ms']['mean'] for d in data if d['variant'] == v), 0) 
                    for v in variants]
        
        bars = ax1.bar(labels, latencies, color=['steelblue', 'coral'], alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Mean Latency (ms)', fontweight='bold')
        ax1.set_title('Query Latency', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, lat in zip(bars, latencies):
            if lat > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{lat:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Memory
        memory = [next((d['memory_mb']['peak'] for d in data if d['variant'] == v), 0) 
                 for v in variants]
        
        bars = ax2.bar(labels, memory, color=['steelblue', 'coral'], alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Peak Memory (MB)', fontweight='bold')
        ax2.set_title('Memory Usage', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, mem in zip(bars, memory):
            if mem > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{mem:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'plot_ac_query_processors_{timestamp}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_file}")


def plot_latency_lines(query_metrics, output_dir, metric='mean'):
    """Line plots for latency across document sizes (SelfIndex, ES, RocksDB)."""
    
    print(f"\nGenerating Line Plot - {metric.upper()} Latency...")
    
    variants = ['tfidf_custom', 'es_tfidf', 'tfidf_rocksdb']
    labels = ['SelfIndex (Custom)', 'Elasticsearch', 'RocksDB']
    markers = ['o', 's', '^']
    colors = ['steelblue', 'gold', 'coral']
    
    sizes = [1000, 5000, 10000]
    datasets = ['news', 'wiki']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{metric.upper()} Latency vs Document Size', fontsize=16, fontweight='bold')
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        
        for variant, label, marker, color in zip(variants, labels, markers, colors):
            latencies = []
            for size in sizes:
                data = filter_metrics(query_metrics, dataset, size, [variant])
                if data:
                    lat = data[0]['latency_ms'][metric]
                    latencies.append(lat)
                else:
                    latencies.append(None)
            
            # Plot line
            valid_sizes = [s for s, lat in zip(sizes, latencies) if lat is not None]
            valid_latencies = [lat for lat in latencies if lat is not None]
            
            ax.plot(valid_sizes, valid_latencies, marker=marker, linewidth=2, 
                   markersize=8, label=label, color=color)
            
            # Add values next to points
            for size, lat in zip(valid_sizes, valid_latencies):
                ax.text(size, lat, f'  {lat:.2f}', fontsize=8, fontweight='bold',
                       verticalalignment='center')
        
        ax.set_xlabel('Document Count', fontweight='bold', fontsize=11)
        ax.set_ylabel(f'{metric.upper()} Latency (ms)', fontweight='bold', fontsize=11)
        ax.set_title(f'{dataset.upper()} Dataset', fontweight='bold', fontsize=12)
        ax.set_xscale('log')
        ax.set_xticks(sizes)
        ax.set_xticklabels([f'{s:,}' for s in sizes])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'plot_latency_{metric}_lines_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def plot_scalability(query_metrics, indexing_metrics, output_dir):
    """Scalability analysis across document sizes."""
    
    print("\nGenerating Scalability Plots...")
    
    variant = 'tfidf_custom'
    sizes = [1000, 5000, 10000]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Scalability Analysis: TF-IDF Index', fontsize=16, fontweight='bold')
    
    for dataset in ['news', 'wiki']:
        # Index size vs doc count
        index_sizes = []
        for size in sizes:
            data = filter_metrics(indexing_metrics, dataset, size, [variant])
            if data:
                index_sizes.append(data[0]['index_size_mb'])
            else:
                index_sizes.append(0)
        
        line = axes[0, 0].plot(sizes, index_sizes, 'o-', label=dataset.upper(), 
                               linewidth=2, markersize=8)
        
        # Add values
        for size, val in zip(sizes, index_sizes):
            if val > 0:
                axes[0, 0].text(size, val, f' {val:.1f}', fontsize=8, fontweight='bold',
                               color=line[0].get_color())
        
        # Latency vs doc count
        latencies = []
        for size in sizes:
            data = filter_metrics(query_metrics, dataset, size, [variant])
            if data:
                latencies.append(data[0]['latency_ms']['mean'])
            else:
                latencies.append(0)
        
        line = axes[0, 1].plot(sizes, latencies, 'o-', label=dataset.upper(), 
                               linewidth=2, markersize=8)
        
        for size, val in zip(sizes, latencies):
            if val > 0:
                axes[0, 1].text(size, val, f' {val:.2f}', fontsize=8, fontweight='bold',
                               color=line[0].get_color())
        
        # Throughput vs doc count
        throughputs = []
        for size in sizes:
            data = filter_metrics(query_metrics, dataset, size, [variant])
            if data:
                throughputs.append(data[0]['throughput_qps'])
            else:
                throughputs.append(0)
        
        line = axes[1, 0].plot(sizes, throughputs, 'o-', label=dataset.upper(), 
                               linewidth=2, markersize=8)
        
        for size, val in zip(sizes, throughputs):
            if val > 0:
                axes[1, 0].text(size, val, f' {val:.1f}', fontsize=8, fontweight='bold',
                               color=line[0].get_color())
        
        # Indexing time vs doc count
        index_times = []
        for size in sizes:
            data = filter_metrics(indexing_metrics, dataset, size, [variant])
            if data:
                index_times.append(data[0]['indexing_time_seconds'])
            else:
                index_times.append(0)
        
        line = axes[1, 1].plot(sizes, index_times, 'o-', label=dataset.upper(), 
                               linewidth=2, markersize=8)
        
        for size, val in zip(sizes, index_times):
            if val > 0:
                axes[1, 1].text(size, val, f' {val:.1f}', fontsize=8, fontweight='bold',
                               color=line[0].get_color())
    
    # Configure subplots
    axes[0, 0].set_xlabel('Document Count', fontweight='bold')
    axes[0, 0].set_ylabel('Index Size (MB)', fontweight='bold')
    axes[0, 0].set_title('Index Size Scalability', fontweight='bold')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_xticks(sizes)
    axes[0, 0].set_xticklabels([f'{s:,}' for s in sizes])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Document Count', fontweight='bold')
    axes[0, 1].set_ylabel('Mean Latency (ms)', fontweight='bold')
    axes[0, 1].set_title('Query Latency Scalability', fontweight='bold')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xticks(sizes)
    axes[0, 1].set_xticklabels([f'{s:,}' for s in sizes])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Document Count', fontweight='bold')
    axes[1, 0].set_ylabel('Throughput (qps)', fontweight='bold')
    axes[1, 0].set_title('Query Throughput Scalability', fontweight='bold')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_xticks(sizes)
    axes[1, 0].set_xticklabels([f'{s:,}' for s in sizes])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Document Count', fontweight='bold')
    axes[1, 1].set_ylabel('Indexing Time (seconds)', fontweight='bold')
    axes[1, 1].set_title('Indexing Time Scalability', fontweight='bold')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_xticks(sizes)
    axes[1, 1].set_xticklabels([f'{s:,}' for s in sizes])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'plot_scalability_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")

def plot_avg_latency_by_metric(query_metrics, output_dir, metric='mean'):
    """Line plot showing average latency for each index type across document sizes."""
    
    print(f"\nGenerating Line Plot - Average {metric.upper()} Latency Comparison...")
    
    variants = ['tfidf_custom', 'es_tfidf', 'tfidf_rocksdb']
    labels = ['SelfIndex (Custom)', 'Elasticsearch', 'RocksDB']
    markers = ['o', 's', '^']
    colors = ['steelblue', 'gold', 'coral']
    
    sizes = [1000, 5000, 10000]
    datasets = ['news', 'wiki']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(f'Average {metric.upper()} Latency vs Document Size', 
                 fontsize=16, fontweight='bold')
    
    for variant, label, marker, color in zip(variants, labels, markers, colors):
        avg_latencies = []
        
        for size in sizes:
            # Get latencies from both datasets
            latencies_for_size = []
            
            for dataset in datasets:
                data = filter_metrics(query_metrics, dataset, size, [variant])
                if data:
                    latencies_for_size.append(data[0]['latency_ms'][metric])
            
            # Calculate average across datasets
            if latencies_for_size:
                avg_lat = sum(latencies_for_size) / len(latencies_for_size)
                avg_latencies.append(avg_lat)
            else:
                avg_latencies.append(None)
        
        # Plot line
        valid_sizes = [s for s, lat in zip(sizes, avg_latencies) if lat is not None]
        valid_latencies = [lat for lat in avg_latencies if lat is not None]
        
        ax.plot(valid_sizes, valid_latencies, marker=marker, linewidth=2.5, 
               markersize=10, label=label, color=color, alpha=0.9)
        
        # Add values next to points
        for size, lat in zip(valid_sizes, valid_latencies):
            ax.text(size, lat, f'  {lat:.2f}', fontsize=9, fontweight='bold',
                   verticalalignment='center', color=color)
    
    ax.set_xlabel('Document Count', fontweight='bold', fontsize=12)
    ax.set_ylabel(f'Average {metric.upper()} Latency (ms)', fontweight='bold', fontsize=12)
    ax.set_xscale('log')
    ax.set_xticks(sizes)
    ax.set_xticklabels([f'{s:,}' for s in sizes])
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'plot_avg_{metric}_latency_comparison_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")

def create_summary_tables(indexing_metrics, query_metrics, output_dir):
    """Create comprehensive summary tables."""
    
    print("\nGenerating summary tables...")
    
    # Combine metrics
    combined = []
    for idx_m in indexing_metrics:
        key = f"{idx_m['dataset']}_{idx_m['doc_count']}_{idx_m['variant']}"
        
        # Find matching query metrics
        query_m = next((q for q in query_metrics 
                       if q['dataset'] == idx_m['dataset'] 
                       and q['doc_count'] == idx_m['doc_count']
                       and q['variant'] == idx_m['variant']), None)
        
        row = {
            'Dataset': idx_m['dataset'],
            'Docs': idx_m['doc_count'],
            'Variant': idx_m['variant'],
            'Index Size (MB)': idx_m['index_size_mb'],
            'Indexing Time (s)': idx_m['indexing_time_seconds'],
            'Memory Used (MB)': idx_m.get('memory_used_mb', 'N/A'),
        }
        
        if query_m:
            row.update({
                'Mean Latency (ms)': query_m['latency_ms']['mean'],
                'P95 Latency (ms)': query_m['latency_ms']['p95'],
                'P99 Latency (ms)': query_m['latency_ms']['p99'],
                'Throughput (qps)': query_m['throughput_qps'],
            })
        
        combined.append(row)
    
    df = pd.DataFrame(combined)
    
    # Save as CSV
    csv_file = output_dir / f"summary_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_file, index=False)
    print(f"✅ Saved CSV: {csv_file}")
    
    # Save as Markdown
    md_file = output_dir / f"summary_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(md_file, 'w') as f:
        f.write(df.to_markdown(index=False))
    print(f"✅ Saved Markdown: {md_file}")


def main():
    print("="*70)
    print("GENERATING ASSIGNMENT PLOTS - UPDATED VERSION")
    print("="*70)
    
    results_dir = Path('results')
    output_dir = Path('plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    indexing_metrics, query_metrics = load_latest_metrics(results_dir)
    
    if not indexing_metrics:
        print("\n❌ No metrics found. Run create_all_indices.py first.")
        return
    
    if not query_metrics:
        print("\n⚠️  No query metrics found. Some plots will be skipped.")
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    plot_c_memory_footprint(indexing_metrics, output_dir)
    
    if query_metrics:
        plot_a_datastore_comparison(query_metrics, output_dir)
        plot_ab_compression(query_metrics, indexing_metrics, output_dir)
        plot_a_skip_pointers(query_metrics, output_dir)
        plot_ac_query_processors(query_metrics, output_dir)
        
        plot_latency_lines(query_metrics, output_dir, metric='mean')
        plot_latency_lines(query_metrics, output_dir, metric='p95')
        plot_latency_lines(query_metrics, output_dir, metric='p99')
        
        plot_scalability(query_metrics, indexing_metrics, output_dir)
        create_summary_tables(indexing_metrics, query_metrics, output_dir)
    
    print("\n" + "="*70)
    print("✅ ALL PLOTS GENERATED!")
    print(f"Output directory: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()