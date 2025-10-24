#!/usr/bin/env python
"""
Generate all required plots for assignment.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load benchmark results
with open("results/benchmark_results.json", 'r') as f:
    results = json.load(f)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def plot_c_memory_footprint():
    """Plot.C: Memory footprint vs index type (x=1,2,3)"""
    # Filter for x=1,2,3
    data = [r for r in results if r['index_name'] in ['idx_boolean', 'idx_wordcount', 'idx_tfidf']]
    
    names = [r['description'] for r in data]
    sizes = [r['index_size_mb'] for r in data]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, sizes, color=['coral', 'steelblue', 'seagreen'], alpha=0.8)
    
    plt.xlabel('Index Type', fontweight='bold')
    plt.ylabel('Index Size (MB)', fontweight='bold')
    plt.title('Plot.C: Memory Footprint vs Index Type', fontweight='bold')
    
    # Add values on bars
    for bar, size in zip(bars, sizes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{size:.2f} MB', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/plot_c_memory_footprint.png', dpi=300)
    print("✅ Generated: plots/plot_c_memory_footprint.png")

def plot_ab_compression():
    """Plot.AB: Latency & Throughput with compression (z=1,2)"""
    # Filter for compression variants
    data = [r for r in results if r['index_name'] in ['idx_tfidf', 'idx_compressed', 'idx_gzip']]
    
    names = ['No Compression\n(z=0)', 'Gap Encoding\n(z=1)', 'Gzip\n(z=2)']
    latencies = [r['latency_mean'] for r in data]
    throughputs = [r['throughput_qps'] for r in data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Latency
    ax1.bar(names, latencies, color=['steelblue', 'coral', 'seagreen'], alpha=0.8)
    ax1.set_ylabel('Mean Latency (ms)', fontweight='bold')
    ax1.set_title('Latency vs Compression', fontweight='bold')
    
    # Throughput
    ax2.bar(names, throughputs, color=['steelblue', 'coral', 'seagreen'], alpha=0.8)
    ax2.set_ylabel('Throughput (queries/sec)', fontweight='bold')
    ax2.set_title('Throughput vs Compression', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/plot_ab_compression.png', dpi=300)
    print("✅ Generated: plots/plot_ab_compression.png")

def plot_a_skip_pointers():
    """Plot.A: Latency with/without skip pointers (i=0,1)"""
    # Filter for skip pointer comparison
    data = [r for r in results if r['index_name'] in ['idx_tfidf', 'idx_tfidf_skip']]
    
    names = ['Without Skip Pointers\n(i=0)', 'With Skip Pointers\n(i=1)']
    latencies = [r['latency_mean'] for r in data]
    p95 = [r['latency_p95'] for r in data]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, latencies, width, label='Mean', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, p95, width, label='P95', color='coral', alpha=0.8)
    
    ax.set_ylabel('Latency (ms)', fontweight='bold')
    ax.set_title('Plot.A: Latency with/without Skip Pointers', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('plots/plot_a_skip_pointers.png', dpi=300)
    print("✅ Generated: plots/plot_a_skip_pointers.png")

def plot_ac_query_processors():
    """Plot.AC: Latency & Memory for query processors (q=T,D)"""
    # Filter for query processor comparison
    data = [r for r in results if r['index_name'] in ['idx_tfidf', 'idx_docatat']]
    
    names = ['Term-at-a-Time\n(q=T)', 'Document-at-a-Time\n(q=D)']
    latencies = [r['latency_mean'] for r in data]
    memory = [r['memory_mb'] for r in data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Latency
    ax1.bar(names, latencies, color=['steelblue', 'coral'], alpha=0.8)
    ax1.set_ylabel('Mean Latency (ms)', fontweight='bold')
    ax1.set_title('Latency vs Query Processor', fontweight='bold')
    
    # Memory
    ax2.bar(names, memory, color=['steelblue', 'coral'], alpha=0.8)
    ax2.set_ylabel('Memory (MB)', fontweight='bold')
    ax2.set_title('Memory vs Query Processor', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/plot_ac_query_processors.png', dpi=300)
    print("✅ Generated: plots/plot_ac_query_processors.png")

def plot_optimizations():
    """Bonus: Compare all optimizations"""
    # Filter optimization variants
    data = [r for r in results if 'tfidf' in r['index_name'].lower()]
    
    names = [r['index_name'].replace('idx_', '') for r in data]
    latencies = [r['latency_mean'] for r in data]
    
    plt.figure(figsize=(12, 6))
    plt.bar(names, latencies, color='steelblue', alpha=0.8)
    plt.xlabel('Configuration', fontweight='bold')
    plt.ylabel('Mean Latency (ms)', fontweight='bold')
    plt.title('Latency Comparison: All Optimizations', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('plots/plot_all_optimizations.png', dpi=300)
    print("✅ Generated: plots/plot_all_optimizations.png")

def main():
    print("="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    Path("plots").mkdir(exist_ok=True)
    
    plot_c_memory_footprint()
    plot_ab_compression()
    plot_a_skip_pointers()
    plot_ac_query_processors()
    plot_optimizations()
    
    print("\n✅ All plots generated in plots/ directory")

if __name__ == "__main__":
    main()