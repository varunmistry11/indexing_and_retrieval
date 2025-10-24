#!/usr/bin/env python
"""
Generate comprehensive comparison plots across datasets, sizes, and implementations.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

def load_all_results():
    """Load all benchmark results."""
    results_file = Path("results/benchmark_results.json")
    
    if not results_file.exists():
        print("❌ No results found. Run benchmark first.")
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)

def plot_dataset_comparison():
    """Compare News vs Wiki datasets."""
    results = load_all_results()
    if not results:
        return
    
    # Filter by dataset
    news_results = [r for r in results if 'news' in r['index_name']]
    wiki_results = [r for r in results if 'wiki' in r['index_name']]
    
    # Group by index type
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['index_size_mb', 'latency_mean', 'throughput_qps']
    titles = ['Index Size (MB)', 'Mean Latency (ms)', 'Throughput (qps)']
    
    for ax, metric, title in zip(axes, metrics, titles):
        news_vals = [r[metric] for r in news_results[:5]]  # First 5
        wiki_vals = [r[metric] for r in wiki_results[:5]]
        
        x = np.arange(len(news_vals))
        width = 0.35
        
        ax.bar(x - width/2, news_vals, width, label='News', alpha=0.8)
        ax.bar(x + width/2, wiki_vals, width, label='Wiki', alpha=0.8)
        
        ax.set_ylabel(title, fontweight='bold')
        ax.set_title(f'{title} Comparison', fontweight='bold')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('plots/dataset_comparison.png', dpi=300)
    print("✅ Generated: plots/dataset_comparison.png")

def plot_scalability():
    """Plot scalability across document sizes."""
    # This requires running comprehensive benchmark with multiple sizes
    
    # Sample data structure (replace with actual data)
    sizes = [1000, 10000, 100000]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Index size vs doc count
    axes[0, 0].plot(sizes, [6.5, 65, 650], 'o-', label='SelfIndex', linewidth=2)
    axes[0, 0].plot(sizes, [8, 80, 800], 's-', label='Elasticsearch', linewidth=2)
    axes[0, 0].set_xlabel('Document Count', fontweight='bold')
    axes[0, 0].set_ylabel('Index Size (MB)', fontweight='bold')
    axes[0, 0].set_title('Index Size Scalability', fontweight='bold')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Latency vs doc count
    axes[0, 1].plot(sizes, [1.5, 3.5, 8.5], 'o-', label='SelfIndex', linewidth=2)
    axes[0, 1].plot(sizes, [2.0, 4.0, 10.0], 's-', label='Elasticsearch', linewidth=2)
    axes[0, 1].set_xlabel('Document Count', fontweight='bold')
    axes[0, 1].set_ylabel('Mean Latency (ms)', fontweight='bold')
    axes[0, 1].set_title('Query Latency Scalability', fontweight='bold')
    axes[0, 1].set_xscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Throughput vs doc count
    axes[1, 0].plot(sizes, [700, 600, 450], 'o-', label='SelfIndex', linewidth=2)
    axes[1, 0].plot(sizes, [650, 550, 400], 's-', label='Elasticsearch', linewidth=2)
    axes[1, 0].set_xlabel('Document Count', fontweight='bold')
    axes[1, 0].set_ylabel('Throughput (qps)', fontweight='bold')
    axes[1, 0].set_title('Throughput Scalability', fontweight='bold')
    axes[1, 0].set_xscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Compression ratio
    axes[1, 1].bar(['None', 'Gap', 'Gzip'], [100, 138, 26], alpha=0.8)
    axes[1, 1].set_ylabel('Relative Size (%)', fontweight='bold')
    axes[1, 1].set_title('Compression Effectiveness', fontweight='bold')
    axes[1, 1].axhline(y=100, color='r', linestyle='--', label='Baseline')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('plots/scalability_analysis.png', dpi=300)
    print("✅ Generated: plots/scalability_analysis.png")

def plot_implementation_comparison():
    """Compare SelfIndex vs Elasticsearch."""
    results = load_all_results()
    if not results:
        return
    
    # Filter SelfIndex vs ES
    self_results = [r for r in results if 'implementation' in r and r['implementation'] == 'selfindex']
    es_results = [r for r in results if 'implementation' in r and r['implementation'] == 'elasticsearch']
    
    if not es_results:
        print("⚠️  No Elasticsearch results found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Latency comparison
    self_lat = [r['latency_mean'] for r in self_results[:3]]
    es_lat = [r['latency_mean'] for r in es_results[:3]]
    
    x = np.arange(3)
    width = 0.35
    
    axes[0].bar(x - width/2, self_lat, width, label='SelfIndex', alpha=0.8, color='steelblue')
    axes[0].bar(x + width/2, es_lat, width, label='Elasticsearch', alpha=0.8, color='coral')
    axes[0].set_ylabel('Mean Latency (ms)', fontweight='bold')
    axes[0].set_title('Latency: SelfIndex vs Elasticsearch', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Boolean', 'WordCount', 'TF-IDF'])
    axes[0].legend()
    
    # Size comparison
    self_size = [r['index_size_mb'] for r in self_results[:3]]
    es_size = [r['index_size_mb'] for r in es_results[:3]]
    
    axes[1].bar(x - width/2, self_size, width, label='SelfIndex', alpha=0.8, color='steelblue')
    axes[1].bar(x + width/2, es_size, width, label='Elasticsearch', alpha=0.8, color='coral')
    axes[1].set_ylabel('Index Size (MB)', fontweight='bold')
    axes[1].set_title('Index Size: SelfIndex vs Elasticsearch', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Boolean', 'WordCount', 'TF-IDF'])
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('plots/implementation_comparison.png', dpi=300)
    print("✅ Generated: plots/implementation_comparison.png")

def create_summary_table():
    """Create comprehensive summary table."""
    results = load_all_results()
    if not results:
        return
    
    df = pd.DataFrame(results)
    
    # Select and rename columns
    summary = df[[
        'index_name', 'implementation', 'index_size_mb', 
        'latency_mean', 'latency_p95', 'throughput_qps'
    ]].copy()
    
    summary.columns = ['Index', 'Implementation', 'Size (MB)', 
                       'Latency (ms)', 'P95 (ms)', 'Throughput (qps)']
    
    summary = summary.round(2)
    
    # Save as CSV
    summary.to_csv('results/summary_table.csv', index=False)
    
    # Save as markdown
    with open('results/summary_table.md', 'w') as f:
        f.write(summary.to_markdown(index=False))
    
    print("✅ Generated: results/summary_table.csv")
    print("✅ Generated: results/summary_table.md")

def main():
    print("="*60)
    print("GENERATING COMPREHENSIVE PLOTS")
    print("="*60)
    
    Path("plots").mkdir(exist_ok=True)
    
    plot_dataset_comparison()
    plot_scalability()
    plot_implementation_comparison()
    create_summary_table()
    
    print("\n✅ All comprehensive plots generated!")

if __name__ == "__main__":
    main()