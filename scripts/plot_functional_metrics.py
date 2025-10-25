#!/usr/bin/env python
"""
Generate plots for functional metrics (precision, recall, F1).
File: scripts/plot_functional_metrics.py
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


def load_latest_functional_metrics(results_dir):
    """Load the most recent functional metrics file."""
    
    functional_dir = Path(results_dir) / 'functional'
    
    if not functional_dir.exists():
        print("❌ No functional metrics directory found")
        return None
    
    metric_files = sorted(functional_dir.glob('all_functional_metrics_*.json'))
    
    if not metric_files:
        print("❌ No functional metrics found")
        return None
    
    with open(metric_files[-1], 'r') as f:
        metrics = json.load(f)
    
    print(f"Loaded functional metrics: {metric_files[-1]}")
    return metrics


def filter_metrics(metrics, dataset, size, variants):
    """Filter metrics for specific dataset, size, and variants."""
    filtered = []
    for m in metrics:
        if m['dataset'] == dataset and m['doc_count'] == size and m['variant'] in variants:
            filtered.append(m)
    return filtered


def plot_precision_at_k_comparison(metrics, output_dir):
    """Bar chart comparing Precision@K across variants."""
    
    print("\nGenerating Precision@K Comparison...")
    
    # Use news, 5000 docs as example
    dataset = 'news'
    size = 5000
    
    variants = ['boolean_custom', 'wordcount_custom', 'tfidf_custom', 
                'tfidf_rocksdb', 'tfidf_skip']
    labels = ['Boolean', 'WordCount', 'TF-IDF', 'RocksDB', 'Skip Ptrs']
    
    data = filter_metrics(metrics, dataset, size, variants)
    
    if not data:
        print("⚠️  No data for precision comparison")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Precision@K Comparison vs ES Baseline - {dataset.upper()} ({size:,} docs)', 
                 fontsize=16, fontweight='bold')
    
    for idx, k in enumerate([5, 10, 20]):
        ax = axes[idx]
        
        precisions = [next((d[f'precision@{k}'] for d in data if d['variant'] == v), 0) 
                     for v in variants]
        
        bars = ax.bar(labels, precisions, color='steelblue', alpha=0.8, edgecolor='black')
        
        # Add values on bars
        for bar, prec in zip(bars, precisions):
            if prec > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{prec:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax.set_ylabel(f'Precision@{k}', fontweight='bold', fontsize=11)
        ax.set_title(f'Precision@{k}', fontweight='bold', fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Target: 0.8')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'plot_precision_at_k_comparison_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def plot_f1_score_comparison(metrics, output_dir):
    """Bar chart comparing F1@10 across all variants."""
    
    print("\nGenerating F1@10 Comparison...")
    
    variants = ['boolean_custom', 'wordcount_custom', 'tfidf_custom', 
                'tfidf_rocksdb', 'tfidf_skip', 'tfidf_gap', 
                'tfidf_gzip', 'tfidf_threshold', 'tfidf_earlystop']
    
    labels = ['Boolean', 'WordCount', 'TF-IDF', 'RocksDB', 
              'Skip', 'Gap', 'Gzip', 'Threshold', 'EarlyStop']
    
    colors = ['coral', 'steelblue', 'seagreen', 'gold', 
              'purple', 'orange', 'pink', 'brown', 'cyan']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('F1@10 Score vs ES Baseline', fontsize=16, fontweight='bold')
    
    datasets = ['news', 'wiki']
    sizes = [1000, 5000, 10000]
    
    for row, dataset in enumerate(datasets):
        for col, size in enumerate(sizes):
            ax = axes[row, col]
            
            data = filter_metrics(metrics, dataset, size, variants)
            
            if data:
                f1_scores = [next((d['f1@10'] for d in data if d['variant'] == v), 0) 
                            for v in variants]
                
                bars = ax.bar(range(len(labels)), f1_scores, color=colors, 
                             alpha=0.8, edgecolor='black')
                
                # Add values on bars
                for bar, f1 in zip(bars, f1_scores):
                    if f1 > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                               f'{f1:.2f}', ha='center', va='bottom', 
                               fontweight='bold', fontsize=7)
                
                ax.set_ylabel('F1@10', fontweight='bold', fontsize=11)
                ax.set_title(f'{dataset.upper()} - {size:,} docs', fontweight='bold', fontsize=12)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
                ax.set_ylim(0, 1.0)
                ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
                ax.grid(axis='y', alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'plot_f1_score_comparison_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def plot_precision_recall_curves(metrics, output_dir):
    """Line plot showing Precision and Recall at different K values."""
    
    print("\nGenerating Precision-Recall Curves...")
    
    variants = ['tfidf_custom', 'tfidf_rocksdb', 'tfidf_skip']
    labels = ['TF-IDF Custom', 'RocksDB', 'Skip Pointers']
    colors = ['steelblue', 'coral', 'seagreen']
    markers = ['o', 's', '^']
    
    dataset = 'news'
    size = 5000
    
    data = filter_metrics(metrics, dataset, size, variants)
    
    if not data:
        print("⚠️  No data for precision-recall curves")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Precision and Recall at Different K - {dataset.upper()} ({size:,} docs)', 
                 fontsize=16, fontweight='bold')
    
    k_values = [5, 10, 20]
    
    # Precision curves
    for variant, label, color, marker in zip(variants, labels, colors, markers):
        var_data = next((d for d in data if d['variant'] == variant), None)
        
        if var_data:
            precisions = [var_data[f'precision@{k}'] for k in k_values]
            
            ax1.plot(k_values, precisions, marker=marker, linewidth=2, 
                    markersize=8, label=label, color=color)
            
            # Add values
            for k, prec in zip(k_values, precisions):
                ax1.text(k, prec, f' {prec:.3f}', fontsize=8, fontweight='bold',
                        verticalalignment='center', color=color)
    
    ax1.set_xlabel('K (Top-K results)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Precision@K', fontweight='bold', fontsize=11)
    ax1.set_title('Precision vs K', fontweight='bold', fontsize=12)
    ax1.set_xticks(k_values)
    ax1.set_ylim(0, 1.0)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Recall curves
    for variant, label, color, marker in zip(variants, labels, colors, markers):
        var_data = next((d for d in data if d['variant'] == variant), None)
        
        if var_data:
            recalls = [var_data[f'recall@{k}'] for k in k_values]
            
            ax2.plot(k_values, recalls, marker=marker, linewidth=2, 
                    markersize=8, label=label, color=color)
            
            # Add values
            for k, rec in zip(k_values, recalls):
                ax2.text(k, rec, f' {rec:.3f}', fontsize=8, fontweight='bold',
                        verticalalignment='center', color=color)
    
    ax2.set_xlabel('K (Top-K results)', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Recall@K', fontweight='bold', fontsize=11)
    ax2.set_title('Recall vs K', fontweight='bold', fontsize=12)
    ax2.set_xticks(k_values)
    ax2.set_ylim(0, 1.0)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'plot_precision_recall_curves_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def plot_f1_heatmap(metrics, output_dir):
    """Heatmap of F1@10 scores across datasets and variants."""
    
    print("\nGenerating F1@10 Heatmap...")
    
    variants = ['boolean_custom', 'wordcount_custom', 'tfidf_custom', 
                'tfidf_rocksdb', 'tfidf_skip', 'tfidf_gap']
    
    labels = ['Boolean', 'WordCount', 'TF-IDF', 'RocksDB', 'Skip', 'Gap']
    
    datasets = ['news', 'wiki']
    sizes = [1000, 5000, 10000]
    
    # Create matrix: rows = dataset_size, cols = variants
    row_labels = [f'{ds}_{sz}' for ds in datasets for sz in sizes]
    f1_matrix = []
    
    for dataset in datasets:
        for size in sizes:
            row = []
            data = filter_metrics(metrics, dataset, size, variants)
            
            for variant in variants:
                var_data = next((d for d in data if d['variant'] == variant), None)
                f1 = var_data['f1@10'] if var_data else 0
                row.append(f1)
            
            f1_matrix.append(row)
    
    f1_matrix = np.array(f1_matrix)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(f1_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(row_labels)
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add values
    for i in range(len(row_labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f'{f1_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('F1@10 Score Heatmap (vs ES Baseline)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Index Variant', fontweight='bold', fontsize=11)
    ax.set_ylabel('Dataset_Size', fontweight='bold', fontsize=11)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('F1@10 Score', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'plot_f1_heatmap_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")


def create_functional_summary_table(metrics, output_dir):
    """Create summary table for functional metrics."""
    
    print("\nGenerating functional metrics summary table...")
    
    # Create rows
    rows = []
    for m in metrics:
        row = {
            'Dataset': m['dataset'],
            'Docs': m['doc_count'],
            'Variant': m['variant'],
            'Precision@5': m['precision@5'],
            'Precision@10': m['precision@10'],
            'Precision@20': m['precision@20'],
            'Recall@10': m['recall@10'],
            'F1@10': m['f1@10'],
            'Success Rate': f"{m['successful_queries']}/{m['total_queries']}"
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save as CSV
    csv_file = output_dir / f"functional_metrics_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_file, index=False)
    print(f"✅ Saved CSV: {csv_file}")
    
    # Save as Markdown
    md_file = output_dir / f"functional_metrics_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(md_file, 'w') as f:
        f.write(df.to_markdown(index=False))
    print(f"✅ Saved Markdown: {md_file}")


def main():
    print("="*70)
    print("GENERATING FUNCTIONAL METRICS PLOTS")
    print("="*70)
    
    results_dir = Path('results')
    output_dir = Path('plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    metrics = load_latest_functional_metrics(results_dir)
    
    if not metrics:
        print("\n❌ No functional metrics found. Run evaluate_functional_metrics.py first.")
        return
    
    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    plot_precision_at_k_comparison(metrics, output_dir)
    plot_f1_score_comparison(metrics, output_dir)
    plot_precision_recall_curves(metrics, output_dir)
    plot_f1_heatmap(metrics, output_dir)
    create_functional_summary_table(metrics, output_dir)
    
    print("\n" + "="*70)
    print("✅ ALL FUNCTIONAL METRICS PLOTS GENERATED!")
    print(f"Output directory: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()