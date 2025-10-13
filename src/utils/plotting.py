import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class Plotter:
    """Handles all plotting operations."""
    
    def __init__(self, config):
        """
        Initialize plotter.
        
        Args:
            config: Hydra configuration object
        """
        self.config = config
        self.output_dir = Path(config.paths.plots_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_word_frequencies(
        self,
        word_freq: Dict[str, int],
        title: str = "Word Frequency Distribution",
        filename: str = "word_frequencies.png",
        top_n: int = 50
    ):
        """
        Plot word frequency distribution.
        
        Args:
            word_freq: Dictionary mapping words to frequencies
            title: Plot title
            filename: Output filename
            top_n: Number of top words to plot
        """
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
        words, freqs = zip(*sorted_words)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(14, 8))
        bars = ax.bar(range(len(words)), freqs, color='steelblue', alpha=0.8)
        
        ax.set_xlabel('Words', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, freq) in enumerate(zip(bars, freqs)):
            if i < 10:  # Only label top 10 to avoid clutter
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{freq:,}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved word frequency plot to {output_path}")
    
    def plot_word_frequency_comparison(
        self,
        freq_before: Dict[str, int],
        freq_after: Dict[str, int],
        filename: str = "word_freq_comparison.png",
        top_n: int = 30
    ):
        """
        Compare word frequencies before and after preprocessing.
        
        Args:
            freq_before: Word frequencies before preprocessing
            freq_after: Word frequencies after preprocessing
            filename: Output filename
            top_n: Number of words to compare
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Before preprocessing
        sorted_before = sorted(freq_before.items(), key=lambda x: x[1], reverse=True)[:top_n]
        words_before, freqs_before = zip(*sorted_before)
        
        ax1.barh(range(len(words_before)), freqs_before, color='coral', alpha=0.8)
        ax1.set_yticks(range(len(words_before)))
        ax1.set_yticklabels(words_before)
        ax1.set_xlabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Before Preprocessing', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        
        # After preprocessing
        sorted_after = sorted(freq_after.items(), key=lambda x: x[1], reverse=True)[:top_n]
        words_after, freqs_after = zip(*sorted_after)
        
        ax2.barh(range(len(words_after)), freqs_after, color='steelblue', alpha=0.8)
        ax2.set_yticks(range(len(words_after)))
        ax2.set_yticklabels(words_after)
        ax2.set_xlabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('After Preprocessing', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved comparison plot to {output_path}")
    
    def plot_latency_distribution(
        self,
        latencies: List[float],
        title: str = "Query Latency Distribution",
        filename: str = "latency_distribution.png"
    ):
        """
        Plot latency distribution with percentiles.
        
        Args:
            latencies: List of latency values in milliseconds
            title: Plot title
            filename: Output filename
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        ax1.hist(latencies, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(np.percentile(latencies, 50), color='green', linestyle='--', 
                   linewidth=2, label=f'P50: {np.percentile(latencies, 50):.2f}ms')
        ax1.axvline(np.percentile(latencies, 95), color='orange', linestyle='--',
                   linewidth=2, label=f'P95: {np.percentile(latencies, 95):.2f}ms')
        ax1.axvline(np.percentile(latencies, 99), color='red', linestyle='--',
                   linewidth=2, label=f'P99: {np.percentile(latencies, 99):.2f}ms')
        ax1.set_xlabel('Latency (ms)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(latencies, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2),
                   whiskerprops=dict(linewidth=1.5),
                   capprops=dict(linewidth=1.5))
        ax2.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
        ax2.set_title('Latency Box Plot', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved latency distribution plot to {output_path}")
    
    def plot_throughput_comparison(
        self,
        experiments: Dict[str, float],
        title: str = "Throughput Comparison",
        filename: str = "throughput_comparison.png"
    ):
        """
        Compare throughput across experiments.
        
        Args:
            experiments: Dictionary mapping experiment names to throughput (qps)
            title: Plot title
            filename: Output filename
        """
        names = list(experiments.keys())
        throughputs = list(experiments.values())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(names, throughputs, color='steelblue', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Throughput (queries/sec)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticklabels(names, rotation=45, ha='right')
        
        # Add value labels
        for bar, throughput in zip(bars, throughputs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{throughput:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved throughput comparison to {output_path}")
    
    def plot_memory_footprint(
        self,
        experiments: Dict[str, float],
        title: str = "Memory Footprint Comparison",
        filename: str = "memory_comparison.png"
    ):
        """
        Compare memory footprint across experiments.
        
        Args:
            experiments: Dictionary mapping experiment names to memory usage (MB)
            title: Plot title
            filename: Output filename
        """
        names = list(experiments.keys())
        memory = list(experiments.values())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(names, memory, color='coral', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticklabels(names, rotation=45, ha='right')
        
        # Add value labels
        for bar, mem in zip(bars, memory):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{mem:.1f} MB', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved memory comparison to {output_path}")