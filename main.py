#!/usr/bin/env python
"""
Main entry point for the Indexing and Retrieval Assignment.
Uses Fire for CLI and Hydra for configuration management.
"""

import os
import sys
import logging
from pathlib import Path
import fire
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from dotenv import load_dotenv

# Load .env variables and register resolver
load_dotenv()
OmegaConf.register_new_resolver("env", os.getenv)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.indices.elasticsearch_index import ElasticsearchIndex
from src.data.data_loader import DataLoader
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.utils.benchmark import Benchmarker
from src.utils.plotting import Plotter


class IndexingCLI:
    """CLI for Indexing and Retrieval Assignment."""
    
    def __init__(self, config_path: str = "conf", config_name: str = "config"):
        """
        Initialize CLI with configuration.
        
        Args:
            config_path: Path to config directory
            config_name: Name of main config file
        """
        self.config_path = config_path
        self.config_name = config_name
        self.config = None
        self.index_instance = None
        self.logger = None
    
    def _init_config(self, overrides=None):
        """Initialize Hydra configuration."""
        with hydra.initialize(version_base=None, config_path=self.config_path):
            if overrides:
                self.config = hydra.compose(config_name=self.config_name, overrides=overrides)
            else:
                self.config = hydra.compose(config_name=self.config_name)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level),
            format=self.config.logging.format
        )
        self.logger = logging.getLogger(__name__)
        
        # Create necessary directories
        for path_key in ['data_dir', 'index_storage', 'results_dir', 'plots_dir']:
            if path_key in self.config.paths:
                Path(self.config.paths[path_key]).mkdir(parents=True, exist_ok=True)
    
    def _get_index_instance(self):
        """Get or create index instance."""
        if self.index_instance is None:
            # Check implementation type
            implementation = None
            if hasattr(self.config.index, 'implementation'):
                implementation = self.config.index.implementation
            
            # Determine which index to use
            if implementation == 'self':
                # Use SelfIndex
                from src.indices.self_index import SelfIndex
                self.index_instance = SelfIndex(self.config)
                self.logger.info("Using SelfIndex implementation")
            elif self.config.datastore.name == 'elasticsearch':
                # Use Elasticsearch
                self.index_instance = ElasticsearchIndex(self.config)
                self.logger.info("Using Elasticsearch implementation")
            elif self.config.datastore.name == 'custom':
                # Custom datastore defaults to SelfIndex
                from src.indices.self_index import SelfIndex
                self.index_instance = SelfIndex(self.config)
                self.logger.info("Using SelfIndex with custom datastore")
            else:
                raise NotImplementedError(
                    f"Datastore {self.config.datastore.name} not yet implemented"
                )
        return self.index_instance
    
    def setup(self):
        """Setup the environment and verify connections."""
        self._init_config()
        self.logger.info("="*60)
        self.logger.info("INDEXING AND RETRIEVAL ASSIGNMENT - SETUP")
        self.logger.info("="*60)
        
        # Test Elasticsearch connection
        try:
            index = self._get_index_instance()
            indices = index.list_indices()
            self.logger.info(f"✓ Connected to Elasticsearch")
            self.logger.info(f"✓ Found {len(indices)} existing indices")
            
            # Verify data files
            for dataset_name in ['wiki', 'news']:
                with hydra.initialize(version_base=None, config_path=self.config_path):
                    cfg = hydra.compose(config_name=self.config_name, overrides=[f"dataset={dataset_name}"])
                
                data_path = Path(cfg.dataset.source_file)
                if data_path.exists():
                    self.logger.info(f"✓ Found {dataset_name} dataset at {data_path}")
                else:
                    self.logger.warning(f"✗ Dataset not found: {data_path}")
            
            self.logger.info("\nSetup completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            self.logger.error("Please ensure Elasticsearch is running: sudo systemctl start elasticsearch")
            return False
        
        return True
    
    def create_index(self, dataset: str = None, index_type: str = None, 
                    index_name: str = None, config_name: str = None):
        """
        Create an index from a dataset.
        
        Args:
            dataset: Dataset to use (wiki, news, combined)
            index_type: Type of index (boolean, wordcount, tfidf, self_boolean, etc.)
            index_name: Custom name for the index (optional)
            config_name: Config file to use (config or config_self)
        """
        overrides = []
        if dataset:
            overrides.append(f"dataset={dataset}")
        if index_type:
            overrides.append(f"index={index_type}")
        
        # Use config_self if specified or if using self index
        if config_name:
            self.config_name = config_name
        elif index_type and 'self_' in index_type:
            self.config_name = 'config_self'
        
        self._init_config(overrides)
        
        # Determine actual implementation
        implementation = 'elasticsearch'
        if hasattr(self.config.index, 'implementation'):
            implementation = self.config.index.implementation
        
        self.logger.info("="*60)
        self.logger.info("CREATING INDEX")
        self.logger.info("="*60)
        self.logger.info(f"Configuration: {self.config_name}")
        self.logger.info(f"Dataset: {self.config.dataset.name}")
        self.logger.info(f"Index Type: {self.config.index.type}")
        
        # Display actual implementation
        if implementation == 'self':
            self.logger.info(f"Implementation: SelfIndex (Custom)")
            storage_format = getattr(self.config.datastore, 'format', 'pickle')
            self.logger.info(f"Storage Format: {storage_format}")
            self.logger.info(f"Query Processor: {self.config.index.query_proc}")
            self.logger.info(f"Optimization: {self.config.index.optimization}")
            if hasattr(self.config.index, 'compression'):
                comp_type = self.config.index.compression.type if hasattr(self.config.index.compression, 'type') else self.config.index.compression
                self.logger.info(f"Compression: {comp_type}")
        else:
            self.logger.info(f"Implementation: Elasticsearch")
            self.logger.info(f"Datastore: {self.config.datastore.name}")
        
        # Generate index name
        if index_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            index_name = f"{self.config.dataset.name}_{self.config.index.type.lower()}_{timestamp}"
        
        self.logger.info(f"Index Name: {index_name}")
        
        # Load data
        self.logger.info("\nLoading dataset...")
        data_loader = DataLoader(self.config)
        documents = list(data_loader.load_dataset())
        
        self.logger.info(f"Loaded {len(documents)} documents")
        
        # Create index
        self.logger.info("\nCreating index...")
        index = self._get_index_instance()
        
        # Benchmark indexing if enabled
        if self.config.benchmark.measure_latency:
            benchmarker = Benchmarker(self.config)
            results = benchmarker.benchmark_indexing(index, index_name, documents)
        else:
            index.create_index(index_name, documents)
        
        self.logger.info(f"\n✓ Index '{index_name}' created successfully!")
        return index_name
    
    def run_experiment(self, dataset: str = None, index_type: str = None, datastore: str = None):
        """
        Run a complete experiment: create index, benchmark queries, generate plots.
        
        Args:
            dataset: Dataset to use (wiki, news, combined)
            index_type: Type of index (boolean, wordcount, tfidf)
            datastore: Datastore to use (elasticsearch, custom, etc.)
        """
        overrides = []
        if dataset:
            overrides.append(f"dataset={dataset}")
        if index_type:
            overrides.append(f"index={index_type}")
        if datastore:
            overrides.append(f"datastore={datastore}")
        
        self._init_config(overrides)
        
        experiment_name = f"{self.config.dataset.name}_{self.config.index.type.lower()}"
        if self.config.experiment.timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{experiment_name}_{timestamp}"
        
        self.logger.info("="*60)
        self.logger.info(f"RUNNING EXPERIMENT: {experiment_name}")
        self.logger.info("="*60)
        
        # Step 1: Create index
        self.logger.info("\n[1/3] Creating index...")
        index_name = self.create_index(dataset, index_type)
        
        # Step 2: Benchmark queries
        self.logger.info("\n[2/3] Benchmarking queries...")
        index = self._get_index_instance()
        benchmarker = Benchmarker(self.config)
        queries = benchmarker.generate_test_queries()
        
        results = benchmarker.benchmark_queries(index, index_name, queries, experiment_name)
        
        # Save results
        if self.config.experiment.save_results:
            results_dir = Path(self.config.paths.results_dir)
            results.save(results_dir)
        
        # Step 3: Generate plots
        self.logger.info("\n[3/3] Generating plots...")
        plotter = Plotter(self.config)
        
        plotter.plot_latency_distribution(
            results.latencies,
            title=f"Query Latency Distribution - {experiment_name}",
            filename=f"{experiment_name}_latency.png"
        )
        
        self.logger.info(f"\n✓ Experiment '{experiment_name}' completed successfully!")
        self.logger.info(f"Results saved to: {self.config.paths.results_dir}")
        self.logger.info(f"Plots saved to: {self.config.paths.plots_dir}")
    
    def query_index(self, query: str, index_name: str = None, dataset: str = None, 
                    index_type: str = None, max_results: int = None, config_name: str = None):
        """
        Query an existing index.
        
        Args:
            query: Query string
            index_name: Name of the index to query
            dataset: Dataset name (if index_name not provided)
            index_type: Index type (if index_name not provided)
            max_results: Maximum number of results to return
            config_name: Config file to use (config or config_self)
        """
        overrides = []
        if dataset:
            overrides.append(f"dataset={dataset}")
        if index_type:
            overrides.append(f"index={index_type}")
        
        # Auto-select config
        if config_name:
            self.config_name = config_name
        elif index_type and 'self_' in index_type:
            self.config_name = 'config_self'

        self._init_config(overrides)
        
        # If no index name provided, list available indices
        if index_name is None:
            index = self._get_index_instance()
            indices = index.list_indices()
            
            if not indices:
                self.logger.error("No indices found. Please create an index first.")
                return
            
            self.logger.info(f"Available indices: {', '.join(indices)}")
            
            # Use most recent index matching dataset/type if possible
            matching = [idx for idx in indices if self.config.dataset.name in idx]
            if matching:
                index_name = matching[-1]
                self.logger.info(f"Using index: {index_name}")
            else:
                self.logger.error("Please specify an index name with --index_name")
                return
        
        # Execute query
        self.logger.info(f"Querying index '{index_name}' with: {query}")
        
        index = self._get_index_instance()
        results_json = index.query(query, index_name, max_results)
        
        # Pretty print results
        import json
        results = json.loads(results_json)
        
        self.logger.info("\n" + "="*60)
        self.logger.info("QUERY RESULTS")
        self.logger.info("="*60)
        self.logger.info(f"Query: {results.get('query', 'N/A')}")
        self.logger.info(f"Processed Query: {results.get('processed_query', 'N/A')}")
        self.logger.info(f"Total Hits: {results.get('total_hits', 0)}")
        self.logger.info(f"Max Score: {results.get('max_score', 0)}")
        self.logger.info("\nTop Results:")
        
        for i, doc in enumerate(results.get('documents', []), 1):
            self.logger.info(f"\n{i}. Score: {doc.get('score', 0):.4f}")
            source = doc.get('source', {})
            self.logger.info(f"   Title: {source.get('title', 'N/A')[:100]}")
            self.logger.info(f"   ID: {doc.get('id', 'N/A')}")
            if 'url' in source:
                self.logger.info(f"   URL: {source['url']}")
        
        return results
    
    def generate_word_plots(self, dataset: str = None, sample_size: int = None):
        """
        Generate word frequency plots with and without preprocessing.
        
        Args:
            dataset: Dataset to analyze (wiki, news, combined)
            sample_size: Number of documents to sample
        """
        overrides = []
        if dataset:
            overrides.append(f"dataset={dataset}")
        
        self._init_config(overrides)
        
        if sample_size is None:
            sample_size = 1000
        
        self.logger.info("="*60)
        self.logger.info("GENERATING WORD FREQUENCY PLOTS")
        self.logger.info("="*60)
        self.logger.info(f"Dataset: {self.config.dataset.name}")
        self.logger.info(f"Sample Size: {sample_size}")
        
        # Load sample texts
        self.logger.info("\nLoading sample texts...")
        data_loader = DataLoader(self.config)
        texts = data_loader.load_sample_texts(sample_size)
        
        self.logger.info(f"Loaded {len(texts)} texts")
        
        # Calculate frequencies without preprocessing
        self.logger.info("\nCalculating word frequencies without preprocessing...")
        
        # Temporarily disable preprocessing
        original_config = self.config.preprocessing.copy()
        self.config.preprocessing.lowercase = False
        self.config.preprocessing.remove_punctuation = False
        self.config.preprocessing.remove_stopwords = False
        self.config.preprocessing.stemming = False
        
        preprocessor_raw = TextPreprocessor(self.config)
        freq_before = preprocessor_raw.get_word_frequencies(texts)
        
        # Restore preprocessing config
        self.config.preprocessing = original_config
        
        # Calculate frequencies with preprocessing
        self.logger.info("Calculating word frequencies with preprocessing...")
        preprocessor = TextPreprocessor(self.config)
        freq_after = preprocessor.get_word_frequencies(texts)
        
        # Generate plots
        self.logger.info("\nGenerating plots...")
        plotter = Plotter(self.config)
        
        plotter.plot_word_frequencies(
            freq_before,
            title=f"Word Frequencies Before Preprocessing - {self.config.dataset.name}",
            filename=f"{self.config.dataset.name}_freq_before.png"
        )
        
        plotter.plot_word_frequencies(
            freq_after,
            title=f"Word Frequencies After Preprocessing - {self.config.dataset.name}",
            filename=f"{self.config.dataset.name}_freq_after.png"
        )
        
        plotter.plot_word_frequency_comparison(
            freq_before,
            freq_after,
            filename=f"{self.config.dataset.name}_freq_comparison.png"
        )
        
        self.logger.info(f"\n✓ Plots saved to: {self.config.paths.plots_dir}")
        
        # Print statistics
        self.logger.info("\n" + "="*60)
        self.logger.info("STATISTICS")
        self.logger.info("="*60)
        self.logger.info(f"Unique words before preprocessing: {len(freq_before):,}")
        self.logger.info(f"Unique words after preprocessing: {len(freq_after):,}")
        self.logger.info(f"Reduction: {(1 - len(freq_after)/len(freq_before))*100:.1f}%")
    
    def benchmark(self, experiment_name: str = None, dataset: str = None, 
                  index_type: str = None, index_name: str = None, config_name: str = None):
        """
        Run performance benchmarks on an existing index.
        
        Args:
            experiment_name: Name for the benchmark experiment
            dataset: Dataset name
            index_type: Index type
            index_name: Specific index to benchmark
            config_name: Config file to use (config or config_self)
        """
        overrides = []
        if dataset:
            overrides.append(f"dataset={dataset}")
        if index_type:
            overrides.append(f"index={index_type}")
        
        # Auto-select config
        if config_name:
            self.config_name = config_name
        elif index_type and 'self_' in index_type:
            self.config_name = 'config_self'

        self._init_config(overrides)
        
        if experiment_name is None:
            experiment_name = f"benchmark_{self.config.dataset.name}_{self.config.index.type.lower()}"
        
        # Get index name
        if index_name is None:
            index = self._get_index_instance()
            indices = index.list_indices()
            
            if not indices:
                self.logger.error("No indices found. Please create an index first.")
                return
            
            matching = [idx for idx in indices if self.config.dataset.name in idx]
            if matching:
                index_name = matching[-1]
                self.logger.info(f"Using index: {index_name}")
            else:
                self.logger.error("Please specify an index name with --index_name")
                return
        
        self.logger.info("="*60)
        self.logger.info(f"BENCHMARKING: {experiment_name}")
        self.logger.info("="*60)
        
        # Generate test queries
        benchmarker = Benchmarker(self.config)
        queries = benchmarker.generate_test_queries()
        
        # Run benchmark
        index = self._get_index_instance()
        results = benchmarker.benchmark_queries(index, index_name, queries, experiment_name)
        
        # Save results
        if self.config.experiment.save_results:
            results_dir = Path(self.config.paths.results_dir)
            results.save(results_dir)
        
        # Generate plots
        plotter = Plotter(self.config)
        plotter.plot_latency_distribution(
            results.latencies,
            title=f"Latency Distribution - {experiment_name}",
            filename=f"{experiment_name}_latency.png"
        )
        
        self.logger.info(f"\n✓ Benchmark completed!")
        self.logger.info(f"Results saved to: {self.config.paths.results_dir}")
    
    def list_indices(self, datastore: str = None):
        """
        List all available indices.
        
        Args:
            datastore: Datastore to query (elasticsearch, custom, etc.)
        """
        overrides = []
        if datastore:
            overrides.append(f"datastore={datastore}")
        
        self._init_config(overrides)
        
        index = self._get_index_instance()
        indices = index.list_indices()
        
        self.logger.info("="*60)
        self.logger.info("AVAILABLE INDICES")
        self.logger.info("="*60)
        
        if not indices:
            self.logger.info("No indices found.")
        else:
            for i, idx in enumerate(indices, 1):
                self.logger.info(f"{i}. {idx}")
        
        return indices
    
    def show_config(self, dataset: str = None, index_type: str = None, datastore: str = None):
        """
        Display current configuration.
        
        Args:
            dataset: Dataset configuration to show
            index_type: Index type configuration to show
            datastore: Datastore configuration to show
        """
        overrides = []
        if dataset:
            overrides.append(f"dataset={dataset}")
        if index_type:
            overrides.append(f"index={index_type}")
        if datastore:
            overrides.append(f"datastore={datastore}")
        
        self._init_config(overrides)
        
        self.logger.info("="*60)
        self.logger.info("CURRENT CONFIGURATION")
        self.logger.info("="*60)
        print(OmegaConf.to_yaml(self.config))
    
    def generate_queries(self, num_queries: int = 1000, dataset: str = None, 
                        output: str = None, show_stats: bool = True):
        """
        Generate test queries for benchmarking.
        
        Args:
            num_queries: Number of queries to generate
            dataset: Dataset to generate queries for
            output: Output file path (optional)
            show_stats: Whether to show query statistics
        """
        overrides = []
        if dataset:
            overrides.append(f"dataset={dataset}")
        
        self._init_config(overrides)
        
        self.logger.info("="*60)
        self.logger.info("GENERATING TEST QUERIES")
        self.logger.info("="*60)
        self.logger.info(f"Dataset: {self.config.dataset.name}")
        self.logger.info(f"Number of queries: {num_queries}")
        
        from src.utils.query_generator import QueryGenerator
        
        # Generate queries
        query_gen = QueryGenerator(self.config, seed=42)
        queries = query_gen.generate_queries(num_queries)
        
        # Show statistics
        if show_stats:
            stats = query_gen.get_query_statistics(queries)
            self.logger.info("\n" + "="*60)
            self.logger.info("QUERY STATISTICS")
            self.logger.info("="*60)
            self.logger.info(f"Total queries: {stats['total_queries']}")
            self.logger.info(f"Single term queries: {stats['single_term']}")
            self.logger.info(f"Queries with AND: {stats['with_and']}")
            self.logger.info(f"Queries with OR: {stats['with_or']}")
            self.logger.info(f"Queries with NOT: {stats['with_not']}")
            self.logger.info(f"Phrase queries: {stats['phrase_queries']}")
            self.logger.info(f"Complex queries: {stats['complex_queries']}")
            self.logger.info(f"Avg terms per query: {stats['avg_terms_per_query']:.2f}")
            self.logger.info(f"Unique terms: {stats['unique_terms']}")
            
            # Show sample queries
            self.logger.info("\nSample queries (first 20):")
            for i, query in enumerate(queries[:20], 1):
                self.logger.info(f"  {i:2d}. {query}")
        
        # Save to file if requested
        if output:
            output_path = Path(output)
            query_gen.save_queries(queries, output_path)
            self.logger.info(f"\n✓ Saved queries to: {output}")
        else:
            # Save to default location
            cache_dir = Path(self.config.paths.data_dir) / "query_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            output_path = cache_dir / f"queries_{self.config.dataset.name}_{num_queries}.json"
            query_gen.save_queries(queries, output_path)
            self.logger.info(f"\n✓ Saved queries to: {output_path}")
        
        return queries
    
    def test_query_parsing(self, query: str = None):
        """
        Test query parsing and show how it converts to Elasticsearch.
        
        Args:
            query: Query string to test (if None, tests several examples)
        """
        self._init_config()
        
        from src.utils.query_parser import BooleanQueryParser
        import json
        
        parser = BooleanQueryParser()
        
        if query:
            test_queries = [query]
        else:
            # Test with various query types
            test_queries = [
                "python",
                '"machine learning"',
                "python AND java",
                "python OR java",
                "python AND NOT beginner",
                "(python OR java) AND programming",
                "climate AND environment AND NOT pollution",
                '("natural language" OR NLP) AND processing',
                "(election OR vote) AND government",
            ]
        
        self.logger.info("="*60)
        self.logger.info("QUERY PARSING TEST")
        self.logger.info("="*60)
        
        for test_query in test_queries:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Query: {test_query}")
            self.logger.info(f"{'='*60}")
            
            # Parse and convert
            es_query = parser.parse_to_elasticsearch(test_query)
            
            # Pretty print the Elasticsearch query
            self.logger.info("Elasticsearch Query DSL:")
            print(json.dumps(es_query, indent=2))
            
            # Show explanation
            self.logger.info("\nParsing Explanation:")
            explanation = parser.explain_query(test_query)
            for line in explanation.split('\n'):
                if line.strip():
                    self.logger.info(f"  {line}")
    
    def delete_index(self, index_name: str, datastore: str = None):
        """
        Delete an index.
        
        Args:
            index_name: Name of the index to delete
            datastore: Datastore where the index exists
        """
        overrides = []
        if datastore:
            overrides.append(f"datastore={datastore}")
        
        self._init_config(overrides)
        
        self.logger.info(f"Deleting index: {index_name}")
        
        index = self._get_index_instance()
        index.delete_index(index_name)
        
        self.logger.info(f"✓ Index '{index_name}' deleted successfully")


def main():
    """Main entry point."""
    fire.Fire(IndexingCLI)


if __name__ == "__main__":
    main()