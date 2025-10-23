import time
import psutil
import logging
import json
from typing import List, Dict, Any, Callable
from pathlib import Path
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class BenchmarkResults:
    """Container for benchmark results."""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().isoformat()
        self.queries = []
        self.latencies = []
        self.memory_usage = []
        self.results_count = []
        
    def add_query_result(self, query: str, latency: float, memory_mb: float, num_results: int):
        """Add a single query result."""
        self.queries.append(query)
        self.latencies.append(latency)
        self.memory_usage.append(memory_mb)
        self.results_count.append(num_results)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate and return statistics."""
        latencies = np.array(self.latencies)
        
        stats = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "total_queries": len(self.queries),
            "latency_ms": {
                "mean": float(np.mean(latencies)),
                "median": float(np.median(latencies)),
                "std": float(np.std(latencies)),
                "min": float(np.min(latencies)),
                "max": float(np.max(latencies)),
                "p95": float(np.percentile(latencies, 95)),
                "p99": float(np.percentile(latencies, 99))
            },
            "memory_mb": {
                "mean": float(np.mean(self.memory_usage)),
                "max": float(np.max(self.memory_usage))
            },
            "throughput_qps": len(self.queries) / sum(latencies) * 1000 if sum(latencies) > 0 else 0
        }
        
        return stats
    
    def save(self, output_dir: Path):
        """Save benchmark results to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        detailed_path = output_dir / f"{self.experiment_name}_detailed.json"
        detailed_data = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "queries": self.queries,
            "latencies_ms": self.latencies,
            "memory_mb": self.memory_usage,
            "results_count": self.results_count
        }
        
        with open(detailed_path, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        
        # Save statistics
        stats_path = output_dir / f"{self.experiment_name}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.get_statistics(), f, indent=2)
        
        logger.info(f"Saved benchmark results to {output_dir}")


class Benchmarker:
    """Handles benchmarking of index operations."""
    
    def __init__(self, config):
        """
        Initialize benchmarker.
        
        Args:
            config: Hydra configuration object
        """
        self.config = config
        self.process = psutil.Process()
    
    def benchmark_queries(
        self,
        index_instance,
        index_id: str,
        queries: List[str],
        experiment_name: str = None
    ) -> BenchmarkResults:
        """
        Benchmark a set of queries.
        
        Args:
            index_instance: Index instance to query
            index_id: Index identifier
            queries: List of query strings
            experiment_name: Name for this experiment
            
        Returns:
            BenchmarkResults object
        """
        if experiment_name is None:
            experiment_name = self.config.experiment.name
        
        results = BenchmarkResults(experiment_name)
        
        logger.info(f"Starting benchmark: {experiment_name}")
        logger.info(f"Total queries: {len(queries)}")
        
        # Warmup queries
        warmup_count = min(self.config.benchmark.warmup_queries, len(queries))
        logger.info(f"Running {warmup_count} warmup queries...")
        
        for i in range(warmup_count):
            try:
                index_instance.query(queries[i], index_id)
            except Exception as e:
                logger.warning(f"Warmup query {i} failed: {e}")
        
        # Actual benchmark
        logger.info("Running benchmark queries...")
        
        for i, query in enumerate(queries):
            try:
                # Measure memory before query
                mem_before = self.process.memory_info().rss / 1024 / 1024  # MB
                
                # Execute query and measure time
                start_time = time.perf_counter()
                result_json = index_instance.query(query, index_id)
                end_time = time.perf_counter()
                
                # Measure memory after query
                mem_after = self.process.memory_info().rss / 1024 / 1024  # MB
                
                # Parse results to count
                try:
                    result_data = json.loads(result_json)
                    num_results = len(result_data.get('documents', []))
                except:
                    num_results = 0
                
                # Calculate metrics
                latency_ms = (end_time - start_time) * 1000
                memory_mb = max(mem_before, mem_after)
                
                # Store results
                results.add_query_result(query, latency_ms, memory_mb, num_results)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(queries)} queries")
                    
            except Exception as e:
                logger.error(f"Query {i} failed: {e}")
                results.add_query_result(query, 0, 0, 0)
        
        # Print summary
        stats = results.get_statistics()
        logger.info("\n" + "="*50)
        logger.info("BENCHMARK RESULTS")
        logger.info("="*50)
        logger.info(f"Experiment: {experiment_name}")
        logger.info(f"Total Queries: {stats['total_queries']}")
        logger.info(f"Mean Latency: {stats['latency_ms']['mean']:.2f} ms")
        logger.info(f"P95 Latency: {stats['latency_ms']['p95']:.2f} ms")
        logger.info(f"P99 Latency: {stats['latency_ms']['p99']:.2f} ms")
        logger.info(f"Throughput: {stats['throughput_qps']:.2f} queries/sec")
        logger.info(f"Max Memory: {stats['memory_mb']['max']:.2f} MB")
        logger.info("="*50)
        
        return results
    
    def benchmark_indexing(
        self,
        index_instance,
        index_id: str,
        documents: List[tuple],
        experiment_name: str = None
    ) -> Dict[str, Any]:
        """
        Benchmark indexing operation.
        
        Args:
            index_instance: Index instance
            index_id: Index identifier
            documents: List of (doc_id, doc_content) tuples
            experiment_name: Name for this experiment
            
        Returns:
            Dictionary with benchmark results
        """
        if experiment_name is None:
            experiment_name = f"{self.config.experiment.name}_indexing"
        
        logger.info(f"Benchmarking indexing: {experiment_name}")
        logger.info(f"Documents to index: {len(documents)}")
        
        # Measure memory before
        mem_before = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure indexing time
        start_time = time.perf_counter()
        index_instance.create_index(index_id, documents)
        end_time = time.perf_counter()
        
        # Measure memory after
        mem_after = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate metrics
        indexing_time = end_time - start_time
        throughput = len(documents) / indexing_time if indexing_time > 0 else 0
        memory_increase = mem_after - mem_before
        
        results = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "document_count": len(documents),
            "indexing_time_seconds": indexing_time,
            "throughput_docs_per_sec": throughput,
            "memory_before_mb": mem_before,
            "memory_after_mb": mem_after,
            "memory_increase_mb": memory_increase
        }
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("INDEXING BENCHMARK RESULTS")
        logger.info("="*50)
        logger.info(f"Documents: {results['document_count']}")
        logger.info(f"Time: {results['indexing_time_seconds']:.2f} seconds")
        logger.info(f"Throughput: {results['throughput_docs_per_sec']:.2f} docs/sec")
        logger.info(f"Memory increase: {results['memory_increase_mb']:.2f} MB")
        logger.info("="*50)
        
        # Save results
        if self.config.experiment.save_results:
            output_dir = Path(self.config.paths.results_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"{experiment_name}.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Saved indexing results to {output_path}")
        
        return results
    
    def generate_test_queries(self, num_queries: int = None, use_cached: bool = True) -> List[str]:
        """
        Generate test queries for benchmarking.
        
        Args:
            num_queries: Number of queries to generate
            use_cached: Whether to use cached queries if available
            
        Returns:
            List of query strings
        """
        if num_queries is None:
            num_queries = self.config.benchmark.num_queries
        
        # Import here to avoid circular dependency
        from src.utils.query_generator import QueryGenerator
        
        # Create cache directory
        cache_dir = Path(self.config.paths.data_dir) / "query_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file path
        cache_file = cache_dir / f"queries_{self.config.dataset.name}_{num_queries}.json"
        
        # Try to load from cache
        if use_cached and cache_file.exists():
            logger.info(f"Loading cached queries from {cache_file}")
            query_gen = QueryGenerator(self.config)
            queries = query_gen.load_queries(cache_file)
            
            # If cached queries match the requested count, use them
            if len(queries) == num_queries:
                logger.info(f"Using {len(queries)} cached queries")
                return queries
            else:
                logger.info(f"Cached queries count mismatch, generating new queries")
        
        # Generate new queries
        logger.info(f"Generating {num_queries} new queries for {self.config.dataset.name} dataset")
        query_gen = QueryGenerator(self.config, seed=42)  # Fixed seed for reproducibility
        queries = query_gen.generate_queries(num_queries)
        
        # Save to cache
        query_gen.save_queries(queries, cache_file)
        
        # Print statistics
        stats = query_gen.get_query_statistics(queries)
        logger.info(f"Query Statistics:")
        logger.info(f"  Total queries: {stats['total_queries']}")
        logger.info(f"  Single term: {stats['single_term']}")
        logger.info(f"  With AND: {stats['with_and']}")
        logger.info(f"  With OR: {stats['with_or']}")
        logger.info(f"  With NOT: {stats['with_not']}")
        logger.info(f"  Phrase queries: {stats['phrase_queries']}")
        logger.info(f"  Complex queries: {stats['complex_queries']}")
        logger.info(f"  Avg terms per query: {stats['avg_terms_per_query']:.2f}")
        logger.info(f"  Unique terms: {stats['unique_terms']}")
        
        return queries