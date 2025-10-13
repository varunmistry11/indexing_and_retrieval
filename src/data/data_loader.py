import json
import logging
from pathlib import Path
from typing import Iterator, Tuple, List
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading datasets from various sources."""
    
    def __init__(self, config):
        """
        Initialize data loader.
        
        Args:
            config: Hydra configuration object
        """
        self.config = config
    
    def load_dataset(self) -> Iterator[Tuple[str, str]]:
        """
        Load dataset based on configuration.
        
        Yields:
            Tuples of (doc_id, doc_content_json_string)
        """
        dataset_path = Path(self.config.dataset.source_file)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        logger.info(f"Loading dataset from: {dataset_path}")
        
        # Count total lines for progress bar
        total_lines = self._count_lines(dataset_path)
        
        # Apply sample size if specified
        max_docs = self.config.dataset.sample_size
        if max_docs is not None:
            total_lines = min(total_lines, max_docs)
        
        logger.info(f"Loading {total_lines} documents")
        
        # Load and yield documents
        with open(dataset_path, 'r', encoding='utf-8') as f:
            pbar = tqdm(
                total=total_lines, 
                desc="Loading documents", 
                disable=not self.config.indexing.show_progress
            )
            
            for i, line in enumerate(f):
                if max_docs is not None and i >= max_docs:
                    break
                
                try:
                    doc = json.loads(line.strip())
                    doc_id = doc.get(self.config.dataset.fields.id_field, f"doc_{i}")
                    
                    # Yield as (doc_id, json_string)
                    yield (doc_id, json.dumps(doc))
                    pbar.update(1)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing line {i}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing line {i}: {e}")
                    continue
            
            pbar.close()
    
    def _count_lines(self, filepath: Path) -> int:
        """
        Count lines in a file efficiently.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Number of lines in the file
        """
        logger.info(f"Counting lines in {filepath}...")
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for _ in f:
                count += 1
        logger.info(f"Found {count:,} lines")
        return count
    
    def load_sample_texts(self, sample_size: int = 1000) -> List[str]:
        """
        Load a sample of texts for analysis.
        
        Args:
            sample_size: Number of texts to load
            
        Returns:
            List of text strings
        """
        texts = []
        text_field = self.config.dataset.fields.text_field
        
        logger.info(f"Loading {sample_size} sample texts...")
        
        for doc_id, doc_json in self.load_dataset():
            if len(texts) >= sample_size:
                break
            
            try:
                doc = json.loads(doc_json)
                if text_field in doc and doc[text_field]:
                    texts.append(doc[text_field])
            except Exception as e:
                logger.warning(f"Error loading document {doc_id}: {e}")
                continue
        
        logger.info(f"Loaded {len(texts)} sample texts")
        return texts
    
    def get_dataset_stats(self) -> dict:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            "total_documents": 0,
            "fields": set(),
            "sample_documents": []
        }
        
        logger.info("Gathering dataset statistics...")
        
        for i, (doc_id, doc_json) in enumerate(self.load_dataset()):
            stats["total_documents"] += 1
            
            # Collect field names
            try:
                doc = json.loads(doc_json)
                stats["fields"].update(doc.keys())
                
                # Store first 3 documents as samples
                if i < 3:
                    stats["sample_documents"].append(doc)
            except Exception as e:
                logger.warning(f"Error processing document {doc_id}: {e}")
            
            # Limit counting for very large datasets
            if stats["total_documents"] >= 100000:
                logger.info("Large dataset detected, stopping count at 100k...")
                break
        
        stats["fields"] = list(stats["fields"])
        
        logger.info(f"Dataset statistics: {stats['total_documents']} documents")
        return stats