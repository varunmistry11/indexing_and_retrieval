"""
RocksDB-based index implementation (y=2) using rocksdict.
Uses RocksDB as the datastore instead of pickle/JSON.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Iterable
from rocksdict import Rdict, Options, AccessType

from src.index_base import IndexBase
from src.preprocessing.text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class RocksDBIndex(IndexBase):
    """
    Index using RocksDB as datastore (y=2, DB2).
    Stores term → postings mappings in RocksDB key-value store.
    """
    
    def __init__(self, config):
        """Initialize RocksDB index."""
        super().__init__(
            core='SelfIndex',
            info=config.index.type,
            dstore='DB2',  # RocksDB
            qproc='TERMatat',
            compr='NONE',
            optim='Null'
        )
        
        self.config = config
        self.preprocessor = TextPreprocessor(config)
        self.db = None
        self.db_path = None
    
    def create_index(self, index_id: str, files: Iterable[Tuple[str, str]]) -> None:
        """Create index and store in RocksDB."""
        logger.info(f"Creating RocksDB index: {index_id}")
        
        # Setup RocksDB path
        self.db_path = Path(self.config.paths.index_storage) / 'rocksdb' / index_id
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Create RocksDB instance
        opts = Options()
        opts.create_if_missing(True)
        self.db = Rdict(str(self.db_path), options=opts)
        
        # Build inverted index in memory first
        inverted_index = {}
        doc_count = 0
        
        for file_id, content in files:
            if not content:
                continue
            
            # Parse JSON if needed
            try:
                doc = json.loads(content) if isinstance(content, str) else content
                text = doc.get('text', content)
            except:
                text = content
            
            tokens = self.preprocessor.preprocess(text)
            if not tokens:
                continue
            
            # Track term positions and frequencies
            term_positions = {}
            for pos, token in enumerate(tokens):
                if token not in term_positions:
                    term_positions[token] = []
                term_positions[token].append(pos)
            
            # Add to inverted index
            for term, positions in term_positions.items():
                if term not in inverted_index:
                    inverted_index[term] = []
                
                inverted_index[term].append({
                    'doc_id': file_id,
                    'term_freq': len(positions),
                    'positions': positions
                })
            
            doc_count += 1
            if doc_count % 100 == 0:
                logger.info(f"Processed {doc_count} documents...")
        
        # Write to RocksDB
        logger.info(f"Writing {len(inverted_index)} terms to RocksDB...")
        
        for term, postings in inverted_index.items():
            # Serialize postings to JSON
            postings_json = json.dumps(postings)
            self.db[term] = postings_json
        
        # Store metadata
        metadata = {
            'index_id': index_id,
            'doc_count': doc_count,
            'term_count': len(inverted_index)
        }
        self.db['__metadata__'] = json.dumps(metadata)
        
        logger.info(f"✅ Created RocksDB index with {doc_count} documents, {len(inverted_index)} terms")
    
    def load_index(self, serialized_index_dump: str) -> None:
        """Load RocksDB index."""
        self.db_path = Path(serialized_index_dump)
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"RocksDB not found: {self.db_path}")
        
        # Open in read-only mode
        opts = Options()
        self.db = Rdict(str(self.db_path), options=opts, access_type=AccessType.read_only())
        
        logger.info(f"Loaded RocksDB index from {self.db_path}")
    
    def query(self, query: str, index_id: str = None, max_results: int = None) -> str:
        """Query the RocksDB index."""
        # Auto-load if index_id provided and not loaded
        if index_id and not self.db:
            index_path = Path(self.config.paths.index_storage) / 'rocksdb' / index_id
            if index_path.exists():
                self.load_index(str(index_path))
        
        if not self.db:
            raise ValueError("No index loaded. Call load_index() first or provide index_id.")
        
        k = max_results or 10
        
        # Preprocess query
        query_terms = self.preprocessor.preprocess(query)
        
        # Get postings from RocksDB and calculate scores
        doc_scores = {}
        
        for term in query_terms:
            postings_json = self.db.get(term)
            
            if postings_json:
                postings = json.loads(postings_json)
                
                # Calculate BM25-like score (simplified)
                for posting in postings:
                    doc_id = posting['doc_id']
                    tf = posting['term_freq']
                    
                    # Simple TF scoring
                    score = tf
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
        
        # Sort by score
        results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format output
        output = {
            'results': [
                {'rank': i+1, 'doc_id': doc_id, 'score': float(score)}
                for i, (doc_id, score) in enumerate(results[:k])
            ],
            'total_results': len(results),
            'query_time': 0
        }
        
        return json.dumps(output)
    
    def delete_index(self, index_id: str) -> None:
        """Delete RocksDB index."""
        if self.db:
            self.db.close()
            self.db = None
        
        import shutil
        db_path = Path(self.config.paths.index_storage) / 'rocksdb' / index_id
        if db_path.exists():
            shutil.rmtree(db_path)
            logger.info(f"Deleted RocksDB index: {index_id}")
    
    def list_indices(self) -> Iterable[str]:
        """List RocksDB indices."""
        rocksdb_dir = Path(self.config.paths.index_storage) / 'rocksdb'
        if not rocksdb_dir.exists():
            return []
        
        return [d.name for d in rocksdb_dir.iterdir() if d.is_dir()]
    
    def list_indexed_files(self, index_id: str) -> Iterable[str]:
        """List files in index."""
        # Would need to iterate through all keys
        return []
    
    def update_index(self, index_id: str, remove_files: Iterable[Tuple[str, str]], 
                     add_files: Iterable[Tuple[str, str]]) -> None:
        """Update not implemented for RocksDB."""
        raise NotImplementedError("Update not implemented for RocksDB index")