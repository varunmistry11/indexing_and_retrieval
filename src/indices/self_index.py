"""
Phase 3: Main SelfIndex Class Implementation
File: src/indices/self_index.py

This is the main class that integrates all Phase 1 and Phase 2 components
and implements the IndexBase interface.
"""

import logging
import json
import pickle
import gzip
from pathlib import Path
from typing import List, Dict, Optional, Set, Any, Iterable, Tuple
from datetime import datetime
import time

from src.index_base import IndexBase
from src.selfindex import (
    InvertedIndex,
    DocumentStore,
    CollectionStatistics,
    BooleanOperations,
    TermAtATimeProcessor,
    DocumentAtATimeProcessor,
    BooleanQueryProcessor,
    PhraseQueryProcessor,
    QueryResult
)
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.utils.query_parser import BooleanQueryParser

logger = logging.getLogger(__name__)


class SelfIndex(IndexBase):
    """
    Custom inverted index implementation (SelfIndex-v1.0).
    
    Supports multiple configurations:
    - Index types (x=1,2,3): BOOLEAN, WORDCOUNT, TFIDF
    - Storage formats (y=1,2): Custom objects, external DB
    - Compression (z=1,2): Custom encoding, external library
    - Optimizations (i=0,1): Skip pointers
    - Query processors (q=T,D): Term-at-a-time, Document-at-a-time
    """
    
    def __init__(self, config):
        """
        Initialize SelfIndex with configuration.
        
        Args:
            config: Hydra configuration object
        """
        # Extract configuration parameters
        self.index_type = config.index.type  # BOOLEAN, WORDCOUNT, TFIDF
        self.storage_format = config.get('datastore', {}).get('format', 'json')
        self.compression_type = config.get('compression', {}).get('type', 'NONE')
        self.use_skip_pointers = config.index.get('optimization', 'Null') == 'Skipping'
        self.query_processor_type = config.index.get('query_proc', 'TERMatat')
        
        # Map to enum values for parent class
        info = self.index_type  # BOOLEAN, WORDCOUNT, TFIDF
        dstore = 'CUSTOM'  # Always CUSTOM for SelfIndex
        compr = self.compression_type if self.compression_type != 'NONE' else 'NONE'
        qproc = self.query_processor_type  # TERMatat or DOCatat
        optim = 'Skipping' if self.use_skip_pointers else 'Null'
        
        # Initialize parent class
        super().__init__(
            core='SelfIndex',
            info=info,
            dstore=dstore,
            qproc=qproc,
            compr=compr,
            optim=optim
        )
        
        # Store configuration
        self.config = config
        
        # Storage paths
        self.storage_dir = Path(config.paths.index_storage) / 'selfindex'
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.inverted_index: Optional[InvertedIndex] = None
        self.doc_store: Optional[DocumentStore] = None
        self.stats: Optional[CollectionStatistics] = None
        
        # Query processors
        self.term_processor: Optional[TermAtATimeProcessor] = None
        self.doc_processor: Optional[DocumentAtATimeProcessor] = None
        self.bool_processor: Optional[BooleanQueryProcessor] = None
        
        # Text preprocessing
        self.preprocessor = TextPreprocessor(config)
        self.query_parser = BooleanQueryParser()
        
        # Metadata
        self.index_name: Optional[str] = None
        self.created_at: Optional[datetime] = None
        self.document_count = 0
        
        logger.info(f"Initialized SelfIndex with type={self.index_type}, "
                    f"processor={self.query_processor_type}, "
                    f"skip_pointers={self.use_skip_pointers}")
    
    def create_index(
        self,
        index_id: str,
        files: Iterable[Tuple[str, str]]
    ) -> None:
        """
        Create a new index from files.
        
        Args:
            index_id: The unique identifier for the index
            files: Iterable of tuples (file_id, content)
        """
        start_time = time.time()
        
        # Use index_id as the index name
        self.index_name = index_id
        
        # Check if index exists
        index_path = self._get_index_path(self.index_name)
        if index_path.exists():
            logger.warning(f"Index {self.index_name} already exists. Overwriting.")
            import shutil
            shutil.rmtree(index_path)
        
        logger.info(f"Creating index: {self.index_name}")
        
        # Initialize components
        self.inverted_index = InvertedIndex(use_skip_pointers=self.use_skip_pointers)
        self.doc_store = DocumentStore()
        self.stats = CollectionStatistics()
        
        # Process files
        processed_count = 0
        
        for file_id, content in files:
            try:
                if not content:
                    continue
                
                # Preprocess text
                tokens = self.preprocessor.preprocess(content)
                
                if not tokens:
                    continue
                
                # Add to document store
                metadata = {'length': len(tokens)}
                internal_id = self.doc_store.add_document(file_id, metadata)
                
                # Add to inverted index
                self.inverted_index.add_document(internal_id, tokens)
                
                # Update statistics
                self.stats.add_document(internal_id, len(tokens))
                
                processed_count += 1
                
                if processed_count % 1000 == 0:
                    logger.info(f"Processed {processed_count} documents...")
                    
            except Exception as e:
                logger.error(f"Error processing document {file_id}: {e}")
        
        # Finalize index (build skip pointers if needed)
        if self.use_skip_pointers:
            logger.info("Building skip pointers...")
            self.inverted_index.finalize()
        
        # Initialize query processors
        self._initialize_query_processors()
        
        # Update document count and timestamp
        self.document_count = processed_count
        self.created_at = datetime.now()
        
        # Save index to disk
        logger.info("Saving index to disk...")
        self._save_index()
        
        # Calculate statistics
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Index creation complete. Processed {processed_count} documents in {duration:.2f}s")
    
    def load_index(self, serialized_index_dump: str) -> None:
        """
        Load an existing index from disk.
        
        Args:
            serialized_index_dump: Path to dump of serialized index
        """
        index_path = Path(serialized_index_dump)
        
        if not index_path.exists():
            logger.error(f"Index not found at {index_path}")
            raise FileNotFoundError(f"Index not found at {index_path}")
        
        logger.info(f"Loading index from: {index_path}")
        
        try:
            # Load metadata
            metadata_file = index_path / 'metadata.json'
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            self.index_name = metadata['index_name']
            self.document_count = metadata['document_count']
            
            # Handle created_at which might be None or string
            created_at_str = metadata.get('created_at')
            if created_at_str and isinstance(created_at_str, str):
                self.created_at = datetime.fromisoformat(created_at_str)
            else:
                self.created_at = None
            
            # Load inverted index
            index_file = index_path / 'inverted_index.dat'
            self.inverted_index = self._load_component(index_file, InvertedIndex)
            
            # Load document store
            doc_store_file = index_path / 'doc_store.dat'
            self.doc_store = self._load_component(doc_store_file, DocumentStore)
            
            # Load statistics
            stats_file = index_path / 'statistics.dat'
            self.stats = self._load_component(stats_file, CollectionStatistics)
            
            # Initialize query processors
            self._initialize_query_processors()
            
            logger.info(f"Successfully loaded index: {self.index_name}")
            logger.info(f"Documents: {self.document_count}, "
                       f"Vocabulary: {self.inverted_index.get_vocabulary_size()}")
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise
    
    def update_index(
        self,
        index_id: str,
        remove_files: Iterable[Tuple[str, str]],
        add_files: Iterable[Tuple[str, str]]
    ) -> None:
        """
        Update an existing index.
        
        Args:
            index_id: The unique identifier for the index
            remove_files: Iterable of tuples (file_id, content) to remove
            add_files: Iterable of tuples (file_id, content) to add
        """
        # Load the index first
        index_path = self._get_index_path(index_id)
        if not index_path.exists():
            raise ValueError(f"Index {index_id} not found")
        
        self.load_index(str(index_path))
        
        # Remove files (not fully implemented - would require tracking and rebuilding)
        if remove_files:
            logger.warning("Document removal not fully implemented")
            # For now, just log the files to be removed
            for file_id, _ in remove_files:
                logger.info(f"Should remove: {file_id}")
        
        # Add new files
        updated_count = 0
        for file_id, content in add_files:
            try:
                if not content:
                    continue
                
                tokens = self.preprocessor.preprocess(content)
                if not tokens:
                    continue
                
                metadata = {'length': len(tokens)}
                internal_id = self.doc_store.add_document(file_id, metadata)
                self.inverted_index.add_document(internal_id, tokens)
                self.stats.add_document(internal_id, len(tokens))
                
                updated_count += 1
                
            except Exception as e:
                logger.error(f"Error updating document: {e}")
        
        # Update document count
        self.document_count += updated_count
        
        # Rebuild skip pointers if needed
        if self.use_skip_pointers:
            self.inverted_index.finalize()
        
        # Re-initialize query processors
        self._initialize_query_processors()
        
        # Save updated index
        self._save_index()
        
        logger.info(f"Updated index with {updated_count} new documents")
    
    def query(self, query: str) -> str:
        """
        Query the index and return results as JSON string.
        
        Args:
            query: Input query string
            
        Returns:
            JSON string with results
        """
        if not self.inverted_index:
            raise ValueError("No index loaded. Call load_index() first.")
        
        start_time = time.time()
        
        # Default k value
        k = 10
        
        # Parse query
        try:
            # Use the parser's internal method to get the tree structure
            query_tree = self.query_parser._parse_boolean_expression(query)
            logger.debug(f"Parsed query tree: {query_tree}")
        except Exception as e:
            logger.error(f"Error parsing query '{query}': {e}")
            # Fallback to simple term query
            query_tree = {'type': 'TERM', 'value': query}
        
        # Process query based on type
        if self._is_boolean_query(query_tree) or query_tree.get('type') == 'PHRASE':
            # Use boolean processor for boolean queries and phrase queries
            if query_tree.get('type') == 'PHRASE':
                # Handle phrase query
                phrase_terms = query_tree.get('value', '').split()
                processed_terms = []
                for term in phrase_terms:
                    processed = self.preprocessor.preprocess(term)
                    processed_terms.extend(processed)
                
                if not processed_terms:
                    return json.dumps({'results': [], 'query_time': 0})
                
                # Get postings for each term
                term_postings = [self.inverted_index.get_postings(term) for term in processed_terms]
                # Filter out None values
                term_postings = [tp for tp in term_postings if tp is not None]
                
                if not term_postings:
                    return json.dumps({'results': [], 'query_time': 0})
                
                result_postings = PhraseQueryProcessor.phrase_query(term_postings)
                results = [
                    QueryResult(doc_id=doc_id, score=1.0)
                    for doc_id in result_postings.get_doc_ids()
                ]
            else:
                # Regular boolean query
                result_postings = self.bool_processor.process_complex_query(query_tree)
                results = [
                    QueryResult(doc_id=doc_id, score=1.0)
                    for doc_id in result_postings.get_doc_ids()
                ]
        else:
            # Use scoring processor (Term-at-a-time or Document-at-a-time)
            query_terms = self._extract_terms(query_tree)
            
            # Preprocess query terms
            processed_terms = []
            for term in query_terms:
                processed = self.preprocessor.preprocess(term)
                processed_terms.extend(processed)
            
            if not processed_terms:
                return json.dumps({'results': [], 'query_time': 0})
            
            # Choose processor
            if self.query_processor_type == 'TERMatat':
                results = self.term_processor.process_query(processed_terms, k=k)
            else:  # DOCatat
                results = self.doc_processor.process_query(processed_terms, k=k)
        
        # Convert to output format
        output_results = []
        for i, result in enumerate(results[:k]):
            internal_id = result.doc_id
            doc_metadata = self.doc_store.get_document(internal_id)
            
            if doc_metadata:
                output_results.append({
                    'rank': i + 1,
                    'doc_id': self.doc_store.get_external_id(internal_id),
                    'score': result.score,
                })
        
        query_time = time.time() - start_time
        logger.info(f"Query '{query}' returned {len(output_results)} results in {query_time:.3f}s")
        
        # Return as JSON string
        return json.dumps({
            'results': output_results,
            'query_time': query_time,
            'total_results': len(output_results)
        })
    
    def delete_index(self, index_id: str) -> None:
        """
        Delete an index from disk.
        
        Args:
            index_id: Name of the index to delete
        """
        index_path = self._get_index_path(index_id)
        
        if not index_path.exists():
            logger.warning(f"Index {index_id} not found")
            raise FileNotFoundError(f"Index {index_id} not found")
        
        try:
            import shutil
            shutil.rmtree(index_path)
            logger.info(f"Deleted index: {index_id}")
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            raise
    
    def list_indices(self) -> Iterable[str]:
        """
        List all available indices.
        
        Returns:
            Iterable of index IDs
        """
        indices = []
        
        for index_dir in self.storage_dir.iterdir():
            if index_dir.is_dir():
                indices.append(index_dir.name)
        
        return indices
    
    def list_indexed_files(self, index_id: str) -> Iterable[str]:
        """
        List all files indexed in the given index.
        
        Args:
            index_id: The unique identifier for the index
            
        Returns:
            Iterable of file IDs
        """
        # Load the index if not already loaded
        if not self.doc_store or self.index_name != index_id:
            index_path = self._get_index_path(index_id)
            self.load_index(str(index_path))
        
        # Return all external document IDs
        file_ids = []
        for internal_id in self.doc_store.documents.keys():
            external_id = self.doc_store.get_external_id(internal_id)
            if external_id:
                file_ids.append(external_id)
        
        return file_ids
    
    # Additional helper method for compatibility
        
        if not index_path.exists():
            logger.warning(f"Index {index_name} not found")
            return False
        
        try:
            import shutil
            shutil.rmtree(index_path)
            logger.info(f"Deleted index: {index_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            return False
    
    def list_indices(self) -> List[Dict[str, Any]]:
        """
        List all available indices.
        
        Returns:
            List of index metadata dictionaries
        """
        indices = []
        
        for index_dir in self.storage_dir.iterdir():
            if index_dir.is_dir():
                metadata_file = index_dir / 'metadata.json'
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        indices.append(metadata)
                    except Exception as e:
                        logger.error(f"Error reading metadata for {index_dir.name}: {e}")
        
        return indices
    
    # Additional helper method for compatibility
    def create_index_from_documents(
        self,
        dataset_name: str,
        documents: List[Dict[str, Any]],
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Convenience method for creating index from document list.
        This is for compatibility with test code.
        
        Args:
            dataset_name: Name of the dataset/index
            documents: List of documents with 'id', 'text', and optional metadata
            force: Whether to overwrite existing index
            
        Returns:
            Dictionary with indexing statistics
        """
        start_time = time.time()
        
        # Generate index name
        version = self._generate_version_string()
        index_id = f"selfindex_{dataset_name}_{version}_{int(time.time())}"
        
        # Convert documents to files format
        files = []
        for doc in documents:
            doc_id = doc.get('id', doc.get('_id', str(len(files))))
            text = doc.get('text', doc.get('content', ''))
            if text:
                files.append((doc_id, text))
        
        # Create index
        self.create_index(index_id, files)
        
        # Calculate statistics
        end_time = time.time()
        duration = end_time - start_time
        
        stats = {
            'index_name': self.index_name,
            'documents_processed': len(files),
            'documents_skipped': len(documents) - len(files),
            'total_documents': len(documents),
            'vocabulary_size': self.inverted_index.get_vocabulary_size(),
            'total_tokens': self.stats.total_terms,
            'avg_document_length': self.stats.avg_document_length,
            'index_size_bytes': self._get_index_size(),
            'indexing_time_seconds': duration,
            'documents_per_second': len(files) / duration if duration > 0 else 0,
            'created_at': self.created_at.isoformat(),
        }
        
        return stats
    
    def query_with_params(
        self,
        query_string: str,
        k: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Convenience method for querying with parameters.
        This is for compatibility with test code.
        
        Args:
            query_string: Query string
            k: Number of results to return
            **kwargs: Additional parameters
            
        Returns:
            List of result dictionaries
        """
        # Call the main query method
        json_result = self.query(query_string)
        result_dict = json.loads(json_result)
        
        # Return limited results
        return result_dict['results'][:k]
    
    # Helper methods
    
    def _generate_version_string(self) -> str:
        """Generate version string based on configuration (x.y.z.i.q)."""
        x = {'BOOLEAN': '1', 'WORDCOUNT': '2', 'TFIDF': '3'}.get(self.index_type, '1')
        y = '1' if self.storage_format in ['json', 'pickle'] else '2'
        z = {'NONE': '0', 'CODE': '1', 'CLIB': '2'}.get(self.compression_type, '0')
        i = '1' if self.use_skip_pointers else '0'
        q = 'T' if self.query_processor_type == 'TERMatat' else 'D'
        
        return f"{x}.{y}.{z}.{i}.{q}"
    
    def _get_index_path(self, index_name: str) -> Path:
        """Get path to index directory."""
        return self.storage_dir / index_name
    
    def _save_index(self):
        """Save index to disk."""
        index_path = self._get_index_path(self.index_name)
        index_path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            'index_name': self.index_name,
            'index_type': self.index_type,
            'document_count': self.document_count,
            'vocabulary_size': self.inverted_index.get_vocabulary_size(),
            'created_at': self.created_at.isoformat() if self.created_at else datetime.now().isoformat(),
            'version': self._generate_version_string(),
            'configuration': {
                'skip_pointers': self.use_skip_pointers,
                'query_processor': self.query_processor_type,
                'compression': self.compression_type,
            }
        }
        
        with open(index_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save components
        self._save_component(self.inverted_index, index_path / 'inverted_index.dat')
        self._save_component(self.doc_store, index_path / 'doc_store.dat')
        self._save_component(self.stats, index_path / 'statistics.dat')
        
        logger.info(f"Index saved to {index_path}")
    
    def _save_component(self, component, file_path: Path):
        """Save a component to disk with optional compression."""
        data = component.to_dict()
        
        if self.compression_type == 'CLIB':
            # Use gzip compression
            with gzip.open(str(file_path) + '.gz', 'wb') as f:
                pickle.dump(data, f)
        elif self.storage_format == 'json':
            with open(file_path, 'w') as f:
                json.dump(data, f)
        else:  # pickle
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
    
    def _load_component(self, file_path: Path, component_class):
        """Load a component from disk."""
        # Check for compressed version
        compressed_path = Path(str(file_path) + '.gz')
        
        if compressed_path.exists():
            with gzip.open(compressed_path, 'rb') as f:
                data = pickle.load(f)
        elif file_path.suffix == '.json' or self.storage_format == 'json':
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        
        return component_class.from_dict(data)
    
    def _get_index_size(self) -> int:
        """Calculate total size of index on disk."""
        index_path = self._get_index_path(self.index_name)
        total_size = 0
        
        for file_path in index_path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size
    
    def _initialize_query_processors(self):
        """Initialize query processors based on index type."""
        # Determine scoring method
        if self.index_type == 'BOOLEAN':
            scoring_method = 'boolean'
        elif self.index_type == 'WORDCOUNT':
            scoring_method = 'tf'
        else:  # TFIDF
            scoring_method = 'bm25'
        
        # Initialize processors
        self.term_processor = TermAtATimeProcessor(
            self.inverted_index,
            self.stats,
            scoring_method=scoring_method
        )
        
        self.doc_processor = DocumentAtATimeProcessor(
            self.inverted_index,
            self.stats,
            scoring_method=scoring_method
        )
        
        self.bool_processor = BooleanQueryProcessor(
            self.inverted_index,
            self.doc_store.get_all_internal_ids()
        )
    
    def _is_boolean_query(self, query_tree: Dict) -> bool:
        """Check if query contains boolean operators."""
        if not isinstance(query_tree, dict):
            return False
        
        node_type = query_tree.get('type', '')
        return node_type in ['AND', 'OR', 'NOT', 'PHRASE']
    
    def _extract_terms(self, query_tree: Dict) -> List[str]:
        """Extract all terms from query tree."""
        if not isinstance(query_tree, dict):
            return []
        
        node_type = query_tree.get('type', '')
        
        if node_type == 'TERM':
            return [query_tree.get('value', '')]
        elif node_type == 'PHRASE':
            # For phrase, split the value into terms
            phrase_value = query_tree.get('value', '')
            return phrase_value.split()
        elif node_type in ['AND', 'OR']:
            terms = []
            for child in query_tree.get('children', []):
                terms.extend(self._extract_terms(child))
            return terms
        elif node_type == 'NOT':
            children = query_tree.get('children', [])
            if children:
                return self._extract_terms(children[0])
        
        return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive index statistics."""
        if not self.inverted_index:
            return {}
        
        return {
            'index_name': self.index_name,
            'index_type': self.index_type,
            'document_count': self.document_count,
            'vocabulary_size': self.inverted_index.get_vocabulary_size(),
            'total_tokens': self.stats.total_terms,
            'avg_document_length': self.stats.avg_document_length,
            'index_size_bytes': self._get_index_size(),
            'use_skip_pointers': self.use_skip_pointers,
            'query_processor': self.query_processor_type,
            'compression': self.compression_type,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }