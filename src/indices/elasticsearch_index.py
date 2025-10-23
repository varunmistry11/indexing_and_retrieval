import json
import logging
from typing import Iterable, List, Dict, Any
from pathlib import Path
from elasticsearch import Elasticsearch, helpers
from datetime import datetime
import os
from omegaconf import OmegaConf

from src.index_base import IndexBase, IndexInfo, DataStore, Compression, QueryProc, Optimizations
from src.preprocessing.text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class ElasticsearchIndex(IndexBase):
    """Elasticsearch-based index implementation."""
    
    def __init__(self, config):
        """
        Initialize Elasticsearch index.
        
        Args:
            config: Hydra configuration object
        """
        self.config = config
        self.preprocessor = TextPreprocessor(config)
        
        # Extract config values with defaults
        # Handle None values from YAML (null) by converting to string 'Null'
        index_type = str(getattr(config.index, 'type', 'BOOLEAN'))
        query_proc = str(getattr(config.index, 'query_proc', 'TERMatat'))
        compression = str(getattr(config.index, 'compression', 'NONE'))
        optimization = getattr(config.index, 'optimization', None)
        # Convert None to 'Null' string for enum
        optimization = 'Null' if optimization is None else str(optimization)
        
        # Initialize base class with proper parameters
        super().__init__(
            core='ESIndex',
            info=index_type,
            dstore='DB1',
            qproc=query_proc,
            compr=compression,
            optim=optimization
        )
        
        # Initialize Elasticsearch client
        try:
            self.es = self._create_es_client()
        except Exception as e:
            logger.error(f"Failed to create Elasticsearch client: {e}")
            raise
        
        # Store index metadata
        self.indices_metadata = {}
        self._load_metadata()
    
    def _create_es_client(self) -> Elasticsearch:
        """Create and return Elasticsearch client."""
        ds_config = self.config.datastore
        
        es_config = {
            'hosts': [f"{ds_config.scheme}://{ds_config.host}:{ds_config.port}"],
            'timeout': ds_config.timeout,
            'max_retries': ds_config.max_retries,
            'retry_on_timeout': ds_config.retry_on_timeout,
        }
        
        # Handle authentication
        use_auth = ds_config.use_auth
        if isinstance(use_auth, str):
            use_auth = use_auth.lower() == 'true'
        
        if use_auth:
            es_config['basic_auth'] = (ds_config.username, ds_config.password)
            logger.info("Elasticsearch authentication enabled")
        
        # For HTTPS connections
        if ds_config.scheme == 'https':
            verify_certs = getattr(ds_config, 'verify_certs', False)
            if isinstance(verify_certs, str):
                verify_certs = verify_certs.lower() == 'true'
            
            es_config['verify_certs'] = verify_certs
            es_config['ssl_show_warn'] = getattr(ds_config, 'ssl_show_warn', False)
            
            if not verify_certs:
                logger.warning("SSL certificate verification is disabled")
        
        logger.info(f"Connecting to Elasticsearch at {ds_config.scheme}://{ds_config.host}:{ds_config.port}")
        
        return Elasticsearch(**es_config)
    
    def _get_analyzer_settings(self) -> dict:
        """Get custom analyzer settings based on preprocessing config."""
        filters = ['lowercase']
        
        if self.config.preprocessing.remove_stopwords:
            filters.append('stop')
        
        if self.config.preprocessing.stemming:
            filters.append('porter_stem')
        
        return {
            "analysis": {
                "analyzer": {
                    "custom_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": filters
                    }
                }
            }
        }
    
    def _load_metadata(self):
        """Load index metadata from disk."""
        metadata_path = Path(self.config.paths.index_storage) / "es_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.indices_metadata = json.load(f)
    
    def _save_metadata(self):
        """Save index metadata to disk."""
        metadata_path = Path(self.config.paths.index_storage) / "es_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(self.indices_metadata, f, indent=2)
    
    def create_index(self, index_id: str, files: Iterable[tuple[str, str]]) -> None:
        """
        Create an index for the given files.
        
        Args:
            index_id: The unique identifier for the index
            files: An iterable of tuples (file_id, file_content)
        """
        logger.info(f"Creating index: {index_id}")
        
        # Check if index already exists
        if self.es.indices.exists(index=index_id):
            logger.warning(f"Index {index_id} already exists. Deleting it first.")
            self.delete_index(index_id)
        
        # Create index with settings - Convert OmegaConf to dict
        from omegaconf import OmegaConf
        
        index_body = {
            "settings": {
                **OmegaConf.to_container(self.config.index.settings, resolve=True),
                **self._get_analyzer_settings()
            },
            "mappings": OmegaConf.to_container(self.config.index.mappings, resolve=True)
        }
        
        self.es.indices.create(index=index_id, body=index_body)
        logger.info(f"Index {index_id} created successfully")
        
        # Index documents
        doc_count = self._bulk_index_documents(index_id, files)
        
        # Save metadata
        self.indices_metadata[index_id] = {
            "created_at": datetime.now().isoformat(),
            "document_count": doc_count,
            "index_type": self.config.index.type,
            "dataset": self.config.dataset.name,
            "version": self.identifier_short
        }
        self._save_metadata()
        
        logger.info(f"Indexed {doc_count} documents to {index_id}")
    
    def _bulk_index_documents(self, index_id: str, files: Iterable[tuple[str, str]]) -> int:
        """
        Bulk index documents into Elasticsearch.
        
        Args:
            index_id: Index identifier
            files: Iterable of (file_id, file_content) tuples
            
        Returns:
            Number of documents indexed
        """
        def generate_actions():
            for file_id, content in files:
                # Parse JSON content
                try:
                    doc = json.loads(content) if isinstance(content, str) else content
                    
                    # Preprocess text fields
                    if 'text' in doc:
                        doc['preprocessed_text'] = ' '.join(
                            self.preprocessor.preprocess(doc['text'])
                        )
                    
                    if 'title' in doc:
                        doc['preprocessed_title'] = ' '.join(
                            self.preprocessor.preprocess(doc['title'])
                        )
                    
                    yield {
                        "_index": index_id,
                        "_id": doc.get(self.config.dataset.fields.id_field, file_id),
                        "_source": doc
                    }
                except Exception as e:
                    logger.error(f"Error processing document {file_id}: {e}")
                    continue
        
        # Perform bulk indexing
        success_count = 0
        error_count = 0
        
        for success, info in helpers.parallel_bulk(
            self.es,
            generate_actions(),
            chunk_size=self.config.datastore.bulk_size,
            request_timeout=self.config.datastore.bulk_timeout
        ):
            if success:
                success_count += 1
            else:
                error_count += 1
                logger.error(f"Failed to index document: {info}")
            
            if self.config.indexing.show_progress and success_count % 1000 == 0:
                logger.info(f"Indexed {success_count} documents...")
        
        if error_count > 0:
            logger.warning(f"Failed to index {error_count} documents")
        
        # Refresh index
        self.es.indices.refresh(index=index_id)
        
        return success_count
    
    def load_index(self, serialized_index_dump: str) -> None:
        """
        Load an already created index into memory.
        
        Args:
            serialized_index_dump: Path to dump of serialized index
        """
        # For Elasticsearch, indices are always "loaded" via the client
        # This method mainly validates that the index exists
        if not self.es.indices.exists(index=serialized_index_dump):
            raise ValueError(f"Index {serialized_index_dump} does not exist")
        
        logger.info(f"Index {serialized_index_dump} is ready")
    
    def update_index(
        self,
        index_id: str,
        remove_files: Iterable[tuple[str, str]],
        add_files: Iterable[tuple[str, str]]
    ) -> None:
        """
        Update an index by removing and adding files.
        
        Args:
            index_id: The unique identifier for the index
            remove_files: Iterable of (file_id, file_content) tuples to remove
            add_files: Iterable of (file_id, file_content) tuples to add
        """
        logger.info(f"Updating index: {index_id}")
        
        # Remove documents
        remove_count = 0
        for file_id, _ in remove_files:
            try:
                self.es.delete(index=index_id, id=file_id)
                remove_count += 1
            except Exception as e:
                logger.error(f"Error removing document {file_id}: {e}")
        
        logger.info(f"Removed {remove_count} documents")
        
        # Add documents
        add_count = self._bulk_index_documents(index_id, add_files)
        logger.info(f"Added {add_count} documents")
        
        # Update metadata
        if index_id in self.indices_metadata:
            self.indices_metadata[index_id]["updated_at"] = datetime.now().isoformat()
            self.indices_metadata[index_id]["document_count"] += (add_count - remove_count)
            self._save_metadata()
    
    def query(self, query: str, index_id: str = None, max_results: int = None) -> str:
        """
        Query the index and return results as JSON string.
        
        Args:
            index_id: Index identifier
            query: Query string
            max_results: Maximum number of results to return
            
        Returns:
            JSON string with results
        """
        if index_id is None:
            raise ValueError("index_id is required for Elasticsearch queries")
        if max_results is None:
            max_results = self.config.query.max_results
        
        # Preprocess query
        processed_query = self.preprocessor.preprocess_query(query)
        
        # Build Elasticsearch query
        es_query = self._build_es_query(processed_query)
        
        # Execute search
        try:
            response = self.es.search(
                index=index_id,
                body={
                    "query": es_query,
                    "size": max_results,
                    "explain": self.config.query.explain
                }
            )
            
            # Format results
            results = {
                "query": query,
                "processed_query": processed_query,
                "total_hits": response['hits']['total']['value'],
                "max_score": response['hits']['max_score'],
                "documents": []
            }
            
            for hit in response['hits']['hits']:
                doc = {
                    "id": hit['_id'],
                    "score": hit['_score'],
                    "source": hit['_source']
                }
                if self.config.query.explain:
                    doc['explanation'] = hit.get('_explanation')
                results["documents"].append(doc)
            
            return json.dumps(results, indent=2)
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            return json.dumps({"error": str(e)})
    
    def _build_es_query(self, query: str) -> dict:
        """
        Build Elasticsearch query from processed query string.
        
        Args:
            query: Preprocessed query string
            
        Returns:
            Elasticsearch query dict
        """
        from src.utils.query_parser import BooleanQueryParser
        
        # Initialize parser
        parser = BooleanQueryParser()
        
        # Fields to search (with boosting)
        fields = ["title^2", "text", "preprocessed_title^2", "preprocessed_text"]
        
        # Parse and convert to Elasticsearch query
        try:
            es_query = parser.parse_to_elasticsearch(query, fields)
            logger.debug(f"Parsed query '{query}' to ES query: {es_query}")
            return es_query
        except Exception as e:
            logger.error(f"Error parsing query '{query}': {e}")
            # Fallback to simple multi_match
            return {
                "multi_match": {
                    "query": query,
                    "fields": fields,
                    "operator": self.config.query.default_operator.lower()
                }
            }
    
    def delete_index(self, index_id: str) -> None:
        """
        Delete the index with the given index_id.
        
        Args:
            index_id: Index identifier
        """
        if self.es.indices.exists(index=index_id):
            self.es.indices.delete(index=index_id)
            logger.info(f"Deleted index: {index_id}")
            
            # Remove from metadata
            if index_id in self.indices_metadata:
                del self.indices_metadata[index_id]
                self._save_metadata()
        else:
            logger.warning(f"Index {index_id} does not exist")
    
    def list_indices(self) -> List[str]:
        """
        List all indices.
        
        Returns:
            List of index identifiers
        """
        # Get all indices from Elasticsearch
        indices = self.es.indices.get_alias(index="*")
        return list(indices.keys())
    
    def list_indexed_files(self, index_id: str) -> List[str]:
        """
        List all files indexed in the given index.
        
        Args:
            index_id: Index identifier
            
        Returns:
            List of file IDs
        """
        try:
            # Scroll through all documents to get IDs
            file_ids = []
            
            response = self.es.search(
                index=index_id,
                body={
                    "query": {"match_all": {}},
                    "_source": False,
                    "size": 10000
                },
                scroll='2m'
            )
            
            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']
            
            while hits:
                file_ids.extend([hit['_id'] for hit in hits])
                
                response = self.es.scroll(scroll_id=scroll_id, scroll='2m')
                scroll_id = response['_scroll_id']
                hits = response['hits']['hits']
            
            # Clear scroll
            self.es.clear_scroll(scroll_id=scroll_id)
            
            return file_ids
            
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []


import re