"""
Core inverted index data structure for SelfIndex.
"""

from typing import Dict, List, Optional, Set
from collections import defaultdict
import logging
import math

from .postings import PostingsList, PostingsListWithSkips
from .compression import CompressionManager

logger = logging.getLogger(__name__)


class InvertedIndex:
    """
    Core inverted index structure.
    Maps terms to postings lists.
    """
    
    def __init__(self, use_skip_pointers: bool = False, compression_type: str = 'NONE'):
        """
        Initialize inverted index.
        
        Args:
            use_skip_pointers: Whether to use skip pointers (optimization)
            compression_type: Type of compression to use (NONE, CODE, CLIB)
        """
        # Term -> PostingsList mapping
        self.dictionary: Dict[str, PostingsList] = {}
        
        # Configuration
        self.use_skip_pointers = use_skip_pointers
        self.compression_type = compression_type
        
        # Compression manager
        if compression_type == 'CODE':
            self.compression_manager = CompressionManager(compression_type='CODE')
        else:
            self.compression_manager = None

        # Statistics
        self.num_documents = 0
        self.num_terms = 0
        self.total_tokens = 0
    
    def add_document(self, doc_id: int, tokens: List[str]):
        """
        Add a document to the index.
        
        Args:
            doc_id: Document identifier (integer)
            tokens: List of tokens (terms) in the document
        """
        # Track unique terms in this document
        term_positions = defaultdict(list)
        
        # Collect positions for each term
        for position, token in enumerate(tokens):
            term_positions[token].append(position)
        
        # Add to postings lists
        for term, positions in term_positions.items():
            if term not in self.dictionary:
                if self.use_skip_pointers:
                    self.dictionary[term] = PostingsListWithSkips()
                else:
                    self.dictionary[term] = PostingsList()
                self.num_terms += 1
            
            # Add all positions at once (more efficient)
            self.dictionary[term].add_posting_batch(doc_id, positions)
        
        # Update statistics
        self.num_documents += 1
        self.total_tokens += len(tokens)
    
    def get_postings(self, term: str) -> Optional[PostingsList]:
        """
        Get postings list for a term.
        
        Args:
            term: The term to look up
            
        Returns:
            PostingsList if term exists, None otherwise
        """
        return self.dictionary.get(term)
    
    def get_document_frequency(self, term: str) -> int:
        """
        Get document frequency (number of documents containing term).
        
        Args:
            term: The term to look up
            
        Returns:
            Number of documents containing the term
        """
        postings = self.get_postings(term)
        return postings.document_frequency() if postings else 0
    
    def get_term_frequency(self, term: str, doc_id: int) -> int:
        """
        Get term frequency in a specific document.
        
        Args:
            term: The term to look up
            doc_id: Document identifier
            
        Returns:
            Number of times term appears in document
        """
        postings = self.get_postings(term)
        return postings.get_term_frequency(doc_id) if postings else 0
    
    def get_positions(self, term: str, doc_id: int) -> List[int]:
        """
        Get positions of a term in a specific document.
        
        Args:
            term: The term to look up
            doc_id: Document identifier
            
        Returns:
            List of positions where term appears
        """
        postings = self.get_postings(term)
        return postings.get_positions(doc_id) if postings else []
    
    def contains_term(self, term: str) -> bool:
        """Check if term exists in vocabulary."""
        return term in self.dictionary
    
    def get_vocabulary(self) -> Set[str]:
        """Get all terms in the index."""
        return set(self.dictionary.keys())
    
    def get_vocabulary_size(self) -> int:
        """Get size of vocabulary (number of unique terms)."""
        return len(self.dictionary)
    
    def finalize(self):
        """
        Finalize index after all documents are added.
        Builds skip pointers if enabled.
        """
        if self.use_skip_pointers:
            logger.info("Building skip pointers...")
            for term, postings in self.dictionary.items():
                if isinstance(postings, PostingsListWithSkips):
                    postings.build_skip_pointers()
            logger.info("Skip pointers built")
    
    def get_statistics(self) -> Dict:
        """Get index statistics."""
        avg_postings_length = (
            sum(len(postings) for postings in self.dictionary.values()) / len(self.dictionary)
            if self.dictionary else 0
        )
        
        return {
            'num_documents': self.num_documents,
            'num_terms': self.num_terms,
            'vocabulary_size': len(self.dictionary),
            'total_tokens': self.total_tokens,
            'avg_document_length': self.total_tokens / self.num_documents if self.num_documents > 0 else 0,
            'avg_postings_length': avg_postings_length
        }
    
    def to_dict(self) -> dict:
        """Convert index to dictionary for serialization."""
        dictionary_data = {}
    
        for term, postings in self.dictionary.items():
            postings_dict = postings.to_dict()
            
            # Apply compression if enabled
            if self.compression_manager:
                postings_dict = self.compression_manager.compress_postings_list(postings_dict)
            
            dictionary_data[term] = postings_dict
        
        return {
            'dictionary': dictionary_data,
            'statistics': self.get_statistics(),
            'use_skip_pointers': self.use_skip_pointers,
            'compression_type': self.compression_type
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'InvertedIndex':
        """Create index from dictionary."""
        compression_type = data.get('compression_type', 'NONE')
        index = cls(
            use_skip_pointers=data['use_skip_pointers'],
            compression_type=compression_type
        )

        # Initialize compression manager for decompression
        if compression_type == 'CODE':
            compression_manager = CompressionManager(compression_type='CODE')
        else:
            compression_manager = None
        
        # Restore dictionary
        for term, postings_data in data['dictionary'].items():
            # Decompress if needed
            if compression_manager and postings_data.get('compressed', False):
                postings_data = compression_manager.decompress_postings_list(postings_data)

            if index.use_skip_pointers:
                postings = PostingsListWithSkips()
            else:
                postings = PostingsList()
            
            postings.postings = [
                PostingEntry.from_dict(p) for p in postings_data['postings']
            ]
            index.dictionary[term] = postings
        
        # Restore statistics
        stats = data['statistics']
        index.num_documents = stats['num_documents']
        index.num_terms = stats['num_terms']
        index.total_tokens = stats['total_tokens']
        
        # Rebuild skip pointers if needed
        if index.use_skip_pointers:
            for postings in index.dictionary.values():
                if isinstance(postings, PostingsListWithSkips):
                    postings.build_skip_pointers()
        
        return index


class DocumentStore:
    """
    Store document metadata.
    Separate from inverted index for efficiency.
    """
    
    def __init__(self):
        """Initialize document store."""
        self.documents: Dict[int, Dict] = {}
        self._doc_id_to_internal: Dict[str, int] = {}  # Map external ID to internal ID
        self._internal_to_doc_id: Dict[int, str] = {}  # Map internal ID to external ID
        self._next_internal_id = 0
    
    def add_document(self, doc_id: str, metadata: Dict) -> int:
        """
        Add document metadata.
        
        Args:
            doc_id: External document identifier (string)
            metadata: Document metadata (title, url, length, etc.)
            
        Returns:
            Internal document ID (integer)
        """
        # Get or create internal ID
        if doc_id in self._doc_id_to_internal:
            internal_id = self._doc_id_to_internal[doc_id]
        else:
            internal_id = self._next_internal_id
            self._doc_id_to_internal[doc_id] = internal_id
            self._internal_to_doc_id[internal_id] = doc_id
            self._next_internal_id += 1
        
        # Store metadata with internal ID
        self.documents[internal_id] = {
            'doc_id': doc_id,
            'length': metadata.get('length', 0),
            'title': metadata.get('title', ''),
            'url': metadata.get('url', ''),
            **metadata
        }
        
        return internal_id
    
    def get_document(self, internal_id: int) -> Optional[Dict]:
        """Get document metadata by internal ID."""
        return self.documents.get(internal_id)
    
    def get_external_id(self, internal_id: int) -> Optional[str]:
        """Get external document ID from internal ID."""
        return self._internal_to_doc_id.get(internal_id)
    
    def get_internal_id(self, doc_id: str) -> Optional[int]:
        """Get internal ID from external document ID."""
        return self._doc_id_to_internal.get(doc_id)
    
    def get_document_length(self, internal_id: int) -> int:
        """Get document length (number of tokens)."""
        doc = self.get_document(internal_id)
        return doc['length'] if doc else 0
    
    def get_all_internal_ids(self) -> Set[int]:
        """Get set of all internal document IDs."""
        return set(self.documents.keys())
    
    def get_document_count(self) -> int:
        """Get total number of documents."""
        return len(self.documents)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'documents': self.documents,
            'doc_id_to_internal': self._doc_id_to_internal,
            'internal_to_doc_id': self._internal_to_doc_id,
            'next_internal_id': self._next_internal_id
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DocumentStore':
        """Create from dictionary."""
        store = cls()
        store.documents = data['documents']
        store._doc_id_to_internal = data['doc_id_to_internal']
        store._internal_to_doc_id = {int(k): v for k, v in data['internal_to_doc_id'].items()}
        store._next_internal_id = data['next_internal_id']
        return store


class CollectionStatistics:
    """
    Store collection-level statistics for ranking.
    Used for TF-IDF and BM25 calculations.
    """
    
    def __init__(self):
        """Initialize collection statistics."""
        self.num_documents = 0
        self.total_terms = 0
        self.avg_document_length = 0.0
        self.document_lengths: Dict[int, int] = {}
        
        # IDF cache
        self._idf_cache: Dict[str, float] = {}
    
    def add_document(self, doc_id: int, doc_length: int):
        """
        Add document statistics.
        
        Args:
            doc_id: Internal document ID
            doc_length: Number of tokens in document
        """
        self.document_lengths[doc_id] = doc_length
        self.num_documents += 1
        self.total_terms += doc_length
        
        # Recalculate average
        self.avg_document_length = self.total_terms / self.num_documents
        
        # Invalidate IDF cache
        self._idf_cache.clear()
    
    def calculate_idf(self, term: str, document_frequency: int) -> float:
        """
        Calculate IDF (Inverse Document Frequency) for a term.
        
        IDF = log(N / df)
        where N is total documents and df is document frequency
        
        Args:
            term: The term
            document_frequency: Number of documents containing the term
            
        Returns:
            IDF score
        """
        if term in self._idf_cache:
            return self._idf_cache[term]
        
        if document_frequency == 0 or self.num_documents == 0:
            idf = 0.0
        else:
            idf = math.log(self.num_documents / document_frequency)
        
        self._idf_cache[term] = idf
        return idf
    
    def calculate_bm25_score(
        self,
        term_frequency: int,
        document_frequency: int,
        doc_length: int,
        k1: float = 1.5,
        b: float = 0.75
    ) -> float:
        """
        Calculate BM25 score for a term in a document.
        
        BM25 = IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (DL / avgDL)))
        
        Args:
            term_frequency: Term frequency in document
            document_frequency: Document frequency of term
            doc_length: Length of current document
            k1: BM25 parameter (default: 1.5)
            b: BM25 parameter (default: 0.75)
            
        Returns:
            BM25 score
        """
        if term_frequency == 0:
            return 0.0
        
        # Calculate IDF
        idf = self.calculate_idf('', document_frequency)
        
        # Calculate normalized document length
        norm_doc_length = doc_length / self.avg_document_length if self.avg_document_length > 0 else 1.0
        
        # Calculate BM25
        numerator = term_frequency * (k1 + 1)
        denominator = term_frequency + k1 * (1 - b + b * norm_doc_length)
        
        return idf * (numerator / denominator)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'num_documents': self.num_documents,
            'total_terms': self.total_terms,
            'avg_document_length': self.avg_document_length,
            'document_lengths': self.document_lengths
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CollectionStatistics':
        """Create from dictionary."""
        stats = cls()
        stats.num_documents = data['num_documents']
        stats.total_terms = data['total_terms']
        stats.avg_document_length = data['avg_document_length']
        stats.document_lengths = {int(k): v for k, v in data['document_lengths'].items()}
        return stats


# Fix import for PostingEntry
from .postings import PostingEntry