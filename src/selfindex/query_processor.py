"""
Query processing engines: Term-at-a-time and Document-at-a-time.
"""

from typing import List, Dict, Tuple, Set
from abc import ABC, abstractmethod
import logging

from .inverted_index import InvertedIndex, CollectionStatistics
from .postings import PostingsList
from .boolean_ops import BooleanOperations, PhraseQueryProcessor

logger = logging.getLogger(__name__)


class QueryResult:
    """Container for query results with scores."""
    
    def __init__(self, doc_id: int, score: float):
        self.doc_id = doc_id
        self.score = score
    
    def __repr__(self):
        return f"QueryResult(doc_id={self.doc_id}, score={self.score:.4f})"
    
    def __lt__(self, other):
        # For sorting by score (descending)
        return self.score > other.score


class QueryProcessor(ABC):
    """Abstract base class for query processors."""
    
    def __init__(self, index: InvertedIndex, stats: CollectionStatistics):
        """
        Initialize query processor.
        
        Args:
            index: InvertedIndex to query
            stats: Collection statistics for scoring
        """
        self.index = index
        self.stats = stats
    
    @abstractmethod
    def process_query(self, query_terms: List[str], k: int = 10) -> List[QueryResult]:
        """
        Process a query and return top-k results.
        
        Args:
            query_terms: List of query terms
            k: Number of results to return
            
        Returns:
            List of QueryResult objects sorted by score
        """
        pass


class TermAtATimeProcessor(QueryProcessor):
    """
    Term-at-a-time query processing (q=T).
    Processes one term at a time, accumulating scores.
    """
    
    def __init__(self, index: InvertedIndex, stats: CollectionStatistics, 
                 scoring_method: str = 'bm25'):
        """
        Initialize term-at-a-time processor.
        
        Args:
            index: InvertedIndex to query
            stats: Collection statistics
            scoring_method: 'boolean', 'tf', or 'bm25'
        """
        super().__init__(index, stats)
        self.scoring_method = scoring_method
    
    def process_query(self, query_terms: List[str], k: int = 10) -> List[QueryResult]:
        """
        Process query using term-at-a-time algorithm.
        
        Algorithm:
        1. For each query term:
            a. Get postings list
            b. For each document in postings:
                - Calculate term score
                - Accumulate in document score
        2. Sort documents by score
        3. Return top-k
        
        Args:
            query_terms: List of query terms
            k: Number of results to return
            
        Returns:
            List of top-k QueryResult objects
        """
        # Accumulator for document scores
        scores: Dict[int, float] = {}
        
        # Process each term
        for term in query_terms:
            postings = self.index.get_postings(term)
            
            if not postings:
                continue
            
            df = postings.document_frequency()
            
            # Score each document containing this term
            for posting in postings:
                doc_id = posting.doc_id
                tf = posting.term_freq
                doc_length = self.stats.document_lengths.get(doc_id, 1)
                
                # Calculate score based on method
                if self.scoring_method == 'boolean':
                    term_score = 1.0
                elif self.scoring_method == 'tf':
                    term_score = tf
                elif self.scoring_method == 'bm25':
                    term_score = self.stats.calculate_bm25_score(tf, df, doc_length)
                else:
                    term_score = 1.0
                
                # Accumulate score
                if doc_id in scores:
                    scores[doc_id] += term_score
                else:
                    scores[doc_id] = term_score
        
        # Convert to QueryResult objects and sort
        results = [QueryResult(doc_id, score) for doc_id, score in scores.items()]
        results.sort()  # Uses __lt__ for descending score order
        
        return results[:k]


class DocumentAtATimeProcessor(QueryProcessor):
    """
    Document-at-a-time query processing (q=D).
    Processes all terms for one document at a time.
    """
    
    def __init__(self, index: InvertedIndex, stats: CollectionStatistics,
                 scoring_method: str = 'bm25'):
        """
        Initialize document-at-a-time processor.
        
        Args:
            index: InvertedIndex to query
            stats: Collection statistics
            scoring_method: 'boolean', 'tf', or 'bm25'
        """
        super().__init__(index, stats)
        self.scoring_method = scoring_method
    
    def process_query(self, query_terms: List[str], k: int = 10) -> List[QueryResult]:
        """
        Process query using document-at-a-time algorithm.
        
        Algorithm:
        1. Get postings lists for all query terms
        2. Find all candidate documents (union of all postings)
        3. For each document:
            a. Calculate score from all query terms
        4. Sort and return top-k
        
        Args:
            query_terms: List of query terms
            k: Number of results to return
            
        Returns:
            List of top-k QueryResult objects
        """
        # Get postings for all terms
        term_postings = []
        term_dfs = []
        
        for term in query_terms:
            postings = self.index.get_postings(term)
            if postings:
                term_postings.append(postings)
                term_dfs.append(postings.document_frequency())
        
        if not term_postings:
            return []
        
        # Get all candidate documents (union)
        candidate_docs = BooleanOperations.union_many(term_postings)
        
        results = []
        
        # Score each candidate document
        for posting in candidate_docs.postings:
            doc_id = posting.doc_id
            doc_score = 0.0
            doc_length = self.stats.document_lengths.get(doc_id, 1)
            
            # Calculate score from each query term
            for i, term in enumerate(query_terms):
                tf = term_postings[i].get_term_frequency(doc_id)
                
                if tf > 0:
                    if self.scoring_method == 'boolean':
                        term_score = 1.0
                    elif self.scoring_method == 'tf':
                        term_score = tf
                    elif self.scoring_method == 'bm25':
                        term_score = self.stats.calculate_bm25_score(
                            tf, term_dfs[i], doc_length
                        )
                    else:
                        term_score = 1.0
                    
                    doc_score += term_score
            
            results.append(QueryResult(doc_id, doc_score))
        
        # Sort by score and return top-k
        results.sort()
        return results[:k]


class BooleanQueryProcessor:
    """
    Process boolean queries with AND, OR, NOT operations.
    """
    
    def __init__(self, index: InvertedIndex, all_doc_ids: Set[int]):
        """
        Initialize boolean query processor.
        
        Args:
            index: InvertedIndex to query
            all_doc_ids: Set of all document IDs (for NOT operation)
        """
        self.index = index
        self.all_doc_ids = all_doc_ids
    
    def process_and_query(self, terms: List[str]) -> PostingsList:
        """Process AND query (all terms must be present)."""
        postings_lists = []
        
        for term in terms:
            postings = self.index.get_postings(term)
            if not postings or len(postings) == 0:
                # If any term not found, result is empty
                return PostingsList()
            postings_lists.append(postings)
        
        return BooleanOperations.intersect_many(postings_lists)
    
    def process_or_query(self, terms: List[str]) -> PostingsList:
        """Process OR query (at least one term must be present)."""
        postings_lists = []
        
        for term in terms:
            postings = self.index.get_postings(term)
            if postings and len(postings) > 0:
                postings_lists.append(postings)
        
        if not postings_lists:
            return PostingsList()
        
        return BooleanOperations.union_many(postings_lists)
    
    def process_not_query(self, term: str) -> PostingsList:
        """Process NOT query (term must not be present)."""
        postings = self.index.get_postings(term)
        
        if not postings:
            # Term not in index, return all documents
            result = PostingsList()
            for doc_id in sorted(self.all_doc_ids):
                result.add_posting_batch(doc_id, [])
            return result
        
        return BooleanOperations.negate(postings, self.all_doc_ids)
    
    def process_and_not_query(self, positive_terms: List[str], 
                              negative_terms: List[str]) -> PostingsList:
        """
        Process AND NOT query (positive terms AND NOT negative terms).
        
        Args:
            positive_terms: Terms that must be present
            negative_terms: Terms that must not be present
            
        Returns:
            PostingsList of matching documents
        """
        # Get documents containing positive terms
        positive_results = self.process_and_query(positive_terms)
        
        if len(positive_results) == 0:
            return PostingsList()
        
        # Get documents containing negative terms
        negative_results = self.process_or_query(negative_terms)
        
        if len(negative_results) == 0:
            return positive_results
        
        # Remove negative documents from positive results
        return BooleanOperations.and_not(positive_results, negative_results)
    
    def process_phrase_query(self, terms: List[str]) -> PostingsList:
        """
        Process phrase query (terms must appear consecutively).
        
        Args:
            terms: List of terms in phrase (in order)
            
        Returns:
            PostingsList of documents containing the phrase
        """
        # Get postings for each term
        term_postings = []
        
        for term in terms:
            postings = self.index.get_postings(term)
            if not postings or len(postings) == 0:
                return PostingsList()
            term_postings.append(postings)
        
        return PhraseQueryProcessor.phrase_query(term_postings)
    
    def process_complex_query(self, query_tree: dict) -> PostingsList:
        """
        Process complex boolean query from parsed query tree.
        
        Args:
            query_tree: Parsed query tree from QueryParser
            
        Returns:
            PostingsList of matching documents
        """
        query_type = query_tree.get('type')
        
        if query_type == 'TERM':
            # Simple term query
            term = query_tree['value']
            postings = self.index.get_postings(term)
            return postings if postings else PostingsList()
        
        elif query_type == 'PHRASE':
            # Phrase query
            phrase = query_tree['value']
            terms = phrase.split()
            return self.process_phrase_query(terms)
        
        elif query_type == 'AND':
            # AND operation
            children = query_tree['children']
            child_results = [self.process_complex_query(child) for child in children]
            return BooleanOperations.intersect_many(child_results)
        
        elif query_type == 'OR':
            # OR operation
            children = query_tree['children']
            child_results = [self.process_complex_query(child) for child in children]
            return BooleanOperations.union_many(child_results)
        
        elif query_type == 'NOT':
            # NOT operation
            child = query_tree['children'][0]
            child_result = self.process_complex_query(child)
            return BooleanOperations.negate(child_result, self.all_doc_ids)
        
        else:
            logger.warning(f"Unknown query type: {query_type}")
            return PostingsList()