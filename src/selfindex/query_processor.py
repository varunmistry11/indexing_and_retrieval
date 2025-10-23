"""
Query processing engines: Term-at-a-time and Document-at-a-time.
"""

from typing import List, Dict, Tuple, Set
from collections import deque
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

class TopKTracker:
    """
    Tracks top-k results and detects stabilization for early stopping.
    """
    
    def __init__(self, k: int, stability_window: int = 3):
        """
        Initialize tracker.
        
        Args:
            k: Number of top results to track
            stability_window: Number of iterations without change to consider stable
        """
        self.k = k
        self.stability_window = stability_window
        self.current_top_k = []
        self.history = deque(maxlen=stability_window)
    
    def update(self, results: List['QueryResult']):
        """
        Update with new results.
        
        Args:
            results: List of QueryResult objects (should be sorted by score)
        """
        # Get top-k doc IDs
        top_k_ids = frozenset(r.doc_id for r in results[:self.k])
        self.history.append(top_k_ids)
        self.current_top_k = results[:self.k]
    
    def has_stabilized(self) -> bool:
        """
        Check if top-k has stabilized.
        
        Returns:
            True if top-k hasn't changed for stability_window iterations
        """
        if len(self.history) < self.stability_window:
            return False
        
        # Check if all recent top-k sets are identical
        first = self.history[0]
        return all(s == first for s in self.history)
    
    def get_top_k(self) -> List['QueryResult']:
        """Get current top-k results."""
        return self.current_top_k

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
    
    def process_query_with_threshold(self, query_terms: List[str], k: int = 10, 
                                     threshold: float = 0.1) -> List[QueryResult]:
        """
        Process query using term-at-a-time with score thresholding.
        Only accumulates scores above threshold.
        
        Algorithm:
        1. For each query term:
            a. Get postings list
            b. For each document in postings:
                - Calculate term score
                - Only accumulate if score > threshold
        2. Sort documents by score
        3. Return top-k
        
        Args:
            query_terms: List of query terms
            k: Number of results to return
            threshold: Minimum score threshold (relative to max possible score)
            
        Returns:
            List of top-k QueryResult objects
        """
        # Accumulator for document scores
        scores: Dict[int, float] = {}
        
        # Calculate approximate max possible score per term for normalization
        max_scores = {}
        for term in query_terms:
            postings = self.index.get_postings(term)
            if postings:
                df = postings.document_frequency()
                # Estimate max score for this term
                if self.scoring_method == 'bm25':
                    # Approximate max BM25 score
                    max_tf = max(p.term_freq for p in postings)
                    max_scores[term] = self.stats.calculate_bm25_score(max_tf, df, 1)
                else:
                    max_scores[term] = df  # Simple approximation
        
        # Calculate absolute threshold
        max_possible = sum(max_scores.values()) if max_scores else 1.0
        abs_threshold = threshold * max_possible
        
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
                
                # Only accumulate if above threshold
                if term_score > abs_threshold / len(query_terms):  # Per-term threshold
                    if doc_id in scores:
                        scores[doc_id] += term_score
                    else:
                        scores[doc_id] = term_score
        
        # Convert to QueryResult objects and sort
        results = [QueryResult(doc_id, score) for doc_id, score in scores.items()]
        results.sort()  # Uses __lt__ for descending score order
        
        return results[:k]
    
    def process_query_with_early_stop(self, query_terms: List[str], k: int = 10,
                                      stability_window: int = 3) -> List[QueryResult]:
        """
        Process query using term-at-a-time with early stopping.
        Stops when top-k results stabilize.
        
        Algorithm:
        1. For each query term:
            a. Get postings and accumulate scores
            b. Check if top-k has stabilized
            c. Stop early if stable for N consecutive terms
        2. Return top-k
        
        Args:
            query_terms: List of query terms
            k: Number of results to return
            stability_window: Number of terms to check for stability
            
        Returns:
            List of top-k QueryResult objects
        """
        # Accumulator for document scores
        scores: Dict[int, float] = {}
        tracker = TopKTracker(k, stability_window)
        
        # Process each term
        for term_idx, term in enumerate(query_terms):
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
            
            # Update tracker and check for stability (after processing at least stability_window terms)
            if term_idx >= stability_window - 1:
                current_results = [QueryResult(doc_id, score) for doc_id, score in scores.items()]
                current_results.sort()
                tracker.update(current_results)
                
                if tracker.has_stabilized():
                    logger.info(f"Early stopping at term {term_idx + 1}/{len(query_terms)}")
                    break
        
        # Convert to QueryResult objects and sort
        results = [QueryResult(doc_id, score) for doc_id, score in scores.items()]
        results.sort()
        
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
    
    def process_query_with_threshold(self, query_terms: List[str], k: int = 10,
                                     threshold: float = 0.1) -> List[QueryResult]:
        """
        Process query using document-at-a-time with score thresholding.
        Skips documents unlikely to meet threshold.
        
        Args:
            query_terms: List of query terms
            k: Number of results to return
            threshold: Minimum score threshold (relative to max possible)
            
        Returns:
            List of top-k QueryResult objects
        """
        # Get postings for all terms
        term_postings = []
        term_dfs = []
        max_term_scores = []
        
        for term in query_terms:
            postings = self.index.get_postings(term)
            if postings:
                term_postings.append(postings)
                df = postings.document_frequency()
                term_dfs.append(df)
                
                # Calculate max possible score for this term
                if self.scoring_method == 'bm25':
                    max_tf = max(p.term_freq for p in postings)
                    max_score = self.stats.calculate_bm25_score(max_tf, df, 1)
                else:
                    max_score = df
                max_term_scores.append(max_score)
        
        if not term_postings:
            return []
        
        # Calculate absolute threshold
        max_possible = sum(max_term_scores)
        abs_threshold = threshold * max_possible
        
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
            
            # Only add if above threshold
            if doc_score >= abs_threshold:
                results.append(QueryResult(doc_id, doc_score))
        
        # Sort by score and return top-k
        results.sort()
        return results[:k]
    
    def process_query_with_early_stop(self, query_terms: List[str], k: int = 10,
                                      stability_window: int = 3) -> List[QueryResult]:
        """
        Process query using document-at-a-time with early stopping.
        Sorts candidates by potential max score and stops when remaining docs can't enter top-k.
        
        Args:
            query_terms: List of query terms
            k: Number of results to return
            stability_window: Not used in document-at-a-time (here for API consistency)
            
        Returns:
            List of top-k QueryResult objects
        """
        # Get postings for all terms
        term_postings = []
        term_dfs = []
        term_max_scores = []
        
        for term in query_terms:
            postings = self.index.get_postings(term)
            if postings:
                term_postings.append(postings)
                df = postings.document_frequency()
                term_dfs.append(df)
                
                # Calculate max score for this term
                if self.scoring_method == 'bm25':
                    max_tf = max(p.term_freq for p in postings)
                    max_score = self.stats.calculate_bm25_score(max_tf, df, 1)
                else:
                    max_score = df
                term_max_scores.append(max_score)
        
        if not term_postings:
            return []
        
        # Get candidate documents
        candidate_docs = BooleanOperations.union_many(term_postings)
        
        # Calculate upper bound score for each document
        doc_upper_bounds = {}
        for posting in candidate_docs.postings:
            doc_id = posting.doc_id
            upper_bound = 0.0
            
            # Sum max scores for terms present in this document
            for i, term_pl in enumerate(term_postings):
                if term_pl.get_term_frequency(doc_id) > 0:
                    upper_bound += term_max_scores[i]
            
            doc_upper_bounds[doc_id] = upper_bound
        
        # Sort candidates by upper bound (descending)
        sorted_candidates = sorted(doc_upper_bounds.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        min_top_k_score = 0.0
        
        # Process candidates in order of upper bound
        for doc_id, upper_bound in sorted_candidates:
            # Early stop: if this document's upper bound can't beat min top-k score
            if len(results) >= k and upper_bound <= min_top_k_score:
                logger.info(f"Early stopping: processed {len(results)} docs, skipped {len(sorted_candidates) - len(results)}")
                break
            
            # Calculate actual score
            doc_score = 0.0
            doc_length = self.stats.document_lengths.get(doc_id, 1)
            
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
            
            # Update min top-k score
            if len(results) >= k:
                results.sort()
                results = results[:k]
                min_top_k_score = results[-1].score if results else 0.0
        
        # Final sort and return top-k
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