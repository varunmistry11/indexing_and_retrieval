"""
Unit tests for Phase 2: Boolean Operations & Query Processing
Run with: pytest tests/test_phase2_query_processing.py -v
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.selfindex import (
    InvertedIndex, DocumentStore, CollectionStatistics,
    PostingsList, BooleanOperations, PhraseQueryProcessor,
    TermAtATimeProcessor, DocumentAtATimeProcessor, BooleanQueryProcessor
)


class TestBooleanOperations:
    """Test boolean operations on postings lists."""
    
    def setup_method(self):
        """Setup test data before each test."""
        # Create test postings lists
        self.list1 = PostingsList()
        self.list1.add_posting_batch(doc_id=1, positions=[1, 2])
        self.list1.add_posting_batch(doc_id=3, positions=[5])
        self.list1.add_posting_batch(doc_id=5, positions=[10])
        
        self.list2 = PostingsList()
        self.list2.add_posting_batch(doc_id=2, positions=[3])
        self.list2.add_posting_batch(doc_id=3, positions=[6])
        self.list2.add_posting_batch(doc_id=4, positions=[8])
    
    def test_intersect_empty_lists(self):
        """Test intersection with empty lists."""
        empty = PostingsList()
        result = BooleanOperations.intersect(empty, self.list1)
        assert len(result) == 0
    
    def test_intersect_no_overlap(self):
        """Test intersection with no overlapping documents."""
        list_a = PostingsList()
        list_a.add_posting(doc_id=1, position=1)
        list_a.add_posting(doc_id=2, position=2)
        
        list_b = PostingsList()
        list_b.add_posting(doc_id=3, position=3)
        list_b.add_posting(doc_id=4, position=4)
        
        result = BooleanOperations.intersect(list_a, list_b)
        assert len(result) == 0
    
    def test_intersect_with_overlap(self):
        """Test intersection with overlapping documents."""
        result = BooleanOperations.intersect(self.list1, self.list2)
        
        # Only doc_id=3 is in both lists
        assert len(result) == 1
        assert result.get_doc_ids() == [3]
        
        # Positions should be merged
        positions = result.get_positions(3)
        assert sorted(positions) == [5, 6]
    
    def test_intersect_many(self):
        """Test intersection of multiple lists."""
        list3 = PostingsList()
        list3.add_posting_batch(doc_id=3, positions=[7])
        list3.add_posting_batch(doc_id=5, positions=[11])
        
        result = BooleanOperations.intersect_many([self.list1, self.list2, list3])
        
        # Only doc_id=3 is in all three lists
        assert len(result) == 1
        assert result.get_doc_ids() == [3]
    
    def test_union_empty_lists(self):
        """Test union with empty lists."""
        empty = PostingsList()
        result = BooleanOperations.union(empty, self.list1)
        assert len(result) == len(self.list1)
    
    def test_union_no_overlap(self):
        """Test union with no overlapping documents."""
        list_a = PostingsList()
        list_a.add_posting(doc_id=1, position=1)
        list_a.add_posting(doc_id=2, position=2)
        
        list_b = PostingsList()
        list_b.add_posting(doc_id=3, position=3)
        list_b.add_posting(doc_id=4, position=4)
        
        result = BooleanOperations.union(list_a, list_b)
        assert len(result) == 4
        assert result.get_doc_ids() == [1, 2, 3, 4]
    
    def test_union_with_overlap(self):
        """Test union with overlapping documents."""
        result = BooleanOperations.union(self.list1, self.list2)
        
        # Should have all documents from both lists
        assert len(result) == 5
        assert result.get_doc_ids() == [1, 2, 3, 4, 5]
        
        # Doc 3 positions should be merged
        positions = result.get_positions(3)
        assert sorted(positions) == [5, 6]
    
    def test_union_many(self):
        """Test union of multiple lists."""
        list3 = PostingsList()
        list3.add_posting(doc_id=6, position=12)
        
        result = BooleanOperations.union_many([self.list1, self.list2, list3])
        
        assert len(result) == 6
        assert result.get_doc_ids() == [1, 2, 3, 4, 5, 6]
    
    def test_negate(self):
        """Test negation operation."""
        all_docs = {1, 2, 3, 4, 5, 6, 7}
        result = BooleanOperations.negate(self.list1, all_docs)
        
        # Should return docs not in list1
        assert result.get_doc_ids() == [2, 4, 6, 7]
    
    def test_and_not(self):
        """Test AND NOT operation."""
        result = BooleanOperations.and_not(self.list1, self.list2)
        
        # list1 has [1, 3, 5], list2 has [2, 3, 4]
        # Result should be [1, 5] (docs in list1 but not in list2)
        assert result.get_doc_ids() == [1, 5]


class TestPhraseQueryProcessor:
    """Test phrase query processing."""
    
    def setup_method(self):
        """Setup test data."""
        # Create postings for phrase "natural language processing"
        self.natural = PostingsList()
        self.natural.add_posting_batch(doc_id=1, positions=[0, 10])
        self.natural.add_posting_batch(doc_id=2, positions=[5])
        
        self.language = PostingsList()
        self.language.add_posting_batch(doc_id=1, positions=[1, 11])
        self.language.add_posting_batch(doc_id=2, positions=[8])
        
        self.processing = PostingsList()
        self.processing.add_posting_batch(doc_id=1, positions=[2, 12])
        self.processing.add_posting_batch(doc_id=3, positions=[3])
    
    def test_phrase_query_simple(self):
        """Test simple phrase query."""
        result = PhraseQueryProcessor.phrase_query([self.natural, self.language])
        
        # "natural language" appears at positions 0-1 and 10-11 in doc 1
        assert len(result) == 1
        assert result.get_doc_ids() == [1]
        assert result.get_positions(1) == [0, 10]
    
    def test_phrase_query_three_terms(self):
        """Test phrase query with three terms."""
        result = PhraseQueryProcessor.phrase_query([
            self.natural, self.language, self.processing
        ])
        
        # "natural language processing" appears at positions 0-2 and 10-12 in doc 1
        assert len(result) == 1
        assert result.get_doc_ids() == [1]
        assert result.get_positions(1) == [0, 10]
    
    def test_phrase_query_no_match(self):
        """Test phrase query with no matches."""
        # Doc 2 has "natural" at 5, we need a non-consecutive position
        other_term = PostingsList()
        other_term.add_posting(doc_id=2, position=7)  # Not consecutive with position 5
        
        result = PhraseQueryProcessor.phrase_query([self.natural, other_term])
        
        # No consecutive occurrences
        assert len(result) == 0

class TestTermAtATimeProcessor:
    """Test term-at-a-time query processing."""
    
    def setup_method(self):
        """Setup test index."""
        self.index = InvertedIndex()
        self.doc_store = DocumentStore()
        self.stats = CollectionStatistics()
        
        # Add test documents
        docs = [
            (0, ["cat", "dog", "pet"]),
            (1, ["dog", "pet", "animal"]),
            (2, ["cat", "pet", "animal"]),
        ]
        
        for doc_id, tokens in docs:
            self.index.add_document(doc_id, tokens)
            self.doc_store.add_document(f"doc{doc_id}", {'length': len(tokens)})
            self.stats.add_document(doc_id, len(tokens))
        
        self.processor = TermAtATimeProcessor(self.index, self.stats, scoring_method='boolean')
    
    def test_single_term_query(self):
        """Test query with single term."""
        results = self.processor.process_query(["cat"], k=10)
        
        # "cat" appears in docs 0 and 2
        assert len(results) == 2
        doc_ids = [r.doc_id for r in results]
        assert set(doc_ids) == {0, 2}
    
    def test_multi_term_query(self):
        """Test query with multiple terms."""
        results = self.processor.process_query(["cat", "dog"], k=10)
        
        # All docs should match (each has at least one term)
        assert len(results) == 3
        
        # Doc 0 should score highest (has both terms)
        assert results[0].doc_id == 0
    
    def test_no_results(self):
        """Test query with no matching documents."""
        results = self.processor.process_query(["nonexistent"], k=10)
        assert len(results) == 0
    
    def test_top_k_limiting(self):
        """Test that only top-k results are returned."""
        results = self.processor.process_query(["pet"], k=2)
        
        # "pet" appears in all 3 docs, but we only want top 2
        assert len(results) == 2
    
    def test_tf_scoring(self):
        """Test with TF scoring."""
        processor = TermAtATimeProcessor(self.index, self.stats, scoring_method='tf')
        results = processor.process_query(["pet"], k=10)
        
        # All docs have "pet" once, so scores should be equal
        assert len(results) == 3
        scores = [r.score for r in results]
        assert all(s == scores[0] for s in scores)
    
    def test_bm25_scoring(self):
        """Test with BM25 scoring."""
        processor = TermAtATimeProcessor(self.index, self.stats, scoring_method='bm25')
        results = processor.process_query(["cat"], k=10)
        
        # Should get BM25 scores
        assert len(results) == 2
        assert all(r.score > 0 for r in results)


class TestDocumentAtATimeProcessor:
    """Test document-at-a-time query processing."""
    
    def setup_method(self):
        """Setup test index."""
        self.index = InvertedIndex()
        self.doc_store = DocumentStore()
        self.stats = CollectionStatistics()
        
        # Add test documents
        docs = [
            (0, ["python", "programming", "language"]),
            (1, ["java", "programming", "language"]),
            (2, ["python", "java", "comparison"]),
        ]
        
        for doc_id, tokens in docs:
            self.index.add_document(doc_id, tokens)
            self.doc_store.add_document(f"doc{doc_id}", {'length': len(tokens)})
            self.stats.add_document(doc_id, len(tokens))
        
        self.processor = DocumentAtATimeProcessor(self.index, self.stats, scoring_method='boolean')
    
    def test_single_term_query(self):
        """Test single term query."""
        results = self.processor.process_query(["python"], k=10)
        
        # "python" in docs 0 and 2
        assert len(results) == 2
        doc_ids = [r.doc_id for r in results]
        assert set(doc_ids) == {0, 2}
    
    def test_multi_term_query(self):
        """Test multi-term query."""
        results = self.processor.process_query(["python", "java"], k=10)
        
        # All docs match at least one term
        assert len(results) == 3
        
        # Doc 2 has both terms, should score highest
        assert results[0].doc_id == 2
    
    def test_query_with_no_results(self):
        """Test query returning no results."""
        results = self.processor.process_query(["nonexistent"], k=10)
        assert len(results) == 0


class TestBooleanQueryProcessor:
    """Test boolean query processor."""
    
    def setup_method(self):
        """Setup test index."""
        self.index = InvertedIndex()
        
        # Add test documents
        docs = [
            (0, ["machine", "learning", "python"]),
            (1, ["deep", "learning", "neural"]),
            (2, ["machine", "learning", "neural"]),
            (3, ["python", "programming"]),
        ]
        
        for doc_id, tokens in docs:
            self.index.add_document(doc_id, tokens)
        
        self.all_docs = {0, 1, 2, 3}
        self.processor = BooleanQueryProcessor(self.index, self.all_docs)
    
    def test_and_query(self):
        """Test AND query."""
        result = self.processor.process_and_query(["machine", "learning"])
        
        # Docs 0 and 2 have both terms
        assert result.get_doc_ids() == [0, 2]
    
    def test_and_query_no_results(self):
        """Test AND query with no results."""
        result = self.processor.process_and_query(["machine", "deep"])
        
        # No document has both terms
        assert len(result) == 0
    
    def test_or_query(self):
        """Test OR query."""
        result = self.processor.process_or_query(["python", "neural"])
        
        # Docs with "python" or "neural"
        doc_ids = result.get_doc_ids()
        assert set(doc_ids) == {0, 1, 2, 3}
    
    def test_not_query(self):
        """Test NOT query."""
        result = self.processor.process_not_query("python")
        
        # Docs without "python" (1 and 2)
        doc_ids = result.get_doc_ids()
        assert set(doc_ids) == {1, 2}
    
    def test_and_not_query(self):
        """Test AND NOT query."""
        result = self.processor.process_and_not_query(
            positive_terms=["learning"],
            negative_terms=["python"]
        )
        
        # "learning" AND NOT "python"
        # Docs 1 and 2 have "learning", doc 0 also has "python"
        doc_ids = result.get_doc_ids()
        assert set(doc_ids) == {1, 2}
    
    def test_phrase_query(self):
        """Test phrase query."""
        # Add document with phrase
        self.index.add_document(doc_id=4, tokens=["machine", "learning", "algorithm"])
        self.all_docs.add(4)
        
        result = self.processor.process_phrase_query(["machine", "learning"])
        
        # Docs 0, 2, and 4 have "machine learning" as consecutive terms
        doc_ids = result.get_doc_ids()
        assert 0 in doc_ids
        assert 2 in doc_ids
        assert 4 in doc_ids
    
    def test_complex_query_term(self):
        """Test complex query with single term."""
        query_tree = {
            'type': 'TERM',
            'value': 'python'
        }
        
        result = self.processor.process_complex_query(query_tree)
        doc_ids = result.get_doc_ids()
        assert set(doc_ids) == {0, 3}
    
    def test_complex_query_and(self):
        """Test complex query with AND."""
        query_tree = {
            'type': 'AND',
            'children': [
                {'type': 'TERM', 'value': 'machine'},
                {'type': 'TERM', 'value': 'learning'}
            ]
        }
        
        result = self.processor.process_complex_query(query_tree)
        assert result.get_doc_ids() == [0, 2]
    
    def test_complex_query_or(self):
        """Test complex query with OR."""
        query_tree = {
            'type': 'OR',
            'children': [
                {'type': 'TERM', 'value': 'deep'},
                {'type': 'TERM', 'value': 'python'}
            ]
        }
        
        result = self.processor.process_complex_query(query_tree)
        doc_ids = result.get_doc_ids()
        assert set(doc_ids) == {0, 1, 3}
    
    def test_complex_query_not(self):
        """Test complex query with NOT."""
        query_tree = {
            'type': 'NOT',
            'children': [
                {'type': 'TERM', 'value': 'python'}
            ]
        }
        
        result = self.processor.process_complex_query(query_tree)
        doc_ids = result.get_doc_ids()
        assert set(doc_ids) == {1, 2}
    
    def test_complex_nested_query(self):
        """Test complex nested query."""
        # (machine OR deep) AND learning
        query_tree = {
            'type': 'AND',
            'children': [
                {
                    'type': 'OR',
                    'children': [
                        {'type': 'TERM', 'value': 'machine'},
                        {'type': 'TERM', 'value': 'deep'}
                    ]
                },
                {'type': 'TERM', 'value': 'learning'}
            ]
        }
        
        result = self.processor.process_complex_query(query_tree)
        doc_ids = result.get_doc_ids()
        # Docs 0, 1, 2 have (machine OR deep) AND learning
        assert set(doc_ids) == {0, 1, 2}


class TestIntegration:
    """Integration tests for complete query processing."""
    
    def test_complete_query_workflow(self):
        """Test complete workflow from indexing to querying."""
        # Create index
        index = InvertedIndex()
        doc_store = DocumentStore()
        stats = CollectionStatistics()
        
        # Add documents
        documents = [
            ("doc1", "the quick brown fox jumps over the lazy dog"),
            ("doc2", "the lazy dog sleeps all day"),
            ("doc3", "the quick fox is very clever"),
        ]
        
        for doc_id, text in documents:
            tokens = text.lower().split()
            internal_id = doc_store.add_document(doc_id, {'length': len(tokens)})
            index.add_document(internal_id, tokens)
            stats.add_document(internal_id, len(tokens))
        
        # Test term-at-a-time
        tat_processor = TermAtATimeProcessor(index, stats, scoring_method='bm25')
        results = tat_processor.process_query(["quick"], k=10)
        
        assert len(results) == 2  # docs 0 and 2
        assert all(r.score > 0 for r in results)
        
        # Test document-at-a-time
        dat_processor = DocumentAtATimeProcessor(index, stats, scoring_method='bm25')
        results = dat_processor.process_query(["lazy"], k=10)
        
        assert len(results) == 2  # docs 0 and 1
        
        # Test boolean queries
        all_docs = doc_store.get_all_internal_ids()
        bool_processor = BooleanQueryProcessor(index, all_docs)
        
        # AND query
        result = bool_processor.process_and_query(["quick", "fox"])
        assert len(result) == 2  # docs 0 and 2
        
        # Phrase query
        result = bool_processor.process_phrase_query(["lazy", "dog"])
        assert len(result) == 2  # docs 0 and 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])