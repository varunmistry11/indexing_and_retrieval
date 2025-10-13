"""
Unit tests for Phase 1: Core Data Structures
Run with: pytest tests/test_phase1_data_structures.py -v
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.selfindex.postings import PostingEntry, PostingsList, PostingsListWithSkips
from src.selfindex.inverted_index import InvertedIndex, DocumentStore, CollectionStatistics


class TestPostingEntry:
    """Test PostingEntry class."""
    
    def test_create_posting_entry(self):
        """Test creating a posting entry."""
        entry = PostingEntry(doc_id=1, term_freq=3, positions=[5, 10, 15])
        assert entry.doc_id == 1
        assert entry.term_freq == 3
        assert entry.positions == [5, 10, 15]
    
    def test_posting_entry_comparison(self):
        """Test posting entry comparison."""
        entry1 = PostingEntry(doc_id=1)
        entry2 = PostingEntry(doc_id=2)
        entry3 = PostingEntry(doc_id=1)
        
        assert entry1 < entry2
        assert not entry2 < entry1
        assert entry1 == entry3
    
    def test_posting_entry_serialization(self):
        """Test serialization and deserialization."""
        entry = PostingEntry(doc_id=5, term_freq=2, positions=[10, 20])
        data = entry.to_dict()
        
        assert data['doc_id'] == 5
        assert data['term_freq'] == 2
        assert data['positions'] == [10, 20]
        
        restored = PostingEntry.from_dict(data)
        assert restored.doc_id == entry.doc_id
        assert restored.term_freq == entry.term_freq
        assert restored.positions == entry.positions


class TestPostingsList:
    """Test PostingsList class."""
    
    def test_create_empty_postings_list(self):
        """Test creating an empty postings list."""
        pl = PostingsList()
        assert len(pl) == 0
        assert pl.document_frequency() == 0
    
    def test_add_single_posting(self):
        """Test adding a single posting."""
        pl = PostingsList()
        pl.add_posting(doc_id=1, position=5)
        
        assert len(pl) == 1
        assert pl.document_frequency() == 1
        assert pl.get_term_frequency(1) == 1
        assert pl.get_positions(1) == [5]
    
    def test_add_multiple_postings_same_doc(self):
        """Test adding multiple postings for same document."""
        pl = PostingsList()
        pl.add_posting(doc_id=1, position=5)
        pl.add_posting(doc_id=1, position=10)
        pl.add_posting(doc_id=1, position=15)
        
        assert len(pl) == 1  # Still one document
        assert pl.get_term_frequency(1) == 3
        assert pl.get_positions(1) == [5, 10, 15]
    
    def test_add_postings_multiple_docs(self):
        """Test adding postings for multiple documents."""
        pl = PostingsList()
        pl.add_posting(doc_id=1, position=5)
        pl.add_posting(doc_id=2, position=10)
        pl.add_posting(doc_id=3, position=15)
        
        assert len(pl) == 3
        assert pl.document_frequency() == 3
        assert pl.get_doc_ids() == [1, 2, 3]
    
    def test_add_posting_batch(self):
        """Test batch adding positions."""
        pl = PostingsList()
        pl.add_posting_batch(doc_id=1, positions=[5, 10, 15, 20])
        
        assert len(pl) == 1
        assert pl.get_term_frequency(1) == 4
        assert pl.get_positions(1) == [5, 10, 15, 20]
    
    def test_get_nonexistent_posting(self):
        """Test getting posting for non-existent document."""
        pl = PostingsList()
        pl.add_posting(doc_id=1, position=5)
        
        assert pl.get_posting(999) is None
        assert pl.get_term_frequency(999) == 0
        assert pl.get_positions(999) == []
    
    def test_total_term_frequency(self):
        """Test calculating total term frequency."""
        pl = PostingsList()
        pl.add_posting_batch(doc_id=1, positions=[5, 10, 15])
        pl.add_posting_batch(doc_id=2, positions=[20, 25])
        pl.add_posting_batch(doc_id=3, positions=[30])
        
        assert pl.total_term_frequency() == 6
    
    def test_postings_list_iteration(self):
        """Test iterating over postings list."""
        pl = PostingsList()
        pl.add_posting(doc_id=1, position=5)
        pl.add_posting(doc_id=2, position=10)
        pl.add_posting(doc_id=3, position=15)
        
        doc_ids = [p.doc_id for p in pl]
        assert doc_ids == [1, 2, 3]
    
    def test_postings_list_serialization(self):
        """Test serialization and deserialization."""
        pl = PostingsList()
        pl.add_posting_batch(doc_id=1, positions=[5, 10])
        pl.add_posting_batch(doc_id=2, positions=[15, 20, 25])
        
        data = pl.to_dict()
        restored = PostingsList.from_dict(data)
        
        assert len(restored) == len(pl)
        assert restored.get_doc_ids() == pl.get_doc_ids()
        assert restored.get_positions(1) == pl.get_positions(1)
        assert restored.get_positions(2) == pl.get_positions(2)
    
    def test_merge_postings_lists(self):
        """Test merging two postings lists."""
        pl1 = PostingsList()
        pl1.add_posting_batch(doc_id=1, positions=[5, 10])
        pl1.add_posting_batch(doc_id=3, positions=[15])
        
        pl2 = PostingsList()
        pl2.add_posting_batch(doc_id=2, positions=[20])
        pl2.add_posting_batch(doc_id=3, positions=[25, 30])
        
        merged = pl1.merge_with(pl2)
        
        assert len(merged) == 3
        assert merged.get_doc_ids() == [1, 2, 3]
        assert merged.get_positions(1) == [5, 10]
        assert merged.get_positions(2) == [20]
        assert merged.get_positions(3) == [15, 25, 30]  # Merged positions


class TestPostingsListWithSkips:
    """Test PostingsListWithSkips class."""
    
    def test_create_with_skip_pointers(self):
        """Test creating postings list with skip pointers."""
        pl = PostingsListWithSkips(skip_interval=2)
        assert pl.skip_interval == 2
        assert not pl.has_skips()
    
    def test_build_skip_pointers(self):
        """Test building skip pointers."""
        pl = PostingsListWithSkips(skip_interval=3)
        
        # Add 10 postings
        for i in range(1, 11):
            pl.add_posting(doc_id=i, position=i*10)
        
        pl.build_skip_pointers()
        
        assert pl.has_skips()
        assert len(pl.skip_pointers) > 0
    
    def test_automatic_skip_interval(self):
        """Test automatic skip interval calculation (sqrt(n))."""
        pl = PostingsListWithSkips()  # No interval specified
        
        # Add 16 postings (sqrt(16) = 4)
        for i in range(1, 17):
            pl.add_posting(doc_id=i, position=i*10)
        
        pl.build_skip_pointers()
        
        # Should create skip pointers at interval ~4
        assert pl.skip_interval == 4


class TestInvertedIndex:
    """Test InvertedIndex class."""
    
    def test_create_empty_index(self):
        """Test creating an empty index."""
        index = InvertedIndex()
        assert index.num_documents == 0
        assert index.num_terms == 0
        assert index.get_vocabulary_size() == 0
    
    def test_add_single_document(self):
        """Test adding a single document."""
        index = InvertedIndex()
        tokens = ["hello", "world", "hello"]
        
        index.add_document(doc_id=1, tokens=tokens)
        
        assert index.num_documents == 1
        assert index.contains_term("hello")
        assert index.contains_term("world")
        assert index.get_document_frequency("hello") == 1
        assert index.get_term_frequency("hello", 1) == 2
        assert index.get_term_frequency("world", 1) == 1
    
    def test_add_multiple_documents(self):
        """Test adding multiple documents."""
        index = InvertedIndex()
        
        index.add_document(doc_id=1, tokens=["cat", "dog", "cat"])
        index.add_document(doc_id=2, tokens=["dog", "bird"])
        index.add_document(doc_id=3, tokens=["cat", "bird", "fish"])
        
        assert index.num_documents == 3
        assert index.get_document_frequency("cat") == 2  # In docs 1 and 3
        assert index.get_document_frequency("dog") == 2  # In docs 1 and 2
        assert index.get_document_frequency("bird") == 2  # In docs 2 and 3
        assert index.get_document_frequency("fish") == 1  # Only in doc 3
    
    def test_get_positions(self):
        """Test getting term positions in documents."""
        index = InvertedIndex()
        tokens = ["the", "cat", "sat", "on", "the", "mat"]
        
        index.add_document(doc_id=1, tokens=tokens)
        
        positions_the = index.get_positions("the", 1)
        assert positions_the == [0, 4]  # "the" at positions 0 and 4
        
        positions_cat = index.get_positions("cat", 1)
        assert positions_cat == [1]
    
    def test_vocabulary(self):
        """Test getting vocabulary."""
        index = InvertedIndex()
        
        index.add_document(doc_id=1, tokens=["apple", "banana"])
        index.add_document(doc_id=2, tokens=["cherry", "apple"])
        
        vocab = index.get_vocabulary()
        assert vocab == {"apple", "banana", "cherry"}
        assert index.get_vocabulary_size() == 3
    
    def test_index_statistics(self):
        """Test getting index statistics."""
        index = InvertedIndex()
        
        index.add_document(doc_id=1, tokens=["a", "b", "c"])
        index.add_document(doc_id=2, tokens=["a", "b"])
        
        stats = index.get_statistics()
        
        assert stats['num_documents'] == 2
        assert stats['total_tokens'] == 5
        assert stats['avg_document_length'] == 2.5
        assert stats['vocabulary_size'] == 3
    
    def test_index_with_skip_pointers(self):
        """Test creating index with skip pointers."""
        index = InvertedIndex(use_skip_pointers=True)
        
        # Add multiple documents (at least 4 to get skip pointers)
        for doc_id in range(1, 10):
            index.add_document(doc_id=doc_id, tokens=["test"])
        
        index.finalize()
        
        postings = index.get_postings("test")
        assert isinstance(postings, PostingsListWithSkips)
        assert postings.has_skips()
    
    def test_index_serialization(self):
        """Test index serialization and deserialization."""
        index = InvertedIndex()
        
        index.add_document(doc_id=1, tokens=["hello", "world"])
        index.add_document(doc_id=2, tokens=["hello", "there"])
        
        # Serialize
        data = index.to_dict()
        
        # Deserialize
        restored = InvertedIndex.from_dict(data)
        
        assert restored.num_documents == index.num_documents
        assert restored.get_vocabulary() == index.get_vocabulary()
        assert restored.get_document_frequency("hello") == 2
        assert restored.get_term_frequency("hello", 1) == 1


class TestDocumentStore:
    """Test DocumentStore class."""
    
    def test_create_empty_store(self):
        """Test creating empty document store."""
        store = DocumentStore()
        assert store.get_document_count() == 0
    
    def test_add_document(self):
        """Test adding a document."""
        store = DocumentStore()
        
        metadata = {
            'title': 'Test Document',
            'url': 'http://example.com',
            'length': 100
        }
        
        internal_id = store.add_document(doc_id='doc1', metadata=metadata)
        
        assert internal_id == 0  # First document gets ID 0
        assert store.get_document_count() == 1
        
        doc = store.get_document(internal_id)
        assert doc['title'] == 'Test Document'
        assert doc['length'] == 100
    
    def test_add_multiple_documents(self):
        """Test adding multiple documents."""
        store = DocumentStore()
        
        id1 = store.add_document('doc1', {'title': 'Doc 1', 'length': 50})
        id2 = store.add_document('doc2', {'title': 'Doc 2', 'length': 75})
        id3 = store.add_document('doc3', {'title': 'Doc 3', 'length': 100})
        
        assert id1 == 0
        assert id2 == 1
        assert id3 == 2
        assert store.get_document_count() == 3
    
    def test_external_internal_id_mapping(self):
        """Test mapping between external and internal IDs."""
        store = DocumentStore()
        
        internal_id = store.add_document('external_doc_id', {'length': 50})
        
        assert store.get_external_id(internal_id) == 'external_doc_id'
        assert store.get_internal_id('external_doc_id') == internal_id
    
    def test_get_document_length(self):
        """Test getting document length."""
        store = DocumentStore()
        
        internal_id = store.add_document('doc1', {'length': 150})
        
        assert store.get_document_length(internal_id) == 150
    
    def test_get_all_internal_ids(self):
        """Test getting all internal IDs."""
        store = DocumentStore()
        
        id1 = store.add_document('doc1', {'length': 50})
        id2 = store.add_document('doc2', {'length': 75})
        id3 = store.add_document('doc3', {'length': 100})
        
        all_ids = store.get_all_internal_ids()
        assert all_ids == {id1, id2, id3}
    
    def test_store_serialization(self):
        """Test document store serialization."""
        store = DocumentStore()
        
        store.add_document('doc1', {'title': 'Test', 'length': 50})
        store.add_document('doc2', {'title': 'Test 2', 'length': 75})
        
        # Serialize
        data = store.to_dict()
        
        # Deserialize
        restored = DocumentStore.from_dict(data)
        
        assert restored.get_document_count() == store.get_document_count()
        assert restored.get_external_id(0) == 'doc1'
        assert restored.get_external_id(1) == 'doc2'


class TestCollectionStatistics:
    """Test CollectionStatistics class."""
    
    def test_create_empty_statistics(self):
        """Test creating empty statistics."""
        stats = CollectionStatistics()
        assert stats.num_documents == 0
        assert stats.avg_document_length == 0.0
    
    def test_add_documents(self):
        """Test adding document statistics."""
        stats = CollectionStatistics()
        
        stats.add_document(doc_id=0, doc_length=100)
        stats.add_document(doc_id=1, doc_length=200)
        stats.add_document(doc_id=2, doc_length=150)
        
        assert stats.num_documents == 3
        assert stats.total_terms == 450
        assert stats.avg_document_length == 150.0
    
    def test_calculate_idf(self):
        """Test IDF calculation."""
        stats = CollectionStatistics()
        
        # Add 10 documents
        for i in range(10):
            stats.add_document(doc_id=i, doc_length=100)
        
        # Term appears in 2 documents
        idf = stats.calculate_idf(term='test', document_frequency=2)
        
        import math
        expected_idf = math.log(10 / 2)
        assert abs(idf - expected_idf) < 0.001
    
    def test_idf_caching(self):
        """Test that IDF values are cached."""
        stats = CollectionStatistics()
        
        for i in range(100):
            stats.add_document(doc_id=i, doc_length=100)
        
        # Calculate IDF twice
        idf1 = stats.calculate_idf('term', 10)
        idf2 = stats.calculate_idf('term', 10)
        
        # Should be exactly the same (cached)
        assert idf1 == idf2
        assert 'term' in stats._idf_cache
    
    def test_calculate_bm25_score(self):
        """Test BM25 score calculation."""
        stats = CollectionStatistics()
        
        # Add documents
        for i in range(10):
            stats.add_document(doc_id=i, doc_length=100)
        
        # Calculate BM25 score
        score = stats.calculate_bm25_score(
            term_frequency=3,
            document_frequency=2,
            doc_length=100
        )
        
        # Score should be positive
        assert score > 0
    
    def test_bm25_zero_term_frequency(self):
        """Test BM25 with zero term frequency."""
        stats = CollectionStatistics()
        
        for i in range(10):
            stats.add_document(doc_id=i, doc_length=100)
        
        score = stats.calculate_bm25_score(
            term_frequency=0,
            document_frequency=5,
            doc_length=100
        )
        
        assert score == 0.0
    
    def test_statistics_serialization(self):
        """Test statistics serialization."""
        stats = CollectionStatistics()
        
        stats.add_document(doc_id=0, doc_length=100)
        stats.add_document(doc_id=1, doc_length=200)
        
        # Serialize
        data = stats.to_dict()
        
        # Deserialize
        restored = CollectionStatistics.from_dict(data)
        
        assert restored.num_documents == stats.num_documents
        assert restored.avg_document_length == stats.avg_document_length
        assert restored.document_lengths == stats.document_lengths


class TestIntegration:
    """Integration tests for all components working together."""
    
    def test_build_simple_index(self):
        """Test building a simple index with all components."""
        # Create components
        index = InvertedIndex()
        doc_store = DocumentStore()
        stats = CollectionStatistics()
        
        # Add documents
        docs = [
            ("doc1", ["the", "cat", "sat", "on", "the", "mat"]),
            ("doc2", ["the", "dog", "sat", "on", "the", "log"]),
            ("doc3", ["the", "cat", "and", "the", "dog"])
        ]
        
        for doc_id, tokens in docs:
            internal_id = doc_store.add_document(
                doc_id=doc_id,
                metadata={'length': len(tokens)}
            )
            index.add_document(doc_id=internal_id, tokens=tokens)
            stats.add_document(doc_id=internal_id, doc_length=len(tokens))
        
        # Verify index
        assert index.num_documents == 3
        assert index.get_document_frequency("the") == 3  # In all docs
        assert index.get_document_frequency("cat") == 2  # In doc1 and doc3
        assert index.get_document_frequency("dog") == 2  # In doc2 and doc3
        
        # Verify document store
        assert doc_store.get_document_count() == 3
        assert doc_store.get_external_id(0) == "doc1"
        
        # Verify statistics
        assert stats.num_documents == 3
        assert stats.avg_document_length == (6 + 6 + 5) / 3
    
    def test_phrase_query_setup(self):
        """Test that positions are correctly stored for phrase queries."""
        index = InvertedIndex()
        
        tokens = ["natural", "language", "processing", "is", "natural", "language"]
        index.add_document(doc_id=0, tokens=tokens)
        
        # Check positions
        natural_positions = index.get_positions("natural", 0)
        language_positions = index.get_positions("language", 0)
        
        assert natural_positions == [0, 4]
        assert language_positions == [1, 5]
        
        # For phrase "natural language", positions should be consecutive
        # Check that positions differ by 1
        for nat_pos in natural_positions:
            if nat_pos + 1 in language_positions:
                # Found phrase occurrence
                assert True
                return
        
        assert False, "Phrase 'natural language' not found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])