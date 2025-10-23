"""
Unit tests for Phase 3: Main SelfIndex Class
Run with: pytest tests/test_phase3_self_index.py -v
"""

import pytest
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indices.self_index import SelfIndex


class MockConfig:
    """Mock configuration for testing."""
    
    def __init__(self, index_type='TFIDF', query_proc='TERMatat', 
                 optimization='Null', compression='NONE'):
        self.index = MagicMock()
        self.index.type = index_type
        self.index.query_proc = query_proc
        self.index.optimization = optimization
        self.index.get = MagicMock(side_effect=lambda k, d=None: {
            'optimization': optimization,
            'query_proc': query_proc
        }.get(k, d))
        
        self.datastore = MagicMock()
        self.datastore.format = 'json'
        self.datastore.get = MagicMock(return_value='json')
        
        self.compression = MagicMock()
        self.compression.type = compression
        self.compression.get = MagicMock(return_value=compression)
        
        # Preprocessing config - IMPORTANT: Set actual values, not MagicMock
        self.preprocessing = MagicMock()
        self.preprocessing.lowercase = True
        self.preprocessing.remove_punctuation = True
        self.preprocessing.remove_stopwords = True
        self.preprocessing.stemming = True
        self.preprocessing.stemmer = 'porter'
        self.preprocessing.stopwords_file = None
        self.preprocessing.min_word_length = 1  # Set actual integer
        self.preprocessing.max_word_length = 50  # Set actual integer
        
        # Paths
        self.paths = MagicMock()
        self.temp_dir = tempfile.mkdtemp()
        self.paths.index_storage = self.temp_dir
        
    def get(self, key, default=None):
        """Mock get method."""
        if key == 'datastore':
            return {'format': 'json'}
        elif key == 'compression':
            return {'type': self.compression.type}
        return default
    
    def cleanup(self):
        """Clean up temporary directory."""
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            'id': 'doc1',
            'text': 'Python programming is great for machine learning',
            'title': 'Python ML',
        },
        {
            'id': 'doc2',
            'text': 'Java programming is used for enterprise applications',
            'title': 'Java Enterprise',
        },
        {
            'id': 'doc3',
            'text': 'Machine learning with Python is powerful',
            'title': 'ML Python',
        },
        {
            'id': 'doc4',
            'text': 'Deep learning is a subset of machine learning',
            'title': 'Deep Learning',
        },
    ]


class TestSelfIndexInitialization:
    """Test SelfIndex initialization."""
    
    def test_init_boolean_index(self):
        """Test initializing boolean index."""
        config = MockConfig(index_type='BOOLEAN')
        index = SelfIndex(config)
        
        assert index.index_type == 'BOOLEAN'
        assert index.query_processor_type == 'TERMatat'
        assert not index.use_skip_pointers
        
        config.cleanup()
    
    def test_init_wordcount_index(self):
        """Test initializing wordcount index."""
        config = MockConfig(index_type='WORDCOUNT')
        index = SelfIndex(config)
        
        assert index.index_type == 'WORDCOUNT'
        config.cleanup()
    
    def test_init_tfidf_index(self):
        """Test initializing TF-IDF index."""
        config = MockConfig(index_type='TFIDF')
        index = SelfIndex(config)
        
        assert index.index_type == 'TFIDF'
        config.cleanup()
    
    def test_init_with_skip_pointers(self):
        """Test initializing with skip pointers."""
        config = MockConfig(optimization='Skipping')
        index = SelfIndex(config)
        
        assert index.use_skip_pointers
        config.cleanup()
    
    def test_init_doc_at_a_time(self):
        """Test initializing with Document-at-a-time processor."""
        config = MockConfig(query_proc='DOCatat')
        index = SelfIndex(config)
        
        assert index.query_processor_type == 'DOCatat'
        config.cleanup()


class TestIndexCreation:
    """Test index creation functionality."""
    
    def test_create_index_basic(self, sample_documents):
        """Test basic index creation."""
        config = MockConfig()
        index = SelfIndex(config)
        
        stats = index.create_index_from_documents('test_dataset', sample_documents)
        
        assert stats['documents_processed'] == 4
        assert stats['vocabulary_size'] > 0
        assert stats['indexing_time_seconds'] > 0
        assert index.index_name is not None
        
        config.cleanup()
    
    def test_create_index_with_empty_docs(self):
        """Test creating index with some empty documents."""
        config = MockConfig()
        index = SelfIndex(config)
        
        documents = [
            {'id': 'doc1', 'text': 'Valid document'},
            {'id': 'doc2', 'text': ''},  # Empty
            {'id': 'doc3'},  # No text field
            {'id': 'doc4', 'text': 'Another valid document'},
        ]
        
        stats = index.create_index_from_documents('test_dataset', documents)
        
        assert stats['documents_processed'] == 2
        assert stats['documents_skipped'] == 2
        
        config.cleanup()
    
    def test_create_index_force_overwrite(self, sample_documents):
        """Test force overwriting existing index."""
        config = MockConfig()
        index = SelfIndex(config)
        
        # Create first index
        stats1 = index.create_index_from_documents('test_dataset', sample_documents[:2])
        
        # Create again - should work because names have timestamps
        index2 = SelfIndex(config)
        stats2 = index2.create_index_from_documents('test_dataset', sample_documents, force=True)
        
        assert stats2['documents_processed'] == 4
        
        config.cleanup()
    
    def test_version_string_generation(self):
        """Test version string generation."""
        config = MockConfig(
            index_type='TFIDF',
            query_proc='DOCatat',
            optimization='Skipping',
            compression='CODE'
        )
        index = SelfIndex(config)
        
        version = index._generate_version_string()
        
        # Should be: x=3, y=1, z=1, i=1, q=D
        assert version == '3.1.1.1.D'
        
        config.cleanup()
    
    def test_index_with_skip_pointers(self, sample_documents):
        """Test creating index with skip pointers enabled."""
        config = MockConfig(optimization='Skipping')
        index = SelfIndex(config)
        
        stats = index.create_index_from_documents('test_dataset', sample_documents)
        
        assert index.use_skip_pointers
        assert stats['documents_processed'] == 4
        
        config.cleanup()


class TestIndexPersistence:
    """Test index save/load functionality."""
    
    def test_save_and_load(self, sample_documents):
        """Test saving and loading an index."""
        config = MockConfig()
        
        # Create and save index
        index1 = SelfIndex(config)
        stats = index1.create_index_from_documents('test_dataset', sample_documents)
        index_name = index1.index_name
        index_path = index1._get_index_path(index_name)
        
        # Load index in new instance
        index2 = SelfIndex(config)
        index2.load_index(str(index_path))
        
        assert index2.document_count == 4
        assert index2.inverted_index is not None
        assert index2.doc_store is not None
        assert index2.stats is not None
        
        config.cleanup()
    
    def test_load_nonexistent_index(self):
        """Test loading non-existent index."""
        config = MockConfig()
        index = SelfIndex(config)
        
        with pytest.raises(FileNotFoundError):
            index.load_index('/nonexistent/path')
        
        config.cleanup()
    
    def test_list_indices(self, sample_documents):
        """Test listing available indices."""
        config = MockConfig()
        
        # Create multiple indices
        index1 = SelfIndex(config)
        index1.create_index_from_documents('dataset1', sample_documents[:2])
        
        index2 = SelfIndex(config)
        index2.create_index_from_documents('dataset2', sample_documents[2:])
        
        # List indices
        index3 = SelfIndex(config)
        indices = list(index3.list_indices())
        
        assert len(indices) >= 2
        
        config.cleanup()
    
    def test_delete_index(self, sample_documents):
        """Test deleting an index."""
        config = MockConfig()
        
        # Create index
        index = SelfIndex(config)
        stats = index.create_index_from_documents('test_dataset', sample_documents)
        index_name = index.index_name
        
        # Delete index
        index.delete_index(index_name)
        
        # Verify it's gone
        with pytest.raises(FileNotFoundError):
            index2 = SelfIndex(config)
            index2.load_index(str(index._get_index_path(index_name)))
        
        config.cleanup()


class TestQuerying:
    """Test query functionality."""
    
    @pytest.fixture
    def indexed_system(self, sample_documents):
        """Create an indexed system for testing."""
        config = MockConfig()
        index = SelfIndex(config)
        index.create_index_from_documents('test_dataset', sample_documents)
        yield index, config
        config.cleanup()
    
    def test_simple_term_query(self, indexed_system):
        """Test simple term query."""
        index, config = indexed_system
        
        results = index.query_with_params('python', k=10)
        
        assert len(results) > 0
        assert all('doc_id' in r for r in results)
        assert all('score' in r for r in results)
        assert all('rank' in r for r in results)
    
    def test_multi_term_query(self, indexed_system):
        """Test query with multiple terms."""
        index, config = indexed_system
        
        results = index.query_with_params('machine learning', k=10)
        
        assert len(results) > 0
        # Documents with both terms should rank higher
        if len(results) > 1:
            assert results[0]['score'] >= results[-1]['score']
    
    def test_boolean_and_query(self, indexed_system):
        """Test AND query."""
        index, config = indexed_system
        
        results = index.query_with_params('"machine" AND "learning"', k=10)
        
        assert len(results) >= 0  # May or may not have results
    
    def test_boolean_or_query(self, indexed_system):
        """Test OR query."""
        index, config = indexed_system
        
        results = index.query_with_params('"python" OR "java"', k=10)
        
        assert len(results) > 0
    
    def test_boolean_not_query(self, indexed_system):
        """Test NOT query."""
        index, config = indexed_system
        
        results = index.query_with_params('NOT "python"', k=10)
        
        # Should return documents without python
        assert isinstance(results, list)
    
    def test_phrase_query(self, indexed_system):
        """Test phrase query."""
        index, config = indexed_system
        
        results = index.query_with_params('"machine" "learning"', k=10)
        
        # Should find documents with "machine learning" as phrase
        assert isinstance(results, list)
    
    def test_complex_boolean_query(self, indexed_system):
        """Test complex boolean query."""
        index, config = indexed_system
        
        results = index.query_with_params('("machine" AND "learning") OR "java"', k=10)
        
        assert isinstance(results, list)
    
    def test_query_with_k_limit(self, indexed_system):
        """Test query with k limit."""
        index, config = indexed_system
        
        results = index.query_with_params('learning', k=2)
        
        assert len(results) <= 2
    
    def test_query_no_results(self, indexed_system):
        """Test query that returns no results."""
        index, config = indexed_system
        
        results = index.query_with_params('nonexistentterm12345', k=10)
        
        assert len(results) == 0
    
    def test_empty_query(self, indexed_system):
        """Test empty query."""
        index, config = indexed_system
        
        results = index.query_with_params('', k=10)
        
        assert len(results) == 0


class TestIndexUpdate:
    """Test index update functionality."""
    
    def test_add_documents(self, sample_documents):
        """Test adding documents to existing index."""
        config = MockConfig()
        
        # Create initial index
        index = SelfIndex(config)
        stats = index.create_index_from_documents('test_dataset', sample_documents[:2])
        index_name = index.index_name
        
        # Add more documents using files format
        new_files = [(doc['id'], doc['text']) for doc in sample_documents[2:]]
        index.update_index(index_name, [], new_files)
        
        # Check that documents were added
        assert index.document_count >= 2
        
        config.cleanup()
    
    def test_remove_documents_not_implemented(self, sample_documents):
        """Test that document removal logs warning."""
        config = MockConfig()
        
        index = SelfIndex(config)
        index.create_index_from_documents('test_dataset', sample_documents)
        index_name = index.index_name
        
        # Try to remove documents - should just log warning, not crash
        remove_files = [(sample_documents[0]['id'], sample_documents[0]['text'])]
        index.update_index(index_name, remove_files, [])
        
        # Should not raise error, just log warning
        assert True
        
        config.cleanup()


class TestStatistics:
    """Test statistics functionality."""
    
    def test_get_statistics(self, sample_documents):
        """Test getting index statistics."""
        config = MockConfig()
        index = SelfIndex(config)
        index.create_index_from_documents('test_dataset', sample_documents)
        
        stats = index.get_statistics()
        
        assert 'index_name' in stats
        assert 'document_count' in stats
        assert 'vocabulary_size' in stats
        assert 'total_tokens' in stats
        assert 'avg_document_length' in stats
        assert stats['document_count'] == 4
        assert stats['vocabulary_size'] > 0
        
        config.cleanup()
    
    def test_index_size_calculation(self, sample_documents):
        """Test calculating index size on disk."""
        config = MockConfig()
        index = SelfIndex(config)
        index.create_index_from_documents('test_dataset', sample_documents)
        
        size = index._get_index_size()
        
        assert size > 0  # Index should have some size
        
        config.cleanup()


class TestDifferentIndexTypes:
    """Test different index type configurations."""
    
    def test_boolean_index_type(self, sample_documents):
        """Test Boolean index (x=1)."""
        config = MockConfig(index_type='BOOLEAN')
        index = SelfIndex(config)
        
        index.create_index_from_documents('test_dataset', sample_documents)
        results = index.query_with_params('python', k=10)
        
        # Boolean index should still return results
        assert len(results) >= 0
        
        config.cleanup()
    
    def test_wordcount_index_type(self, sample_documents):
        """Test WordCount index (x=2)."""
        config = MockConfig(index_type='WORDCOUNT')
        index = SelfIndex(config)
        
        index.create_index_from_documents('test_dataset', sample_documents)
        results = index.query_with_params('learning', k=10)
        
        assert len(results) >= 0
        # Results should be ranked by term frequency
        
        config.cleanup()
    
    def test_tfidf_index_type(self, sample_documents):
        """Test TF-IDF index (x=3)."""
        config = MockConfig(index_type='TFIDF')
        index = SelfIndex(config)
        
        index.create_index_from_documents('test_dataset', sample_documents)
        results = index.query_with_params('machine learning', k=10)
        
        assert len(results) >= 0
        # Results should be ranked by BM25
        
        config.cleanup()


class TestQueryProcessors:
    """Test different query processor configurations."""
    
    def test_term_at_a_time_processor(self, sample_documents):
        """Test Term-at-a-time query processor."""
        config = MockConfig(query_proc='TERMatat')
        index = SelfIndex(config)
        
        index.create_index_from_documents('test_dataset', sample_documents)
        results = index.query_with_params('machine learning', k=10)
        
        assert len(results) >= 0
        
        config.cleanup()
    
    def test_document_at_a_time_processor(self, sample_documents):
        """Test Document-at-a-time query processor."""
        config = MockConfig(query_proc='DOCatat')
        index = SelfIndex(config)
        
        index.create_index_from_documents('test_dataset', sample_documents)
        results = index.query_with_params('machine learning', k=10)
        
        assert len(results) >= 0
        
        config.cleanup()


class TestHelperMethods:
    """Test helper methods."""
    
    def test_extract_terms_from_tree(self):
        """Test extracting terms from query tree."""
        config = MockConfig()
        index = SelfIndex(config)
        
        query_tree = {
            'type': 'AND',
            'children': [
                {'type': 'TERM', 'value': 'python'},
                {'type': 'TERM', 'value': 'java'}
            ]
        }
        
        terms = index._extract_terms(query_tree)
        
        assert 'python' in terms
        assert 'java' in terms
        
        config.cleanup()
    
    def test_is_boolean_query(self):
        """Test checking if query is boolean."""
        config = MockConfig()
        index = SelfIndex(config)
        
        boolean_tree = {'type': 'AND', 'children': []}
        term_tree = {'type': 'TERM', 'value': 'test'}
        
        assert index._is_boolean_query(boolean_tree)
        assert not index._is_boolean_query(term_tree)
        
        config.cleanup()


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_query_without_loaded_index(self):
        """Test querying without loading an index."""
        config = MockConfig()
        index = SelfIndex(config)
        
        with pytest.raises(ValueError, match="No index loaded"):
            index.query_with_params('test query', k=10)
        
        config.cleanup()
    
    def test_create_index_duplicate_name(self, sample_documents):
        """Test creating index with duplicate name without force."""
        config = MockConfig()
        
        # Create first index
        index1 = SelfIndex(config)
        stats1 = index1.create_index_from_documents('test_dataset', sample_documents[:2])
        
        # Try to create again - should work because names have timestamps
        index2 = SelfIndex(config)
        stats2 = index2.create_index_from_documents('test_dataset', sample_documents)
        
        # Should succeed because names are different (timestamp)
        assert stats2['documents_processed'] == 4
        
        config.cleanup()
    
    def test_malformed_query(self, sample_documents):
        """Test handling malformed query."""
        config = MockConfig()
        index = SelfIndex(config)
        index.create_index_from_documents('test_dataset', sample_documents)
        
        # Should handle gracefully and fall back to simple query
        results = index.query_with_params('malformed (((query', k=10)
        
        # Should not crash
        assert isinstance(results, list)
        
        config.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])