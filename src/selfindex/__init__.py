"""
SelfIndex - Custom inverted index implementation.
"""

from .postings import PostingEntry, PostingsList, PostingsListWithSkips
from .inverted_index import InvertedIndex, DocumentStore, CollectionStatistics
from .boolean_ops import BooleanOperations, PhraseQueryProcessor
from .query_processor import (
    QueryResult,
    QueryProcessor,
    TermAtATimeProcessor,
    DocumentAtATimeProcessor,
    BooleanQueryProcessor
)

__all__ = [
    'PostingEntry',
    'PostingsList',
    'PostingsListWithSkips',
    'InvertedIndex',
    'DocumentStore',
    'CollectionStatistics',
    
    'BooleanOperations',
    'PhraseQueryProcessor',
    'QueryResult',
    'QueryProcessor',
    'TermAtATimeProcessor',
    'DocumentAtATimeProcessor',
    'BooleanQueryProcessor',
]