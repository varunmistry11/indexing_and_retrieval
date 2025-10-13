"""Index implementations."""

from .elasticsearch_index import ElasticsearchIndex
from .self_index import SelfIndex

__all__ = ['ElasticsearchIndex', 'SelfIndex']
