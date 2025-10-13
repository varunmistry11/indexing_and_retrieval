"""
Postings list data structures and operations for SelfIndex.
"""

from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field
import bisect


@dataclass
class PostingEntry:
    """
    Single posting entry for a term in a document.
    
    Attributes:
        doc_id: Document identifier
        term_freq: Number of times term appears in document (for x=2, x=3)
        positions: List of positions where term appears (for phrase queries)
    """
    doc_id: int
    term_freq: int = 1
    positions: List[int] = field(default_factory=list)
    
    def __lt__(self, other):
        """Compare by doc_id for sorting."""
        return self.doc_id < other.doc_id
    
    def __eq__(self, other):
        """Equality based on doc_id."""
        return self.doc_id == other.doc_id
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'doc_id': self.doc_id,
            'term_freq': self.term_freq,
            'positions': self.positions
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PostingEntry':
        """Create from dictionary."""
        return cls(
            doc_id=data['doc_id'],
            term_freq=data['term_freq'],
            positions=data['positions']
        )


class PostingsList:
    """
    Postings list for a single term.
    Maintains sorted list of documents containing the term.
    """
    
    def __init__(self):
        """Initialize empty postings list."""
        self.postings: List[PostingEntry] = []
        self._doc_id_set: Optional[Set[int]] = None  # Cache for fast lookup
    
    def add_posting(self, doc_id: int, position: int):
        """
        Add a posting for a term occurrence in a document.
        
        Args:
            doc_id: Document identifier
            position: Position of term in document
        """
        # Check if document already exists in postings
        if self.postings and self.postings[-1].doc_id == doc_id:
            # Same document as last entry, update it
            self.postings[-1].term_freq += 1
            self.postings[-1].positions.append(position)
        else:
            # New document or not in order, need to search
            # Use binary search to find or insert
            entry = PostingEntry(doc_id=doc_id, term_freq=1, positions=[position])
            
            # Find position using binary search
            idx = bisect.bisect_left([p.doc_id for p in self.postings], doc_id)
            
            if idx < len(self.postings) and self.postings[idx].doc_id == doc_id:
                # Document exists, update it
                self.postings[idx].term_freq += 1
                self.postings[idx].positions.append(position)
            else:
                # Insert new entry
                self.postings.insert(idx, entry)
        
        # Invalidate cache
        self._doc_id_set = None
    
    def add_posting_batch(self, doc_id: int, positions: List[int]):
        """
        Add multiple positions for a term in a document at once.
        More efficient than calling add_posting multiple times.
        
        Args:
            doc_id: Document identifier
            positions: List of positions where term appears
        """
        if not positions:
            return
        
        entry = PostingEntry(
            doc_id=doc_id,
            term_freq=len(positions),
            positions=sorted(positions)
        )
        
        # Find position using binary search
        idx = bisect.bisect_left([p.doc_id for p in self.postings], doc_id)
        
        if idx < len(self.postings) and self.postings[idx].doc_id == doc_id:
            # Document exists, update it
            self.postings[idx].term_freq += len(positions)
            self.postings[idx].positions.extend(positions)
            self.postings[idx].positions.sort()
        else:
            # Insert new entry
            self.postings.insert(idx, entry)
        
        # Invalidate cache
        self._doc_id_set = None
    
    def get_doc_ids(self) -> List[int]:
        """Get list of all document IDs containing this term."""
        return [p.doc_id for p in self.postings]
    
    def get_doc_id_set(self) -> Set[int]:
        """Get set of document IDs for fast membership testing."""
        if self._doc_id_set is None:
            self._doc_id_set = {p.doc_id for p in self.postings}
        return self._doc_id_set
    
    def get_posting(self, doc_id: int) -> Optional[PostingEntry]:
        """
        Get posting entry for a specific document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            PostingEntry if found, None otherwise
        """
        # Binary search
        idx = bisect.bisect_left([p.doc_id for p in self.postings], doc_id)
        if idx < len(self.postings) and self.postings[idx].doc_id == doc_id:
            return self.postings[idx]
        return None
    
    def get_term_frequency(self, doc_id: int) -> int:
        """Get term frequency in a specific document."""
        posting = self.get_posting(doc_id)
        return posting.term_freq if posting else 0
    
    def get_positions(self, doc_id: int) -> List[int]:
        """Get positions of term in a specific document."""
        posting = self.get_posting(doc_id)
        return posting.positions if posting else []
    
    def document_frequency(self) -> int:
        """Get number of documents containing this term."""
        return len(self.postings)
    
    def total_term_frequency(self) -> int:
        """Get total occurrences of term across all documents."""
        return sum(p.term_freq for p in self.postings)
    
    def __len__(self) -> int:
        """Number of documents containing this term."""
        return len(self.postings)
    
    def __iter__(self):
        """Iterate over postings."""
        return iter(self.postings)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'postings': [p.to_dict() for p in self.postings],
            'df': self.document_frequency(),
            'total_tf': self.total_term_frequency()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PostingsList':
        """Create from dictionary."""
        pl = cls()
        pl.postings = [PostingEntry.from_dict(p) for p in data['postings']]
        return pl
    
    def merge_with(self, other: 'PostingsList') -> 'PostingsList':
        """
        Merge this postings list with another.
        Used for index updates.
        
        Args:
            other: Another PostingsList to merge
            
        Returns:
            New merged PostingsList
        """
        merged = PostingsList()
        
        i, j = 0, 0
        while i < len(self.postings) and j < len(other.postings):
            if self.postings[i].doc_id < other.postings[j].doc_id:
                merged.postings.append(self.postings[i])
                i += 1
            elif self.postings[i].doc_id > other.postings[j].doc_id:
                merged.postings.append(other.postings[j])
                j += 1
            else:
                # Same document, merge positions
                entry = PostingEntry(
                    doc_id=self.postings[i].doc_id,
                    term_freq=self.postings[i].term_freq + other.postings[j].term_freq,
                    positions=sorted(self.postings[i].positions + other.postings[j].positions)
                )
                merged.postings.append(entry)
                i += 1
                j += 1
        
        # Add remaining
        merged.postings.extend(self.postings[i:])
        merged.postings.extend(other.postings[j:])
        
        return merged

class PostingsListWithSkips(PostingsList):
    """
    Postings list with skip pointers for faster intersection.
    Used when optimization='Skipping' (o=sp).
    """
    
    def __init__(self, skip_interval: Optional[int] = None):
        """
        Initialize postings list with skip pointers.
        
        Args:
            skip_interval: Interval between skip pointers (default: sqrt(n))
        """
        super().__init__()
        self.skip_interval = skip_interval
        self.skip_pointers: dict = {}  # Maps position -> skip_target position
    
    def build_skip_pointers(self):
        """Build skip pointers after all postings are added."""
        if not self.postings:
            return
        
        n = len(self.postings)
        
        # Default skip interval: square root of list length
        if self.skip_interval is None:
            import math
            self.skip_interval = max(1, int(math.sqrt(n)))
        
        # Build skip pointers - store as a dictionary for O(1) lookup
        self.skip_pointers = {}
        for i in range(0, n, self.skip_interval):
            skip_to = i + self.skip_interval
            if skip_to < n:
                self.skip_pointers[i] = skip_to
    
    def has_skips(self) -> bool:
        """Check if skip pointers are built."""
        return len(self.skip_pointers) > 0
    
    def get_skip_target(self, position: int) -> Optional[int]:
        """
        Get skip target for a given position.
        
        Args:
            position: Current position in postings list
            
        Returns:
            Target position to skip to, or None if no skip available
        """
        return self.skip_pointers.get(position)
