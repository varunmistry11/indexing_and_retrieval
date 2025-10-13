"""
Boolean operations on postings lists (AND, OR, NOT).
Implements efficient set operations for query processing.
"""

from typing import List, Set, Optional
import logging

from .postings import PostingsList, PostingsListWithSkips, PostingEntry

logger = logging.getLogger(__name__)


class BooleanOperations:
    """Implements boolean operations on postings lists."""
    
    @staticmethod
    def intersect(list1: PostingsList, list2: PostingsList) -> PostingsList:
        """
        Intersect two postings lists (AND operation).
        Uses two-pointer algorithm for O(n + m) complexity.
        
        Args:
            list1: First postings list
            list2: Second postings list
            
        Returns:
            New PostingsList containing documents in both lists
        """
        result = PostingsList()
        
        if not list1.postings or not list2.postings:
            return result
        
        i, j = 0, 0
        
        while i < len(list1.postings) and j < len(list2.postings):
            doc_id1 = list1.postings[i].doc_id
            doc_id2 = list2.postings[j].doc_id
            
            if doc_id1 == doc_id2:
                # Document in both lists - add to result
                # Merge positions from both lists
                positions = sorted(
                    list1.postings[i].positions + list2.postings[j].positions
                )
                result.add_posting_batch(doc_id1, positions)
                i += 1
                j += 1
            elif doc_id1 < doc_id2:
                i += 1
            else:
                j += 1
        
        return result
    
    @staticmethod
    def intersect_with_skips(list1: PostingsListWithSkips, 
                            list2: PostingsListWithSkips) -> PostingsList:
        """
        Intersect two postings lists using skip pointers for optimization.
        
        Args:
            list1: First postings list with skip pointers
            list2: Second postings list with skip pointers
            
        Returns:
            New PostingsList containing documents in both lists
        """
        result = PostingsList()
        
        if not list1.postings or not list2.postings:
            return result
        
        # Ensure skip pointers are built
        if not list1.has_skips():
            list1.build_skip_pointers()
        if not list2.has_skips():
            list2.build_skip_pointers()
        
        i, j = 0, 0
        
        while i < len(list1.postings) and j < len(list2.postings):
            doc_id1 = list1.postings[i].doc_id
            doc_id2 = list2.postings[j].doc_id
            
            if doc_id1 == doc_id2:
                # Match found
                positions = sorted(
                    list1.postings[i].positions + list2.postings[j].positions
                )
                result.add_posting_batch(doc_id1, positions)
                i += 1
                j += 1
            elif doc_id1 < doc_id2:
                # Try to skip in list1
                skip_target = list1.get_skip_target(i)
                if skip_target and list1.postings[skip_target].doc_id <= doc_id2:
                    i = skip_target
                else:
                    i += 1
            else:
                # Try to skip in list2
                skip_target = list2.get_skip_target(j)
                if skip_target and list2.postings[skip_target].doc_id <= doc_id1:
                    j = skip_target
                else:
                    j += 1
        
        return result
    
    @staticmethod
    def intersect_many(postings_lists: List[PostingsList]) -> PostingsList:
        """
        Intersect multiple postings lists.
        Optimizes by processing shortest list first.
        
        Args:
            postings_lists: List of PostingsList objects
            
        Returns:
            PostingsList containing documents in all lists
        """
        if not postings_lists:
            return PostingsList()
        
        if len(postings_lists) == 1:
            return postings_lists[0]
        
        # Sort by length (shortest first for efficiency)
        sorted_lists = sorted(postings_lists, key=len)
        
        # Start with shortest list
        result = sorted_lists[0]
        
        # Intersect with remaining lists
        for pl in sorted_lists[1:]:
            # Use skip pointers if available
            if isinstance(result, PostingsListWithSkips) and isinstance(pl, PostingsListWithSkips):
                result = BooleanOperations.intersect_with_skips(result, pl)
            else:
                result = BooleanOperations.intersect(result, pl)
            
            # Early termination if result becomes empty
            if len(result) == 0:
                break
        
        return result
    
    @staticmethod
    def union(list1: PostingsList, list2: PostingsList) -> PostingsList:
        """
        Union two postings lists (OR operation).
        Uses two-pointer merge algorithm for O(n + m) complexity.
        
        Args:
            list1: First postings list
            list2: Second postings list
            
        Returns:
            New PostingsList containing documents in either list
        """
        result = PostingsList()
        
        if not list1.postings:
            return list2
        if not list2.postings:
            return list1
        
        i, j = 0, 0
        
        while i < len(list1.postings) or j < len(list2.postings):
            if i >= len(list1.postings):
                # Only list2 has remaining items
                result.postings.extend(list2.postings[j:])
                break
            elif j >= len(list2.postings):
                # Only list1 has remaining items
                result.postings.extend(list1.postings[i:])
                break
            else:
                doc_id1 = list1.postings[i].doc_id
                doc_id2 = list2.postings[j].doc_id
                
                if doc_id1 < doc_id2:
                    result.postings.append(list1.postings[i])
                    i += 1
                elif doc_id1 > doc_id2:
                    result.postings.append(list2.postings[j])
                    j += 1
                else:
                    # Document in both - merge positions
                    positions = sorted(
                        list1.postings[i].positions + list2.postings[j].positions
                    )
                    entry = PostingEntry(
                        doc_id=doc_id1,
                        term_freq=list1.postings[i].term_freq + list2.postings[j].term_freq,
                        positions=positions
                    )
                    result.postings.append(entry)
                    i += 1
                    j += 1
        
        return result
    
    @staticmethod
    def union_many(postings_lists: List[PostingsList]) -> PostingsList:
        """
        Union multiple postings lists.
        
        Args:
            postings_lists: List of PostingsList objects
            
        Returns:
            PostingsList containing documents in any list
        """
        if not postings_lists:
            return PostingsList()
        
        if len(postings_lists) == 1:
            return postings_lists[0]
        
        # Start with first list
        result = postings_lists[0]
        
        # Union with remaining lists
        for pl in postings_lists[1:]:
            result = BooleanOperations.union(result, pl)
        
        return result
    
    @staticmethod
    def negate(postings_list: PostingsList, all_doc_ids: Set[int]) -> PostingsList:
        """
        Negate a postings list (NOT operation).
        Returns documents NOT in the postings list.
        
        Args:
            postings_list: PostingsList to negate
            all_doc_ids: Set of all document IDs in the collection
            
        Returns:
            PostingsList containing documents not in input list
        """
        result = PostingsList()
        docs_in_list = set(postings_list.get_doc_ids())
        
        # Add all documents that are NOT in the postings list
        for doc_id in sorted(all_doc_ids - docs_in_list):
            result.add_posting(doc_id, position=0)  # Position doesn't matter for boolean ops
        
        return result
    
    @staticmethod
    def and_not(list1: PostingsList, list2: PostingsList) -> PostingsList:
        """
        Compute list1 AND NOT list2.
        More efficient than separate AND and NOT operations.
        
        Args:
            list1: Positive postings list
            list2: Postings list to exclude
            
        Returns:
            PostingsList containing documents in list1 but not in list2
        """
        result = PostingsList()
        
        if not list1.postings:
            return result
        
        if not list2.postings:
            return list1
        
        # Get set of doc IDs to exclude for fast lookup
        exclude_docs = list2.get_doc_id_set()
        
        # Add documents from list1 that are not in list2
        for posting in list1.postings:
            if posting.doc_id not in exclude_docs:
                result.postings.append(posting)
        
        return result


class PhraseQueryProcessor:
    """Process phrase queries using positional information."""
        
    @staticmethod
    def phrase_query(term_postings: List[PostingsList]) -> PostingsList:
        """
        Process a phrase query.
        Returns documents where terms appear consecutively.
        
        Args:
            term_postings: List of PostingsList for each term in phrase (in order)
            
        Returns:
            PostingsList containing documents with the phrase
        """
        if not term_postings:
            return PostingsList()
        
        if len(term_postings) == 1:
            return term_postings[0]
        
        # First, find documents containing all terms (intersection)
        candidate_docs = BooleanOperations.intersect_many(term_postings)
        
        if len(candidate_docs) == 0:
            return PostingsList()
        
        result = PostingsList()
        
        # Get the list of candidate document IDs
        candidate_doc_ids = candidate_docs.get_doc_ids()
        
        # For each candidate document, check if terms appear consecutively
        for doc_id in candidate_doc_ids:
            # Get positions of first term
            first_positions = term_postings[0].get_positions(doc_id)
            
            # For each position of the first term, check if remaining terms follow
            phrase_positions = []
            
            for start_pos in first_positions:
                # Check if all subsequent terms appear at consecutive positions
                is_phrase = True
                current_pos = start_pos
                
                for i in range(1, len(term_postings)):
                    # Next term should be at current_pos + 1
                    next_positions = term_postings[i].get_positions(doc_id)
                    if (current_pos + 1) not in next_positions:
                        is_phrase = False
                        break
                    current_pos += 1
                
                if is_phrase:
                    phrase_positions.append(start_pos)
            
            # If phrase found in document, add to results
            if phrase_positions:
                result.add_posting_batch(doc_id, phrase_positions)
        
        return result
    
    @staticmethod
    def proximity_query(term_postings: List[PostingsList], 
                       max_distance: int) -> PostingsList:
        """
        Process a proximity query.
        Returns documents where terms appear within max_distance of each other.
        
        Args:
            term_postings: List of PostingsList for each term
            max_distance: Maximum distance between terms
            
        Returns:
            PostingsList containing documents matching proximity
        """
        if not term_postings or max_distance < 0:
            return PostingsList()
        
        if len(term_postings) == 1:
            return term_postings[0]
        
        # Find documents containing all terms
        candidate_docs = BooleanOperations.intersect_many(term_postings)
        
        if len(candidate_docs) == 0:
            return PostingsList()
        
        result = PostingsList()
        
        # For each candidate document, check proximity
        for candidate in candidate_docs.postings:
            doc_id = candidate.doc_id
            
            # Get all positions for all terms
            all_positions = []
            for term_pl in term_postings:
                positions = term_pl.get_positions(doc_id)
                all_positions.append(positions)
            
            # Check if any combination of positions satisfies proximity
            matched_positions = []
            
            # For simplicity, check if min and max positions are within distance
            flat_positions = [pos for positions in all_positions for pos in positions]
            if flat_positions:
                min_pos = min(flat_positions)
                max_pos = max(flat_positions)
                
                if max_pos - min_pos <= max_distance:
                    matched_positions = flat_positions
            
            if matched_positions:
                result.add_posting_batch(doc_id, sorted(matched_positions))
        
        return result