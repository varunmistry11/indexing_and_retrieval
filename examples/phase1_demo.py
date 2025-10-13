#!/usr/bin/env python
"""
Demo script for Phase 1: Core Data Structures

This demonstrates the functionality of the core SelfIndex components.
Run with: python examples/phase1_demo.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.selfindex import InvertedIndex, DocumentStore, CollectionStatistics
import json


def demo_basic_index():
    """Demo 1: Basic inverted index operations."""
    print("="*60)
    print("DEMO 1: Basic Inverted Index")
    print("="*60)
    
    # Create index
    index = InvertedIndex()
    
    # Add documents
    docs = [
        (0, ["the", "quick", "brown", "fox"]),
        (1, ["the", "lazy", "dog"]),
        (2, ["the", "quick", "fox"])
    ]
    
    for doc_id, tokens in docs:
        index.add_document(doc_id, tokens)
    
    # Query the index
    print(f"\nDocuments indexed: {index.num_documents}")
    print(f"Vocabulary size: {index.get_vocabulary_size()}")
    print(f"Vocabulary: {sorted(index.get_vocabulary())}")
    
    # Term lookup
    term = "quick"
    postings = index.get_postings(term)
    print(f"\nTerm: '{term}'")
    print(f"  Document frequency: {postings.document_frequency()}")
    print(f"  Documents: {postings.get_doc_ids()}")
    
    for posting in postings:
        print(f"    Doc {posting.doc_id}: TF={posting.term_freq}, positions={posting.positions}")
    
    # Statistics
    stats = index.get_statistics()
    print(f"\nIndex Statistics:")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Avg document length: {stats['avg_document_length']:.2f}")
    print(f"  Avg postings length: {stats['avg_postings_length']:.2f}")


def demo_skip_pointers():
    """Demo 2: Index with skip pointers."""
    print("\n" + "="*60)
    print("DEMO 2: Skip Pointers")
    print("="*60)
    
    # Create index with skip pointers
    index = InvertedIndex(use_skip_pointers=True)
    
    # Add many documents with the same term
    print("\nAdding 100 documents containing 'test'...")
    for i in range(100):
        index.add_document(doc_id=i, tokens=["test", "document", str(i)])
    
    # Finalize to build skip pointers
    index.finalize()
    
    # Check skip pointers
    postings = index.get_postings("test")
    print(f"Postings list length: {len(postings)}")
    print(f"Has skip pointers: {postings.has_skips()}")
    print(f"Skip interval: {postings.skip_interval}")
    print(f"Number of skip pointers: {len(postings.skip_pointers)}")
    
    # Show some skip pointers
    print("\nSkip pointer samples:")
    for i, (from_pos, to_pos) in enumerate(postings.skip_pointers[:5]):
        print(f"  Skip {i}: position {from_pos} -> position {to_pos}")


def demo_document_store():
    """Demo 3: Document store with metadata."""
    print("\n" + "="*60)
    print("DEMO 3: Document Store")
    print("="*60)
    
    # Create document store
    doc_store = DocumentStore()
    
    # Add documents with metadata
    documents = [
        ("article_1", {
            'title': 'Introduction to Information Retrieval',
            'url': 'http://example.com/article1',
            'length': 150,
            'author': 'John Doe'
        }),
        ("article_2", {
            'title': 'Advanced Search Techniques',
            'url': 'http://example.com/article2',
            'length': 200,
            'author': 'Jane Smith'
        }),
        ("article_3", {
            'title': 'Building Search Engines',
            'url': 'http://example.com/article3',
            'length': 175,
            'author': 'Bob Johnson'
        })
    ]
    
    print("\nAdding documents:")
    for doc_id, metadata in documents:
        internal_id = doc_store.add_document(doc_id, metadata)
        print(f"  {doc_id} -> internal ID {internal_id}")
    
    # Retrieve documents
    print("\nRetrieving documents:")
    for internal_id in range(doc_store.get_document_count()):
        doc = doc_store.get_document(internal_id)
        external_id = doc_store.get_external_id(internal_id)
        print(f"\n  Internal ID {internal_id} ({external_id}):")
        print(f"    Title: {doc['title']}")
        print(f"    Author: {doc['author']}")
        print(f"    Length: {doc['length']} tokens")
        print(f"    URL: {doc['url']}")
    
    # ID mapping
    print("\n\nID Mapping:")
    print(f"  External 'article_2' -> Internal {doc_store.get_internal_id('article_2')}")
    print(f"  Internal 1 -> External '{doc_store.get_external_id(1)}'")


def demo_collection_statistics():
    """Demo 4: Collection statistics and ranking."""
    print("\n" + "="*60)
    print("DEMO 4: Collection Statistics & Ranking")
    print("="*60)
    
    # Create statistics
    stats = CollectionStatistics()
    
    # Add documents
    doc_lengths = [100, 150, 120, 200, 80]
    print(f"\nAdding {len(doc_lengths)} documents:")
    for doc_id, length in enumerate(doc_lengths):
        stats.add_document(doc_id, length)
        print(f"  Doc {doc_id}: {length} tokens")
    
    # Show statistics
    print(f"\nCollection Statistics:")
    print(f"  Number of documents: {stats.num_documents}")
    print(f"  Total terms: {stats.total_terms}")
    print(f"  Average document length: {stats.avg_document_length:.2f}")
    
    # Calculate IDF
    print("\n\nIDF Calculations:")
    test_cases = [
        ("common", 4),  # Appears in 4 docs
        ("medium", 2),  # Appears in 2 docs
        ("rare", 1),    # Appears in 1 doc
    ]
    
    for term, df in test_cases:
        idf = stats.calculate_idf(term, df)
        print(f"  Term '{term}' (DF={df}): IDF = {idf:.4f}")
    
    # Calculate BM25
    print("\n\nBM25 Scores (for term with DF=2):")
    for doc_id in range(3):
        doc_length = doc_lengths[doc_id]
        for tf in [1, 2, 5]:
            bm25 = stats.calculate_bm25_score(
                term_frequency=tf,
                document_frequency=2,
                doc_length=doc_length
            )
            print(f"  Doc {doc_id} (length={doc_length}), TF={tf}: BM25 = {bm25:.4f}")


def demo_complete_system():
    """Demo 5: All components working together."""
    print("\n" + "="*60)
    print("DEMO 5: Complete System Integration")
    print("="*60)
    
    # Create all components
    index = InvertedIndex()
    doc_store = DocumentStore()
    stats = CollectionStatistics()
    
    # Sample documents
    documents = [
        ("news_1", "The quick brown fox jumps over the lazy dog", 
         "Fox Jumps Over Dog", "http://example.com/news1"),
        ("news_2", "The lazy dog sleeps all day long", 
         "Lazy Dog Sleeps", "http://example.com/news2"),
        ("news_3", "The quick fox is very clever and fast",
         "Clever Fox", "http://example.com/news3"),
        ("news_4", "A brown dog runs in the park",
         "Dog Runs in Park", "http://example.com/news4"),
    ]
    
    print("\nIndexing documents:")
    for doc_id, text, title, url in documents:
        # Tokenize (simple split for demo)
        tokens = text.lower().split()
        
        # Add to document store
        internal_id = doc_store.add_document(
            doc_id=doc_id,
            metadata={
                'title': title,
                'url': url,
                'length': len(tokens),
                'text': text
            }
        )
        
        # Add to index
        index.add_document(doc_id=internal_id, tokens=tokens)
        
        # Add to statistics
        stats.add_document(doc_id=internal_id, doc_length=len(tokens))
        
        print(f"  {doc_id}: {title}")
    
    # Query: Find documents containing "dog"
    query_term = "dog"
    print(f"\n\nQuery: Find documents containing '{query_term}'")
    print("-" * 60)
    
    postings = index.get_postings(query_term)
    if postings:
        print(f"Found {postings.document_frequency()} documents:\n")
        
        # Get and rank results
        results = []
        df = postings.document_frequency()
        
        for posting in postings:
            doc_id = posting.doc_id
            doc = doc_store.get_document(doc_id)
            
            # Calculate BM25 score
            bm25_score = stats.calculate_bm25_score(
                term_frequency=posting.term_freq,
                document_frequency=df,
                doc_length=doc['length']
            )
            
            results.append({
                'doc_id': doc_id,
                'external_id': doc_store.get_external_id(doc_id),
                'title': doc['title'],
                'tf': posting.term_freq,
                'positions': posting.positions,
                'bm25': bm25_score
            })
        
        # Sort by BM25 score
        results.sort(key=lambda x: x['bm25'], reverse=True)
        
        # Display results
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']} ({result['external_id']})")
            print(f"   BM25 Score: {result['bm25']:.4f}")
            print(f"   Term Frequency: {result['tf']}")
            print(f"   Positions: {result['positions']}")
            print()


def demo_serialization():
    """Demo 6: Serialization and persistence."""
    print("\n" + "="*60)
    print("DEMO 6: Serialization & Persistence")
    print("="*60)
    
    # Create and populate index
    print("\nCreating index...")
    index = InvertedIndex()
    
    docs = [
        (0, ["python", "programming", "language"]),
        (1, ["java", "programming", "language"]),
        (2, ["python", "is", "awesome"]),
    ]
    
    for doc_id, tokens in docs:
        index.add_document(doc_id, tokens)
    
    print(f"Index created: {index.num_documents} documents, {index.get_vocabulary_size()} terms")
    
    # Serialize to dictionary
    print("\nSerializing to dictionary...")
    data = index.to_dict()
    
    # Save to file
    output_file = "demo_index.json"
    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    import os
    size_kb = os.path.getsize(output_file) / 1024
    print(f"File size: {size_kb:.2f} KB")
    
    # Load from file
    print(f"\nLoading from {output_file}...")
    with open(output_file, 'r') as f:
        loaded_data = json.load(f)
    
    # Restore index
    print("Restoring index...")
    restored_index = InvertedIndex.from_dict(loaded_data)
    
    # Verify
    print(f"\nVerification:")
    print(f"  Original documents: {index.num_documents}")
    print(f"  Restored documents: {restored_index.num_documents}")
    print(f"  Original vocabulary: {index.get_vocabulary_size()}")
    print(f"  Restored vocabulary: {restored_index.get_vocabulary_size()}")
    
    # Compare vocabularies
    orig_vocab = index.get_vocabulary()
    restored_vocab = restored_index.get_vocabulary()
    print(f"  Vocabularies match: {orig_vocab == restored_vocab}")
    
    # Clean up
    os.remove(output_file)
    print(f"\nCleaned up {output_file}")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("Phase 1: Core Data Structures - Demo")
    print("="*60)
    
    try:
        demo_basic_index()
        demo_skip_pointers()
        demo_document_store()
        demo_collection_statistics()
        demo_complete_system()
        demo_serialization()
        
        print("\n" + "="*60)
        print("All demos completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()