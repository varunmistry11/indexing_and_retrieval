#!/usr/bin/env python
"""
Demo script for Phase 2: Boolean Operations & Query Processing

This demonstrates boolean operations, phrase queries, and query processors.
Run with: python examples/phase2_demo.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.selfindex import (
    InvertedIndex, DocumentStore, CollectionStatistics,
    BooleanOperations, PhraseQueryProcessor,
    TermAtATimeProcessor, DocumentAtATimeProcessor,
    BooleanQueryProcessor
)


def demo_boolean_operations():
    """Demo 1: Boolean operations on postings lists."""
    print("="*60)
    print("DEMO 1: Boolean Operations")
    print("="*60)
    
    # Create index
    index = InvertedIndex()
    
    docs = [
        (0, ["python", "programming", "language"]),
        (1, ["java", "programming", "language"]),
        (2, ["python", "java", "comparison"]),
        (3, ["python", "tutorial"]),
    ]
    
    for doc_id, tokens in docs:
        index.add_document(doc_id, tokens)
    
    print("\nDocuments:")
    for doc_id, tokens in docs:
        print(f"  Doc {doc_id}: {' '.join(tokens)}")
    
    # AND operation
    print("\n\nAND Operation: python AND java")
    python_postings = index.get_postings("python")
    java_postings = index.get_postings("java")
    and_result = BooleanOperations.intersect(python_postings, java_postings)
    print(f"  Result: {and_result.get_doc_ids()}")
    
    # OR operation
    print("\nOR Operation: python OR java")
    or_result = BooleanOperations.union(python_postings, java_postings)
    print(f"  Result: {or_result.get_doc_ids()}")
    
    # NOT operation
    print("\nNOT Operation: NOT python")
    all_docs = {0, 1, 2, 3}
    not_result = BooleanOperations.negate(python_postings, all_docs)
    print(f"  Result: {not_result.get_doc_ids()}")
    
    # AND NOT operation
    print("\nAND NOT Operation: programming AND NOT python")
    programming_postings = index.get_postings("programming")
    and_not_result = BooleanOperations.and_not(programming_postings, python_postings)
    print(f"  Result: {and_not_result.get_doc_ids()}")


def demo_phrase_queries():
    """Demo 2: Phrase query processing."""
    print("\n" + "="*60)
    print("DEMO 2: Phrase Queries")
    print("="*60)
    
    # Create index
    index = InvertedIndex()
    
    docs = [
        (0, ["natural", "language", "processing", "is", "fun"]),
        (1, ["language", "processing", "techniques"]),
        (2, ["natural", "language", "understanding"]),
        (3, ["processing", "natural", "language", "data"]),
    ]
    
    print("\nDocuments:")
    for doc_id, tokens in docs:
        print(f"  Doc {doc_id}: {' '.join(tokens)}")
    
    for doc_id, tokens in docs:
        index.add_document(doc_id, tokens)
    
    # Phrase query: "natural language"
    print("\n\nPhrase Query: 'natural language'")
    natural = index.get_postings("natural")
    language = index.get_postings("language")
    result = PhraseQueryProcessor.phrase_query([natural, language])
    
    print(f"  Documents with phrase: {result.get_doc_ids()}")
    for doc_id in result.get_doc_ids():
        positions = result.get_positions(doc_id)
        print(f"    Doc {doc_id} at positions: {positions}")
    
    # Phrase query: "language processing"
    print("\nPhrase Query: 'language processing'")
    processing = index.get_postings("processing")
    result = PhraseQueryProcessor.phrase_query([language, processing])
    
    print(f"  Documents with phrase: {result.get_doc_ids()}")
    for doc_id in result.get_doc_ids():
        positions = result.get_positions(doc_id)
        print(f"    Doc {doc_id} at positions: {positions}")


def demo_term_at_a_time():
    """Demo 3: Term-at-a-time query processing."""
    print("\n" + "="*60)
    print("DEMO 3: Term-at-a-Time Query Processing")
    print("="*60)
    
    # Create index with statistics
    index = InvertedIndex()
    doc_store = DocumentStore()
    stats = CollectionStatistics()
    
    documents = [
        ("article1", ["machine", "learning", "algorithms"]),
        ("article2", ["deep", "learning", "neural", "networks"]),
        ("article3", ["machine", "learning", "applications"]),
        ("article4", ["supervised", "learning", "methods"]),
    ]
    
    print("\nDocuments:")
    for doc_id, tokens in documents:
        print(f"  {doc_id}: {' '.join(tokens)}")
    
    for i, (doc_id, tokens) in enumerate(documents):
        index.add_document(i, tokens)
        doc_store.add_document(doc_id, {'length': len(tokens)})
        stats.add_document(i, len(tokens))
    
    # Create processor
    processor = TermAtATimeProcessor(index, stats, scoring_method='bm25')
    
    # Query 1: Single term
    print("\n\nQuery: 'learning'")
    results = processor.process_query(["learning"], k=10)
    print(f"  Found {len(results)} documents:")
    for i, result in enumerate(results, 1):
        external_id = doc_store.get_external_id(result.doc_id)
        print(f"    {i}. {external_id} (score: {result.score:.4f})")
    
    # Query 2: Multiple terms
    print("\nQuery: 'machine learning'")
    results = processor.process_query(["machine", "learning"], k=10)
    print(f"  Found {len(results)} documents:")
    for i, result in enumerate(results, 1):
        external_id = doc_store.get_external_id(result.doc_id)
        print(f"    {i}. {external_id} (score: {result.score:.4f})")


def demo_document_at_a_time():
    """Demo 4: Document-at-a-time query processing."""
    print("\n" + "="*60)
    print("DEMO 4: Document-at-a-Time Query Processing")
    print("="*60)
    
    # Create index
    index = InvertedIndex()
    doc_store = DocumentStore()
    stats = CollectionStatistics()
    
    documents = [
        ("paper1", ["information", "retrieval", "systems"]),
        ("paper2", ["search", "engines", "ranking"]),
        ("paper3", ["information", "retrieval", "evaluation"]),
        ("paper4", ["search", "algorithms", "ranking"]),
    ]
    
    print("\nDocuments:")
    for doc_id, tokens in documents:
        print(f"  {doc_id}: {' '.join(tokens)}")
    
    for i, (doc_id, tokens) in enumerate(documents):
        index.add_document(i, tokens)
        doc_store.add_document(doc_id, {'length': len(tokens)})
        stats.add_document(i, len(tokens))
    
    # Create processor
    processor = DocumentAtATimeProcessor(index, stats, scoring_method='bm25')
    
    # Query
    print("\n\nQuery: 'search ranking'")
    results = processor.process_query(["search", "ranking"], k=10)
    print(f"  Found {len(results)} documents:")
    for i, result in enumerate(results, 1):
        external_id = doc_store.get_external_id(result.doc_id)
        print(f"    {i}. {external_id} (score: {result.score:.4f})")


def demo_complex_boolean_queries():
    """Demo 5: Complex boolean queries."""
    print("\n" + "="*60)
    print("DEMO 5: Complex Boolean Queries")
    print("="*60)
    
    # Create index
    index = InvertedIndex()
    doc_store = DocumentStore()
    
    documents = [
        ("doc1", ["python", "machine", "learning", "tutorial"]),
        ("doc2", ["java", "machine", "learning", "guide"]),
        ("doc3", ["python", "programming", "basics"]),
        ("doc4", ["machine", "learning", "algorithms"]),
        ("doc5", ["python", "data", "science"]),
    ]
    
    print("\nDocuments:")
    for doc_id, tokens in documents:
        print(f"  {doc_id}: {' '.join(tokens)}")
    
    for i, (doc_id, tokens) in enumerate(documents):
        index.add_document(i, tokens)
        doc_store.add_document(doc_id, {'length': len(tokens)})
    
    all_docs = set(range(len(documents)))
    processor = BooleanQueryProcessor(index, all_docs)
    
    # Query 1: Simple AND
    print("\n\nQuery: python AND learning")
    result = processor.process_and_query(["python", "learning"])
    doc_ids = [doc_store.get_external_id(d) for d in result.get_doc_ids()]
    print(f"  Results: {doc_ids}")
    
    # Query 2: OR query
    print("\nQuery: java OR data")
    result = processor.process_or_query(["java", "data"])
    doc_ids = [doc_store.get_external_id(d) for d in result.get_doc_ids()]
    print(f"  Results: {doc_ids}")
    
    # Query 3: AND NOT
    print("\nQuery: machine AND learning AND NOT python")
    result = processor.process_and_not_query(
        positive_terms=["machine", "learning"],
        negative_terms=["python"]
    )
    doc_ids = [doc_store.get_external_id(d) for d in result.get_doc_ids()]
    print(f"  Results: {doc_ids}")
    
    # Query 4: Complex nested query
    print("\nQuery: (python OR java) AND machine")
    query_tree = {
        'type': 'AND',
        'children': [
            {
                'type': 'OR',
                'children': [
                    {'type': 'TERM', 'value': 'java'}
                ]
            },
            {'type': 'TERM', 'value': 'machine'}
        ]
    }
    result = processor.process_complex_query(query_tree)
    doc_ids = [doc_store.get_external_id(d) for d in result.get_doc_ids()]
    print(f"  Results: {doc_ids}")


def demo_comparison():
    """Demo 6: Compare Term-at-a-time vs Document-at-a-time."""
    print("\n" + "="*60)
    print("DEMO 6: Comparing Query Processing Methods")
    print("="*60)
    
    # Create index
    index = InvertedIndex()
    doc_store = DocumentStore()
    stats = CollectionStatistics()
    
    documents = [
        ("news1", ["election", "results", "announced", "today"]),
        ("news2", ["election", "campaign", "updates"]),
        ("news3", ["results", "show", "close", "race"]),
        ("news4", ["campaign", "trail", "events"]),
    ]
    
    print("\nDocuments:")
    for doc_id, tokens in documents:
        print(f"  {doc_id}: {' '.join(tokens)}")
    
    for i, (doc_id, tokens) in enumerate(documents):
        index.add_document(i, tokens)
        doc_store.add_document(doc_id, {'length': len(tokens)})
        stats.add_document(i, len(tokens))
    
    # Create both processors
    tat = TermAtATimeProcessor(index, stats, scoring_method='bm25')
    dat = DocumentAtATimeProcessor(index, stats, scoring_method='bm25')
    
    # Query
    query_terms = ["election", "results"]
    print(f"\n\nQuery: '{' '.join(query_terms)}'")
    
    # Term-at-a-time
    print("\nTerm-at-a-Time Results:")
    tat_results = tat.process_query(query_terms, k=10)
    for i, result in enumerate(tat_results, 1):
        external_id = doc_store.get_external_id(result.doc_id)
        print(f"  {i}. {external_id} (score: {result.score:.4f})")
    
    # Document-at-a-time
    print("\nDocument-at-a-Time Results:")
    dat_results = dat.process_query(query_terms, k=10)
    for i, result in enumerate(dat_results, 1):
        external_id = doc_store.get_external_id(result.doc_id)
        print(f"  {i}. {external_id} (score: {result.score:.4f})")
    
    # Compare
    print("\nComparison:")
    print(f"  Both methods return same number of results: {len(tat_results) == len(dat_results)}")
    
    # Check if top results are the same
    if tat_results and dat_results:
        tat_top = tat_results[0].doc_id
        dat_top = dat_results[0].doc_id
        print(f"  Same top result: {tat_top == dat_top}")
        print(f"  Scores identical: {abs(tat_results[0].score - dat_results[0].score) < 0.0001}")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("Phase 2: Boolean Operations & Query Processing - Demo")
    print("="*60)
    
    try:
        demo_boolean_operations()
        demo_phrase_queries()
        demo_term_at_a_time()
        demo_document_at_a_time()
        demo_complex_boolean_queries()
        demo_comparison()
        
        print("\n" + "="*60)
        print("All demos completed successfully!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("  ✓ Boolean operations (AND, OR, NOT)")
        print("  ✓ Phrase query processing")
        print("  ✓ Term-at-a-time query processing")
        print("  ✓ Document-at-a-time query processing")
        print("  ✓ Complex nested boolean queries")
        print("  ✓ BM25 scoring")
        print("\nPhase 2 Implementation Complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
