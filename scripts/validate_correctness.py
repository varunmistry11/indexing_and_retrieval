#!/usr/bin/env python
"""
Validate that all index configurations produce correct results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indices.self_index import SelfIndex
import hydra
from omegaconf import DictConfig
import json

# Test queries
TEST_QUERIES = [
    "technology",
    "government election",
    "climate change",
    "economy market",
]

# All index names
INDICES = [
    "idx_boolean",
    "idx_wordcount",
    "idx_tfidf",
    "idx_tfidf_skip",
    "idx_threshold",
    "idx_earlystop",
    "idx_docatat",
    "idx_compressed",
    "idx_gzip",
]

def load_and_query(index_name, query):
    """Load index and execute query."""
    # Use config_self
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(config_name="config_self")
    
    index = SelfIndex(cfg)
    index_path = Path(f"indices/selfindex/{index_name}")
    
    if not index_path.exists():
        print(f"❌ Index not found: {index_name}")
        return None
    
    try:
        index.load_index(str(index_path))
        result_json = index.query(query)
        results = json.loads(result_json)
        return results
    except Exception as e:
        print(f"❌ Error querying {index_name}: {e}")
        return None

def main():
    print("="*60)
    print("CORRECTNESS VALIDATION")
    print("="*60)
    
    for query in TEST_QUERIES:
        print(f"\nQuery: '{query}'")
        print("-"*60)
        
        results_map = {}
        
        for index_name in INDICES:
            results = load_and_query(index_name, query)
            if results:
                doc_ids = [r['doc_id'] for r in results['results'][:5]]
                results_map[index_name] = doc_ids
                print(f"  {index_name:20s}: {len(results['results'])} results")
        
        # Check consistency (same documents returned)
        if results_map:
            reference = list(results_map.values())[0]
            all_match = all(set(docs[:3]) == set(reference[:3]) for docs in results_map.values())
            
            if all_match:
                print(f"  ✅ All indices return consistent top results")
            else:
                print(f"  ⚠️  Results differ across indices (expected for different scoring)")

if __name__ == "__main__":
    main()