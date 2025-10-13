import random
import json
import logging
from pathlib import Path
from typing import List, Dict, Set
import itertools

logger = logging.getLogger(__name__)


class QueryGenerator:
    """
    Generate diverse test queries for benchmarking.
    Ensures reproducibility and dataset-appropriate queries.
    """
    
    def __init__(self, config, seed: int = 42):
        """
        Initialize query generator.
        
        Args:
            config: Hydra configuration object
            seed: Random seed for reproducibility
        """
        self.config = config
        self.seed = seed
        random.seed(seed)
        
        # Load dataset-specific terms
        self.base_terms = self._get_dataset_terms()
        
        # Query patterns
        self.boolean_operators = ['AND', 'OR', 'NOT']
        self.query_types = [
            'single_term',
            'two_term_and',
            'two_term_or',
            'three_term_and',
            'three_term_or',
            'phrase',
            'not_query',
            'complex_boolean'
        ]
    
    def _get_dataset_terms(self) -> Dict[str, List[str]]:
        """Get relevant terms for each dataset type."""
        
        # Base terms for different datasets
        wiki_terms = {
            'geography': ['country', 'city', 'mountain', 'river', 'ocean', 'continent', 
                         'island', 'capital', 'region', 'territory'],
            'history': ['war', 'battle', 'revolution', 'empire', 'dynasty', 'treaty',
                       'independence', 'ancient', 'medieval', 'century'],
            'science': ['physics', 'chemistry', 'biology', 'theory', 'experiment',
                       'discovery', 'research', 'scientist', 'laboratory', 'hypothesis'],
            'arts': ['music', 'painting', 'literature', 'artist', 'composer', 'novel',
                    'film', 'theater', 'sculpture', 'poetry'],
            'politics': ['government', 'election', 'president', 'parliament', 'democracy',
                        'law', 'constitution', 'minister', 'party', 'vote'],
            'technology': ['computer', 'internet', 'software', 'network', 'digital',
                          'technology', 'system', 'algorithm', 'data', 'programming'],
            'sports': ['football', 'basketball', 'olympic', 'championship', 'team',
                      'player', 'competition', 'tournament', 'league', 'medal'],
            'education': ['university', 'school', 'student', 'education', 'professor',
                         'college', 'degree', 'research', 'learning', 'academic']
        }
        
        news_terms = {
            'politics': ['election', 'president', 'government', 'policy', 'minister',
                        'vote', 'campaign', 'senate', 'congress', 'bill'],
            'economy': ['market', 'stock', 'economy', 'trade', 'business', 'company',
                       'growth', 'inflation', 'investment', 'finance'],
            'international': ['country', 'nation', 'global', 'international', 'foreign',
                            'diplomacy', 'summit', 'agreement', 'treaty', 'relations'],
            'technology': ['technology', 'digital', 'innovation', 'startup', 'app',
                          'platform', 'software', 'device', 'data', 'internet'],
            'health': ['health', 'medical', 'hospital', 'doctor', 'treatment', 'disease',
                      'patient', 'vaccine', 'pandemic', 'care'],
            'environment': ['climate', 'environment', 'pollution', 'energy', 'emissions',
                          'renewable', 'sustainability', 'carbon', 'temperature', 'weather'],
            'sports': ['team', 'player', 'game', 'match', 'victory', 'championship',
                      'season', 'coach', 'score', 'tournament'],
            'crime': ['police', 'arrest', 'investigation', 'court', 'trial', 'crime',
                     'suspect', 'evidence', 'justice', 'lawsuit']
        }
        
        # Combine based on dataset
        dataset_name = self.config.dataset.name.lower()
        
        if 'wiki' in dataset_name:
            return wiki_terms
        elif 'news' in dataset_name:
            return news_terms
        elif 'combined' in dataset_name:
            # Merge both dictionaries
            combined = {}
            for key in set(wiki_terms.keys()) | set(news_terms.keys()):
                combined[key] = list(set(
                    wiki_terms.get(key, []) + news_terms.get(key, [])
                ))
            return combined
        else:
            # Default to combined
            return {**wiki_terms, **news_terms}
    
    def generate_queries(self, num_queries: int = 1000, 
                        query_type_distribution: Dict[str, float] = None) -> List[str]:
        """
        Generate diverse test queries.
        
        Args:
            num_queries: Number of queries to generate
            query_type_distribution: Distribution of query types (if None, use uniform)
            
        Returns:
            List of query strings
        """
        logger.info(f"Generating {num_queries} test queries for {self.config.dataset.name} dataset")
        
        # Default distribution if not provided
        if query_type_distribution is None:
            query_type_distribution = {
                'single_term': 0.20,      # 20% single term
                'two_term_and': 0.15,     # 15% two terms with AND
                'two_term_or': 0.10,      # 10% two terms with OR
                'three_term_and': 0.10,   # 10% three terms with AND
                'three_term_or': 0.10,    # 10% three terms with OR
                'phrase': 0.15,           # 15% phrase queries
                'not_query': 0.10,        # 10% NOT queries
                'complex_boolean': 0.10   # 10% complex boolean
            }
        
        queries = []
        
        for query_type, proportion in query_type_distribution.items():
            count = int(num_queries * proportion)
            
            if query_type == 'single_term':
                queries.extend(self._generate_single_term_queries(count))
            elif query_type == 'two_term_and':
                queries.extend(self._generate_two_term_queries(count, 'AND'))
            elif query_type == 'two_term_or':
                queries.extend(self._generate_two_term_queries(count, 'OR'))
            elif query_type == 'three_term_and':
                queries.extend(self._generate_three_term_queries(count, 'AND'))
            elif query_type == 'three_term_or':
                queries.extend(self._generate_three_term_queries(count, 'OR'))
            elif query_type == 'phrase':
                queries.extend(self._generate_phrase_queries(count))
            elif query_type == 'not_query':
                queries.extend(self._generate_not_queries(count))
            elif query_type == 'complex_boolean':
                queries.extend(self._generate_complex_queries(count))
        
        # Shuffle with seed for reproducibility
        random.shuffle(queries)
        
        # Ensure exact count
        queries = queries[:num_queries]
        
        logger.info(f"Generated {len(queries)} queries")
        return queries
    
    def _generate_single_term_queries(self, count: int) -> List[str]:
        """Generate single-term queries."""
        queries = []
        all_terms = [term for terms in self.base_terms.values() for term in terms]
        
        for _ in range(count):
            term = random.choice(all_terms)
            queries.append(term)
        
        return queries
    
    def _generate_two_term_queries(self, count: int, operator: str) -> List[str]:
        """Generate two-term queries with specified operator."""
        queries = []
        all_terms = [term for terms in self.base_terms.values() for term in terms]
        
        for _ in range(count):
            # Sometimes pick from same category, sometimes different
            if random.random() < 0.5:
                # Same category
                category = random.choice(list(self.base_terms.keys()))
                terms = random.sample(self.base_terms[category], min(2, len(self.base_terms[category])))
            else:
                # Different terms
                terms = random.sample(all_terms, 2)
            
            queries.append(f"{terms[0]} {operator} {terms[1]}")
        
        return queries
    
    def _generate_three_term_queries(self, count: int, operator: str) -> List[str]:
        """Generate three-term queries with specified operator."""
        queries = []
        all_terms = [term for terms in self.base_terms.values() for term in terms]
        
        for _ in range(count):
            terms = random.sample(all_terms, 3)
            queries.append(f"{terms[0]} {operator} {terms[1]} {operator} {terms[2]}")
        
        return queries
    
    def _generate_phrase_queries(self, count: int) -> List[str]:
        """Generate phrase queries."""
        queries = []
        
        # Common two-word phrases by category
        phrases = []
        for category, terms in self.base_terms.items():
            for i in range(len(terms) - 1):
                phrases.append(f'"{terms[i]} {terms[i+1]}"')
        
        for _ in range(count):
            if phrases:
                queries.append(random.choice(phrases))
            else:
                # Fallback to random two terms
                all_terms = [term for terms in self.base_terms.values() for term in terms]
                terms = random.sample(all_terms, 2)
                queries.append(f'"{terms[0]} {terms[1]}"')
        
        return queries
    
    def _generate_not_queries(self, count: int) -> List[str]:
        """Generate NOT queries."""
        queries = []
        all_terms = [term for terms in self.base_terms.values() for term in terms]
        
        for _ in range(count):
            terms = random.sample(all_terms, 2)
            # Format: term1 AND NOT term2
            queries.append(f"{terms[0]} AND NOT {terms[1]}")
        
        return queries
    
    def _generate_complex_queries(self, count: int) -> List[str]:
        """Generate complex boolean queries with parentheses."""
        queries = []
        all_terms = [term for terms in self.base_terms.values() for term in terms]
        
        patterns = [
            "({0} OR {1}) AND {2}",
            "{0} AND ({1} OR {2})",
            "({0} AND {1}) OR {2}",
            "({0} OR {1}) AND NOT {2}",
            "{0} AND ({1} OR {2}) AND {3}",
        ]
        
        for _ in range(count):
            pattern = random.choice(patterns)
            num_terms = pattern.count('{')
            terms = random.sample(all_terms, num_terms)
            queries.append(pattern.format(*terms))
        
        return queries
    
    def save_queries(self, queries: List[str], output_path: Path):
        """
        Save queries to file for reproducibility.
        
        Args:
            queries: List of query strings
            output_path: Path to save queries
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'seed': self.seed,
                'dataset': self.config.dataset.name,
                'num_queries': len(queries),
                'queries': queries
            }, f, indent=2)
        
        logger.info(f"Saved {len(queries)} queries to {output_path}")
    
    def load_queries(self, input_path: Path) -> List[str]:
        """
        Load queries from file.
        
        Args:
            input_path: Path to load queries from
            
        Returns:
            List of query strings
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data['queries'])} queries from {input_path}")
        return data['queries']
    
    def get_query_statistics(self, queries: List[str]) -> Dict:
        """
        Get statistics about generated queries.
        
        Args:
            queries: List of query strings
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_queries': len(queries),
            'single_term': 0,
            'with_and': 0,
            'with_or': 0,
            'with_not': 0,
            'phrase_queries': 0,
            'complex_queries': 0,
            'avg_terms_per_query': 0,
            'unique_terms': set()
        }
        
        total_terms = 0
        
        for query in queries:
            # Count operators
            if ' AND ' in query:
                stats['with_and'] += 1
            if ' OR ' in query:
                stats['with_or'] += 1
            if ' NOT ' in query:
                stats['with_not'] += 1
            if '"' in query:
                stats['phrase_queries'] += 1
            if '(' in query:
                stats['complex_queries'] += 1
            
            # Count terms
            terms = query.replace('(', '').replace(')', '').replace('"', '')
            for op in ['AND', 'OR', 'NOT']:
                terms = terms.replace(f' {op} ', ' ')
            term_list = terms.split()
            total_terms += len(term_list)
            stats['unique_terms'].update(term_list)
            
            # Single term
            if ' ' not in query.strip('"'):
                stats['single_term'] += 1
        
        stats['avg_terms_per_query'] = total_terms / len(queries) if queries else 0
        stats['unique_terms'] = len(stats['unique_terms'])
        
        return stats