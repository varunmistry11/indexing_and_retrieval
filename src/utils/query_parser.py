import re
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class BooleanQueryParser:
    """
    Parse and convert boolean queries to Elasticsearch query format.
    Supports: AND, OR, NOT, phrase queries, and parentheses.
    """
    
    def __init__(self):
        self.operators = {'AND', 'OR', 'NOT'}
    
    def parse_to_elasticsearch(self, query: str, fields: List[str] = None) -> Dict:
        """
        Parse a boolean query string and convert to Elasticsearch query DSL.
        
        Args:
            query: Query string (e.g., "(python OR java) AND NOT beginner")
            fields: List of fields to search (default: ["title^2", "text"])
            
        Returns:
            Elasticsearch query dict
        """
        if fields is None:
            fields = ["title^2", "text"]
        
        query = query.strip()
        
        # Check if it's a simple query or complex
        if not any(op in query for op in ['AND', 'OR', 'NOT', '(', ')']):
            # Simple query - just terms or phrase
            return self._simple_query(query, fields)
        
        # Complex boolean query - parse it
        try:
            parsed = self._parse_boolean_expression(query)
            es_query = self._convert_to_es_query(parsed, fields)
            return es_query
        except Exception as e:
            logger.warning(f"Failed to parse complex query '{query}': {e}. Using simple query.")
            return self._simple_query(query, fields)
    
    def _simple_query(self, query: str, fields: List[str]) -> Dict:
        """Handle simple queries (no boolean operators)."""
        query = query.strip()
        
        # Check if it's a phrase query
        if query.startswith('"') and query.endswith('"'):
            phrase = query.strip('"')
            return {
                "bool": {
                    "should": [
                        {"match_phrase": {field.split('^')[0]: phrase}}
                        for field in fields
                    ],
                    "minimum_should_match": 1
                }
            }
        
        # Simple multi-match query
        return {
            "multi_match": {
                "query": query,
                "fields": fields,
                "type": "best_fields"
            }
        }
    
    def _parse_boolean_expression(self, query: str) -> Dict:
        """
        Parse a boolean expression into a tree structure.
        
        Returns a dict representing the query tree:
        {
            'type': 'AND' | 'OR' | 'NOT' | 'TERM' | 'PHRASE',
            'value': str (for TERM/PHRASE) or None,
            'children': List[Dict] (for AND/OR/NOT)
        }
        """
        # Tokenize the query
        tokens = self._tokenize(query)
        
        # Parse tokens into expression tree
        expr, _ = self._parse_or_expression(tokens, 0)
        return expr
    
    def _tokenize(self, query: str) -> List[str]:
        """Tokenize query string preserving operators, parentheses, and phrases."""
        tokens = []
        current_token = []
        in_phrase = False
        i = 0
        
        while i < len(query):
            char = query[i]
            
            if char == '"':
                if in_phrase:
                    # End of phrase
                    current_token.append(char)
                    tokens.append(''.join(current_token))
                    current_token = []
                    in_phrase = False
                else:
                    # Start of phrase
                    if current_token:
                        tokens.append(''.join(current_token).strip())
                        current_token = []
                    current_token.append(char)
                    in_phrase = True
                i += 1
            elif in_phrase:
                current_token.append(char)
                i += 1
            elif char in '()':
                if current_token:
                    tokens.append(''.join(current_token).strip())
                    current_token = []
                tokens.append(char)
                i += 1
            elif char == ' ':
                if current_token:
                    token = ''.join(current_token).strip()
                    if token:
                        tokens.append(token)
                    current_token = []
                i += 1
            else:
                current_token.append(char)
                i += 1
        
        if current_token:
            tokens.append(''.join(current_token).strip())
        
        # Remove empty tokens
        tokens = [t for t in tokens if t]
        
        return tokens
    
    def _parse_or_expression(self, tokens: List[str], pos: int) -> Tuple[Dict, int]:
        """Parse OR expression (lowest precedence)."""
        left, pos = self._parse_and_expression(tokens, pos)
        
        while pos < len(tokens) and tokens[pos] == 'OR':
            pos += 1  # Skip OR
            right, pos = self._parse_and_expression(tokens, pos)
            left = {
                'type': 'OR',
                'children': [left, right]
            }
        
        return left, pos
    
    def _parse_and_expression(self, tokens: List[str], pos: int) -> Tuple[Dict, int]:
        """Parse AND expression (medium precedence)."""
        left, pos = self._parse_not_expression(tokens, pos)
        
        while pos < len(tokens) and tokens[pos] == 'AND':
            pos += 1  # Skip AND
            right, pos = self._parse_not_expression(tokens, pos)
            left = {
                'type': 'AND',
                'children': [left, right]
            }
        
        return left, pos
    
    def _parse_not_expression(self, tokens: List[str], pos: int) -> Tuple[Dict, int]:
        """Parse NOT expression (high precedence)."""
        if pos < len(tokens) and tokens[pos] == 'NOT':
            pos += 1  # Skip NOT
            expr, pos = self._parse_primary(tokens, pos)
            return {
                'type': 'NOT',
                'children': [expr]
            }, pos
        
        return self._parse_primary(tokens, pos)
    
    def _parse_primary(self, tokens: List[str], pos: int) -> Tuple[Dict, int]:
        """Parse primary expression (parentheses or term)."""
        if pos >= len(tokens):
            raise ValueError("Unexpected end of query")
        
        token = tokens[pos]
        
        # Handle parentheses
        if token == '(':
            pos += 1  # Skip (
            expr, pos = self._parse_or_expression(tokens, pos)
            if pos >= len(tokens) or tokens[pos] != ')':
                raise ValueError("Missing closing parenthesis")
            pos += 1  # Skip )
            return expr, pos
        
        # Handle phrase query
        if token.startswith('"') and token.endswith('"'):
            return {
                'type': 'PHRASE',
                'value': token.strip('"')
            }, pos + 1
        
        # Handle term
        if token not in self.operators and token not in '()':
            return {
                'type': 'TERM',
                'value': token
            }, pos + 1
        
        raise ValueError(f"Unexpected token: {token}")
    
    def _convert_to_es_query(self, expr: Dict, fields: List[str]) -> Dict:
        """Convert parsed expression tree to Elasticsearch query DSL."""
        expr_type = expr['type']
        
        if expr_type == 'TERM':
            # Simple term query
            return {
                "multi_match": {
                    "query": expr['value'],
                    "fields": fields,
                    "type": "best_fields"
                }
            }
        
        elif expr_type == 'PHRASE':
            # Phrase query
            return {
                "bool": {
                    "should": [
                        {"match_phrase": {field.split('^')[0]: expr['value']}}
                        for field in fields
                    ],
                    "minimum_should_match": 1
                }
            }
        
        elif expr_type == 'AND':
            # AND query - all children must match
            children_queries = [
                self._convert_to_es_query(child, fields)
                for child in expr['children']
            ]
            return {
                "bool": {
                    "must": children_queries
                }
            }
        
        elif expr_type == 'OR':
            # OR query - at least one child must match
            children_queries = [
                self._convert_to_es_query(child, fields)
                for child in expr['children']
            ]
            return {
                "bool": {
                    "should": children_queries,
                    "minimum_should_match": 1
                }
            }
        
        elif expr_type == 'NOT':
            # NOT query - must not match
            child_query = self._convert_to_es_query(expr['children'][0], fields)
            return {
                "bool": {
                    "must_not": [child_query]
                }
            }
        
        else:
            raise ValueError(f"Unknown expression type: {expr_type}")
    
    def explain_query(self, query: str) -> str:
        """
        Explain how a query will be parsed.
        Useful for debugging.
        """
        try:
            tokens = self._tokenize(query)
            parsed = self._parse_boolean_expression(query)
            
            explanation = f"Query: {query}\n"
            explanation += f"Tokens: {tokens}\n"
            explanation += f"Parsed tree: {self._format_tree(parsed)}\n"
            
            return explanation
        except Exception as e:
            return f"Failed to parse query: {e}"
    
    def _format_tree(self, expr: Dict, indent: int = 0) -> str:
        """Format expression tree for display."""
        spaces = "  " * indent
        
        if expr['type'] in ['TERM', 'PHRASE']:
            return f"{spaces}{expr['type']}: {expr['value']}\n"
        else:
            result = f"{spaces}{expr['type']}:\n"
            for child in expr['children']:
                result += self._format_tree(child, indent + 1)
            return result


def test_query_parser():
    """Test the query parser with various queries."""
    parser = BooleanQueryParser()
    
    test_queries = [
        "python",
        '"machine learning"',
        "python AND java",
        "python OR java",
        "python AND NOT beginner",
        "(python OR java) AND programming",
        "machine AND learning AND NOT supervised",
        '("natural language" OR NLP) AND processing',
        "((A OR B) AND C) OR D",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        # Parse and convert
        es_query = parser.parse_to_elasticsearch(query)
        
        # Pretty print the result
        import json
        print(json.dumps(es_query, indent=2))
        
        # Show explanation
        print("\nExplanation:")
        print(parser.explain_query(query))


if __name__ == "__main__":
    test_query_parser()