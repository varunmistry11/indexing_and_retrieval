import re
import string
from typing import List, Set
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    """Handles all text preprocessing tasks."""
    
    def __init__(self, config):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Hydra config object with preprocessing settings
        """
        self.config = config
        self.stemmer = PorterStemmer() if config.preprocessing.stemming else None
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Load stopwords if needed
        if config.preprocessing.remove_stopwords:
            self.stopwords = set(stopwords.words('english'))
        else:
            self.stopwords = set()
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
    
    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess text according to configuration.
        
        Args:
            text: Input text string
            
        Returns:
            List of processed tokens
        """
        if not text:
            return []
        
        # Lowercase
        if self.config.preprocessing.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.config.preprocessing.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter by length
        tokens = [
            token for token in tokens
            if self.config.preprocessing.min_word_length <= len(token) <= self.config.preprocessing.max_word_length
        ]
        
        # Remove stopwords
        if self.config.preprocessing.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        # Stemming
        if self.config.preprocessing.stemming and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        return tokens
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess query while preserving operators.
        
        Args:
            query: Query string
            
        Returns:
            Preprocessed query string
        """
        # Define operators to preserve
        operators = {'AND', 'OR', 'NOT', '(', ')', '"'}
        
        # Split by spaces while preserving quoted phrases
        parts = []
        in_quote = False
        current_part = []
        
        for char in query:
            if char == '"':
                if in_quote and current_part:
                    # End of quoted phrase
                    phrase = ''.join(current_part)
                    parts.append(f'"{phrase}"')
                    current_part = []
                in_quote = not in_quote
            elif char == ' ' and not in_quote:
                if current_part:
                    parts.append(''.join(current_part))
                    current_part = []
            else:
                current_part.append(char)
        
        if current_part:
            parts.append(''.join(current_part))
        
        # Process each part
        processed_parts = []
        for part in parts:
            if part in operators:
                processed_parts.append(part)
            elif part.startswith('"') and part.endswith('"'):
                # Process quoted phrase
                phrase_content = part[1:-1]
                processed_tokens = self.preprocess(phrase_content)
                if processed_tokens:
                    processed_parts.append(f'"{" ".join(processed_tokens)}"')
            else:
                # Process regular term
                processed_tokens = self.preprocess(part)
                if processed_tokens:
                    processed_parts.append(processed_tokens[0])
        
        return ' '.join(processed_parts)
    
    def get_word_frequencies(self, texts: List[str]) -> dict:
        """
        Calculate word frequencies across multiple texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary mapping words to frequencies
        """
        word_freq = {}
        
        for text in texts:
            tokens = self.preprocess(text)
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
        
        return word_freq