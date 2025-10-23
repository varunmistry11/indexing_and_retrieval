"""
Compression utilities for SelfIndex.
Implements gap encoding and variable byte encoding for postings lists.
"""

from typing import List, Tuple
import struct
import gzip
import logging

logger = logging.getLogger(__name__)


class GapEncoder:
    """
    Gap encoding for postings lists.
    Stores differences between consecutive values instead of absolute values.
    Works best for sorted lists of integers.
    """
    
    @staticmethod
    def encode(values: List[int]) -> List[int]:
        """
        Encode a list of integers as gaps.
        
        Args:
            values: List of integers (must be sorted ascending)
            
        Returns:
            List of gaps (differences between consecutive values)
            
        Example:
            [1, 5, 10, 15, 20] -> [1, 4, 5, 5, 5]
        
        Raises:
            ValueError: If values are not sorted
        """
        if not values:
            return []
        
        # Verify sorted
        for i in range(1, len(values)):
            if values[i] < values[i - 1]:
                raise ValueError("Values must be sorted in ascending order for gap encoding")
        
        gaps = [values[0]]
        for i in range(1, len(values)):
            gap = values[i] - values[i - 1]
            gaps.append(gap)
        
        return gaps
    
    @staticmethod
    def decode(gaps: List[int]) -> List[int]:
        """
        Decode gaps back to original values.
        
        Args:
            gaps: List of gaps
            
        Returns:
            List of original integers
            
        Example:
            [1, 4, 5, 5, 5] -> [1, 5, 10, 15, 20]
        """
        if not gaps:
            return []
        
        values = [gaps[0]]
        for gap in gaps[1:]:
            values.append(values[-1] + gap)
        
        return values


class VariableByteEncoder:
    """
    Variable byte encoding for integers.
    Uses fewer bytes for smaller numbers.
    
    Encoding format:
    - 7 bits for data per byte
    - 1 bit for continuation (1 = more bytes follow, 0 = last byte)
    """
    
    @staticmethod
    def encode_number(n: int) -> bytes:
        """
        Encode a single integer using variable byte encoding.
        
        Args:
            n: Integer to encode (must be non-negative)
            
        Returns:
            Bytes representing the encoded integer
            
        Raises:
            ValueError: If n is negative
        """
        if n < 0:
            raise ValueError("Variable byte encoding only works for non-negative integers")
        
        if n == 0:
            return bytes([0])
        
        result = []
        
        while n > 0:
            # Take lowest 7 bits
            byte = n & 0x7F
            n >>= 7
            
            # Set continuation bit if more bytes follow
            if n > 0:
                byte |= 0x80
            
            result.append(byte)
        
        # Reverse to get big-endian order
        return bytes(reversed(result))
    
    @staticmethod
    def decode_number(data: bytes, offset: int = 0) -> Tuple[int, int]:
        """
        Decode a single integer from variable byte encoding.
        
        Args:
            data: Bytes containing encoded data
            offset: Starting position in data
            
        Returns:
            Tuple of (decoded_integer, bytes_consumed)
        """
        n = 0
        bytes_read = 0
        
        while offset + bytes_read < len(data):
            byte = data[offset + bytes_read]
            bytes_read += 1
            
            # Add 7 bits of data
            n = (n << 7) | (byte & 0x7F)
            
            # Check if this is the last byte (continuation bit = 0)
            if (byte & 0x80) == 0:
                break
        
        return n, bytes_read
    
    @staticmethod
    def encode_list(numbers: List[int]) -> bytes:
        """
        Encode a list of integers.
        
        Args:
            numbers: List of integers to encode
            
        Returns:
            Bytes containing all encoded integers
        """
        result = b''
        for n in numbers:
            result += VariableByteEncoder.encode_number(n)
        return result
    
    @staticmethod
    def decode_list(data: bytes, count: int = None) -> List[int]:
        """
        Decode a list of integers.
        
        Args:
            data: Bytes containing encoded integers
            count: Expected number of integers (if known, for validation)
            
        Returns:
            List of decoded integers
        """
        numbers = []
        offset = 0
        
        while offset < len(data):
            if count is not None and len(numbers) >= count:
                break
            
            n, bytes_read = VariableByteEncoder.decode_number(data, offset)
            numbers.append(n)
            offset += bytes_read
        
        return numbers


class CompressionManager:
    """
    Manager for compression operations in SelfIndex.
    Handles different compression strategies based on configuration.
    """
    
    def __init__(self, compression_type: str = 'NONE', use_variable_byte: bool = True):
        """
        Initialize compression manager.
        
        Args:
            compression_type: Type of compression (NONE, CODE, CLIB)
            use_variable_byte: Whether to use variable byte encoding with gap encoding
        """
        self.compression_type = compression_type
        self.use_variable_byte = use_variable_byte
        
        logger.info(f"Compression manager initialized: type={compression_type}, vbyte={use_variable_byte}")
    
    def compress_postings_list(self, postings_data: dict) -> dict:
        """
        Compress a postings list dictionary.
        
        Args:
            postings_data: Dictionary with postings list data
                          Format: {'postings': [...], 'df': int, 'total_tf': int}
            
        Returns:
            Compressed postings data dictionary
        """
        if self.compression_type == 'NONE':
            return postings_data
        
        if self.compression_type != 'CODE':
            # CLIB compression handled at file level
            return postings_data
        
        # Apply gap encoding + variable byte (z=1, c=CODE)
        compressed_postings = []
        
        for posting in postings_data['postings']:
            doc_id = posting['doc_id']
            term_freq = posting['term_freq']
            positions = posting['positions']
            
            # Apply gap encoding to positions
            if positions:
                gap_positions = GapEncoder.encode(positions)
                
                # Optionally apply variable byte encoding
                if self.use_variable_byte:
                    # Encode as bytes
                    encoded_positions = VariableByteEncoder.encode_list(gap_positions)
                    compressed_posting = {
                        'doc_id': doc_id,
                        'term_freq': term_freq,
                        'positions_compressed': True,
                        'positions_bytes': list(encoded_positions),  # Convert to list for JSON
                        'positions_count': len(positions)
                    }
                else:
                    # Just use gap encoding (still as list)
                    compressed_posting = {
                        'doc_id': doc_id,
                        'term_freq': term_freq,
                        'positions_compressed': True,
                        'positions_gaps': gap_positions,
                        'positions_count': len(positions)
                    }
            else:
                compressed_posting = {
                    'doc_id': doc_id,
                    'term_freq': term_freq,
                    'positions': []
                }
            
            compressed_postings.append(compressed_posting)
        
        return {
            'postings': compressed_postings,
            'df': postings_data['df'],
            'total_tf': postings_data['total_tf'],
            'compressed': True,
            'compression_method': 'gap_vbyte' if self.use_variable_byte else 'gap'
        }
    
    def decompress_postings_list(self, postings_data: dict) -> dict:
        """
        Decompress a postings list dictionary.
        
        Args:
            postings_data: Compressed postings data
            
        Returns:
            Decompressed postings data dictionary
        """
        if not postings_data.get('compressed', False):
            return postings_data
        
        decompressed_postings = []
        
        for posting in postings_data['postings']:
            doc_id = posting['doc_id']
            term_freq = posting['term_freq']
            
            # Decompress positions
            if posting.get('positions_compressed', False):
                if 'positions_bytes' in posting:
                    # Variable byte encoded
                    encoded_bytes = bytes(posting['positions_bytes'])
                    gap_positions = VariableByteEncoder.decode_list(
                        encoded_bytes, 
                        count=posting.get('positions_count')
                    )
                    positions = GapEncoder.decode(gap_positions)
                elif 'positions_gaps' in posting:
                    # Gap encoded only
                    positions = GapEncoder.decode(posting['positions_gaps'])
                else:
                    positions = []
            else:
                positions = posting.get('positions', [])
            
            decompressed_posting = {
                'doc_id': doc_id,
                'term_freq': term_freq,
                'positions': positions
            }
            
            decompressed_postings.append(decompressed_posting)
        
        return {
            'postings': decompressed_postings,
            'df': postings_data['df'],
            'total_tf': postings_data['total_tf']
        }
    
    def get_compression_stats(self, original_data: dict, compressed_data: dict) -> dict:
        """
        Calculate compression statistics.
        
        Args:
            original_data: Original uncompressed data
            compressed_data: Compressed data
            
        Returns:
            Dictionary with compression statistics
        """
        import json
        
        original_size = len(json.dumps(original_data))
        compressed_size = len(json.dumps(compressed_data))
        
        ratio = compressed_size / original_size if original_size > 0 else 1.0
        savings = 1.0 - ratio
        
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': ratio,
            'space_savings': savings,
            'savings_percent': savings * 100
        }


def compress_file_with_gzip(data: bytes, compression_level: int = 6) -> bytes:
    """
    Compress data using gzip (for z=2, c=CLIB).
    
    Args:
        data: Data to compress
        compression_level: Compression level (1-9, higher = better compression)
        
    Returns:
        Compressed data
    """
    return gzip.compress(data, compresslevel=compression_level)


def decompress_file_with_gzip(data: bytes) -> bytes:
    """
    Decompress gzip data.
    
    Args:
        data: Compressed data
        
    Returns:
        Decompressed data
    """
    return gzip.decompress(data)