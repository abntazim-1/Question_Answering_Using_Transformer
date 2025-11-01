"""
Data utilities for tokenization, cleaning, and text processing
"""

import re
from typing import List, Tuple
from transformers import AutoTokenizer


class DataUtils:
    """Utility functions for data preprocessing"""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        """
        Initialize data utilities
        
        Args:
            model_name: Name of the pretrained model for tokenization
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters if needed
        # text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    def tokenize(self, text: str, max_length: int = 512) -> dict:
        """
        Tokenize text using the model's tokenizer
        
        Args:
            text: Text to tokenize
            max_length: Maximum sequence length
            
        Returns:
            Dictionary containing input_ids, attention_mask, etc.
        """
        return self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    
    def tokenize_qa_pair(
        self,
        question: str,
        context: str,
        max_length: int = 512,
        stride: int = 128
    ) -> List[dict]:
        """
        Tokenize question and context pair, handling long contexts
        
        Args:
            question: Question text
            context: Context text
            max_length: Maximum sequence length
            stride: Overlap between chunks
            
        Returns:
            List of tokenized examples
        """
        # Encode question
        question_encoded = self.tokenizer.encode(
            question,
            add_special_tokens=False
        )
        
        # Encode context with sliding window if needed
        context_encoded = self.tokenizer.encode(
            context,
            add_special_tokens=False,
            max_length=max_length - len(question_encoded) - 3,  # Reserve space for special tokens
            truncation=True
        )
        
        # Combine question and context
        inputs = self.tokenizer(
            question,
            context,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return inputs
    
    def find_answer_position(
        self,
        context: str,
        answer_text: str,
        answer_start: int
    ) -> Tuple[int, int]:
        """
        Find start and end token positions of answer in context
        
        Args:
            context: Context text
            answer_text: Answer text
            answer_start: Character-level answer start position
            
        Returns:
            Tuple of (start_token, end_token) positions
        """
        # Encode context
        context_tokens = self.tokenizer.encode(
            context,
            add_special_tokens=False
        )
        
        # Find answer span in original text
        answer_end = answer_start + len(answer_text)
        
        # Map character positions to token positions
        # This is a simplified version - actual implementation may need
        # character-to-token mapping from tokenizer
        decoded_context = self.tokenizer.decode(context_tokens)
        
        # For now, return placeholder
        # In practice, use tokenizer's char_to_token method
        return (0, 1)  # Placeholder
    
    def split_text(self, text: str, max_length: int = 512) -> List[str]:
        """
        Split long text into chunks
        
        Args:
            text: Text to split
            max_length: Maximum length per chunk
            
        Returns:
            List of text chunks
        """
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

