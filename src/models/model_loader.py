"""
Load pretrained transformer models for Question Answering
"""

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    AutoConfig
)
from typing import Optional
import torch


class ModelLoader:
    """Load pretrained transformer models"""
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        cache_dir: Optional[str] = None
    ):
        """
        Initialize model loader
        
        Args:
            model_name: Name or path of pretrained model
            cache_dir: Directory to cache models
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self.config = None
    
    def load_model(self):
        """Load the pretrained model"""
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        return self.model
    
    def load_tokenizer(self):
        """Load the tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        return self.tokenizer
    
    def load_config(self):
        """Load the model configuration"""
        self.config = AutoConfig.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        return self.config
    
    def load_all(self):
        """Load model, tokenizer, and config"""
        self.load_config()
        self.load_tokenizer()
        self.load_model()
        return self.model, self.tokenizer, self.config
    
    def save_model(self, output_path: str):
        """
        Save model and tokenizer to output path
        
        Args:
            output_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.model.save_pretrained(output_path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_path)
    
    def load_from_checkpoint(self, checkpoint_path: str):
        """
        Load model from a checkpoint
        
        Args:
            checkpoint_path: Path to model checkpoint
        """
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            checkpoint_path
        )
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        return self.model, self.tokenizer

