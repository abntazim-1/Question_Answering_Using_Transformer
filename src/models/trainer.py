"""
Training pipeline for fine-tuning Question Answering models
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from typing import Dict, List, Optional
import os


class QADataset(Dataset):
    """Dataset class for Question Answering"""
    
    def __init__(
        self,
        examples: List[Dict],
        tokenizer: AutoTokenizer,
        max_length: int = 512
    ):
        """
        Initialize QA dataset
        
        Args:
            examples: List of examples with 'context', 'question', 'answers'
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize
        encodings = self.tokenizer(
            example['question'],
            example['context'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Extract answer positions
        start_positions = example.get('start_positions', 0)
        end_positions = example.get('end_positions', 0)
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'start_positions': torch.tensor(start_positions, dtype=torch.long),
            'end_positions': torch.tensor(end_positions, dtype=torch.long)
        }


class QATrainer:
    """Trainer for Question Answering models"""
    
    def __init__(
        self,
        model: AutoModelForQuestionAnswering,
        tokenizer: AutoTokenizer,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        training_args: Optional[TrainingArguments] = None
    ):
        """
        Initialize QA Trainer
        
        Args:
            model: Pretrained model for QA
            tokenizer: Tokenizer instance
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            training_args: Training arguments
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Default training arguments
        if training_args is None:
            training_args = TrainingArguments(
                output_dir='./outputs/models',
                num_train_epochs=3,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                learning_rate=3e-5,
                weight_decay=0.01,
                logging_dir='./outputs/logs',
                logging_steps=100,
                eval_steps=500,
                save_steps=1000,
                evaluation_strategy="steps"
            )
        
        self.training_args = training_args
    
    def prepare_dataset(
        self,
        examples: List[Dict],
        max_length: int = 512
    ) -> QADataset:
        """
        Prepare dataset from examples
        
        Args:
            examples: List of examples
            max_length: Maximum sequence length
            
        Returns:
            QADataset instance
        """
        return QADataset(examples, self.tokenizer, max_length)
    
    def train(self):
        """Start training"""
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=default_data_collator
        )
        
        trainer.train()
        return trainer
    
    def save_model(self, output_path: str):
        """
        Save trained model
        
        Args:
            output_path: Path to save the model
        """
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

