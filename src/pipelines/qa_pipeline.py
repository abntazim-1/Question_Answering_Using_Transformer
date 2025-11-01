"""
End-to-end inference pipeline for Question Answering
"""

import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    pipeline
)
from typing import Dict, List, Optional, Tuple
import numpy as np


class QAPipeline:
    """End-to-end Question Answering pipeline"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = 'bert-base-uncased',
        device: Optional[str] = None
    ):
        """
        Initialize QA pipeline
        
        Args:
            model_path: Path to fine-tuned model (if available)
            model_name: Name of pretrained model (if model_path is None)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        
        # Load model and tokenizer
        if model_path:
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model.to(device)
        self.model.eval()
        
        # Alternative: Use HuggingFace pipeline
        # self.pipeline = pipeline(
        #     "question-answering",
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     device=0 if device == 'cuda' else -1
        # )
    
    def predict(
        self,
        question: str,
        context: str,
        max_answer_length: int = 50,
        top_k: int = 1
    ) -> Dict:
        """
        Predict answer for given question and context
        
        Args:
            question: Question text
            context: Context text
            max_answer_length: Maximum answer length
            top_k: Number of top answers to return
            
        Returns:
            Dictionary containing answer, confidence score, and start/end positions
        """
        # Tokenize inputs
        inputs = self.tokenizer(
            question,
            context,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        
        # Get the most likely start and end positions
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores)
        
        # Ensure end >= start
        if end_idx < start_idx:
            end_idx = start_idx
        
        # Decode answer
        answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        # Calculate confidence scores
        start_score = torch.softmax(start_scores, dim=-1)[0][start_idx].item()
        end_score = torch.softmax(end_scores, dim=-1)[0][end_idx].item()
        confidence = (start_score + end_score) / 2
        
        return {
            'answer': answer,
            'score': confidence,
            'start': start_idx.item(),
            'end': end_idx.item()
        }
    
    def predict_batch(
        self,
        questions: List[str],
        contexts: List[str],
        max_answer_length: int = 50
    ) -> List[Dict]:
        """
        Predict answers for a batch of questions
        
        Args:
            questions: List of questions
            contexts: List of contexts
            max_answer_length: Maximum answer length
            
        Returns:
            List of prediction dictionaries
        """
        if len(questions) != len(contexts):
            raise ValueError("Questions and contexts must have the same length")
        
        predictions = []
        for question, context in zip(questions, contexts):
            pred = self.predict(question, context, max_answer_length)
            predictions.append(pred)
        
        return predictions
    
    def predict_with_pipeline(
        self,
        question: str,
        context: str
    ) -> Dict:
        """
        Predict using HuggingFace pipeline (alternative method)
        
        Args:
            question: Question text
            context: Context text
            
        Returns:
            Dictionary containing answer and score
        """
        qa_pipeline = pipeline(
            "question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == 'cuda' else -1
        )
        
        result = qa_pipeline({
            'question': question,
            'context': context
        })
        
        return result

