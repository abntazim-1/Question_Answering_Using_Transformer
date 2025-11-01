"""
Dataset loader for Question Answering datasets (e.g., SQuAD)
"""

import json
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class SquadDatasetLoader:
    """Load and preprocess SQuAD dataset"""
    
    def __init__(self, data_path: str):
        """
        Initialize dataset loader
        
        Args:
            data_path: Path to SQuAD dataset (JSON or CSV format)
        """
        self.data_path = Path(data_path)
        
    def load_from_json(self) -> Dict:
        """
        Load SQuAD dataset from JSON format
        
        Returns:
            Dictionary containing dataset
        """
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def load_from_csv(self) -> pd.DataFrame:
        """
        Load dataset from CSV format
        
        Returns:
            DataFrame containing dataset
        """
        return pd.read_csv(self.data_path)
    
    def extract_qa_pairs(self, data: Dict) -> List[Dict]:
        """
        Extract question-answer pairs from SQuAD format
        
        Args:
            data: SQuAD dataset dictionary
            
        Returns:
            List of dictionaries containing context, question, answer pairs
        """
        qa_pairs = []
        
        if 'data' in data:
            for article in data['data']:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    for qa in paragraph['qas']:
                        question = qa['question']
                        answers = qa.get('answers', [])
                        
                        for answer in answers:
                            qa_pairs.append({
                                'context': context,
                                'question': question,
                                'answer_text': answer['text'],
                                'answer_start': answer['answer_start']
                            })
        
        return qa_pairs
    
    def load(self, format: str = 'auto') -> List[Dict]:
        """
        Load dataset based on file extension
        
        Args:
            format: Format type ('json', 'csv', or 'auto')
            
        Returns:
            Processed dataset as list of dictionaries
        """
        if format == 'auto':
            format = self.data_path.suffix[1:]  # Remove the dot
        
        if format == 'json':
            data = self.load_from_json()
            return self.extract_qa_pairs(data)
        elif format == 'csv':
            df = self.load_from_csv()
            return df.to_dict('records')
        else:
            raise ValueError(f"Unsupported format: {format}")

