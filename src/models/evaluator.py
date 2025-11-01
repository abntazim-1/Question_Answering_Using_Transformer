"""
Evaluation metrics for Question Answering (Exact Match, F1 Score)
"""

import re
from typing import List, Dict
from collections import Counter


class QAEvaluator:
    """Evaluator for Question Answering models"""
    
    @staticmethod
    def normalize_answer(s: str) -> str:
        """
        Normalize answer string for comparison
        
        Args:
            s: Answer string
            
        Returns:
            Normalized answer
        """
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set('!?.')
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    @staticmethod
    def f1_score(prediction: str, ground_truth: str) -> float:
        """
        Calculate F1 score between prediction and ground truth
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            F1 score
        """
        prediction_tokens = QAEvaluator.normalize_answer(prediction).split()
        ground_truth_tokens = QAEvaluator.normalize_answer(ground_truth).split()
        
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
            # If either is no-answer
            return float(num_same == 0)
        
        if num_same == 0:
            return 0.0
        
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1
    
    @staticmethod
    def exact_match_score(prediction: str, ground_truth: str) -> float:
        """
        Calculate Exact Match score
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        return float(
            QAEvaluator.normalize_answer(prediction) == 
            QAEvaluator.normalize_answer(ground_truth)
        )
    
    def evaluate(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate predictions against ground truths
        
        Args:
            predictions: List of predicted answers
            ground_truths: List of ground truth answers
            
        Returns:
            Dictionary containing EM and F1 scores
        """
        if len(predictions) != len(ground_truths):
            raise ValueError(
                "Predictions and ground truths must have the same length"
            )
        
        f1_scores = []
        em_scores = []
        
        for pred, truth in zip(predictions, ground_truths):
            f1_scores.append(self.f1_score(pred, truth))
            em_scores.append(self.exact_match_score(pred, truth))
        
        return {
            'f1': sum(f1_scores) / len(f1_scores),
            'exact_match': sum(em_scores) / len(em_scores)
        }
    
    def evaluate_batch(
        self,
        model_predictions: List[Dict],
        ground_truths: List[Dict]
    ) -> Dict[str, float]:
        """
        Evaluate batch of model predictions
        
        Args:
            model_predictions: List of predictions with 'answer' key
            ground_truths: List of ground truths with 'answer' key
            
        Returns:
            Dictionary containing EM and F1 scores
        """
        predictions = [pred.get('answer', '') for pred in model_predictions]
        truths = [truth.get('answer', '') for truth in ground_truths]
        
        return self.evaluate(predictions, truths)

