"""
Main entry point for training, evaluation, and inference
"""

import argparse
import sys
from pathlib import Path

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.data.dataset_loader import SquadDatasetLoader
from src.data.data_utils import DataUtils
from src.models.model_loader import ModelLoader
from src.models.trainer import QATrainer
from src.models.evaluator import QAEvaluator
from src.pipelines.qa_pipeline import QAPipeline


def train(config_path: str):
    """
    Train the model
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    config = load_config(config_path)
    logger = get_logger(config.get('output.log_dir'))
    
    logger.info("Starting training...")
    
    # Load dataset
    logger.info("Loading dataset...")
    loader = SquadDatasetLoader(config.get('data.train_path'))
    train_data = loader.load()
    
    # Load model
    logger.info("Loading model...")
    model_loader = ModelLoader(
        model_name=config.get('model.name'),
        cache_dir=config.get('model.cache_dir')
    )
    model, tokenizer, _ = model_loader.load_all()
    
    # Prepare data
    logger.info("Preparing data...")
    data_utils = DataUtils(model_name=config.get('model.name'))
    
    # Create trainer
    trainer = QATrainer(model, tokenizer)
    train_dataset = trainer.prepare_dataset(train_data)
    trainer.train_dataset = train_dataset
    
    # Train
    logger.info("Training model...")
    trainer.train()
    
    # Save model
    output_path = Path(config.get('output.model_dir')) / 'final_model'
    trainer.save_model(str(output_path))
    logger.info(f"Model saved to {output_path}")


def evaluate(config_path: str, model_path: str):
    """
    Evaluate the model
    
    Args:
        config_path: Path to configuration file
        model_path: Path to trained model
    """
    config = load_config(config_path)
    logger = get_logger(config.get('output.log_dir'))
    
    logger.info("Starting evaluation...")
    
    # Load dataset
    loader = SquadDatasetLoader(config.get('data.val_path'))
    eval_data = loader.load()
    
    # Load model
    model_loader = ModelLoader()
    model, tokenizer = model_loader.load_from_checkpoint(model_path)
    
    # Create pipeline
    pipeline = QAPipeline(model_path=model_path)
    
    # Evaluate
    evaluator = QAEvaluator()
    predictions = []
    ground_truths = []
    
    for example in eval_data[:100]:  # Evaluate on subset
        pred = pipeline.predict(
            example['question'],
            example['context']
        )
        predictions.append(pred)
        ground_truths.append({'answer': example.get('answer_text', '')})
    
    results = evaluator.evaluate_batch(predictions, ground_truths)
    
    logger.info(f"Evaluation Results: {results}")
    
    # Save results
    results_path = Path(config.get('output.results_dir')) / 'evaluation_results.txt'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        f.write(f"F1 Score: {results['f1']:.4f}\n")
        f.write(f"Exact Match: {results['exact_match']:.4f}\n")


def inference(config_path: str, model_path: str, question: str, context: str):
    """
    Run inference on a question-answer pair
    
    Args:
        config_path: Path to configuration file
        model_path: Path to trained model
        question: Question text
        context: Context text
    """
    config = load_config(config_path)
    logger = get_logger(config.get('output.log_dir'))
    
    logger.info("Running inference...")
    
    # Create pipeline
    pipeline = QAPipeline(model_path=model_path)
    
    # Predict
    result = pipeline.predict(question, context)
    
    print(f"Question: {question}")
    print(f"Context: {context[:100]}...")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['score']:.4f}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Question Answering Using Transformer'
    )
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'evaluate', 'inference'],
        help='Mode: train, evaluate, or inference'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/qa_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help='Path to model (for evaluate/inference)'
    )
    parser.add_argument(
        '--question',
        type=str,
        help='Question text (for inference)'
    )
    parser.add_argument(
        '--context',
        type=str,
        help='Context text (for inference)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args.config)
    elif args.mode == 'evaluate':
        if not args.model_path:
            print("Error: --model_path is required for evaluation")
            sys.exit(1)
        evaluate(args.config, args.model_path)
    elif args.mode == 'inference':
        if not args.model_path or not args.question or not args.context:
            print("Error: --model_path, --question, and --context are required for inference")
            sys.exit(1)
        inference(args.config, args.model_path, args.question, args.context)


if __name__ == '__main__':
    main()

