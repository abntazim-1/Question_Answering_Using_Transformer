# Question Answering Using Transformer

A complete implementation of a Question Answering system using Transformer models (e.g., BERT, RoBERTa) fine-tuned on the SQuAD dataset.

## Project Structure

```
question_answering_transformer/
├── src/
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model loading, training, and evaluation
│   ├── pipelines/         # Inference pipeline
│   ├── utils/             # Utilities (logging, config, exceptions)
│   └── main.py            # Entry point
├── notebooks/             # Jupyter notebooks for exploration
├── configs/               # Configuration files
├── data/                  # Dataset storage
│   ├── raw/              # Original datasets
│   └── processed/        # Preprocessed data
├── outputs/              # Model outputs
│   ├── models/           # Saved models
│   ├── results/          # Evaluation results
│   └── logs/             # Training logs
└── requirements.txt      # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd question_answering_transformer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `configs/qa_config.yaml` to configure:
- Model name and settings
- Training hyperparameters
- Data paths
- Output directories

## Usage

### Training

Train a model on SQuAD dataset:

```bash
python src/main.py --mode train --config configs/qa_config.yaml
```

### Evaluation

Evaluate a trained model:

```bash
python src/main.py --mode evaluate --config configs/qa_config.yaml --model_path outputs/models/final_model
```

### Inference

Run inference on a question-answer pair:

```bash
python src/main.py --mode inference \
  --config configs/qa_config.yaml \
  --model_path outputs/models/final_model \
  --question "What is the capital of France?" \
  --context "Paris is the capital and most populous city of France."
```

## Notebooks

- `notebooks/01_data_exploration.ipynb`: Explore and understand the dataset
- `notebooks/02_model_testing.ipynb`: Quick experiments and debugging

## Features

- **Dataset Loading**: Support for SQuAD format (JSON and CSV)
- **Model Support**: Compatible with any HuggingFace QA model (BERT, RoBERTa, etc.)
- **Training Pipeline**: Complete training pipeline with evaluation
- **Evaluation Metrics**: Exact Match (EM) and F1 Score
- **Inference Pipeline**: Easy-to-use inference interface
- **Logging**: Comprehensive logging system
- **Configuration Management**: YAML-based configuration

## Model Support

The project supports any Question Answering model from HuggingFace, including:
- BERT (bert-base-uncased)
- DistilBERT (distilbert-base-uncased)
- RoBERTa (roberta-base)
- ALBERT (albert-base-v2)
- And many more...

## Data Format

The project expects SQuAD format:
- JSON: Standard SQuAD v1.1 format
- CSV: Columns should include context, question, answer_text, answer_start

## License

[Add your license here]

## Authors

[Add author information here]

