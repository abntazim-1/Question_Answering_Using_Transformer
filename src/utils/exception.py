"""
Custom exception classes for the project
"""


class QAException(Exception):
    """Base exception for Question Answering project"""
    pass


class DatasetLoadError(QAException):
    """Raised when dataset loading fails"""
    pass


class ModelLoadError(QAException):
    """Raised when model loading fails"""
    pass


class TrainingError(QAException):
    """Raised when training fails"""
    pass


class EvaluationError(QAException):
    """Raised when evaluation fails"""
    pass


class ConfigurationError(QAException):
    """Raised when configuration is invalid"""
    pass


class InferenceError(QAException):
    """Raised when inference fails"""
    pass

