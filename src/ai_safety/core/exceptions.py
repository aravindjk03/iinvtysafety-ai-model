"""Explicit exceptions for AI Safety Model Lab."""


class AISafetyError(Exception):
    """Base exception for all AI Safety Model Lab errors."""
    pass


class ModelNotFoundError(AISafetyError):
    """Raised when a requested model cannot be found in the registry or filesystem."""
    pass


class ModelMetadataError(AISafetyError):
    """Raised when model metadata fails validation or is malformed."""
    pass


class UnsupportedModelError(AISafetyError):
    """Raised when an unsupported model architecture, task type, or framework is requested."""
    pass


class ModelLoadError(AISafetyError):
    """Raised when a model cannot be loaded (missing weights, corrupt file, invalid device)."""
    pass


class InvalidInputError(AISafetyError):
    """Raised when input provided for inference or processing is invalid."""
    pass


class PredictionError(AISafetyError):
    """Raised when an error occurs during model inference/prediction."""
    pass


class RegistryError(AISafetyError):
    """Raised when an operation on the model registry violates constraints (e.g. duplicate or overwriting pretrained)."""
    pass
