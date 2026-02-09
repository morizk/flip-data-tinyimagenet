"""
Utility functions for model analysis and summary.
"""
import torch
import torch.nn as nn
from typing import Dict, Any


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model, input_size=(1, 3, 64, 64)):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, height, width)
    """
    print("="*80)
    print("MODEL SUMMARY")
    print("="*80)
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Input size: {input_size}")
    print("\nArchitecture:")
    print(model)
    print("="*80)


def get_model_info(model, input_size=(1, 3, 64, 64)):
    """
    Get model information as a dictionary.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
    
    Returns:
        dict with model information
    """
    return {
        'num_parameters': count_parameters(model),
        'input_size': input_size,
        'architecture': str(model.__class__.__name__)
    }

