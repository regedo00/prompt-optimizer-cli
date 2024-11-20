"""Prompt Optimizer - A tool to optimize prompts for different LLM models."""

from .prompt_optimizer import PromptOptimizer, PromptOptimizerCLI
from .config_manager import ConfigManager

__version__ = "0.1.0"

__all__ = [
    "PromptOptimizer",
    "PromptOptimizerCLI",
    "ConfigManager"
]