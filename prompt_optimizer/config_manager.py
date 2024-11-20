"""
Configuration manager for Prompt Optimizer.
Handles loading and retrieving configuration from YAML files.
"""

import yaml
from typing import Dict, Any
from loguru import logger

class ConfigManager:
    def __init__(self, config_path="prompts_config.yml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def get_system_prompt(self, context: str, model: str, **format_kwargs) -> str:
        """
        Retrieve and format a system prompt

        :param context: Context of the prompt (e.g., 'prompt_generation')
        :param model: Target model
        :param format_kwargs: Formatting arguments
        :return: Formatted system prompt
        """
        try:
            prompt_template = self.config["system_prompts"][model][context]["template"]
            return prompt_template.format(**format_kwargs)
        except KeyError as e:
            logger.error(f"Prompt configuration not found: {e}")
            raise ValueError(f"Invalid prompt configuration: {e}")

    def get_model_config(self, model: str) -> Dict[str, Any]:
        """
        Get model-specific configuration

        :param model: Target model
        :return: Model configuration dictionary
        """
        return self.config["model_configurations"].get(model, {})