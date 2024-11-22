import anthropic
import argparse
import ollama
import openai
import os
import questionary
import sys
import yaml
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.decorators import observe
from loguru import logger
from typing import Any, Dict, Literal
from config_manager import ConfigManager


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)


class PromptOptimizer:
    def __init__(
        self,
        config_manager: ConfigManager,
        openai_api_key: str = None,
        anthropic_api_key: str = None,
    ):
        """
        Initialize PromptOptimizer with configuration and API keys

        :param config_manager: Configuration manager instance
        :param openai_api_key: OpenAI API key
        :param anthropic_api_key: Anthropic API key
        """
        self.config_manager = config_manager
        self._openai_client = None
        self._anthropic_client = None
        self._openai_api_key = openai_api_key
        self._anthropic_api_key = anthropic_api_key
        
        # Cache model configs
        self._model_configs = {}
        self._system_prompts = {}

    @property
    def openai_client(self):
        if self._openai_client is None and self._openai_api_key:
            self._openai_client = openai.OpenAI(api_key=self._openai_api_key)
        return self._openai_client

    @property
    def anthropic_client(self):
        if self._anthropic_client is None and self._anthropic_api_key:
            self._anthropic_client = anthropic.Anthropic(api_key=self._anthropic_api_key)
        return self._anthropic_client

    def _get_model_config(self, model_name: str) -> Dict:
        """Cache and retrieve model configurations"""
        if model_name not in self._model_configs:
            self._model_configs[model_name] = self.config_manager.get_model_config(model_name)
        return self._model_configs[model_name]

    def _get_system_prompt(self, context: str, **kwargs) -> str:
        """Cache and retrieve system prompts"""
        cache_key = f"{context}_{hash(str(kwargs))}"
        if cache_key not in self._system_prompts:
            self._system_prompts[cache_key] = self.config_manager.get_system_prompt(
                context=context, **kwargs
            )
        return self._system_prompts[cache_key]  

    def generate_and_optimize_prompt(
        self,
        task_description: str,
        target_model: Literal["claude", "chatgpt"] = "claude",
    ) -> dict:
        """
        Generate and optimize a prompt using Llama3, tailored for a specific target model

        :param task_description: Natural language description of the desired function
        :param target_model: Model to optimize the prompt for
        :return: Dictionary with prompt details
        """
        logger.info(f"Generating prompt for task: {task_description}")

        try:
            # Generate initial prompt using Llama3
            initial_prompt = self._generate_prompt_with_llama3(
                task_description, target_model
            )

            # Check if we can optimize with target model
            can_optimize = (
                (target_model == "claude" and self.anthropic_client is not None) or
                (target_model == "chatgpt" and self.openai_client is not None)
            )

            if can_optimize:
                # Optimize for target model
                optimized_prompt = self._optimize_for_target_model(
                    initial_prompt, target_model
                )
            else:
                logger.warning(f"Skipping {target_model} optimization due to missing API key")
                optimized_prompt = initial_prompt

            return {
                "original_task": task_description,
                "initial_prompt": initial_prompt,
                "optimized_prompt": optimized_prompt,
                "target_model": target_model,
            }

        except Exception as e:
            logger.error(f"Prompt generation failed: {e}")
            raise

    def _generate_prompt_with_llama3(
        self, task_description: str, target_model: str
    ) -> str:
        """
        Generate a prompt using local Llama3 model

        :param task_description: Original task description
        :param target_model: Target model for prompt optimization
        :return: Generated prompt
        """
        # Retrieve system prompt from configuration
        system_prompt = self.config_manager.get_system_prompt(
            context="prompt_generation",
            model="llama3",
            task_description=task_description,
            target_model=target_model.upper(),
        )

        # Retrieve Llama3 model configuration
        model_config = self.config_manager.get_model_config("llama3")

        response = ollama.chat(
            model=model_config.get("default_model", "llama3"),
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Generate an optimized prompt for: {task_description}",
                },
            ],
        )

        return response["message"]["content"]

    def _optimize_for_target_model(
        self, initial_prompt: str, target_model: Literal["claude", "chatgpt"]
    ) -> str:
        """
        Further optimize the prompt for a specific target model

        :param initial_prompt: Prompt generated by Llama3
        :param target_model: Target model for final optimization
        :return: Final optimized prompt
        """
        if target_model == "claude":
            return self._claude_prompt_optimization(initial_prompt)
        elif target_model == "chatgpt":
            return self._chatgpt_prompt_optimization(initial_prompt)
        else:
            raise ValueError(f"Unsupported target model: {target_model}")

    @observe(name="claude_prompt_optimization")
    def _claude_prompt_optimization(self, initial_prompt: str) -> str:
        """Optimize prompt specifically for Claude's capabilities"""
        if not self.anthropic_client:
            raise ValueError(
                self.config_manager.config["error_messages"]["api_key_missing"][
                    "anthropic"
                ]
            )

        # Retrieve Claude optimization prompt from configuration
        system_prompt = self.config_manager.get_system_prompt(
            context="prompt_optimization",
            model="claude",
            original_prompt=initial_prompt,
        )

        # Retrieve Claude model configuration
        model_config = self.config_manager.get_model_config("claude")

        response = self.anthropic_client.messages.create(
            model=model_config.get("default_model", "claude-3-opus-20240229"),
            max_tokens=model_config.get("generation_settings", {}).get(
                "max_tokens", 300
            ),
            system="You are an expert prompt engineer for Claude.",
            messages=[{"role": "user", "content": system_prompt}],
        )

        # Add Langfuse observation
        langfuse.trace(
            name="claude_response",
            input={
                "system": "You are an expert prompt engineer for Claude.",
                "user": system_prompt,
            },
            output=response.content[0].text,
            metadata={
                "model": model_config.get("default_model"),
                "max_tokens": model_config.get("generation_settings", {}).get(
                    "max_tokens", 300
                ),
            },
        )

        return response.content[0].text

    @observe(name="chatgpt_prompt_optimization")
    def _chatgpt_prompt_optimization(self, initial_prompt: str) -> str:
        """Optimize prompt specifically for ChatGPT's capabilities"""
        if not self.openai_client:
            raise ValueError(
                self.config_manager.config["error_messages"]["api_key_missing"]["openai"]
            )

        # Retrieve ChatGPT optimization prompt from configuration
        system_prompt = self.config_manager.get_system_prompt(
            context="prompt_optimization",
            model="chatgpt",
            original_prompt=initial_prompt,
        )

        # Retrieve ChatGPT model configuration
        model_config = self.config_manager.get_model_config("chatgpt")

        response = self.openai_client.chat.completions.create(
            model=model_config.get("default_model", "gpt-4-turbo"),
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert prompt engineer for ChatGPT.",
                },
                {"role": "user", "content": system_prompt},
            ],
        )

        # Add Langfuse observation with corrected usage handling
        langfuse.trace(
            name="chatgpt_response",
            input={
                "system": "You are an expert prompt engineer for ChatGPT.",
                "user": system_prompt,
            },
            output=response.choices[0].message.content,
            metadata={
                "model": model_config.get("default_model"),
                "usage": response.usage.model_dump() if hasattr(response, "usage") else None,
            },
        )

        return response.choices[0].message.content


class PromptOptimizerCLI:
    def __init__(self):
        """
        Initialize the CLI interface for PromptOptimizer
        """
        # Configure logging
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logger.add("prompt_optimizer.log", rotation="10 MB")

        # Initialize API keys from environment or prompt
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    def _validate_api_keys(self):
        """
        Validate and optionally prompt for missing API keys
        """
        # OpenAI API Key
        if not self.openai_api_key:
            self.openai_api_key = questionary.password(
                "OpenAI API Key (optional, press enter to skip):"
            ).ask()

        # Anthropic API Key
        if not self.anthropic_api_key:
            self.anthropic_api_key = questionary.password(
                "Anthropic API Key (optional, press enter to skip):"
            ).ask()

    def interactive_mode(self):
        """
        Run the prompt optimizer in interactive mode
        """
        # Validate API keys
        self._validate_api_keys()

        # Create ConfigManager instance
        config_manager = ConfigManager()  # Add this line

        # Create PromptOptimizer instance
        optimizer = PromptOptimizer(
            config_manager=config_manager,  # Add this line
            openai_api_key=self.openai_api_key,
            anthropic_api_key=self.anthropic_api_key
        )

        # Interactive prompt for task description
        task_description = questionary.text(
            "Enter the task description for prompt optimization:",
            validate=lambda text: len(text.strip()) > 0,
        ).ask()

        # Interactive model selection
        target_model = questionary.select(
            "Select the target model for prompt optimization:",
            choices=["claude", "chatgpt"],
        ).ask()

        try:
            # Generate and optimize prompt
            result = optimizer.generate_and_optimize_prompt(
                task_description=task_description, target_model=target_model
            )

            # Display results
            self._display_results(result)

        except Exception as e:
            logger.error(f"Prompt optimization failed: {e}")
            questionary.print(f"Error: {e}", style="bold_red")

    def cli_mode(self, args):
        """
        Run the prompt optimizer in CLI mode

        :param args: Parsed command-line arguments
        """
        # Validate API keys
        self._validate_api_keys()

        # Create ConfigManager instance
        config_manager = ConfigManager()

        # Create PromptOptimizer instance
        optimizer = PromptOptimizer(
            config_manager=config_manager,  # Add this line
            openai_api_key=self.openai_api_key,
            anthropic_api_key=self.anthropic_api_key,
        )

        try:
            # Generate and optimize prompt
            result = optimizer.generate_and_optimize_prompt(
                task_description=args.task, target_model=args.model
            )

            # Display results
            self._display_results(result)

        except Exception as e:
            logger.error(f"Prompt optimization failed: {e}")
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    def _display_results(self, result):
        """
        Display prompt optimization results

        :param result: Optimization result dictionary
        """
        # Use rich formatting for better readability
        from rich.console import Console
        from rich.panel import Panel
        from rich.syntax import Syntax

        console = Console()

        console.print(
            Panel.fit(
                f"Prompt Optimization for {result['target_model'].upper()} Model",
                title="Results",
                border_style="bold blue",
            )
        )

        console.print("\n[bold green]Original Task:[/bold green]")
        console.print(result["original_task"])

        console.print("\n[bold blue]Initial Llama3 Generated Prompt:[/bold blue]")
        console.print(Syntax(result["initial_prompt"], "markdown", theme="monokai"))

        console.print("\n[bold magenta]Model-Optimized Prompt:[/bold magenta]")
        console.print(Syntax(result["optimized_prompt"], "markdown", theme="monokai"))

    def run(self):
        """
        Main entry point for the CLI application
        """
        # Create argument parser
        parser = argparse.ArgumentParser(
            description="Prompt Optimization CLI using Llama3"
        )

        # Add arguments
        parser.add_argument(
            "-t", "--task", type=str, help="Task description for prompt optimization"
        )
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            choices=["claude", "chatgpt"],
            default="claude",
            help="Target model for prompt optimization (default: claude)",
        )

        # Parse arguments
        args = parser.parse_args()

        # Run in appropriate mode
        if args.task:
            self.cli_mode(args)
        else:
            self.interactive_mode()


def main():
    # Ensure Ollama is running and Llama3 is available
    try:
        ollama.list()
    except Exception:
        print("Error: Ollama is not running or Llama3 is not installed.")
        print("Please ensure Ollama is running and Llama3 is pulled.")
        sys.exit(1)

    # Run CLI
    cli = PromptOptimizerCLI()
    cli.run()


if __name__ == "__main__":
    main()
