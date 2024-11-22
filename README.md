# 🚀 AI Prompt Optimizer CLI

## Overview

A CLI tool for generating and optimizing prompts using Llama3, with support for Claude and ChatGPT model targeting. This tool helps you craft precise, model-specific prompts to maximize the effectiveness of your AI interactions.

## 🌟 Features

- 🤖 Prompt generation using Llama3
- 🎯 Model-specific prompt optimization
- 📝 Configurable via YAML
- 🔒 Secure API key management
- 📊 Detailed logging and error handling

## 🛠 Prerequisites

- Python 3.10+
- Ollama
- Llama3 model
- API keys (optional) for Claude and ChatGPT
- Langfuse account for analytics (optional)

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/prompt-optimizer.git
cd prompt-optimizer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Ollama and pull Llama3:
```bash
# Install Ollama from https://ollama.ai/
ollama pull llama3
```

## 🚀 Usage

### Interactive Mode

Run the script without arguments for an interactive prompt generation:

```bash
python prompt_optimizer.py
```

### CLI Mode

Optimize prompts directly from the command line:

```bash
# Optimize for Claude
python prompt_optimizer.py -t "Create a function that summarizes text" -m claude

# Optimize for ChatGPT
python prompt_optimizer.py -t "Develop a web scraping script" -m chatgpt
```

### API Key Configuration

Set API keys in `/.env` file:
```bash
OPENAI_API_KEY='your_openai_key'
ANTHROPIC_API_KEY='your_anthropic_key'
LANGFUSE_PUBLIC_KEY='your_langfuse_public_key'
LANGFUSE_SECRET_KEY='your_langfuse_secret_key'
```

## 📝 Configuration

Edit `prompts_config.yml` to customize:
- System prompts
- Model configurations
- Error messages
- Logging settings


🤫 *This README.md file is LLM-generated* 🤫