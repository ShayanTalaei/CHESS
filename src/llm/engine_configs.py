from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from typing import Dict, Any

"""
This module defines configurations for various language models using the langchain library.
Each configuration includes a constructor, parameters, and an optional preprocessing function.
"""

ENGINE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gemini-pro": {
        "constructor": ChatGoogleGenerativeAI,
        "params": {"model": "gemini-pro", "temperature": 0, "convert_system_message_to_human": True},
        "preprocess": lambda x: x.to_messages()
    },
    "gemini-1.5-pro-latest": {
        "constructor": ChatGoogleGenerativeAI,
        "params": {"model": "gemini-1.5-pro-latest", "temperature": 0, "convert_system_message_to_human": True},
        "preprocess": lambda x: x.to_messages()
    },
    "gpt-3.5-turbo-0125": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-3.5-turbo-0125", "temperature": 0}
    },
    "gpt-3.5-turbo-instruct": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-3.5-turbo-instruct", "temperature": 0}
    },
    "gpt-4-1106-preview": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4-1106-preview", "temperature": 0}
    },
    "gpt-4-0125-preview": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4-0125-preview", "temperature": 0}
    },
    "gpt-4-turbo": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4-turbo", "temperature": 0}
    },
    "claude-3-opus-20240229": {
        "constructor": ChatAnthropic,
        "params": {"model": "claude-3-opus-20240229", "temperature": 0}
    },
    "finetuned_nl2sql": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "AI4DS/NL2SQL_DeepSeek_33B",
            "openai_api_key": "EMPTY",
            "openai_api_base": "/v1",
            "max_tokens": 400,
            "temperature": 0,
            "model_kwargs": {
                "stop": ["```\n", ";"]
            }
        }
    },
    "meta-llama/Meta-Llama-3-70B-Instruct": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "meta-llama/Meta-Llama-3-70B-Instruct",
            "openai_api_key": "EMPTY",
            "openai_api_base": "/v1",
            "max_tokens": 600,
            "temperature": 0,
            "model_kwargs": {
                "stop": [""]
            }
        }
    }
}
