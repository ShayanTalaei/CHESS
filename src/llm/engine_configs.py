from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import VertexAI
from google.oauth2 import service_account
from google.cloud import aiplatform
from typing import Dict, Any
import vertexai
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
import os

safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}


GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_REGION = os.getenv("GCP_REGION")
GCP_CREDENTIALS = os.getenv("GCP_CREDENTIALS")

if GCP_CREDENTIALS and GCP_PROJECT and GCP_REGION:
    aiplatform.init(
        project=GCP_PROJECT,
        location=GCP_REGION,
        credentials=service_account.Credentials.from_service_account_file(GCP_CREDENTIALS),
    )
    vertexai.init(
        project=GCP_PROJECT,
        location=GCP_REGION,
        credentials=service_account.Credentials.from_service_account_file(GCP_CREDENTIALS),
    )

"""
This module defines configurations for various language models using the langchain library.
Each configuration includes a constructor, parameters, and an optional preprocessing function.
"""

ENGINE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gemini-pro": {
        "constructor": ChatGoogleGenerativeAI,
        "params": {"model": "gemini-pro", "temperature": 0},
        "preprocess": lambda x: x.to_messages(),
    },
    "gemini-1.5-pro": {
        "constructor": VertexAI,
        "params": {"model": "gemini-1.5-pro", "temperature": 0, "safety_settings": safety_settings},
    },
    "gemini-1.5-pro-002": {
        "constructor": VertexAI,
        "params": {
            "model": "gemini-1.5-pro-002",
            "temperature": 0,
            "safety_settings": safety_settings,
        },
    },
    "gemini-1.5-flash": {
        "constructor": VertexAI,
        "params": {
            "model": "gemini-1.5-flash",
            "temperature": 0,
            "safety_settings": safety_settings,
        },
    },
    "picker_gemini_model": {
        "constructor": VertexAI,
        "params": {
            "model": "projects/613565144741/locations/us-central1/endpoints/7618015791069265920",
            "temperature": 0,
            "safety_settings": safety_settings,
        },
    },
    "gemini-1.5-pro-text2sql": {
        "constructor": VertexAI,
        "params": {
            "model": "projects/618488765595/locations/us-central1/endpoints/1743594544210903040",
            "temperature": 0,
            "safety_settings": safety_settings,
        },
    },
    "cot_picker": {
        "constructor": VertexAI,
        "params": {
            "model": "projects/243839366443/locations/us-central1/endpoints/2772315215344173056",
            "temperature": 0,
            "safety_settings": safety_settings,
        },
    },
    "gpt-3.5-turbo-0125": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-3.5-turbo-0125", "temperature": 0},
    },
    "gpt-3.5-turbo-instruct": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-3.5-turbo-instruct", "temperature": 0},
    },
    "gpt-4-1106-preview": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4-1106-preview", "temperature": 0},
    },
    "gpt-4-0125-preview": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4-0125-preview", "temperature": 0},
    },
    "gpt-4-turbo": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4-turbo", "temperature": 0},
    },
    "gpt-4o": {"constructor": ChatOpenAI, "params": {"model": "gpt-4o", "temperature": 0}},
    "gpt-4o-mini": {
        "constructor": ChatOpenAI,
        "params": {"model": "gpt-4o-mini", "temperature": 0},
    },
    "claude-3-opus-20240229": {
        "constructor": ChatAnthropic,
        "params": {"model": "claude-3-opus-20240229", "temperature": 0},
    },
    # "finetuned_nl2sql": {
    #     "constructor": ChatOpenAI,
    #     "params": {
    #         "model": "AI4DS/NL2SQL_DeepSeek_33B",
    #         "openai_api_key": "EMPTY",
    #         "openai_api_base": "/v1",
    #         "max_tokens": 400,
    #         "temperature": 0,
    #         "stop": ["```\n", ";"]
    #     }
    # },
    "finetuned_nl2sql": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "ft:gpt-4o-mini-2024-07-18:stanford-university::9p4f6Z4W",
            "max_tokens": 400,
            "temperature": 0,
            "stop": ["```\n", ";"],
        },
    },
    "column_selection_finetuning": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "ft:gpt-4o-mini-2024-07-18:stanford-university::9t1Gcj6Y:ckpt-step-1511",
            "max_tokens": 1000,
            "temperature": 0,
            "stop": [";"],
        },
    },
    # "finetuned_nl2sql_cot": {
    #     "constructor": ChatOpenAI,
    #     "params": {
    #         "model": "AI4DS/deepseek-cot",
    #         "openai_api_key": "EMPTY",
    #         "openai_api_base": "/v1",
    #         "max_tokens": 1000,
    #         "temperature": 0,
    #         "stop": [";"]
    #     }
    # },
    # "finetuned_nl2sql_cot": {
    #     "constructor": ChatOpenAI,
    #     "params": {
    #         "model": "ft:gpt-4o-mini-2024-07-18:stanford-university::9oKvRYet",
    #         "max_tokens": 1000,
    #         "temperature": 0,
    #         "stop": [";"]
    #     }
    # },
    "meta-llama/Meta-Llama-3-70B-Instruct": {
        "constructor": ChatOpenAI,
        "params": {
            "model": "meta-llama/Meta-Llama-3-70B-Instruct",
            "openai_api_key": "EMPTY",
            "openai_api_base": "/v1",
            "max_tokens": 600,
            "temperature": 0,
            "model_kwargs": {"stop": [""]},
        },
    },
}
