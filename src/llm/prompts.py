import os
import logging
from typing import Any

from langchain.prompts import (
    PromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

TEMPLATES_ROOT_PATH = "templates"

def load_template(template_name: str) -> str:
    """
    Loads a template from a file.

    Args:
        template_name (str): The name of the template to load.

    Returns:
        str: The content of the template.
    """
    file_name = f"template_{template_name}.txt"
    template_path = os.path.join(TEMPLATES_ROOT_PATH, file_name)
    
    try:
        with open(template_path, "r") as file:
            template = file.read()
        logging.info(f"Template {template_name} loaded successfully.")
        return template
    except FileNotFoundError:
        logging.error(f"Template file not found: {template_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading template {template_name}: {e}")
        raise

def _get_prompt_template(template_name: str, **kwargs: Any) -> HumanMessagePromptTemplate:
    """
    Creates a HumanMessagePromptTemplate based on the provided template name and parameters.

    Args:
        template_name (str): The name of the template.
        **kwargs: Additional parameters for the template.

    Returns:
        HumanMessagePromptTemplate: The configured prompt template.

    Raises:
        ValueError: If the template name is invalid.
    """
    template_configs = {
        "keyword_extraction": {"input_variables": ["HINT", "QUESTION"]},
        "column_filtering": {"input_variables": ["COLUMN_PROFILE", "QUESTION", "HINT"]},
        "column_filtering_with_examples": {"input_variables": ["COLUMN_PROFILE", "QUESTION", "HINT"]},
        "column_filtering_with_examples_llama": {"input_variables": ["COLUMN_PROFILE", "QUESTION", "HINT"]},
        "table_selection": {"input_variables": ["HINT", "QUESTION"], "partial_variables": {"DATABASE_SCHEMA": kwargs.get("schema_string", "")}},
        "column_selection": {"input_variables": ["HINT", "QUESTION"], "partial_variables": {"DATABASE_SCHEMA": kwargs.get("schema_string", "")}},
        "candidate_generation": {"input_variables": ["HINT", "QUESTION"], "partial_variables": {"DATABASE_SCHEMA": kwargs.get("schema_string", "")}},
        "finetuned_candidate_generation": {"input_variables": ["HINT", "QUESTION"], "partial_variables": {"DATABASE_SCHEMA": kwargs.get("schema_string", "")}},
        "revision": {"input_variables": ["SQL", "QUESTION", "MISSING_ENTITIES", "EVIDENCE", "QUERY_RESULT"], "partial_variables": {"DATABASE_SCHEMA": kwargs.get("schema_string", "")}}
    }

    if template_name not in template_configs:
        raise ValueError(f"Invalid template name: {template_name}")

    config = template_configs[template_name]
    input_variables = config["input_variables"]
    partial_variables = config.get("partial_variables", {})

    template_content = load_template(template_name)
    
    human_message_prompt_template = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=template_content,
            input_variables=input_variables,
            partial_variables=partial_variables
        )
    )

    return human_message_prompt_template

def get_prompt(template_name: str, schema_string: str = None) -> ChatPromptTemplate:
    """
    Creates a ChatPromptTemplate based on the provided template name and schema string.

    Args:
        template_name (str): The name of the template.
        schema_string (str, optional): The schema string for the template. Defaults to None.

    Returns:
        ChatPromptTemplate: The combined prompt template.
    """
    human_message_prompt_template = _get_prompt_template(template_name=template_name, schema_string=schema_string)
    
    combined_prompt_template = ChatPromptTemplate.from_messages(
        [human_message_prompt_template]
    )
    
    return combined_prompt_template
