import json
import re
import logging
from typing import Any, Dict, List, Tuple

from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

class PythonListOutputParser(BaseOutputParser):
    """Parses output embedded in markdown code blocks containing Python lists."""
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Any:
        """
        Parses the output to extract Python list content from markdown.

        Args:
            output (str): The output string containing Python list.

        Returns:
            Any: The parsed Python list.
        """
        logging.debug(f"Parsing output with PythonListOutputParser: {output}")
        if "```python" in output:
            output = output.split("```python")[1].split("```")[0]
        output = re.sub(r"^\s+", "", output)
        return eval(output)  # Note: Using eval is potentially unsafe, consider using ast.literal_eval if possible.

class ColumnFilteringOutput(BaseModel):
    """Model for column filtering output."""
    chain_of_thought_reasoning: str = Field(description="One line explanation of why or why not the column information is relevant to the question and the hint.")
    is_column_information_relevant: str = Field(description="Yes or No")

class TableSelectionOutputParser(BaseOutputParser):
    """Parses table selection outputs embedded in markdown code blocks containing JSON."""
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Any:
        """
        Parses the output to extract JSON content from markdown.

        Args:
            output (str): The output string containing JSON.

        Returns:
            Any: The parsed JSON content.
        """
        logging.debug(f"Parsing output with TableSelectionOutputParser: {output}")
        if "```json" in output:
            output = output.split("```json")[1].split("```")[0]
        output = re.sub(r"^\s+", "", output)
        output = output.replace("\n", " ").replace("\t", " ")
        return json.loads(output)

class ColumnSelectionOutput(BaseModel):
    """Model for column selection output."""
    table_columns: Dict[str, Tuple[str, List[str]]] = Field(description="A mapping of table and column names to a tuple containing the reason for the column's selection and a list of keywords for data lookup. If no keywords are required, an empty list is provided.")

class SQLGenerationOutput(BaseModel):
    """Model for SQL generation output."""
    chain_of_thought_reasoning: str = Field(description="Your thought process on how you arrived at the final SQL query.")
    SQL: str = Field(description="The generated SQL query in a single string.")

class MarkDownOutputParser(BaseOutputParser):
    """Parses output embedded in markdown code blocks containing SQL queries."""
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def parse(self, output: str) -> Dict[str, str]:
        """
        Parses the output to extract SQL content from markdown.

        Args:
            output (str): The output string containing SQL query.

        Returns:
            Dict[str, str]: A dictionary with the SQL query.
        """
        logging.debug(f"Parsing output with MarkDownOutputParser: {output}")
        if "```sql" in output:
            output = output.split("```sql")[1].split("```")[0]
        output = re.sub(r"^\s+", "", output)
        return {"SQL": output}

class SQLRevisionOutput(BaseModel):
    """Model for SQL revision output."""
    chain_of_thought_reasoning: str = Field(description="Your thought process on how you arrived at the final SQL query.")
    revised_SQL: str = Field(description="The revised SQL query in a single string.")

def get_parser(parser_name: str) -> BaseOutputParser:
    """
    Returns the appropriate parser based on the provided parser name.

    Args:
        parser_name (str): The name of the parser to retrieve.

    Returns:
        BaseOutputParser: The appropriate parser instance.

    Raises:
        ValueError: If the parser name is invalid.
    """
    parser_configs = {
        "keyword_extraction": PythonListOutputParser,
        "column_filtering": lambda: JsonOutputParser(pydantic_object=ColumnFilteringOutput),
        "table_selection": lambda: JsonOutputParser(pydantic_object=TableSelectionOutputParser),
        "column_selection": lambda: JsonOutputParser(pydantic_object=ColumnSelectionOutput),
        "candidate_generation": lambda: JsonOutputParser(pydantic_object=SQLGenerationOutput),
        "finetuned_candidate_generation": MarkDownOutputParser,
        "revision": lambda: JsonOutputParser(pydantic_object=SQLRevisionOutput),
    }

    if parser_name not in parser_configs:
        logging.error(f"Invalid parser name: {parser_name}")
        raise ValueError(f"Invalid parser name: {parser_name}")

    logging.info(f"Retrieving parser for: {parser_name}")
    parser = parser_configs[parser_name]() if callable(parser_configs[parser_name]) else parser_configs[parser_name]
    return parser
