import logging
from typing import Dict, List

from database_utils.db_catalog.csv_utils import load_tables_description

from runner.database_manager import DatabaseManager
from workflow.system_state import SystemState
from workflow.agents.tool import Tool

import os
from dotenv import load_dotenv

load_dotenv()

class RetrieveContext(Tool):
    """
    Tool for retrieving context information based on the task's question and evidence.
    """

    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k
        
    def _run(self, state: SystemState):
        """
        Executes the context retrieval process.
        
        Args:
            state (SystemState): The current system state.
        """
        
        retrieved_columns = self._find_most_similar_columns(
            question=state.task.question,
            evidence=state.task.evidence,
            keywords=state.keywords,
            top_k=self.top_k
        )
        
        state.schema_with_descriptions = self._format_retrieved_descriptions(retrieved_columns)

        # try:
        #     path = os.path.join(os.getenv("DB_ROOT_DIRECTORY"), state.task.db_id)
        #     state.schema_with_descriptions = load_tables_description(path, use_value_description=True)
        # except Exception as e:
        #     logging.error(f"Error loading tables description: {e}")
        #     state.schema_with_descriptions = {}

    ### Context similarity ###
    
    def _find_most_similar_columns(self, question: str, evidence: str, keywords: List[str], top_k: int) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Finds the most similar columns based on the question and evidence.

        Args:
            question (str): The question string.
            evidence (str): The evidence string.
            keywords (List[str]): The list of keywords.
            top_k (int): The number of top similar columns to retrieve.

        Returns:
            Dict[str, Dict[str, Dict[str, str]]]: A dictionary containing the most similar columns with descriptions.
        """
        logging.info("Finding the most similar columns")
        tables_with_descriptions = {}
        
        for keyword in keywords:
            question_based_query = f"{question} {keyword}"
            evidence_based_query = f"{evidence} {keyword}"
            
            retrieved_question_based_query = DatabaseManager().query_vector_db(question_based_query, top_k=top_k)
            retrieved_evidence_based_query = DatabaseManager().query_vector_db(evidence_based_query, top_k=top_k)
            
            tables_with_descriptions = self._add_description(tables_with_descriptions, retrieved_question_based_query)
            tables_with_descriptions = self._add_description(tables_with_descriptions, retrieved_evidence_based_query)
        
        return tables_with_descriptions

    def _add_description(self, tables_with_descriptions: Dict[str, Dict[str, Dict[str, str]]], 
                         retrieved_descriptions: Dict[str, Dict[str, Dict[str, str]]]) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Adds descriptions to tables from retrieved descriptions.

        Args:
            tables_with_descriptions (Dict[str, Dict[str, Dict[str, str]]]): The current tables with descriptions.
            retrieved_descriptions (Dict[str, Dict[str, Dict[str, str]]]): The retrieved descriptions.

        Returns:
            Dict[str, Dict[str, Dict[str, str]]]: The updated tables with descriptions.
        """
        if retrieved_descriptions is None:
            logging.warning("No descriptions retrieved")
            return tables_with_descriptions
        for table_name, column_descriptions in retrieved_descriptions.items():
            if table_name not in tables_with_descriptions:
                tables_with_descriptions[table_name] = {}
            for column_name, description in column_descriptions.items():
                if (column_name not in tables_with_descriptions[table_name] or 
                    description["score"] > tables_with_descriptions[table_name][column_name]["score"]):
                    tables_with_descriptions[table_name][column_name] = description
        return tables_with_descriptions

    def _format_retrieved_descriptions(self, retrieved_columns: Dict[str, Dict[str, Dict[str, str]]]) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Formats retrieved descriptions by removing the score key.

        Args:
            retrieved_columns (Dict[str, Dict[str, Dict[str, str]]]): The retrieved columns with descriptions.

        Returns:
            Dict[str, Dict[str, Dict[str, str]]]: The formatted descriptions.
        """
        logging.info("Formatting retrieved descriptions")
        for column_descriptions in retrieved_columns.values():
            for column_info in column_descriptions.values():
                column_info.pop("score", None)
        return retrieved_columns

    def _get_updates(self, state: SystemState) -> Dict:
        return {"schema_with_descriptions": state.schema_with_descriptions}