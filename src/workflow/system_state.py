from pydantic import BaseModel
from typing import List, Dict, Any

from runner.task import Task
from runner.database_manager import DatabaseManager
from workflow.sql_meta_info import SQLMetaInfo

import re


class SystemState(BaseModel):
    """
    Represents the state of a graph during execution.

    Attributes:
        task (Task): The task associated with the graph state.
        tentative_schema (Dict[str, Any]): A dictionary representing the tentative schema.
        execution_history (List[Any]): A list representing the execution history.
    """

    executing_tool: str = ""
    task: Task
    tentative_schema: Dict[str, List[str]]
    execution_history: List[Any]
    
    keywords: List[str] = []
    
    similar_columns: Dict[str, List[str]] = {}
    schema_with_examples: Dict[str, Dict[str, List[str]]] = {}
    schema_with_descriptions:  Dict[str, Dict[str, Dict[str, str]]] = {}
    
    SQL_meta_infos: Dict[str, List[SQLMetaInfo]] = {}
    unit_tests: Dict[str, List[str]] = {}
    errors: Dict[str, str] = {}
    
    def add_columns_to_tentative_schema(self, selected_columns: Dict[str, List[str]]) -> None:
        """
        Adds columns to the tentative schema based on selected columns.

        Args:
            tentative_schema (Dict[str, List[str]]): The tentative schema.
            selected_columns (Dict[str, List[str]]): The selected columns to add.
        """
        for table_name, columns in selected_columns.items():
            target_table_name = next((t for t in self.tentative_schema.keys() if t.lower() == table_name.lower()), None)
            if target_table_name:
                for column in columns:
                    if column.lower() not in [c.lower() for c in self.tentative_schema[target_table_name]]:
                        self.tentative_schema[target_table_name].append(column)
            else:
                self.tentative_schema[table_name] = columns
    
    def check_schema_status(self) -> Dict[str, any]:
        """
        Checks for missing tables and columns in the tentative schema.

        Args:
            task (Any): The current task.
            tentative_schema (Dict[str, List[str]]): The tentative schema.

        Returns:
            Dict[str, Any]: A dictionary with the status of missing tables and columns.
        """
        ground_truth_sql = self.task.SQL
        correct_columns = DatabaseManager().get_sql_columns_dict(sql=ground_truth_sql)
        missing_tables = []
        missing_columns = []

        for table_name, cols in correct_columns.items():
            for col in cols:
                selected_table = [table for table in self.tentative_schema.keys() if table.lower() == table_name.lower()]
                if not selected_table:
                    if table_name not in missing_tables:
                        missing_tables.append(table_name)
                else:
                    selected_table = selected_table[0]
                    if col.lower() not in [selected_col.lower() for selected_col in self.tentative_schema[selected_table]]:
                        missing_columns.append(f"'{table_name}'.'{col}'")
        
        status = {
            "missing_table_status": "success" if not missing_tables else "missing_table",
            "missing_tables": missing_tables,
            "missing_column_status": "success" if not missing_columns else "missing_column",
            "missing_columns": missing_columns,
            "correct_columns": correct_columns,
        }
        return status
    
    def add_connections_to_tentative_schema(self):
        """
        Adds connections to the tentative schema.
        """
        DatabaseManager().add_connections_to_tentative_schema(self.tentative_schema)
        
    def get_schema_string(self,
                          schema_type: str = "tentative",
                          include_value_description: bool = True) -> str:

        if schema_type == "tentative":
            schema = self.tentative_schema
        elif schema_type == "complete":
            schema = DatabaseManager().get_db_schema()
        else:
            raise ValueError(f"Unknown schema type: {schema_type}")

        return DatabaseManager().get_database_schema_string(
            schema,
            self.schema_with_examples,
            self.schema_with_descriptions,
            include_value_description=include_value_description
        )
    
    def get_database_schema_for_queries(
        self,
        queries: List[str],
        include_value_description: bool = True
    ) -> str:
        schema_dict_list = []
        for query in queries:
            try:
                schema_dict_list.append(DatabaseManager().get_sql_columns_dict(query))
            except Exception as e:
                print(f"Error in getting database schema for query: {e}")
                schema_dict_list.append({})
        union_schema_dict = DatabaseManager().get_union_schema_dict(schema_dict_list)
        database_info = DatabaseManager().get_database_schema_string(
            union_schema_dict,
            self.schema_with_examples,
            self.schema_with_descriptions,
            include_value_description=include_value_description
        )
        return database_info
    
    def construct_history(self) -> Dict:
        """
        Constructs the history of the executing tool.
        
        Args:
            state (SystemState): The current system state.
        
        Returns:
            Dict: The history of the previous question and SQL pairs.
        """
        history = ""
        # Convert dict_values to a list
        values_list = list(self.SQL_meta_infos.values())

        for index in range(len(values_list) - 1):
            history += f"Step: {index + 1}\n"
            history += f"Original SQL: {self.remove_new_lines(values_list[index][0].SQL)}\n"
            history += f"Feedbacks: {self.remove_new_lines(self._get_feedback_string(values_list[index][0].feedbacks))}\n"
            history += f"Refined SQL: {self.remove_new_lines(values_list[index + 1][0].SQL)}\n"
            history += "\n"

        if not history:
            history = "No history available."

        return history
    
    def remove_new_lines(self, text):
        return re.sub(r'[\r\n]+', '', text)
    
    def _get_feedback_string(self, feedbacks: List[str]) -> str:
        """
        Returns a string representation of the feedbacks.
        
        Args:
            feedbacks (List[str]): The list of feedbacks.
        
        Returns:
            str: The string representation of the feedbacks.
        """
        feedback_string = ""
        for i, feedback in enumerate(feedbacks):
            feedback_string += f"--> {i+1}. {feedback}\n"
        return feedback_string
