import logging
from typing import Dict, List, Any

from llm.models import async_llm_chain_call, get_llm_chain
from llm.prompts import get_prompt
from llm.parsers import get_parser
from workflow.system_state import SystemState
from workflow.agents.tool import Tool
from runner.database_manager import DatabaseManager

class SelectColumns(Tool):
    """
    Tool for selecting columns based on the specified mode and updating the tentative schema.
    """

    def __init__(self, mode: str, template_name: str, engine_config: Dict[str, Any], parser_name: str, sampling_count: int = 1):
        super().__init__()
        self.mode = mode
        self.template_name = template_name
        self.engine_config = engine_config
        self.parser_name = parser_name
        self.sampling_count = sampling_count
        
        self.selected_columns = {}
        self.chain_of_thought_reasoning = ""

    def _run(self, state: SystemState):
        """
        Executes the column selection process.
        
        Args:
            state (SystemState): The current system state.
        """
        if self.mode == "ask_model":                        
            request_kwargs = {
                "DATABASE_SCHEMA": state.get_schema_string(schema_type="tentative"),
                "QUESTION": state.task.question,
                "HINT": state.task.evidence,
            }
            
            response = async_llm_chain_call(
                prompt=get_prompt(template_name=self.template_name),
                engine=get_llm_chain(**self.engine_config),
                parser=get_parser(self.parser_name),
                request_list=[request_kwargs],
                step=self.tool_name,
                sampling_count=self.sampling_count,
            )[0]

            aggregated_result = self.aggregate_columns(response, list(state.tentative_schema.keys()))
            self.chain_of_thought_reasoning = aggregated_result.pop("chain_of_thought_reasoning")
            # self.selected_columns = self.union_schemas(response)
            self.selected_columns = aggregated_result
            
        elif self.mode == "corrects":
            self.chain_of_thought_reasoning = "Columns that are appeared in the gold SQL query."
            self.selected_columns = DatabaseManager().get_sql_columns_dict(state.task.SQL)

        else:
            logging.error(f"Unknown mode for column selection: {self.mode}")
            raise ValueError(f"Unknown mode for column selection: {self.mode}")
        
        state.tentative_schema = self.selected_columns.copy()

    def union_schemas(self, schemas):
        schema_union = {}
        for schema in schemas:
            for table, columns in schema.items():
                table_lower = table.lower()
                col_lower = [col.lower() for col in columns]
                if table_lower not in schema_union:
                    schema_union[table_lower] = []
                schema_union[table_lower] += [col for col in col_lower if col not in schema_union[table_lower]]
        return schema_union

    def aggregate_columns(self, columns_dicts: List[Dict[str, Any]], selected_tables: List[str]) -> Dict[str, List[str]]:
        """
        Aggregates columns from multiple responses and consolidates reasoning.

        Args:
            columns_dicts (List[Dict[str, Any]]): List of dictionaries containing column names and reasoning.
            selected_tables (List[str]): List of selected tables.

        Returns:
            Dict[str, List[str]]: Aggregated result with unique column names and consolidated reasoning.
        """
        logging.info("Aggregating columns from multiple responses")
        columns = {}
        chain_of_thoughts = []
        for column_dict in columns_dicts:
            valid_column_dict = False
            for key, value in column_dict.items():
                if key == "chain_of_thought_reasoning":
                    dict_cot = value
                else:  # key is table name
                    table_name = key
                    if table_name.startswith("`"):
                        table_name = table_name[1:-1]
                    column_names = value
                    if table_name.lower() in [t.lower() for t in selected_tables]:
                        for column_name in column_names:
                            if column_name.startswith("`"):
                                column_name = column_name[1:-1]
                            if table_name not in columns:
                                columns[table_name] = []
                            if column_name.lower() not in [col.lower() for col in columns[table_name]]:
                                columns[table_name].append(column_name)
                            valid_column_dict = True
            if valid_column_dict:
                chain_of_thoughts.append(dict_cot)
        
        aggregated_chain_of_thoughts = "\n----\n".join(chain_of_thoughts)
        aggregation_result = columns
        aggregation_result["chain_of_thought_reasoning"] = aggregated_chain_of_thoughts
        return aggregation_result

    def _get_updates(self, state: SystemState) -> Dict:
        updates = {
            "chain_of_thought_reasoning": self.chain_of_thought_reasoning,
            "selected_columns": self.selected_columns,
            "tentative_schema": state.tentative_schema,
        }
        updates.update(state.check_schema_status())

        return updates