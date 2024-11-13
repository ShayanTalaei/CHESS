import logging
from typing import Dict, List, Any

from llm.models import async_llm_chain_call, get_llm_chain
from llm.prompts import get_prompt
from llm.parsers import get_parser
from workflow.system_state import SystemState
from workflow.agents.tool import Tool
from runner.database_manager import DatabaseManager

class SelectTables(Tool):
    """
    Tool for selecting tables based on the specified mode and updating the tentative schema.
    """

    def __init__(self, mode: str, template_name: str = None, engine_config: str = None, parser_name: str = None, sampling_count: int = 1):
        super().__init__()
        self.mode = mode
        self.template_name = template_name
        self.engine_config = engine_config
        self.parser_name = parser_name
        self.sampling_count = sampling_count
        
        self.selected_tables = []
        self.chain_of_thought_reasoning = ""

    def _run(self, state: SystemState):
        """
        Executes the table selection process.
        
        Args:
            state (SystemState): The current system state.

        Returns:
            Dict[str, Any]: A dictionary containing the updated tentative schema and selected tables.
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

            aggregated_result = self.aggregate_tables(response)
            self.selected_tables = aggregated_result["table_names"]
            self.chain_of_thought_reasoning = aggregated_result["chain_of_thought_reasoning"]
            
        elif self.mode == "corrects":
            self.chain_of_thought_reasoning = "Tables that are appeared in the gold SQL query."
            self.selected_tables = DatabaseManager().get_sql_tables(state.task.SQL)
        else:
            logging.error(f"Unknown mode for table selection: {self.mode}")
            raise ValueError(f"Unknown mode for table selection: {self.mode}")

        state.tentative_schema = {
            table_name: state.tentative_schema.get(table_name, [])
            for table_name in self.selected_tables
        }
        state.add_columns_to_tentative_schema(state.similar_columns)
        state.add_connections_to_tentative_schema()

    def aggregate_tables(self, tables_dicts: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Aggregates tables from multiple responses and consolidates reasoning.

        Args:
            tables_dicts (List[Dict[str, Any]]): List of dictionaries containing table names and reasoning.

        Returns:
            Dict[str, List[str]]: Aggregated result with unique table names and consolidated reasoning.
        """
        logging.info("Aggregating tables from multiple responses")
        tables = []
        chain_of_thoughts = []
        for table_dict in tables_dicts:
            chain_of_thoughts.append(table_dict.get("chain_of_thought_reasoning", ""))
            response_tables = table_dict.get("table_names", [])
            for table in response_tables:
                if table.lower() not in [t.lower() for t in tables]:
                    tables.append(table)
        
        aggregated_chain_of_thoughts = "\n----\n".join(chain_of_thoughts)
        aggregation_result = {
            "table_names": tables,
            "chain_of_thought_reasoning": aggregated_chain_of_thoughts,
        }
        logging.info(f"Aggregated tables: {tables}")
        return aggregation_result

    def _get_updates(self, state: SystemState) -> Dict:
        updates = {
            "chain_of_thought_reasoning": self.chain_of_thought_reasoning,
            "selected_tables": self.selected_tables,
            "tentative_schema": state.tentative_schema,
        }
        updates.update(state.check_schema_status())
        return updates