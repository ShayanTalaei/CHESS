import logging
from typing import Dict

from llm.models import async_llm_chain_call, get_llm_chain
from llm.prompts import get_prompt
from llm.parsers import get_parser
from runner.logger import Logger
from runner.database_manager import DatabaseManager
from workflow.system_state import SystemState
from workflow.agents.tool import Tool

class FilterColumn(Tool):
    """
    Tool for filtering columns based on profiles and updating the tentative schema.
    """

    def __init__(self, template_name: str = None, engine_config: str = None, parser_name: str = None):
        super().__init__()
        self.template_name = template_name
        self.engine_config = engine_config
        self.parser_name = parser_name

    def _run(self, state: SystemState):
        """
        Executes the column filtering process.
        
        Args:
            state (SystemState): The current system state.
        """

        column_profiles = DatabaseManager().get_column_profiles(
            schema_with_examples=state.schema_with_examples, 
            use_value_description=True, 
            with_keys=True, 
            with_references=True,
            tentative_schema=state.tentative_schema
        )

        list_of_kwargs = []
        for table_name, columns in column_profiles.items():
            for column_name, column_profile in columns.items():
                kwargs = {
                    "QUESTION": state.task.question,
                    "HINT": state.task.evidence,
                    "COLUMN_PROFILE": column_profile,
                }
                list_of_kwargs.append(kwargs)

        response = async_llm_chain_call(
            prompt=get_prompt(template_name=self.template_name),
            engine=get_llm_chain(**self.engine_config),
            parser=get_parser(self.parser_name),
            request_list=list_of_kwargs,
            step=self.tool_name, 
            sampling_count=1
        )
        
        index = 0
        tentative_schema = state.tentative_schema
        for table_name, columns in column_profiles.items():
            tentative_schema[table_name] = []
            for column_name, column_profile in columns.items():
                try:
                    chosen = (response[index][0]["is_column_information_relevant"].lower() == "yes")
                    if chosen:
                        tentative_schema[table_name].append(column_name)
                except Exception as e:
                    Logger().log(f"({state.task.db_id}, {state.task.question_id}) Error in column filtering: {e}", "error")
                    logging.error(f"Error in column filtering for table '{table_name}', column '{column_name}': {e}")
                index += 1        
        
        state.add_columns_to_tentative_schema(state.similar_columns)
        state.add_connections_to_tentative_schema()
        

    def _get_updates(self, state: SystemState) -> Dict:
        updates = {"tentative_schema": state.tentative_schema}
        updates.update(state.check_schema_status())
        return updates