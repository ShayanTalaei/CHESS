import logging
from typing import Any, Dict, List

from llm.models import async_llm_chain_call
from runner.logger import Logger
from runner.database_manager import DatabaseManager
from pipeline.utils import node_decorator, get_last_node_result, add_columns_to_tentative_schema
from pipeline.pipeline_manager import PipelineManager

@node_decorator(check_schema_status=True)
def column_filtering(task: Any, tentative_schema: Dict[str, Any], execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Filters columns based on profiles and updates the tentative schema.

    Args:
        task (Any): The task object containing the question and evidence.
        tentative_schema (Dict[str, Any]): The current tentative schema.
        execution_history (List[Dict[str, Any]]): The history of executions.

    Returns:
        Dict[str, Any]: A dictionary containing the updated tentative schema.
    """
    logging.info("Starting column filtering")

    schema_with_examples = get_last_node_result(execution_history, "entity_retrieval")["similar_values"]
    column_profiles = DatabaseManager().get_column_profiles(
        schema_with_examples=schema_with_examples, 
        use_value_description=True, 
        with_keys=True, 
        with_references=True
    )

    list_of_kwargs = []
    for table_name, columns in column_profiles.items():
        for column_name, column_profile in columns.items():
            kwargs = {
                "QUESTION": task.question,
                "HINT": task.evidence,
                "COLUMN_PROFILE": column_profile,
            }
            list_of_kwargs.append(kwargs)

    logging.info("Fetching prompt, engine, and parser from PipelineManager")
    prompt, engine, parser = PipelineManager().get_prompt_engine_parser()
    
    logging.info("Initiating asynchronous LLM chain call for column filtering")
    response = async_llm_chain_call(
        prompt=prompt, 
        engine=engine, 
        parser=parser,
        request_list=list_of_kwargs,
        step="column_filtering", 
        sampling_count=1
    )
    
    index = 0
    tentative_schema = {}
    for table_name, columns in column_profiles.items():
        tentative_schema[table_name] = []
        for column_name, column_profile in columns.items():
            try:
                chosen = (response[index][0]["is_column_information_relevant"].lower() == "yes")
                if chosen:
                    tentative_schema[table_name].append(column_name)
            except Exception as e:
                Logger().log(f"({task.db_id}, {task.question_id}) Error in column filtering: {e}", "error")
                logging.error(f"Error in column filtering for table '{table_name}', column '{column_name}': {e}")
            index += 1
            
    similar_columns = get_last_node_result(execution_history, "entity_retrieval")["similar_columns"]
    add_columns_to_tentative_schema(tentative_schema, similar_columns)
    tentative_schema = DatabaseManager().add_connections_to_tentative_schema(tentative_schema)
    
    result = {"tentative_schema": tentative_schema}
    logging.info("Column filtering completed successfully")
    return result
