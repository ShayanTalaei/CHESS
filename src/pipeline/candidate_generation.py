import logging
from typing import Dict, List, Any

from llm.models import async_llm_chain_call
from runner.database_manager import DatabaseManager
from pipeline.utils import node_decorator, get_last_node_result
from pipeline.pipeline_manager import PipelineManager

@node_decorator(check_schema_status=False)
def candidate_generation(task: Any, tentative_schema: Dict[str, List[str]], execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generates candidate SQL queries based on the task's question and evidence.

    Args:
        task (Any): The task object containing the question and evidence.
        tentative_schema (Dict[str, List[str]]): The current tentative schema.
        execution_history (List[Dict[str, Any]]): The history of executions.

    Returns:
        Dict[str, Any]: A dictionary containing the best SQL query result.
    """
    logging.info("Starting candidate generation")

    schema_with_examples = get_last_node_result(execution_history, node_type="entity_retrieval")["similar_values"]
    schema_with_descriptions = get_last_node_result(execution_history, node_type="context_retrieval")["schema_with_descriptions"]
    
    schema_string = DatabaseManager().get_database_schema_string(
        tentative_schema, 
        schema_with_examples, 
        schema_with_descriptions, 
        include_value_description=True
    )
    
    logging.info("Fetching prompt, engine, and parser from PipelineManager")
    prompt, engine, parser = PipelineManager().get_prompt_engine_parser(schema_string=schema_string)
    
    request_kwargs = {
        "QUESTION": task.question,
        "HINT": task.evidence,
    }
    
    sampling_count = PipelineManager().candidate_generation.get("sampling_count", 1)
    
    logging.info("Initiating asynchronous LLM chain call for candidate generation")
    response = async_llm_chain_call(
        prompt=prompt,
        engine=engine,
        parser=parser,
        request_list=[request_kwargs],
        step="nl_to_sql",
        sampling_count=sampling_count
    )[0]
    
    sqls = [res["SQL"] for res in response]
    sql = DatabaseManager().aggregate_sqls(sqls)
    result = next(res for res in response if res["SQL"] == sql)
    
    logging.info("Candidate generation completed successfully")
    return result
