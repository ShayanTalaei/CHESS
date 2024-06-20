import logging
from typing import Dict, List, Any

from llm.models import async_llm_chain_call
from pipeline.utils import node_decorator, get_last_node_result, add_columns_to_tentative_schema
from pipeline.pipeline_manager import PipelineManager
from runner.database_manager import DatabaseManager

@node_decorator(check_schema_status=True)
def table_selection(task: Any, tentative_schema: Dict[str, List[str]], execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Selects tables based on the specified mode and updates the tentative schema.

    Args:
        task (Any): The task object containing the question and evidence.
        tentative_schema (Dict[str, List[str]]): The current tentative schema.
        execution_history (List[Dict[str, Any]]): The history of executions.

    Returns:
        Dict[str, Any]: A dictionary containing the updated tentative schema and selected tables.
    """
    logging.info("Starting table selection")
    mode = PipelineManager().table_selection["mode"]

    if mode == "ask_model":
        schema_with_examples = get_last_node_result(execution_history, "entity_retrieval")["similar_values"]
        schema_with_descriptions = get_last_node_result(execution_history, "context_retrieval")["schema_with_descriptions"]
        
        schema_string = DatabaseManager().get_database_schema_string(
            tentative_schema,
            schema_with_examples,
            schema_with_descriptions,
            include_value_description=True
        )
        
        logging.info("Fetching prompt, engine, and parser from PipelineManager")
        prompt, engine, parser = PipelineManager().get_prompt_engine_parser(schema_string=schema_string)

        request_kwargs = {
            "HINT": task.evidence,
            "QUESTION": task.question,
        }
        
        sampling_count = PipelineManager().table_selection.get("sampling_count", 1)
        
        logging.info("Initiating asynchronous LLM chain call for table selection")
        response = async_llm_chain_call(
            prompt=prompt,
            engine=engine,
            parser=parser,
            request_list=[request_kwargs],
            step="table_selection",
            sampling_count=sampling_count,
        )[0]

        aggregated_result = aggregate_tables(response)
        table_names = aggregated_result["table_names"]
        result = {
            "chain_of_thought_reasoning": aggregated_result["chain_of_thought_reasoning"],
            "selected_tables": table_names,
        }
    elif mode == "corrects":
        logging.info("Retrieving correct tables from SQL task")
        table_names = DatabaseManager().get_sql_tables(task.sql)
        result = {
            "selected_tables": table_names,
        }
    else:
        logging.error(f"Unknown mode for table selection: {mode}")
        raise ValueError(f"Unknown mode for table selection: {mode}")

    tentative_schema = {
        table_name: tentative_schema.get(table_name, [])
        for table_name in table_names
    }

    similar_columns = get_last_node_result(execution_history, "entity_retrieval")["similar_columns"]
    add_columns_to_tentative_schema(tentative_schema, similar_columns)
    tentative_schema = DatabaseManager().add_connections_to_tentative_schema(tentative_schema)

    result = {"tentative_schema": tentative_schema, 
              **result}
    logging.info("Table selection completed successfully")
    return result

def aggregate_tables(tables_dicts: List[Dict[str, Any]]) -> Dict[str, List[str]]:
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
