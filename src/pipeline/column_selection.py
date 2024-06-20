import logging
from typing import Dict, List, Any

from llm.models import async_llm_chain_call
from runner.database_manager import DatabaseManager
from pipeline.utils import node_decorator, get_last_node_result
from pipeline.pipeline_manager import PipelineManager

@node_decorator(check_schema_status=True)
def column_selection(task: Any, tentative_schema: Dict[str, List[str]], execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Selects columns based on the specified mode and updates the tentative schema.

    Args:
        task (Any): The task object containing the question and evidence.
        tentative_schema (Dict[str, List[str]]): The current tentative schema.
        execution_history (List[Dict[str, Any]]): The history of executions.

    Returns:
        Dict[str, Any]: A dictionary containing the updated tentative schema and selected columns.
    """
    logging.info("Starting column selection")
    mode = PipelineManager().column_selection["mode"]

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
        
        sampling_count = PipelineManager().column_selection.get("sampling_count", 1)
        
        logging.info("Initiating asynchronous LLM chain call for column selection")
        response = async_llm_chain_call(
            prompt=prompt,
            engine=engine,
            parser=parser,
            request_list=[request_kwargs],
            step="column_selection",
            sampling_count=sampling_count,
        )[0]

        aggregated_result = aggregate_columns(response, list(tentative_schema.keys()))
        column_names = aggregated_result
        chain_of_thought_reasoning = aggregated_result.pop("chain_of_thought_reasoning")
        
        result = {
            "tentative_schema": column_names,
            "model_selected_columns": column_names, 
            "chain_of_thought_reasoning": chain_of_thought_reasoning
        }
    elif mode == "corrects":
        logging.info("Retrieving correct columns from SQL task")
        column_names = DatabaseManager().get_sql_columns_dict(task.sql)
        result = {
            "tentative_schema": column_names,
            "selected_columns": column_names
        }
    else:
        logging.error(f"Unknown mode for column selection: {mode}")
        raise ValueError(f"Unknown mode for column selection: {mode}")

    logging.info("Column selection completed successfully")
    return result

def aggregate_columns(columns_dicts: List[Dict[str, Any]], selected_tables: List[str]) -> Dict[str, List[str]]:
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
    
    logging.info(f"Aggregated columns: {columns}")
    return aggregation_result
