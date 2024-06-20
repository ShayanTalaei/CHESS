import difflib
import logging
from typing import Dict, List, Tuple, Any

from llm.models import async_llm_chain_call
from database_utils.schema import DatabaseSchema
from runner.database_manager import DatabaseManager
from pipeline.utils import node_decorator, get_last_node_result
from pipeline.pipeline_manager import PipelineManager

@node_decorator(check_schema_status=False)
def revision(task: Any, tentative_schema: Dict[str, List[str]], execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Revises the predicted SQL query based on task evidence and schema information.

    Args:
        task (Any): The task object containing the question and evidence.
        tentative_schema (Dict[str, List[str]]): The current tentative schema.
        execution_history (List[Dict[str, Any]]): The history of executions.

    Returns:
        Dict[str, Any]: A dictionary containing the revised SQL query and reasoning.
    """
    logging.info("Starting SQL revision")

    schema_with_examples = get_last_node_result(execution_history, "entity_retrieval").get("similar_values", {})
    schema_with_descriptions = get_last_node_result(execution_history, "context_retrieval").get("schema_with_descriptions", {})
    
    complete_schema = DatabaseManager().get_db_schema()
    schema_string = DatabaseManager().get_database_schema_string(
        complete_schema,
        schema_with_examples,
        schema_with_descriptions,
        include_value_description=True
    )

    logging.info("Fetching prompt, engine, and parser from PipelineManager")
    prompt, engine, parser = PipelineManager().get_prompt_engine_parser(schema_string=schema_string)
    
    predicted_query = get_last_node_result(execution_history, "candidate_generation")["SQL"]

    try:
        query_result = DatabaseManager().validate_sql_query(sql=predicted_query)['RESULT']
    except Exception as e:
        query_result = str(e)
        logging.error(f"Error validating SQL query: {e}")
        
    try:
        missing_entities = find_wrong_entities(predicted_query, schema_with_examples)
    except Exception as e:
        missing_entities = {}
        logging.error(f"Error finding wrong entities: {e}")

    request_kwargs = {
        "SQL": predicted_query,
        "QUESTION": task.question,
        "MISSING_ENTITIES": missing_entities,
        "EVIDENCE": task.evidence,
        "QUERY_RESULT": query_result,
    }
    
    sampling_count = PipelineManager().revision.get("sampling_count", 1)
    
    logging.info("Initiating asynchronous LLM chain call for SQL revision")
    response = async_llm_chain_call(
        prompt=prompt, 
        engine=engine, 
        parser=parser,
        request_list=[request_kwargs],
        step="revision",
        sampling_count=sampling_count
    )[0]
    
    revised_sqls = [res["revised_SQL"] for res in response]
    revised_sql = DatabaseManager().aggregate_sqls(sqls=revised_sqls)
    chosen_res = next(res for res in response if res["revised_SQL"] == revised_sql)
    
    result = {
        "chain_of_thought_reasoning": chosen_res.get("chain_of_thought_reasoning", ""),
        "SQL": chosen_res["revised_SQL"],
    }
    
    logging.info("SQL revision completed successfully")
    return result

def find_wrong_entities(sql: str, similar_values: Dict[str, Dict[str, List[str]]], similarity_threshold: float = 0.4) -> str:
    """
    Finds and returns a string listing entities in the SQL that do not match the database schema.

    Args:
        sql (str): The SQL query to check.
        similar_values (Dict[str, Dict[str, List[str]]]): Dictionary of similar values for columns.
        similarity_threshold (float, optional): The similarity threshold for matching values. Defaults to 0.4.

    Returns:
        str: A string listing the mismatched entities and suggestions.
    """
    logging.info("Finding wrong entities in the SQL query")
    wrong_entities = ""
    used_entities = DatabaseManager().get_sql_condition_literals(sql)
    similar_values_database_schema = DatabaseSchema.from_schema_dict_with_examples(similar_values)

    for table_name, column_info in used_entities.items():
        for column_name, column_values in column_info.items():
            target_column_info = similar_values_database_schema.get_column_info(table_name, column_name)
            if not target_column_info:
                continue
            for value in column_values:
                column_similar_values = target_column_info.examples
                if value not in column_similar_values:
                    most_similar_entity, similarity = _find_most_syntactically_similar_value(value, column_similar_values)
                    if similarity > similarity_threshold:
                        wrong_entities += f"Column {column_name} in table {table_name} does not contain the value '{value}'. The correct value is '{most_similar_entity}'.\n"

    for used_table_name, used_column_info in used_entities.items():
        for used_column_name, used_values in used_column_info.items():
            for used_value in used_values:
                for table_name, column_info in similar_values.items():
                    for column_name, column_values in column_info.items():
                        if (used_value in column_values) and (column_name.lower() != used_column_name.lower()):
                            wrong_entities += f"Value {used_value} that you used in the query appears in the column {column_name} of table {table_name}.\n"
    return wrong_entities

def _find_most_syntactically_similar_value(target_value: str, candidate_values: List[str]) -> Tuple[str, float]:
    """
    Finds the most syntactically similar value to the target value from the candidate values.

    Args:
        target_value (str): The target value to match.
        candidate_values (List[str]): The list of candidate values.

    Returns:
        Tuple[str, float]: The most similar value and the similarity score.
    """
    most_similar_entity = max(candidate_values, key=lambda value: difflib.SequenceMatcher(None, value, target_value).ratio(), default=None)
    max_similarity = difflib.SequenceMatcher(None, most_similar_entity, target_value).ratio() if most_similar_entity else 0
    return most_similar_entity, max_similarity
