import logging
from typing import Any, Dict

from llm.models import async_llm_chain_call
from pipeline.utils import node_decorator
from pipeline.pipeline_manager import PipelineManager

@node_decorator(check_schema_status=False)
def keyword_extraction(task: Any, tentative_schema: Dict[str, Any], execution_history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts keywords from the task using an LLM chain call.

    Args:
        task (Any): The task object containing the evidence and question.
        tentative_schema (Dict[str, Any]): The current tentative schema.
        execution_history (Dict[str, Any]): The history of executions.

    Returns:
        Dict[str, Any]: A dictionary containing the extracted keywords.
    """
    request_kwargs = {
        "HINT": task.evidence,
        "QUESTION": task.question,
    }
    
    logging.info("Fetching prompt, engine, and parser from PipelineManager")
    prompt, engine, parser = PipelineManager().get_prompt_engine_parser()
    
    logging.info("Initiating asynchronous LLM chain call for keyword extraction")
    response = async_llm_chain_call(
        prompt=prompt, 
        engine=engine, 
        parser=parser,
        request_list=[request_kwargs],
        step="keyword_extraction",
        sampling_count=1
    )[0]
    
    keywords = response[0]
    result = {"keywords": keywords}
    
    logging.info(f"Keywords extracted: {keywords}")
    return result
