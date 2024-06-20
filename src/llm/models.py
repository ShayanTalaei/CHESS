import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from langchain_core.exceptions import OutputParserException
from langchain.output_parsers import OutputFixingParser

from llm.engine_configs import ENGINE_CONFIGS
from runner.logger import Logger

def get_llm_chain(engine: str, temperature: float = 0, base_uri: str = None) -> Any:
    """
    Returns the appropriate LLM chain based on the provided engine name and temperature.

    Args:
        engine (str): The name of the engine.
        temperature (float): The temperature for the LLM.
        base_uri (str, optional): The base URI for the engine. Defaults to None.

    Returns:
        Any: The LLM chain instance.

    Raises:
        ValueError: If the engine is not supported.
    """
    if engine not in ENGINE_CONFIGS:
        raise ValueError(f"Engine {engine} not supported")
    
    config = ENGINE_CONFIGS[engine]
    constructor = config["constructor"]
    params = config["params"]
    if temperature:
        params["temperature"] = temperature
    
    # Adjust base_uri if provided
    if base_uri and "openai_api_base" in params:
        params["openai_api_base"] = f"{base_uri}/v1"
    
    model = constructor(**params)
    if "preprocess" in config:
        llm_chain = config["preprocess"] | model
    else:
        llm_chain = model
    return llm_chain

def call_llm_chain(prompt: Any, engine: Any, parser: Any, request_kwargs: Dict[str, Any], step: int, log_file_lock: threading.Lock, max_attempts: int = 12, backoff_base: int = 2, jitter_max: int = 60) -> Any:
    """
    Calls the LLM chain with exponential backoff and jitter on failure.

    Args:
        prompt (Any): The prompt to be passed to the chain.
        engine (Any): The engine to be used in the chain.
        parser (Any): The parser to parse the output.
        request_kwargs (Dict[str, Any]): The request arguments.
        step (int): The current step in the process.
        log_file_lock (threading.Lock): The lock for logging into the file.
        max_attempts (int, optional): The maximum number of attempts. Defaults to 12.
        backoff_base (int, optional): The base for exponential backoff. Defaults to 2.
        jitter_max (int, optional): The maximum jitter in seconds. Defaults to 60.

    Returns:
        Any: The output from the chain.

    Raises:
        Exception: If all attempts fail.
    """
    logger = Logger()
    for attempt in range(max_attempts):
        try:
            chain = prompt | engine | parser
            prompt_text = prompt.invoke(request_kwargs).messages[0].content
            output = chain.invoke(request_kwargs)
            with log_file_lock:
                logger.log_conversation(prompt_text, "Human", step)
                logger.log_conversation(output, "AI", step)
            return output
        except OutputParserException as e:
            new_parser = OutputFixingParser.from_llm(parser=parser, llm=engine)
            chain = prompt | engine | new_parser
            if attempt == max_attempts - 1:
                logger.log(f"call_chain: {e}", "error")
                raise e
        except Exception as e:
            if attempt < max_attempts - 1:
                logger.log(f"Failed to invoke the chain {attempt + 1} times.\n{type(e)}\n{e}", "warning")
                sleep_time = (backoff_base ** attempt) + random.uniform(0, jitter_max)
                time.sleep(sleep_time)
            else:
                logger.log(f"Failed to invoke the chain {attempt + 1} times.\n{type(e)} <{e}>\n", "error")
                raise e

def threaded_llm_call(request_id: int, prompt: Any, engine: Any, parser: Any, request_kwargs: Dict[str, Any], step: int, result_queue: queue.Queue, log_file_lock: threading.Lock) -> None:
    """
    Makes a threaded call to the LLM chain and stores the result in a queue.

    Args:
        request_id (int): The ID of the request.
        prompt (Any): The prompt to be passed to the chain.
        engine (Any): The engine to be used in the chain.
        parser (Any): The parser to parse the output.
        request_kwargs (Dict[str, Any]): The request arguments.
        step (int): The current step in the process.
        result_queue (queue.Queue): The queue to store results.
        log_file_lock (threading.Lock): The lock for logging into the file.
    """
    try:
        result = call_llm_chain(prompt, engine, parser, request_kwargs, step, log_file_lock)
        result_queue.put((request_id, result))  # Store a tuple of request ID and its result
    except Exception as e:
        logging.error(f"Exception in thread with request: {request_kwargs}\n{e}")
        result_queue.put((request_id, None))  # Indicate failure for this request

def async_llm_chain_call(prompt: Any, engine: Any, parser: Any, request_list: List[Dict[str, Any]], step: int, sampling_count: int) -> List[List[Any]]:
    """
    Asynchronously calls the LLM chain using multiple threads.

    Args:
        prompt (Any): The prompt to be passed to the chain.
        engine (Any): The engine to be used in the chain.
        parser (Any): The parser to parse the output.
        request_list (List[Dict[str, Any]]): The list of request arguments.
        step (int): The current step in the process.
        sampling_count (int): The number of samples to be taken.

    Returns:
        List[List[Any]]: A list of lists containing the results for each request.
    """
    result_queue = queue.Queue()  # Queue to store results
    log_file_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=len(request_list) * sampling_count) as executor:
        for request_id, request_kwargs in enumerate(request_list):
            for _ in range(sampling_count):
                executor.submit(threaded_llm_call, request_id, prompt, engine, parser, request_kwargs, step, result_queue, log_file_lock)
                time.sleep(0.2)  # Sleep for a short time to avoid rate limiting

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
        
    # Sort results based on their request IDs
    results = sorted(results, key=lambda x: x[0])
    sorted_results = [result[1] for result in results]

    # Group results by sampling_count
    grouped_results = [sorted_results[i * sampling_count: (i + 1) * sampling_count] for i in range(len(request_list))]

    return grouped_results
