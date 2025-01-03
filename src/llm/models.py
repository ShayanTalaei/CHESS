from typing import Any, Dict, List

from langchain_core.exceptions import OutputParserException
from langchain.output_parsers import OutputFixingParser

from llm.engine_configs import ENGINE_CONFIGS
from runner.logger import Logger
from threading_utils import ordered_concurrent_function_calls


def get_llm_chain(engine_name: str, temperature: float = 0, base_uri: str = None) -> Any:
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
    if engine_name not in ENGINE_CONFIGS:
        raise ValueError(f"Engine {engine_name} not supported")

    config = ENGINE_CONFIGS[engine_name]
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


def call_llm_chain(
    prompt: Any,
    engine: Any,
    parser: Any,
    request_kwargs: Dict[str, Any],
    step: int,
    max_attempts: int = 12,
    backoff_base: int = 2,
    jitter_max: int = 60,
) -> Any:
    """
    Calls the LLM chain with exponential backoff and jitter on failure.

    Args:
        prompt (Any): The prompt to be passed to the chain.
        engine (Any): The engine to be used in the chain.
        parser (Any): The parser to parse the output.
        request_kwargs (Dict[str, Any]): The request arguments.
        step (int): The current step in the process.
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
            # chain = prompt | engine | parser
            chain = prompt | engine
            prompt_text = prompt.invoke(request_kwargs).messages[0].content
            output = chain.invoke(request_kwargs)
            if isinstance(output, str):
                if output.strip() == "":
                    engine = get_llm_chain("gemini-1.5-flash")
                    raise OutputParserException("Empty output")
            else:
                if output.content.strip() == "":
                    engine = get_llm_chain("gemini-1.5-flash")
                    raise OutputParserException("Empty output")
            output = parser.invoke(output)
            logger.log_conversation([
                {"text": prompt_text, "from": "Human", "step": step},
                {"text": output, "from": "AI", "step": step},
            ])
            return output
        except OutputParserException as e:
            logger.log(f"OutputParserException: {e}", "warning")
            new_parser = OutputFixingParser.from_llm(parser=parser, llm=engine)
            chain = prompt | engine | new_parser
            if attempt == max_attempts - 1:
                logger.log(f"call_chain: {e}", "error")
                raise e
        except Exception as e:
            # if attempt < max_attempts - 1:
            #     logger.log(f"Failed to invoke the chain {attempt + 1} times.\n{type(e)}\n{e}", "warning")
            #     sleep_time = (backoff_base ** attempt) + random.uniform(0, jitter_max)
            #     time.sleep(sleep_time)
            # else:
            logger.log(
                f"Failed to invoke the chain {attempt + 1} times.\n{type(e)} <{e}>\n", "error"
            )
            raise e


def async_llm_chain_call(
    prompt: Any,
    engine: Any,
    parser: Any,
    request_list: List[Dict[str, Any]],
    step: int,
    sampling_count: int = 1,
) -> List[List[Any]]:
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

    call_list = []
    engine_id = 0
    for request_id, request_kwargs in enumerate(request_list):
        for _ in range(sampling_count):
            call_list.append({
                "function": call_llm_chain,
                "kwargs": {
                    "prompt": prompt,
                    "engine": (
                        engine[engine_id % len(engine)] if isinstance(engine, list) else engine
                    ),
                    "parser": parser,
                    "request_kwargs": request_kwargs,
                    "step": step,
                },
            })
            engine_id += 1

    # Execute the functions concurrently
    results = ordered_concurrent_function_calls(call_list)

    # Group results by sampling_count
    grouped_results = [
        results[i * sampling_count : (i + 1) * sampling_count] for i in range(len(request_list))
    ]

    return grouped_results


def call_engine(
    message: str, engine: Any, max_attempts: int = 12, backoff_base: int = 2, jitter_max: int = 60
) -> Any:
    """
    Calls the LLM chain with exponential backoff and jitter on failure.

    Args:
        message (str): The message to be passed to the chain.
        engine (Any): The engine to be used in the chain.
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
            output = engine.invoke(message)
            return output.content
        except Exception as e:
            # if attempt < max_attempts - 1:
            #     logger.log(f"Failed to invoke the chain {attempt + 1} times.\n{type(e)}\n{e}", "warning")
            #     sleep_time = (backoff_base ** attempt) + random.uniform(0, jitter_max)
            #     time.sleep(sleep_time)
            # else:
            logger.log(
                f"Failed to invoke the chain {attempt + 1} times.\n{type(e)} <{e}>\n", "error"
            )
            raise e
