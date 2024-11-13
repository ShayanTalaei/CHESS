import queue
from concurrent.futures import ThreadPoolExecutor
import logging

def _threaded(func):
    """
    A function that adds threading capabilities to a function.
    The returned function will take two additional arguments: thread_id and result_queue.
    It will run the function and put the result in the result_queue as a tuple (thread_id, result).

    Args:
        func (Callable): The function to be wrapped.

    Returns:
        Callable: The wrapped function.
    """
    def wrapper(*args, thread_id, result_queue, **kwargs):
        try:
            result = func(*args, **kwargs)
            result_queue.put((thread_id, result))
        except Exception as e:
            logging.error(f"Exception in thread with kwargs: {kwargs}\n{e}")
            result_queue.put((thread_id, None))
    return wrapper

def ordered_concurrent_function_calls(call_list: list) -> list:
    """
    Executes multiple functions concurrently using a thread pool, and returns the results in the order of the input list.

    Args:
        call_list (list): A list of dictionaries, each containing:
            'function' (Callable): The function to be called.
            'kwargs' (dict): The keyword arguments to pass to the function.

    Returns:
        list: A list of results from the functions.
    """
    result_queue = queue.Queue()
    with ThreadPoolExecutor(max_workers=len(call_list)) as executor:
        for idx, call in enumerate(call_list):
            func = _threaded(call['function'])
            kwargs = call['kwargs']
            executor.submit(func, thread_id=idx, result_queue=result_queue, **kwargs)

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    # Sort results based on their thread IDs
    results = sorted(results, key=lambda x: x[0])
    sorted_results = [result[1] for result in results]

    return sorted_results
