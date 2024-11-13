import sqlite3
import random
import logging
from typing import Any, Union, List, Dict
from func_timeout import func_timeout, FunctionTimedOut
from multiprocessing import Process, Queue
import threading
from queue import Empty
from enum import Enum

import os

from sqlglot import parse_one, exp

class TimeoutException(Exception):
    pass



def execute_sql(db_path: str, sql: str, fetch: Union[str, int] = "all", timeout: int = 60) -> Any:
    class QueryThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None
            self.exception = None

        def run(self):
            try:
                with sqlite3.connect(db_path, timeout=60) as conn:
                    cursor = conn.cursor()
                    cursor.execute(sql)
                    if fetch == "all":
                        self.result = cursor.fetchall()
                    elif fetch == "one":
                        self.result = cursor.fetchone()
                    elif fetch == "random":
                        samples = cursor.fetchmany(10)
                        self.result = random.choice(samples) if samples else []
                    elif isinstance(fetch, int):
                        self.result = cursor.fetchmany(fetch)
                    else:
                        raise ValueError("Invalid fetch argument. Must be 'all', 'one', 'random', or an integer.")
            except Exception as e:
                self.exception = e
    query_thread = QueryThread()
    query_thread.start()
    query_thread.join(timeout)
    if query_thread.is_alive():
        raise TimeoutError(f"SQL query execution exceeded the timeout of {timeout} seconds.")
    if query_thread.exception:
        # logging.error(f"Error in execute_sql: {query_thread.exception}\nSQL: {sql}")
        raise query_thread.exception
    return query_thread.result


def _clean_sql(sql: str) -> str:
    """
    Cleans the SQL query by removing unwanted characters and whitespace.
    
    Args:
        sql (str): The SQL query string.
        
    Returns:
        str: The cleaned SQL query string.
    """
    return sql.replace('\n', ' ').replace('"', "'").strip("`.")

def create_smaller_db(original_db_path, max_rows=100000):
    if not os.path.exists(original_db_path):
        raise FileNotFoundError("The specified database does not exist.")

    base, ext = os.path.splitext(original_db_path)
    new_db_path = f"{base}_small{ext}"
    conn_orig = sqlite3.connect(original_db_path)
    conn_new = sqlite3.connect(new_db_path)
    cursor_orig = conn_orig.cursor()
    cursor_orig.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor_orig.fetchall()
    cursor_new = conn_new.cursor()
    
    for table in tables:
        if table[0] == "sqlite_sequence":
          continue
        table_name = table[0]
        cursor_orig.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        ddl = cursor_orig.fetchone()[0]
        cursor_new.execute(ddl)
        cursor_orig.execute(f"SELECT * FROM `{table_name}` ORDER BY RANDOM() LIMIT {max_rows}")
        rows = cursor_orig.fetchall()
        cursor_new.executemany(f"INSERT INTO `{table_name}` VALUES ({','.join(['?' for _ in range(len(rows[0]))])})", rows)
        conn_new.commit()

    conn_orig.close()
    conn_new.close()
    return new_db_path

def task(queue, db_path, sql, fetch):
    try:
        result = execute_sql(db_path, sql, fetch)
        queue.put(result)
    except Exception as e:
        queue.put(e)

def subprocess_sql_executor(db_path: str, sql: str, timeout: int = 60):
    queue = Queue()
    process = Process(target=task, args=(queue, db_path, sql, "all"))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        print("Time out in subprocess_sql_executor")
        raise TimeoutError("Execution timed out.")
    else:
        try:
            result = queue.get_nowait()
        except Empty:
            raise Exception("No data returned from the process.")
        
        if isinstance(result, Exception):
            raise result
        return result

# def execute_sql(db_path: str, sql: str, fetch: Union[str, int] = "all") -> Any:
#     """
#     Executes an SQL query on a database and fetches results.
    
#     Args:
#         db_path (str): The path to the database file.
#         sql (str): The SQL query to execute.
#         fetch (Union[str, int]): How to fetch the results. Options are "all", "one", "random", or an integer.
        
#     Returns:
#         Any: The fetched results based on the fetch argument.
    
#     Raises:
#         Exception: If an error occurs during SQL execution.
#     """
#     if fetch == "limited":
#         original_db_path = db_path
#         base, ext = os.path.splitext(db_path)
#         db_path = f"{base}_small{ext}"
#         if not os.path.exists(db_path):
#             print("Creating the smaller db_path")
#             create_smaller_db(original_db_path)
#     try:
#         with sqlite3.connect(db_path, timeout=60) as conn:
#             cursor = conn.cursor()
#             if fetch == "all":
#                 return cursor.execute(sql).fetchall()
#             elif fetch == "one":
#                 return cursor.execute(sql).fetchone()
#             elif fetch == "random":
#                 samples = cursor.execute(sql).fetchmany(10)
#                 return random.choice(samples) if samples else []
#             elif isinstance(fetch, int):
#                 return cursor.execute(sql).fetchmany(fetch)
#             elif fetch == "limited":
#                 return  cursor.execute(sql).fetchall()   
#             else:
#                 raise ValueError("Invalid fetch argument. Must be 'all', 'one', 'random', 'limited', or an integer.")
#     except Exception as e:
#         logging.error(f"Error in execute_sql: {e}\nSQL: {sql}, fetch: {fetch}")
#         raise e

def _compare_sqls_outcomes(db_path: str, predicted_sql: str, ground_truth_sql: str) -> int:
    """
    Compares the outcomes of two SQL queries to check for equivalence.
    
    Args:
        db_path (str): The path to the database file.
        predicted_sql (str): The predicted SQL query.
        ground_truth_sql (str): The ground truth SQL query.
        
    Returns:
        int: 1 if the outcomes are equivalent, 0 otherwise.
    
    Raises:
        Exception: If an error occurs during SQL execution.
    """
    try:
        predicted_res = execute_sql(db_path, predicted_sql)
        ground_truth_res = execute_sql(db_path, ground_truth_sql)
        return int(set(predicted_res) == set(ground_truth_res))
    except Exception as e:
        logging.critical(f"Error comparing SQL outcomes: {e}")
        raise e

def compare_sqls(db_path: str, predicted_sql: str, ground_truth_sql: str, meta_time_out: int = 30) -> Dict[str, Union[int, str]]:
    """
    Compares predicted SQL with ground truth SQL within a timeout.
    
    Args:
        db_path (str): The path to the database file.
        predicted_sql (str): The predicted SQL query.
        ground_truth_sql (str): The ground truth SQL query.
        meta_time_out (int): The timeout for the comparison.
        
    Returns:
        dict: A dictionary with the comparison result and any error message.
    """
    predicted_sql = _clean_sql(predicted_sql)
    try:
        res = func_timeout(meta_time_out, _compare_sqls_outcomes, args=(db_path, predicted_sql, ground_truth_sql))
        error = "incorrect answer" if res == 0 else "--"
    except FunctionTimedOut:
        logging.warning("Comparison timed out.")
        error = "timeout"
        res = 0
    except Exception as e:
        logging.error(f"Error in compare_sqls: {e}")
        error = str(e)
        res = 0
    return {'exec_res': res, 'exec_err': error}

def validate_sql_query(db_path: str, sql: str, max_returned_rows: int = 30) -> Dict[str, Union[str, Any]]:
    """
    Validates an SQL query by executing it and returning the result.
    
    Args:
        db_path (str): The path to the database file.
        sql (str): The SQL query to validate.
        max_returned_rows (int): The maximum number of rows to return.
        
    Returns:
        dict: A dictionary with the SQL query, result, and status.
    """
    try:
        result = execute_sql(db_path, sql, fetch=max_returned_rows)
        return {"SQL": sql, "RESULT": result, "STATUS": "OK"}
    except Exception as e:
        logging.error(f"Error in validate_sql_query: {e}")
        return {"SQL": sql, "RESULT": str(e), "STATUS": "ERROR"}

def aggregate_sqls(db_path: str, sqls: List[str]) -> str:
    """
    Aggregates multiple SQL queries by validating them and clustering based on result sets.
    
    Args:
        db_path (str): The path to the database file.
        sqls (List[str]): A list of SQL queries to aggregate.
        
    Returns:
        str: The shortest SQL query from the largest cluster of equivalent queries.
    """
    results = [validate_sql_query(db_path, sql) for sql in sqls]
    clusters = {}

    # Group queries by unique result sets
    for result in results:
        if result['STATUS'] == 'OK':
            # Using a frozenset as the key to handle unhashable types like lists
            key = frozenset(tuple(row) for row in result['RESULT'])
            if key in clusters:
                clusters[key].append(result['SQL'])
            else:
                clusters[key] = [result['SQL']]
    
    if clusters:
        # Find the largest cluster
        largest_cluster = max(clusters.values(), key=len, default=[])
        # Select the shortest SQL query from the largest cluster
        if largest_cluster:
            return min(largest_cluster, key=len)
    
    logging.warning("No valid SQL clusters found. Returning the first SQL query.")
    return sqls[0]

class ExecutionStatus(Enum):
    SYNTACTICALLY_CORRECT = "SYNTACTICALLY_CORRECT"
    EMPTY_RESULT = "EMPTY_RESULT"
    NONE_RESULT = "NONE_RESULT"
    ZERO_COUNT_RESULT = "ZERO_COUNT_RESULT"
    ALL_NONE_RESULT = "ALL_NONE_RESULT"
    SYNTACTICALLY_INCORRECT = "SYNTACTICALLY_INCORRECT"
    
def get_execution_status(db_path: str, sql: str, execution_result: List = None) -> ExecutionStatus:
    """
    Determines the status of an SQL query execution result.
    
    Args:
        execution_result (List): The result of executing an SQL query.
        
    Returns:
        ExecutionStatus: The status of the execution result.
    """
    if not execution_result:
        try:
            execution_result = execute_sql(db_path, sql, fetch="all")
        except FunctionTimedOut:
            print("Timeout in get_execution_status")
            return ExecutionStatus.SYNTACTICALLY_INCORRECT
        except Exception:
            return ExecutionStatus.SYNTACTICALLY_INCORRECT   
    if (execution_result is None) or (execution_result == []):
        return ExecutionStatus.EMPTY_RESULT
    # elif len(execution_result) == 1:
    #     if execution_result[0] is None or execution_result[0][0] is None:
    #         return ExecutionStatus.NONE_RESULT
    #     elif len(execution_result[0]) == 1 and execution_result[0][0] == 0: # suspicious of a failed agg query
    #         select_expression = list(parse_one(sql, read='sqlite').find_all(exp.Select))[0].expressions[0]
    #         if isinstance(select_expression, exp.Count):
    #             return ExecutionStatus.ZERO_COUNT_RESULT
    # elif all([all([val is None for val in res]) for res in execution_result]):
    #     return ExecutionStatus.ALL_NONE_RESULT
    return ExecutionStatus.SYNTACTICALLY_CORRECT

def run_with_timeout(func, *args, timeouts=[3, 5]):
    def wrapper(stop_event, *args):
        try:
            if not stop_event.is_set():
                result[0] = func(*args)
        except Exception as e:
            result[1] = e

    for attempt, timeout in enumerate(timeouts):
        result = [None, None]
        stop_event = threading.Event()
        thread = threading.Thread(target=wrapper, args=(stop_event, *args))
        thread.start()

        # Wait for the thread to complete or timeout
        thread.join(timeout)

        if thread.is_alive():
            logging.error(f"Function {func.__name__} timed out after {timeout} seconds on attempt {attempt + 1}/{len(timeouts)}")
            stop_event.set()  # Signal the thread to stop
            thread.join()  # Wait for the thread to recognize the stop event
            if attempt == len(timeouts) - 1:
                raise TimeoutException(
                    f"Function {func.__name__} timed out after {timeout} seconds on attempt {attempt + 1}/{len(timeouts)}"
                )
        else:
            if result[1] is not None:
                raise result[1]
            return result[0]

    raise TimeoutException(f"Function {func.__name__} failed to complete after {len(timeouts)} attempts")
