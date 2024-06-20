import logging
from typing import Dict, Any

from runner.logger import Logger
from runner.database_manager import DatabaseManager
from pipeline.utils import node_decorator, get_last_node_result

@node_decorator(check_schema_status=False)
def evaluation(task: Any, tentative_schema: Dict[str, Any], execution_history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates the predicted SQL queries against the ground truth SQL query.

    Args:
        task (Any): The task object containing the question and evidence.
        tentative_schema (Dict[str, Any]): The current tentative schema.
        execution_history (Dict[str, Any]): The history of executions.

    Returns:
        Dict[str, Any]: A dictionary containing the evaluation results.
    """
    logging.info("Starting evaluation")

    ground_truth_sql = task.SQL
    to_evaluate = {
        "candidate_generation": get_last_node_result(execution_history, "candidate_generation"), 
        "revision": get_last_node_result(execution_history, "revision")
    }
    result = {}

    for evaluation_for, node_result in to_evaluate.items():
        predicted_sql = "--"
        evaluation_result = {}

        try:
            if node_result["status"] == "success":
                predicted_sql = node_result["SQL"]
                response = DatabaseManager().compare_sqls(
                    predicted_sql=predicted_sql,
                    ground_truth_sql=ground_truth_sql,
                )

                evaluation_result.update({
                    "exec_res": response["exec_res"],
                    "exec_err": response["exec_err"],
                })
            else:
                evaluation_result.update({
                    "exec_res": "generation error",
                    "exec_err": node_result["error"],
                })
        except Exception as e:
            Logger().log(
                f"Node 'evaluate_sql': {task.db_id}_{task.question_id}\n{type(e)}: {e}\n",
                "error",
            )
            evaluation_result.update({
                "exec_res": "error",
                "exec_err": str(e),
            })

        evaluation_result.update({
            "Question": task.question,
            "Evidence": task.evidence,
            "GOLD_SQL": ground_truth_sql,
            "PREDICTED_SQL": predicted_sql
        })
        result[evaluation_for] = evaluation_result

    logging.info("Evaluation completed successfully")
    return result
