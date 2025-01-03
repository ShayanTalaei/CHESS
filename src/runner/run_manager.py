import asyncio
import os
import sys
import json
from pathlib import Path
from multiprocessing import Pool
from typing import List, Dict, Any, Tuple
from langgraph.graph import StateGraph
from sqlglot import parse_one, exp

from runner.logger import Logger
from runner.task import Task
from runner.database_manager import DatabaseManager
from runner.statistics_manager import StatisticsManager
from workflow.agents.evaluation import ExecutionAccuracy
from workflow.sql_meta_info import SQLMetaInfo

# Temporarily add the path for wisdom imports
wisdom_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))


sys.path.insert(0, wisdom_path)
from wisdom.authz.identities import UNSAFE_SUPER_USER_IDENTITY
from wisdom.platforms.k8s_context import K8sWisdomStaticContext
from wisdom.query_parser.chat.activity_queue import DummyNLQueryActivityQueue
from wisdom.query_parser.processors.nl_query_processor import nl_query_request
from wisdom.query_parser.processors.nl_query_processor.nl_query_context import NLQueryContext
from wisdom.query_parser.processors.nl_query_processor.nl_query_processor import NLQueryProcessor
from wisdom.query_parser.processors.nl_query_processor.nl_query_request import NLQueryRequest
from wisdom.query_parser.processors.nl_query_processor.user_preferences import UserPreferences
from wisdom.query_parser.query.query import Query

sys.path.pop(0)


from workflow.team_builder import build_team
from database_utils.execution import ExecutionStatus
from workflow.system_state import SystemState
import fcntl


class RunManager:
    RESULT_ROOT_PATH = "results"

    def __init__(self, args: Any):
        self.args = args
        self.result_directory = self.get_result_directory()
        self.statistics_manager = StatisticsManager(self.result_directory)
        self.tasks: List[Task] = []
        self.total_number_of_tasks = 0
        self.processed_tasks = 0

        self.wisdom_context = K8sWisdomStaticContext()
        self.nlp_engine: NLQueryProcessor = NLQueryProcessor(self.wisdom_context)
        self.zsheet_store = self.wisdom_context.zsheet_store()
        self.db_id_to_domain_id = {}

    def get_domain_id(self, db_id: str) -> str:
        if db_id not in self.db_id_to_domain_id:
            detailed_zsheet = self.zsheet_store.get_by_name(f"bird_{db_id}")
            assert detailed_zsheet is not None
            self.db_id_to_domain_id[db_id] = detailed_zsheet.zsheet.ref.uuid
        return self.db_id_to_domain_id[db_id]

    def get_result_directory(self) -> str:
        """
        Creates and returns the result directory path based on the input arguments.

        Returns:
            str: The path to the result directory.
        """
        data_mode = self.args.data_mode
        setting_name = self.args.config["setting_name"]
        dataset_name = Path(self.args.data_path).stem
        run_folder_name = str(self.args.run_start_time)
        run_folder_path = (
            Path(self.RESULT_ROOT_PATH) / data_mode / setting_name / dataset_name / run_folder_name
        )

        run_folder_path.mkdir(parents=True, exist_ok=True)

        arg_file_path = run_folder_path / "-args.json"
        with arg_file_path.open("w") as file:
            json.dump(vars(self.args), file, indent=4)

        final_prediction_file = run_folder_path / "-predictions.json"
        with final_prediction_file.open("w") as file:
            json.dump({}, file, indent=4)

        log_folder_path = run_folder_path / "logs"
        log_folder_path.mkdir(exist_ok=True)

        return str(run_folder_path)

    def update_final_predictions(self, question_id: int, final_sql: str = None, db_id: int = None):
        results = {}
        if final_sql:
            temp_results = {str(question_id): final_sql.strip() + "\t----- bird -----\t" + db_id}
        else:
            temp_results = {str(question_id): 0}
        file_path = os.path.join(self.result_directory, "-predictions.json")
        with open(file_path, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                results = json.load(f)
                results.update(temp_results)
                f.seek(0)
                json.dump(results, f, indent=4)
                f.truncate()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def initialize_tasks(self, dataset: List[Dict[str, Any]]):
        """
        Initializes tasks from the provided dataset.

        Args:
            dataset (List[Dict[str, Any]]): The dataset containing task information.
        """
        for i, data in enumerate(dataset):
            if "question_id" not in data:
                data = {"question_id": i, **data}
            if data["question_id"] <= 387:
                continue
            self.update_final_predictions(data["question_id"])
            task = Task(**data)
            self.tasks.append(task)
        self.total_number_of_tasks = len(self.tasks)
        print(f"Total number of tasks: {self.total_number_of_tasks}")

    def run_tasks(self):
        """Runs the tasks using a pool of workers."""
        print(f"Running tasks with {self.args.num_workers} workers.")
        if self.args.num_workers > 1:
            with Pool(self.args.num_workers) as pool:
                for task in self.tasks:
                    pool.apply_async(self.worker, args=(task,), callback=self.task_done)
                pool.close()
                pool.join()
        else:
            for task in self.tasks:
                try:
                    if self.args.wisdom_pipeline:
                        log = self.wisdom_worker(task)
                    else:
                        log = self.worker(task)
                    self.task_done(log)
                except Exception as e:
                    print(f"Error processing task {task.db_id} {task.question_id}: {e}")

    def worker(self, task: Task) -> Tuple[Any, str, int]:
        """
        Worker function to process a single task.

        Args:
            task (Task): The task to be processed.

        Returns:
            tuple: The state of the task processing and task identifiers.
        """
        print(f"Initializing task: {task.db_id} {task.question_id}")
        DatabaseManager(db_mode=self.args.data_mode, db_id=task.db_id)
        logger = Logger(
            db_id=task.db_id, question_id=task.question_id, result_directory=self.result_directory
        )
        logger._set_log_level(self.args.log_level)
        logger.log(f"Processing task: {task.db_id} {task.question_id}", "info")

        team = build_team(self.args.config)
        thread_id = f"{self.args.run_start_time}_{task.db_id}_{task.question_id}"
        thread_config = {"configurable": {"thread_id": thread_id}}
        state_values = SystemState(
            task=task, tentative_schema=DatabaseManager().get_db_schema(), execution_history=[]
        )
        thread_config["recursion_limit"] = 50
        for state_dict in team.stream(state_values, thread_config, stream_mode="values"):
            logger.log(
                "________________________________________________________________________________________"
            )
            continue
        system_state = SystemState(**state_dict)
        return system_state, task.db_id, task.question_id

    def wisdom_sql_post_process(self, sql: str) -> str:
        """
        1. given sql, remove db and catalog using sqlglot
        2. convert dialect to sqlite
        """

        def sql_transform(node: exp.Expression) -> exp.Expression:
            """
            remove db and schema using sqlglot
            """
            if not isinstance(node, exp.Table):
                return node

            # Remove catalog and db from the node
            node.set("catalog", None)
            node.set("db", None)

            return node

        parsed_tree = parse_one(sql, dialect="redshift")
        clean_tree = parsed_tree.transform(sql_transform)
        return clean_tree.sql(dialect="sqlite")

    def wisdom_worker(self, task: Task) -> Tuple[Any, str, int]:

        print(f"Initializing task: {task.db_id} {task.question_id}")
        DatabaseManager(db_mode=self.args.data_mode, db_id=task.db_id)
        logger = Logger(
            db_id=task.db_id, question_id=task.question_id, result_directory=self.result_directory
        )
        logger._set_log_level(self.args.log_level)
        logger.log(f"Processing task: {task.db_id} {task.question_id}", "info")

        domain_id = self.get_domain_id(task.db_id)
        nl_query = Query.from_string(task.question + " " + task.evidence)
        identity = UNSAFE_SUPER_USER_IDENTITY

        nlp_engine_response = asyncio.run(
            self.nlp_engine.async_process(
                NLQueryRequest(
                    query=nl_query,
                    identity=identity,
                    # TODO: skip execution
                    preferences=UserPreferences(skip_cache=True),
                    domain_id=domain_id,
                    skip_summary=True,
                ),
                DummyNLQueryActivityQueue(),
                NLQueryContext(nl_query, domain_id, identity),
            )
        )

        wisdom_sql = nlp_engine_response.code.code_str
        post_processed_sql = self.wisdom_sql_post_process(wisdom_sql)

        system_state = SystemState(
            task=task,
            tentative_schema=DatabaseManager().get_db_schema(),
            execution_history=[],
            SQL_meta_infos={"wisdom_pipeline": [SQLMetaInfo(SQL=post_processed_sql)]},
        )

        evaluation_agent = ExecutionAccuracy()
        system_state = evaluation_agent(system_state)

        return system_state, task.db_id, task.question_id

    def pick_final_sql(self, state: SystemState):
        """
        Picks the final SQL query from the execution history.

        Args:
            state (SystemState): The system state after processing the task.

        Returns:

        """
        execution_history = state.execution_history
        final_sql = ""
        final_sql_execution_status = None
        generated_at_step = None
        for step in execution_history:
            if step["tool_name"] == "generate_candidate":
                sql = step["candidates"][0]["SQL"]
                sql_id = "generate_candidate"
            elif "revise" in step["tool_name"]:
                sql = step["SQL"]
                sql_id = step["tool_name"]
            else:
                continue
            execution_status = DatabaseManager().get_execution_status(sql=sql)
            if not final_sql:
                final_sql = sql
                final_sql_execution_status = execution_status
                generated_at_step = sql_id
            else:
                if execution_status == ExecutionStatus.SYNTACTICALLY_CORRECT:
                    final_sql = sql
                    final_sql_execution_status = execution_status
                    generated_at_step = sql_id
                else:
                    if final_sql_execution_status != ExecutionStatus.SYNTACTICALLY_CORRECT:
                        if execution_status == ExecutionStatus.ZERO_COUNT_RESULT:
                            final_sql = sql
                            final_sql_execution_status = execution_status
                            generated_at_step = sql_id
        final_validation_result = {
            "final_sql": {
                "SQL": final_sql,
                "EXECUTION_STATUS": final_sql_execution_status.value,
                "GENERATED_AT_STEP": generated_at_step,
            }
        }
        if execution_history[-1]["tool_name"] == "evaluation":
            final_validation_result = {
                "final_sql": {
                    **execution_history[-1][generated_at_step],
                    "EXECUTION_STATUS": final_sql_execution_status.value,
                    "GENERATED_AT_STEP": generated_at_step,
                }
            }
            final_validation_result["final_sql"]["SQL"] = final_validation_result["final_sql"].pop(
                "PREDICTED_SQL"
            )

        state.execution_history.append(final_validation_result)
        Logger().dump_history_to_file(state.execution_history)

    def task_done(self, log: Tuple[SystemState, str, int]):
        """
        Callback function when a task is done.

        Args:
            log (tuple): The log information of the task processing.
        """
        state, db_id, question_id = log
        if state is None:
            return
        for step in state.execution_history:
            if "tool_name" in step and step["tool_name"] == "evaluation":
                validation_result = step
                if validation_result.get("tool_name") == "evaluation":
                    for validation_for, result in validation_result.items():
                        if not isinstance(result, dict):
                            continue
                        self.statistics_manager.update_stats(
                            db_id, question_id, validation_for, result
                        )
            if "final_SQL" in step:
                self.statistics_manager.update_stats(
                    db_id, question_id, "final_SQL", step["final_SQL"]
                )
                self.update_final_predictions(
                    question_id, step["final_SQL"]["PREDICTED_SQL"], db_id
                )
        self.statistics_manager.dump_statistics_to_file()
        self.processed_tasks += 1
        self.plot_progress()

    def plot_progress(self, bar_length: int = 100):
        """
        Plots the progress of task processing.

        Args:
            bar_length (int, optional): The length of the progress bar. Defaults to 100.
        """
        processed_ratio = self.processed_tasks / self.total_number_of_tasks
        progress_length = int(processed_ratio * bar_length)
        print("\x1b[1A" + "\x1b[2K" + "\x1b[1A")  # Clear previous line
        print(
            f"[{'=' * progress_length}>{' ' * (bar_length - progress_length)}]"
            f" {self.processed_tasks}/{self.total_number_of_tasks}"
        )

    def generate_sql_files(self):
        """Generates SQL files from the execution history."""
        sqls = {}

        for file in os.listdir(self.result_directory):
            if file.endswith(".json") and "_" in file:
                _index = file.find("_")
                question_id = int(file[:_index])
                db_id = file[_index + 1 : -5]
                with open(os.path.join(self.result_directory, file), "r") as f:
                    exec_history = json.load(f)
                    for step in exec_history:
                        if "SQL" in step:
                            tool_name = step["tool_name"]
                            if tool_name not in sqls:
                                sqls[tool_name] = {}
                            sqls[tool_name][question_id] = step["SQL"]
        for key, value in sqls.items():
            with open(os.path.join(self.result_directory, f"-{key}.json"), "w") as f:
                json.dump(value, f, indent=4)
