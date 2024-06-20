import os
import json
from pathlib import Path
from multiprocessing import Pool
from typing import List, Dict, Any, Tuple

from runner.logger import Logger
from runner.task import Task
from runner.database_manager import DatabaseManager
from runner.statistics_manager import StatisticsManager
from pipeline.workflow_builder import build_pipeline
from pipeline.pipeline_manager import PipelineManager

NUM_WORKERS = 11

class RunManager:
    RESULT_ROOT_PATH = "results"

    def __init__(self, args: Any):
        self.args = args
        self.result_directory = self.get_result_directory()
        self.statistics_manager = StatisticsManager(self.result_directory)
        self.tasks: List[Task] = []
        self.total_number_of_tasks = 0
        self.processed_tasks = 0

    def get_result_directory(self) -> str:
        """
        Creates and returns the result directory path based on the input arguments.
        
        Returns:
            str: The path to the result directory.
        """
        data_mode = self.args.data_mode
        pipeline_nodes = self.args.pipeline_nodes
        dataset_name = Path(self.args.data_path).stem
        run_folder_name = str(self.args.run_start_time)
        run_folder_path = Path(self.RESULT_ROOT_PATH) / data_mode / pipeline_nodes / dataset_name / run_folder_name
        
        run_folder_path.mkdir(parents=True, exist_ok=True)
        
        arg_file_path = run_folder_path / "-args.json"
        with arg_file_path.open('w') as file:
            json.dump(vars(self.args), file, indent=4)
        
        log_folder_path = run_folder_path / "logs"
        log_folder_path.mkdir(exist_ok=True)
        
        return str(run_folder_path)

    def initialize_tasks(self, dataset: List[Dict[str, Any]]):
        """
        Initializes tasks from the provided dataset.
        
        Args:
            dataset (List[Dict[str, Any]]): The dataset containing task information.
        """
        for i, data in enumerate(dataset):
            if "question_id" not in data:
                data = {"question_id": i, **data}
            task = Task(data)
            self.tasks.append(task)
        self.total_number_of_tasks = len(self.tasks)
        print(f"Total number of tasks: {self.total_number_of_tasks}")

    def run_tasks(self):
        """Runs the tasks using a pool of workers."""
        with Pool(NUM_WORKERS) as pool:
            for task in self.tasks:
                pool.apply_async(self.worker, args=(task,), callback=self.task_done)
            pool.close()
            pool.join()

    def worker(self, task: Task) -> Tuple[Any, str, int]:
        """
        Worker function to process a single task.
        
        Args:
            task (Task): The task to be processed.
        
        Returns:
            tuple: The state of the task processing and task identifiers.
        """
        database_manager = DatabaseManager(db_mode=self.args.data_mode, db_id=task.db_id)
        logger = Logger(db_id=task.db_id, question_id=task.question_id, result_directory=self.result_directory)
        logger._set_log_level(self.args.log_level)
        logger.log(f"Processing task: {task.db_id} {task.question_id}", "info")
        pipeline_manager = PipelineManager(json.loads(self.args.pipeline_setup))
        try:
            tentative_schema, execution_history = self.load_checkpoint(task.db_id, task.question_id)
            initial_state = {"keys": {"task": task, 
                                      "tentative_schema": tentative_schema, "execution_history": execution_history}}
            self.app = build_pipeline(self.args.pipeline_nodes)
            for state in self.app.stream(initial_state):
                continue
            return state['__end__'], task.db_id, task.question_id
        except Exception as e:
            logger.log(f"Error processing task: {task.db_id} {task.question_id}\n{e}", "error")
            return None, task.db_id, task.question_id

    def task_done(self, log: Tuple[Any, str, int]):
        """
        Callback function when a task is done.
        
        Args:
            log (tuple): The log information of the task processing.
        """
        state, db_id, question_id = log
        if state is None:
            return
        evaluation_result = state["keys"]['execution_history'][-1]
        if evaluation_result.get("node_type") == "evaluation":
            for evaluation_for, result in evaluation_result.items():
                if evaluation_for in ['node_type', 'status']:
                    continue
                self.statistics_manager.update_stats(db_id, question_id, evaluation_for, result)
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
        print('\x1b[1A' + '\x1b[2K' + '\x1b[1A')  # Clear previous line
        print(f"[{'=' * progress_length}>{' ' * (bar_length - progress_length)}] {self.processed_tasks}/{self.total_number_of_tasks}")

    def load_checkpoint(self, db_id, question_id) -> Dict[str, List[str]]:
        tentative_schema = DatabaseManager().get_db_schema()
        execution_history = []
        if self.args.use_checkpoint:
            checkpoint_file = Path(self.args.checkpoint_dir) / f"{question_id}_{db_id}.json"
            if checkpoint_file.exists():
                with checkpoint_file.open('r') as file:
                    checkpoint = json.load(file)
                    for step in checkpoint:
                        node_type = step["node_type"]
                        if node_type in self.args.checkpoint_nodes:
                            execution_history.append(step)
                        if "tentative_schema" in step:
                            tentative_schema = step["tentative_schema"]
            else:
                Logger().log(f"Checkpoint file not found: {checkpoint_file}", "warning")
        return tentative_schema, execution_history

    def generate_sql_files(self):
        """Generates SQL files from the execution history."""
        sqls = {}
        
        for file in os.listdir(self.result_directory):
            if file.endswith(".json") and "_" in file:
                _index = file.find("_")
                question_id = int(file[:_index])
                db_id = file[_index + 1:-5]
                with open(os.path.join(self.result_directory, file), 'r') as f:
                    exec_history = json.load(f)
                    for step in exec_history:
                        if "SQL" in step:
                            node_type = step["node_type"]
                            if node_type not in sqls:
                                sqls[node_type] = {}
                            sqls[node_type][question_id] = step["SQL"]
        for key, value in sqls.items():
            with open(os.path.join(self.result_directory, f"-{key}.json"), 'w') as f:
                json.dump(value, f, indent=4)
