import argparse
import yaml
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

from runner.run_manager import RunManager

# Temporarily add the path for wisdom imports
wisdom_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
sys.path.insert(0, wisdom_path)
from wisdom.core.logging_util import setup_default_logger

sys.path.pop(0)


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the pipeline with the specified configuration."
    )
    parser.add_argument(
        "--data_mode", type=str, required=True, help="Mode of the data to be processed."
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers to use.")
    parser.add_argument("--log_level", type=str, default="warning", help="Logging level.")
    parser.add_argument(
        "--pick_final_sql",
        type=bool,
        default=False,
        help="Pick the final SQL from the generated SQLs.",
    )
    parser.add_argument(
        "--wisdom_pipeline",
        action="store_true",
        help="Run the wisdom pipeline instead of CHESS.",
    )
    args = parser.parse_args()

    args.run_start_time = datetime.now().isoformat()
    with open(args.config, "r") as file:
        args.config = yaml.safe_load(file)

    return args


def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    """
    Loads the dataset from the specified path.

    Args:
        data_path (str): Path to the data file.

    Returns:
        List[Dict[str, Any]]: The loaded dataset.
    """
    with open(data_path, "r") as file:
        dataset = json.load(file)
    return dataset


def main():
    """
    Main function to run the pipeline with the specified configuration.
    """
    args = parse_arguments()
    setup_default_logger()
    dataset = load_dataset(args.data_path)

    run_manager = RunManager(args)
    run_manager.initialize_tasks(dataset)
    run_manager.run_tasks()
    run_manager.generate_sql_files()


if __name__ == "__main__":
    main()
