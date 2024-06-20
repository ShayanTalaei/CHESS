import inspect
from threading import Lock
from typing import Any, Dict, Tuple

from llm.models import get_llm_chain
from llm.prompts import get_prompt
from llm.parsers import get_parser

class PipelineManager:
    _instance = None
    _lock = Lock()

    def __new__(cls, pipeline_setup: Dict[str, Any] = None):
        """
        Ensures a singleton instance of PipelineManager.

        Args:
            pipeline_setup (Dict[str, Any], optional): The setup dictionary for the pipeline. Required for initialization.

        Returns:
            PipelineManager: The singleton instance of the class.

        Raises:
            ValueError: If the pipeline_setup is not provided during the first initialization.
        """
        if pipeline_setup is not None:
            with cls._lock:
                cls._instance = super(PipelineManager, cls).__new__(cls)
                cls._instance.pipeline_setup = pipeline_setup
                cls._instance._init(pipeline_setup)
        elif cls._instance is None:
            raise ValueError("pipeline_setup dictionary must be provided for initialization")
        return cls._instance

    def _init(self, pipeline_setup: Dict[str, Any]):
        """
        Custom initialization logic using the pipeline_setup dictionary.

        Args:
            pipeline_setup (Dict[str, Any]): The setup dictionary for the pipeline.
        """
        self.keyword_extraction = pipeline_setup.get("keyword_extraction", {})
        self.entity_retrieval = pipeline_setup.get("entity_retrieval", {})
        self.context_retrieval = pipeline_setup.get("context_retrieval", {})
        self.column_filtering = pipeline_setup.get("column_filtering", {})
        self.table_selection = pipeline_setup.get("table_selection", {})
        self.column_selection = pipeline_setup.get("column_selection", {})
        self.candidate_generation = pipeline_setup.get("candidate_generation", {})
        self.revision = pipeline_setup.get("revision", {})
    
    def get_prompt_engine_parser(self, **kwargs: Any) -> Tuple[Any, Any, Any]:
        """
        Retrieves the prompt, engine, and parser for the current node based on the pipeline setup.

        Args:
            **kwargs: Additional keyword arguments for the prompt.

        Returns:
            Tuple[Any, Any, Any]: The prompt, engine, and parser instances.

        Raises:
            ValueError: If the engine is not specified for the node.
        """
        frame = inspect.currentframe()
        caller_frame = frame.f_back
        node_name = caller_frame.f_code.co_name
        
        node_setup = self.pipeline_setup.get(node_name, {})
        try:
            engine_name = node_setup["engine"]
        except KeyError:
            raise ValueError(f"Engine not specified for node {node_name}")
        
        template_name = self.get_template_name(node_name)
        prompt = get_prompt(template_name, **kwargs)
        
        temperature = node_setup.get("temperature", 0)
        base_uri = node_setup.get("base_uri", None)
        engine = get_llm_chain(engine=engine_name, temperature=temperature, base_uri=base_uri)
        
        parser_name = self.get_parser_name(node_name)
        parser = get_parser(parser_name)
        
        return prompt, engine, parser
    
    def get_template_name(self, node_name: str) -> str:
        """
        Determines the appropriate template name for the given node.

        Args:
            node_name (str): The name of the node.

        Returns:
            str: The template name.
        """
        if node_name == "column_filtering":
            engine_name = self.column_filtering.get("engine", None)
            if engine_name and "llama" in engine_name.lower():
                return "column_filtering_with_examples_llama"
            else:
                return "column_filtering_with_examples"
        elif node_name == "candidate_generation":
            engine_name = self.candidate_generation.get("engine", None)
            if engine_name == "finetuned_nl2sql":
                return "finetuned_candidate_generation"
        return node_name
    
    def get_parser_name(self, node_name: str) -> str:
        """
        Determines the appropriate parser name for the given node.

        Args:
            node_name (str): The name of the node.

        Returns:
            str: The parser name.
        """
        if node_name == "candidate_generation":
            engine_name = self.candidate_generation.get("engine", None)
            if engine_name == "finetuned_nl2sql":
                return "finetuned_candidate_generation"
        return node_name
