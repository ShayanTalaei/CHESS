from workflow.agents.agent import Agent

from workflow.agents.unit_tester.tool_kit.generate_unit_test import GenerateUnitTest
from workflow.agents.unit_tester.tool_kit.evaluate import Evaluate


class UnitTester(Agent):
    """
    Agent responsible for generating and evaluating unit tests.
    """
    
    def __init__(self, config: dict):
        """Initialize the tools needed for unit testing"""
        super().__init__(
            name="unit_tester",
            task="generate unit tests then evaluate them",
            config=config,
        )
        
        self.tools = {
            "generate_unit_test": GenerateUnitTest(**config["tools"]["generate_unit_test"]),
            "evaluate": Evaluate(**config["tools"]["evaluate"])
        }
