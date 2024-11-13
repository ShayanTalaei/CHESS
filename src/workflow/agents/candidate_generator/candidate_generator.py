from workflow.agents.agent import Agent
from workflow.system_state import SystemState

from workflow.agents.candidate_generator.tool_kit.generate_candidate import GenerateCandidate
from workflow.agents.candidate_generator.tool_kit.revise import Revise

class CandidateGenerator(Agent):
    """
    Agent responsible for generating candidate sql queries.
    """
    
    def __init__(self, config: dict):
        super().__init__(
            name="Candidate Generator",
            task=("generate candidate sql queries, and revise the predicted SQL query based on task evidence and schema information",
                  "revise the predicted SQL query based on task evidence and schema information"),
            config=config
        )

        self.tools = {
            "generate_candidate": GenerateCandidate(**config["tools"]["generate_candidate"]),
            "revise": Revise(**config["tools"]["revise"])
        }
