from workflow.agents.agent import Agent
from workflow.system_state import SystemState

from workflow.agents.information_retriever.tool_kit.extract_keywords import ExtractKeywords
from workflow.agents.information_retriever.tool_kit.retrieve_entity import RetrieveEntity
from workflow.agents.information_retriever.tool_kit.retrieve_context import RetrieveContext


class InformationRetriever(Agent):
    """
    Agent responsible for retrieving relevant entities and context from the question and hint.
    """
    
    def __init__(self, config: dict):
        """Initialize the tools needed for information retrieval"""
        super().__init__(
            name="Information Retriever",
            task=("retrieve the most important entities and context relevant to the keywords of the question, through ",
                         "extracting keywords, retrieving entities, and retrieving context"),
            config=config
        )
        
        self.tools = {
            "extract_keywords": ExtractKeywords(**config["tools"]["extract_keywords"]),
            "retrieve_entity": RetrieveEntity(**config["tools"]["retrieve_entity"]),
            "retrieve_context": RetrieveContext(**config["tools"]["retrieve_context"])
        }