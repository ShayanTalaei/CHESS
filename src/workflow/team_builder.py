import logging
from typing import Dict, Any

from langgraph.graph import END, StateGraph
from workflow.system_state import SystemState

from workflow.agents.information_retriever.information_retriever import InformationRetriever
from workflow.agents.schema_selector.schema_selector import SchemaSelector
from workflow.agents.candidate_generator.candidate_generator import CandidateGenerator
from workflow.agents.unit_tester.unit_tester import UnitTester

from workflow.agents.evaluation import ExecutionAccuracy

AGENT_CLASSES = {
    "information_retriever": InformationRetriever,
    "schema_selector": SchemaSelector,
    "candidate_generator": CandidateGenerator,
    "unit_tester": UnitTester
}

class CHESSTeamBuilder:
    def __init__(self, config: Dict[str, any]) -> None:
        self.team = StateGraph(SystemState)
        self.config = config
        logging.info("Initialized TeamBuilder")

    def build(self):
        agents = {agent_name: agent_config for agent_name, agent_config in self.config["team_agents"].items() 
                  if agent_name in AGENT_CLASSES}
        self._add_agents(agents)
        self.team.add_node("evaluation", ExecutionAccuracy())
        agents_with_evaluation = list(agents.keys()) + ["evaluation"]
        self.team.set_entry_point(agents_with_evaluation[0])
        connections = [(agents_with_evaluation[i], agents_with_evaluation[i+1]) 
                       for i in range(len(agents_with_evaluation)-1)]
        connections += [(agents_with_evaluation[-1], END)]
        self._add_connections(connections)

    def _add_agents(self, agents: Dict[str, Dict[str, Any]]) -> None:
        """
        Adds agents to the team.

        Args:
            agents (list): A list of agent names.
        """
        for agent_name, agent_config in agents.items():
            agent = AGENT_CLASSES[agent_name](config=agent_config)
            self.team.add_node(agent_name, agent)
            logging.info(f"Added agent: {agent_name}.")


    def _add_connections(self, connections: list) -> None:
        """
        Adds connections between agents in the team.

        Args:
            connections (list): A list of tuples representing the connections.
        """
        for src, dst in connections:
            self.team.add_edge(src, dst)
            logging.info(f"Added connection from {src} to {dst}")

def build_team(config: Dict[str, any]) -> StateGraph:
    """
    Builds and compiles the pipeline based on the provided tools.

    Args:
        pipeline_tools (str): A string of pipeline tool names separated by '+'.

    Returns:
        StateGraph: The compiled team.
    """

    builder = CHESSTeamBuilder(config)
    builder.build()
    team = builder.team.compile()
    logging.info("Team built and compiled successfully")
    return team
