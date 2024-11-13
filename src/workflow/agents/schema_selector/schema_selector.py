from workflow.agents.agent import Agent

from workflow.agents.schema_selector.tool_kit.filter_column import FilterColumn
from workflow.agents.schema_selector.tool_kit.select_tables import SelectTables
from workflow.agents.schema_selector.tool_kit.select_columns import SelectColumns


class SchemaSelector(Agent):
    """
    Agent responsible for selecting appropriate schemas based on the context.
    """
    
    def __init__(self, config: dict):
        """Initialize the tools needed for schema selection"""
        super().__init__(
            name="schema_selector",
            task="narrow down the schema into the most relevant ones through filtering columns, selecting tables and selecting columns",
            config=config,
        )
        
        self.tools = {
            "filter_column": FilterColumn(**config["tools"]["filter_column"]),              
            "select_tables": SelectTables(**config["tools"]["select_tables"]),
            "select_columns": SelectColumns(**config["tools"]["select_columns"])
        }
