from typing import Dict
from pydantic import BaseModel

from llm.models import async_llm_chain_call, get_llm_chain
from llm.prompts import get_prompt
from llm.parsers import get_parser
from workflow.system_state import SystemState
from workflow.sql_meta_info import SQLMetaInfo
from workflow.agents.tool import Tool

class GenerateCandidate(Tool):
    """
    Tool for generating candidate SQL queries based on the task's question and evidence.
    """

    class GeneratorConfig(BaseModel):
        template_name: str
        engine_config: Dict
        parser_name: str
        sampling_count: int
        input_file_path: str = None

    def __init__(self,
                generator_configs: list[Dict]):
        super().__init__()
        self.generator_configs = [self.GeneratorConfig(**config) for config in generator_configs]
        self.generators_queries = {}
        self.next_generator_to_use = "ALL"

    def _run(self, state: SystemState):
        """
        Executes the candidate generation process.
        
        Args:
            state (SystemState): The current system state.
        """
        state.SQL_meta_infos[self.tool_name] = []
        for generator_config in self.generator_configs:
            self.generators_queries[generator_config.template_name] = []
        for generator_config in self.generator_configs:
            if self.next_generator_to_use != "ALL" and generator_config.template_name != self.next_generator_to_use:
                continue
            request_list = []
            for i in range(generator_config.sampling_count):
                try:
                    request_kwargs = {
                        "DATABASE_SCHEMA": state.get_schema_string(schema_type="complete"),
                        "QUESTION": state.task.question,
                        "HINT": state.task.evidence,
                    }
                    request_list.append(request_kwargs)
                except Exception as e:
                    print(f"Error in creating request_kwargs for generator {generator_config.template_name}: {e}")
                    continue
            
            try:
                response = async_llm_chain_call(
                    prompt=get_prompt(template_name=generator_config.template_name),
                    engine=get_llm_chain(**generator_config.engine_config),
                    parser=get_parser(generator_config.parser_name),
                    request_list=request_list,
                    step=f"{self.tool_name}_{generator_config.engine_config['engine_name']}",
                )
                response = [res for sublist in response for res in sublist]
            except Exception as e:
                print(f"Error in generating SQL queries for generator {generator_config.template_name}: {e}")
                continue
            for res in response:
                if not res:
                    continue
                try:
                    sql_meta_info = SQLMetaInfo(**res)
                    # state.SQL_meta_infos[self.tool_name].append(sql_meta_info)
                    self.generators_queries[generator_config.template_name].append(sql_meta_info)
                except Exception as e:
                    print(f"Error in creating SQLMetaInfo for generator {generator_config.template_name}: {e}")
                    continue
            request_list = []
        for generator_config in self.generator_configs:
            if len(self.generators_queries[generator_config.template_name]) > 0:
                state.SQL_meta_infos[self.tool_name] += self.generators_queries[generator_config.template_name]

    def _get_updates(self, state: SystemState) -> Dict:
        SQL_meta_infos = state.SQL_meta_infos[self.tool_name]
        candidates = []
        for i in range(len(SQL_meta_infos)):
            SQL_meta_info = SQL_meta_infos[i]
            if SQL_meta_info.plan:
                candidates.append({
                    "chain_of_thought_reasoning": SQL_meta_info.chain_of_thought_reasoning,
                    "SQL": SQL_meta_info.SQL,
                    "plan": SQL_meta_info.plan
                })
            else:
                candidates.append({
                    "chain_of_thought_reasoning": SQL_meta_info.chain_of_thought_reasoning,
                    "SQL": SQL_meta_info.SQL
                })
        return {
            "node_type": self.tool_name,
            "generation_based_candidates": [{"template_name": generator_config.template_name, "candidates": [candidate.SQL for candidate in self.generators_queries[generator_config.template_name]]} for generator_config in self.generator_configs],
            "candidates": candidates
        }