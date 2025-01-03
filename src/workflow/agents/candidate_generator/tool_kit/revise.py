from typing import Dict

from llm.models import async_llm_chain_call, get_llm_chain
from llm.prompts import get_prompt
from llm.parsers import get_parser
from database_utils.execution import ExecutionStatus
from workflow.system_state import SystemState
from workflow.sql_meta_info import SQLMetaInfo
from workflow.agents.tool import Tool


class Revise(Tool):
    """
    Tool for correcting a SQL query that returns empty set or has a syntax error.
    """

    def __init__(
        self, template_name: str = None, engine_config: str = None, parser_name: str = None
    ):
        super().__init__()
        self.template_name = template_name
        self.engine_config = engine_config
        self.parser_name = parser_name

    def _run(self, state: SystemState):
        """
        Executes the SQL revision process.

        Args:
            state (SystemState): The current system state.
        """
        try:
            key_to_refine = list(state.SQL_meta_infos.keys())[-1]
            target_SQL_meta_infos = state.SQL_meta_infos[key_to_refine]
        except Exception as e:
            print(f"Error in Checker: {e}")
            return
        if key_to_refine.startswith(self.tool_name):
            id = int(key_to_refine[len(self.tool_name) + 1 :])
            SQL_id = self.tool_name + "_" + str(id + 1)
        else:
            SQL_id = self.tool_name + "_1"
        state.SQL_meta_infos[SQL_id] = []
        request_list = []
        for SQL_meta_info in target_SQL_meta_infos:
            try:
                execution_status = SQL_meta_info.execution_status
                if execution_status != ExecutionStatus.SYNTACTICALLY_CORRECT:
                    SQL_meta_info.need_fixing = True
            except Exception:
                SQL_meta_info.need_fixing = True
        need_fixing_SQL_meta_infos = [
            (index, target_SQL_meta_info)
            for index, target_SQL_meta_info in enumerate(target_SQL_meta_infos)
            if target_SQL_meta_info.need_fixing
        ]
        for index, target_SQL_meta_info in need_fixing_SQL_meta_infos:
            try:
                request_kwargs = {
                    "DATABASE_SCHEMA": state.get_schema_string(schema_type="complete"),
                    "QUESTION": state.task.question,
                    "HINT": state.task.evidence,
                    "QUERY": target_SQL_meta_info.SQL,
                    "RESULT": self.get_formatted_execution_result(target_SQL_meta_info),
                }
                request_list.append(request_kwargs)
            except Exception as e:
                print(f"Error in Checker while creating request list: {e}")
                continue

        try:
            response = async_llm_chain_call(
                prompt=get_prompt(template_name=self.template_name),
                engine=get_llm_chain(**self.engine_config),
                parser=get_parser(self.parser_name),
                request_list=request_list,
                step=self.tool_name,
            )
            response = [r[0] for r in response]
        except Exception as e:
            print(f"Error in Checker while getting response: {e}")
            response = []
        index = 0
        for target_SQL_meta_info in target_SQL_meta_infos:
            try:
                if target_SQL_meta_info.need_fixing:
                    refinement_response = response[index]
                    index += 1
                    if "SELECT" not in refinement_response["refined_sql_query"]:
                        refinement_response = {"refined_sql_query": target_SQL_meta_info.SQL}
                else:
                    refinement_response = {"refined_sql_query": target_SQL_meta_info.SQL}
            except Exception as e:
                print(f"Error in Checker while updating SQL meta info: {e}")
                refinement_response = {"refined_sql_query": target_SQL_meta_info.SQL}
            if "refined_sql_query" in refinement_response:
                if refinement_response["refined_sql_query"]:
                    state.SQL_meta_infos[SQL_id].append(
                        SQLMetaInfo(**{"SQL": refinement_response["refined_sql_query"]})
                    )

    def get_formatted_execution_result(self, target_SQL_meta_info: SQLMetaInfo) -> str:
        try:
            execution_result = target_SQL_meta_info.execution_result
            return {"execution_result": execution_result}
        except Exception as e:
            return {"execution_result": str(e)}

    def need_to_fix(self, state: SystemState) -> bool:
        key_to_check = list(state.SQL_meta_infos.keys())[-1]
        SQL_meta_infos = state.SQL_meta_infos[key_to_check]
        needs_fixing = False
        for SQL_meta_info in SQL_meta_infos:
            try:
                execution_status = SQL_meta_info.execution_status
                if execution_status != ExecutionStatus.SYNTACTICALLY_CORRECT:
                    SQL_meta_info.need_fixing = True
                    needs_fixing = True
            except Exception:
                SQL_meta_info.need_fixing = True
                needs_fixing = True

        if self.fixing == self.max_fixing:
            return False
        self.fixing += 1

        return needs_fixing

    def _get_updates(self, state: SystemState) -> Dict:
        original_SQL_id = list(state.SQL_meta_infos.keys())[-2]
        refined_SQL_id = list(state.SQL_meta_infos.keys())[-1]
        target_SQL_meta_infos = state.SQL_meta_infos[refined_SQL_id]
        candidates = []
        for target_SQL_meta_info in target_SQL_meta_infos:
            candidates.append({"refined_query": target_SQL_meta_info.SQL})
        return {
            "original_SQL_id": original_SQL_id,
            "refined_SQL_id": refined_SQL_id,
            "candidates": candidates,
        }
