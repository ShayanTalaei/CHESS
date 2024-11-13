from typing import Dict, List

from llm.models import get_llm_chain, async_llm_chain_call
from llm.prompts import get_prompt
from llm.parsers import get_parser
from workflow.system_state import SystemState
from workflow.sql_meta_info import SQLMetaInfo
from workflow.agents.tool import Tool

HARD_CODES_TEST_CASES = [
    "Only the best answer from the set of candidates that most accurately answers the question, given the database schema and hint should pass this test.",
]

class GenerateUnitTest(Tool):
    
    def __init__(self, template_name: str = None, engine_config: str = None, parser_name: str = None, unit_test_count: int = 5, sampling_count: int = 1):
        super().__init__()
        
        self.template_name = template_name
        self.engine_config = engine_config
        self.parser_name = parser_name
        self.unit_test_count = unit_test_count
        self.sampling_count = sampling_count
        self.candidates = []
        
    def _run(self, state: SystemState):
        try:
            key_to_evaluate = list(state.SQL_meta_infos.keys())[-1]
            target_SQL_meta_infos = state.SQL_meta_infos[key_to_evaluate]
        except Exception as e:
            print(f"Error in UnitTestEvaluator: {e}")
            return
        if len(target_SQL_meta_infos) <= 1:
            state.unit_tests["unit_test_generation"] = []
            return
        database_schema = state.get_database_schema_for_queries(
                [sql_meta_info.SQL for sql_meta_info in target_SQL_meta_infos]
            )
        formatted_candidates = ""
        clusters = self.execution_based_clustering(target_SQL_meta_infos)
        self.candidates = target_SQL_meta_infos
        if len(clusters) == 1:
            state.unit_tests["unit_test_generation"] = []
            return
        index = 0
        for key, candidate_queries in clusters.items():
            formatted_candidates += f"Cluster #{index+1}: \n"
            for candidate_query in candidate_queries:
                formatted_candidates += f"Query: {candidate_query.SQL}\n"
                formatted_candidates += "########\n"
            formatted_candidates += f"Execution result: {self._format_sql_query_result(candidate_queries[-1])}\n"
            formatted_candidates += "=====================\n"
            index += 1
            
        request_kwargs = {
            "HINT": state.task.evidence,
            "QUESTION": state.task.question,
            "DATABASE_SCHEMA": database_schema,
            "CANDIDATE_QUERIES": formatted_candidates,
            "UNIT_TEST_CAP": self.unit_test_count
        }
        
        responses = async_llm_chain_call(
            prompt=get_prompt(template_name=self.template_name),
            engine=get_llm_chain(**self.engine_config),
            parser=get_parser(self.parser_name),
            request_list=[request_kwargs],
            step=self.tool_name,
            sampling_count=self.sampling_count
        )[0]

        state.unit_tests["unit_test_generation"] = []
        for response in responses:
            state.unit_tests["unit_test_generation"].extend(response['unit_tests'])
        state.unit_tests["unit_test_generation"].extend(HARD_CODES_TEST_CASES)

    def execution_based_clustering(self, candidate_queries: List[SQLMetaInfo]) -> list:
        """
        Clusters the generated candidates based on the execution results.
        
        Args:
            state (SystemState): The current system state.
        """
        clusters = {}
        exceptions = []
        for query in candidate_queries:
            try:
                result = str(query.execution_result) if isinstance(query.execution_result, str) else repr(query.execution_result)
            except Exception as e:
                exceptions.append(str(e))
                continue
            if result not in clusters:
                clusters[result] = []
            clusters[result].append(query)
        # sample one query from each cluster
        if not clusters:
            clusters["\n".join(exceptions)] = candidate_queries
        return clusters
    
    def _format_sql_query_result(self, sql_meta_info: SQLMetaInfo) -> str:
        """
        Formats the SQL query to pass to the picker model.
        
        Args:
            sql_meta_info (SQLMetaInfo): The SQL meta information.
        """
        execution_result = sql_meta_info.execution_result
        if execution_result is None:
            return "No results"
        if not isinstance(execution_result, list):
            execution_result = list(execution_result)
        number_of_rows = len(execution_result)
        if number_of_rows == 0:
            number_of_columns = 0
        else:
            number_of_columns = len(execution_result[0])
        if number_of_rows > 20:
            execution_result = execution_result[:20]
        formatted_result = (
            f"Rows: {number_of_rows}, Columns: {number_of_columns}, Results:"
            f" {execution_result}"
        )
        return formatted_result

    def _get_updates(self, state: SystemState) -> Dict:
        return {
            "unit_tests": state.unit_tests,
            "candidates": [sql_meta_info.SQL for sql_meta_info in self.candidates]
            }