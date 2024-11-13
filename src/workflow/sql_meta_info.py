from pydantic import BaseModel, PrivateAttr
from typing import List, Any, Dict

from runner.database_manager import DatabaseManager
from database_utils.execution import ExecutionStatus
from func_timeout import func_timeout, FunctionTimedOut

LAZY_RESULT_TOKEN = "$$$LAZY$$$"

class SQLMetaInfo(BaseModel):
    SQL: str
    plan: str = ""
    chain_of_thought_reasoning: str = ""
    error: str = ""
    need_fixing: bool = False
    evaluations: List[Dict[str, Any]] = []
    feedbacks: List[str] = []
    needs_refinement: bool = False
    refinement_steps: List[str] = []
    
    _execution_result: List[Any] = PrivateAttr(default=[])
    _execution_status: ExecutionStatus = PrivateAttr(default=None)

    
    @property
    def execution_result(self) -> List[Any]:
        if self._execution_result == []:
            try:    
                result = DatabaseManager().execute_sql(self.SQL, "all")
            except FunctionTimedOut:
                print("Timeout in execution_result")
                result = []
            self._execution_result = result
            return result
        elif self._execution_result == LAZY_RESULT_TOKEN:
            return self._retrieve_lazy_result()
        else:
            return self._execution_result
        
    @property
    def execution_status(self) -> ExecutionStatus:
        if self._execution_status is None:
            try:
                result = self._execution_result
            except Exception:
                return ExecutionStatus.SYNTACTICALLY_INCORRECT
            self._execution_status = DatabaseManager().get_execution_status(self.SQL,result)
        return self._execution_status

    @execution_result.setter
    def execution_result(self, result: List[Any]):
        # Customize the setter to store "lazy" if the result is too long
        if self._is_too_long(result):
            self._execution_result = LAZY_RESULT_TOKEN
        else:
            self._execution_result = result

    def _is_too_long(self, result: List[Any]) -> bool:
        #TODO: customize this method's logic
        return len(result) > 50000

    def _retrieve_lazy_result(self) -> List[Any]:
        try:    
            result = DatabaseManager().execute_sql(self.SQL, "all")
        except FunctionTimedOut:
            print("Timeout in execution_result")
            result = []
        return result

