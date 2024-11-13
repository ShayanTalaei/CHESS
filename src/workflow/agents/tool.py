from abc import ABC, abstractmethod
from typing import Dict, Any
import re
import time

from runner.logger import Logger
from workflow.system_state import SystemState

class Tool(ABC):
    
    def __init__(self):
        self.tool_name = camel_to_snake(self.__class__.__name__)
    
    def __call__(self, state: SystemState) -> SystemState:
        Logger().log(f"---START: {self.tool_name}---")
        start_time = time.time()
        state.executing_tool = self.tool_name
        try:
            self._run(state)
            run_status = {
                "status": "success",
            }
        except Exception as e:
            Logger().log(f"Tool '{self.tool_name}'\n{type(e)}: {e}\n", "error", state.task)
            state.errors[self.tool_name] = f"{type(e)}: <{e}>"
            run_status = {
                "status": "error",
                "error": f"{type(e)}: <{e}>",
            }
        run_status["execution_time"] = round(time.time() - start_time, 1)
        
        self._log_run(state, run_status)
        Logger().log(f"---END: {self.tool_name} in {run_status['execution_time']}---")
        return state
    
    @abstractmethod
    def _run(self, state: SystemState) -> Dict:
        pass
    
    def _log_run(self, state: SystemState, run_status: Dict[str, Any]):
        run_log = {"tool_name": self.tool_name}
        if run_status["status"] == "success":
            run_log.update(self._get_updates(state))
        run_log.update(run_status)
        state.execution_history.append(run_log)
        Logger().dump_history_to_file(state.execution_history)
    
    @abstractmethod
    def _get_updates(self, state: SystemState) -> Dict:
        pass
    
def camel_to_snake(name):
    # Insert an underscore before each uppercase letter (excluding the first letter)
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    # Insert an underscore before each uppercase letter followed by lowercase or digits
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    # Convert the entire string to lowercase
    return s2.lower()