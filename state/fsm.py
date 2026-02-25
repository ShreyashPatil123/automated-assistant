from enum import Enum, auto

class FSMState(Enum):
    IDLE = auto()
    PARSING = auto()
    PLANNING = auto()
    EXECUTING = auto()
    VALIDATING = auto()
    RETRYING = auto()
    STEP_FAILED = auto()
    REPLANNING = auto()
    TASK_COMPLETE = auto()
    ABORTED = auto()
    TIMEOUT = auto()
    FAILED = auto()

class StateTracker:
    """Holds the current active state variables of the agent."""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.fsm_state = FSMState.IDLE
        self.session_id = None
        self.task_id = None
        self.current_step_id = None
        self.step_retry_count = 0
        self.replan_count = 0
        self.last_screen_hash = None
        self.repeated_state_count = 0
        self.error_log = []
        self.task_start_time = None
        self.llm_call_count = 0
        self.plan = None          # Current active plan JSON
        self.current_step_idx = -1 # Index in plan steps array

    def transition_to(self, new_state: FSMState):
        """Basic state transition."""
        self.fsm_state = new_state
        
    def get_current_step(self) -> dict:
        if not self.plan or "steps" not in self.plan:
            return None
        if self.current_step_idx < 0 or self.current_step_idx >= len(self.plan["steps"]):
            return None
        return self.plan["steps"][self.current_step_idx]
        
    def advance_step(self) -> bool:
        """Move to the next step. Returns False if task is complete."""
        self.step_retry_count = 0
        self.repeated_state_count = 0
        self.current_step_idx += 1
        
        if self.current_step_idx >= len(self.plan.get("steps", [])):
            self.transition_to(FSMState.TASK_COMPLETE)
            return False
            
        self.current_step_id = self.plan["steps"][self.current_step_idx].get("step_id")
        self.transition_to(FSMState.EXECUTING)
        return True
