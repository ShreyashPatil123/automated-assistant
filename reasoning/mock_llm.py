import logging

class MockLLMClient:
    """A safe fallback LLM client that returns neutral deterministic outputs 
    to prevent application crashes when the real LLM cannot be initialized."""
    
    def __init__(self):
        logging.info("MockLLMClient initialized as a fallback.")
        
    def generate_json(self, prompt: str, schema: dict = None, max_tokens: int = 1024, temperature: float = 0.1) -> dict:
        """Returns safe stub JSON based on the context of the prompt."""
        
        # Simple heuristic to determine what kind of JSON the system wants
        prompt_lower = prompt.lower()
        if "action" in prompt_lower and "screen_state" in prompt_lower:
            return {
                "action_type": "wait",
                "parameters": {"duration_ms": 2000},
                "reasoning": "MockLLMClient fallback decision.",
                "llm_fallback": True
            }
        elif "plan" in prompt_lower or "step_id" in prompt_lower:
            return {
                "task_id": "mock_task_id",
                "steps": [
                    {
                        "step_id": "mock_step_1",
                        "description": "Mock execution step due to missing LLM.",
                        "action_hint": "wait"
                    }
                ],
                "llm_fallback": True
            }
        else:
            return {
                "task_id": "mock_task",
                "parsed_goal": "Fallback task goal due to Mock LLM.",
                "llm_fallback": True
            }
            
    def generate_text(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        return "Mock text generation response."
