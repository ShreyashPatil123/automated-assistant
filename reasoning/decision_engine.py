import os
import json
import logging
from reasoning.llm_client import LLMClient
from state.fsm import StateTracker

logger = logging.getLogger("ladas.decision")

class DecisionEngine:
    def __init__(self, llm_client: LLMClient, config: dict):
        self.llm = llm_client
        self.config = config
        
        # Load prompt template
        template_path = os.path.join(
             os.path.dirname(__file__), 'prompt_templates', 'system_action.txt')
        with open(template_path, 'r') as f:
            self.system_prompt = f.read()

    def parse_action(self, raw_output: str) -> dict:
        action = {}
        try:
            # We first try to find {} if there's trailing garbage
            start = raw_output.find('{')
            end = raw_output.rfind('}')
            if start != -1 and end != -1 and start < end:
                clean_json = raw_output[start:end+1]
                # fix trailing commas
                import re
                clean_json = re.sub(r',\s*\}', '}', clean_json)
                clean_json = re.sub(r',\s*\]', ']', clean_json)
                action = json.loads(clean_json)
            else:
                action = json.loads(raw_output)
        except json.JSONDecodeError as e:
            logger.warning("LLM produced invalid JSON (%s). Raw output: %r", e, raw_output)
            raise ValueError(f"Invalid JSON: {e}")

        if not isinstance(action, dict):
            logger.warning("LLM action is not a JSON object. Parsed: %r", action)
            raise ValueError("Invalid action schema: root must be an object")

        # Unwrap if the LLM put everything inside an "action" key
        if "action" in action and isinstance(action["action"], dict):
            action = action["action"]
            
        # Extract action if it was simplified to "action": "click"
        if "action" in action and isinstance(action["action"], str) and "action_type" not in action:
            action["action_type"] = action.pop("action")

        action_type = action.get("action_type") or action.get("type") or action.get("action_name")
        params = action.get("parameters") or action.get("params") or {}

        if not action_type or not isinstance(action_type, str):
            logger.warning("LLM action missing/invalid action_type. Action: %r", action)
            raise ValueError("Invalid action schema: missing action_type")

        if isinstance(params, list):
            params = {"items": params}
        elif not isinstance(params, dict):
            logger.warning("LLM action parameters must be an object/dict. Action: %r", action)
            raise ValueError("Invalid action schema: parameters must be an object")

        return {
            "action_type": action_type.lower().replace(" ", "_"),
            "parameters": params,
            "reasoning": action.get("reasoning", "")
        }

    def get_next_action(self, 
                        intent: dict,
                        current_step: dict, 
                        step_idx: int, 
                        total_steps: int, 
                        screen_state: dict, 
                        context_history: list,
                        state: StateTracker) -> dict:
        """Determines the next action based on current state and step."""
        max_calls = self.config.get("reasoning", {}).get("max_llm_calls_per_task", 20)
        
        fallback_action = {
            "action_type": "wait",
            "parameters": {"duration_ms": 1000},
            "reasoning": "Fallback no-op due to LLM limit or repeated action parsing failures.",
            "llm_fallback": True
        }
        
        prompt = self.system_prompt.replace("{goal}", intent.get("parsed_goal", "Unknown Goal"))\
                                   .replace("{current_step_idx}", str(step_idx + 1))\
                                   .replace("{total_steps}", str(total_steps))\
                                   .replace("{step_description}", current_step.get("description", "Unknown"))\
                                   .replace("{current_url}", screen_state.get("active_window", {}).get("title", "Unknown URL"))\
                                   .replace("{page_title}", screen_state.get("active_window", {}).get("title", "Desktop Screen"))\
                                   .replace("{timestamp}", screen_state.get("timestamp", ""))\
                                   .replace("{detected_elements_json}", json.dumps(screen_state.get("vision_elements", []), indent=2))\
                                   .replace("{page_text}", json.dumps(screen_state.get("ocr_elements", []), indent=2))\
                                   .replace("{context_history}", json.dumps(context_history, indent=2))
                                   
        reasks = 0
        max_reasks = 2
        
        while reasks <= max_reasks:
            if state.llm_call_count >= max_calls:
                logger.warning(f"Max LLM calls ({max_calls}) reached. Using fallback action.")
                return fallback_action
                
            state.llm_call_count += 1
            
            try:
                raw_text = self.llm.generate_text(prompt)
                return self.parse_action(raw_text)
            except Exception as e:
                logger.exception("LLM generation or parsing failed during decision.")
                reasks += 1
                if reasks <= max_reasks:
                    logger.info("Retrying decision generation...")
                    # Optional: append failure reason to prompt to hint LLM
                    prompt += f"\n\nSystem Error on previous attempt: {str(e)}. Please try again and strictly output valid JSON with action_type and parameters."
                    
        return {
            **fallback_action,
            "reasoning": "Fallback no-op due to repeated invalid action JSON from LLM.",
        }
