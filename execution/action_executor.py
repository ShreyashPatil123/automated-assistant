import time
import logging
import subprocess
from execution.keyboard_controller import KeyboardController
from execution.mouse_controller import MouseController
from execution.failsafe_monitor import failsafe

class ActionExecutor:
    def __init__(self, config: dict):
        self.exec_config = config.get("execution", {})
        self.keyboard = KeyboardController(
            min_delay_ms=self.exec_config.get("min_type_delay_ms", 30),
            max_delay_ms=self.exec_config.get("max_type_delay_ms", 80)
        )
        self.mouse = MouseController(
            cursor_speed_multiplier=self.exec_config.get("cursor_speed_multiplier", 1.0),
            drag_hold_delay_ms=self.exec_config.get("drag_hold_delay_ms", 150),
            drag_release_delay_ms=self.exec_config.get("drag_release_delay_ms", 100)
        )

    def execute(self, action_cmd: dict):
        """Dispatches the action command JSON to the correct controller."""
        failsafe.check()
        if not isinstance(action_cmd, dict):
            raise ValueError(f"Action command must be a dict, got {type(action_cmd)}")
            
        action_type = action_cmd.get("action_type")
        if not action_type or not isinstance(action_type, str):
            raise ValueError("action_cmd missing required 'action_type' field (string)")
        coords = action_cmd.get("coordinates")
        params = action_cmd.get("parameters", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError("action_cmd 'parameters' must be a dict")
        
        # Resolve target bounding box center if coords aren't explicitly provided
        if not coords and "target" in params and isinstance(params["target"], dict) and "bbox" in params["target"]:
            bbox = params["target"]["bbox"]
            if len(bbox) == 4:
                coords = {
                    "x": bbox[0] + (bbox[2] - bbox[0]) / 2,
                    "y": bbox[1] + (bbox[3] - bbox[1]) / 2
                }
        
        pre_wait = action_cmd.get("pre_action_wait_ms", 0) / 1000.0
        post_wait = action_cmd.get("post_action_wait_ms", 0) / 1000.0
        
        if pre_wait > 0:
            time.sleep(pre_wait)
            
        x = coords.get("x") if coords else None
        y = coords.get("y") if coords else None

        pointer_actions = {"click", "double_click", "right_click", "move", "scroll"}
        if action_type in pointer_actions and (x is None or y is None):
            raise ValueError(f"Action '{action_type}' requires coordinates x/y (or a target bbox in parameters)")

        try:
            if self.exec_config.get("dry_run", False):
                logging.info(f"DRY RUN: Executing {action_type} with coords {coords} and params {params}")
                if post_wait > 0:
                    time.sleep(post_wait)
                return

            if action_type == "click":
                self.mouse.click(x, y, button=params.get("button", "left"))
                
            elif action_type == "double_click":
                self.mouse.double_click(x, y, button=params.get("button", "left"))
                
            elif action_type == "right_click":
                self.mouse.click(x, y, button="right")
                
            elif action_type == "move":
                if x is not None and y is not None:
                     self.mouse.move(x, y)
                     
            elif action_type == "drag":
                start = params.get("start_coords")
                end = params.get("end_coords")
                if start and end:
                    self.mouse.drag(start["x"], start["y"], end["x"], end["y"])
                    
            elif action_type == "scroll":
                dy = params.get("amount", -1)
                self.mouse.scroll(x, y, dy)
                
            elif action_type == "type_text":
                # If coords provided, click first to focus
                if x is not None and y is not None:
                    self.mouse.click(x, y)
                text = params.get("text", "")
                if params.get("clear_first", False):
                    # Basic clear: ctrl+a, backspace
                    self.keyboard.hotkey("ctrl", "a")
                    self.keyboard.press_key("backspace")
                self.keyboard.type_text(text)
                
            elif action_type == "press_key":
                self.keyboard.press_key(params.get("key", "enter"))
                
            elif action_type == "hotkey":
                keys = params.get("keys", [])
                if keys:
                    hotkey_str = "+".join(keys).lower()
                    allowed_hotkeys = self.exec_config.get("allowed_hotkeys", [])
                    unsafe_mode = self.exec_config.get("unsafe_mode", False)
                    
                    if not unsafe_mode and allowed_hotkeys and hotkey_str not in allowed_hotkeys:
                         raise PermissionError(f"Hotkey '{hotkey_str}' blocked by safety policy.")
                         
                    self.keyboard.hotkey(*keys)
                    
            elif action_type == "wait":
                wait_time = params.get("duration_ms", 1000) / 1000.0
                time.sleep(wait_time)
                
            elif action_type == "run_command":
                command = params.get("command", "")
                if not isinstance(command, str) or not command.strip():
                    raise ValueError("run_command requires non-empty parameters.command")
                unsafe_mode = self.exec_config.get("unsafe_mode", False)
                allowed_commands = self.exec_config.get("allowed_commands", [])
                
                is_allowed = any(command.startswith(cmd) for cmd in allowed_commands)
                
                if not unsafe_mode and not is_allowed:
                    raise PermissionError(f"Command '{command}' blocked by safety policy.")
                    
                logging.info(f"Running command: {command}")
                subprocess.Popen(command, shell=True)
                
            elif action_type == "screenshot":
                # Handled specifically by the orchestration layer to force a re-captcha
                # No hardware control needed here.
                logging.info("Explicit screenshot requested by LLM. Passing to orchestrator.")
                
            else:
                logging.warning(f"Unknown action type: {action_type}, treating as No-Op.")
                
        except Exception as e:
            logging.error(f"Action execution failed: {e}")
            raise
            
        if post_wait > 0:
            time.sleep(post_wait)
