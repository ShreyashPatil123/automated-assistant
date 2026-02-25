import os
import sys
import yaml
import logging
import uuid
import time
from rich.console import Console

logger = logging.getLogger("ladas")

# LADAS Modules
from capture.capture_manager import CaptureManager
from perception.ocr_engine import OCREngine
from perception.vision_detector import VisionDetector
from perception.state_builder import StateBuilder
from planning.task_planner import TaskPlanner
from reasoning.instruction_parser import InstructionParser
from reasoning.decision_engine import DecisionEngine
from reasoning.llm_client import LLMClient
from reasoning.mock_llm import MockLLMClient
from execution.action_executor import ActionExecutor
from execution.failsafe_monitor import failsafe, FailsafeTriggered
from memory.database import Database
from memory.task_store import TaskStore
from memory.action_log import ActionLog
from state.fsm import StateTracker, FSMState
from config_utils import validate_config

# Setup rich console for terminal output
console = Console()

class LADAS:
    def __init__(self, config_path: str = "config.yaml"):
        # 1. Load Config
        candidate_paths = [config_path]
        if not os.path.isabs(config_path):
            candidate_paths.append(os.path.join(os.path.dirname(__file__), config_path))
            candidate_paths.append(os.path.join(os.path.dirname(__file__), "config.yaml"))

        resolved_config_path = None
        for p in candidate_paths:
            if p and os.path.exists(p):
                resolved_config_path = p
                break

        if not resolved_config_path:
            console.print(
                "[bold red]Config file not found. Tried:\n" + "\n".join(f"- {p}" for p in candidate_paths) + "\nExiting.[/bold red]"
            )
            sys.exit(1)
            
        with open(resolved_config_path, "r") as f:
            raw_config = yaml.safe_load(f) or {}
            self.config = validate_config(raw_config)
            
        # 2. Setup Logging
        os.makedirs("logs", exist_ok=True)
        # Ensure handlers are added once
        if not logger.hasHandlers():
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
            
            # File Handler
            fh = logging.FileHandler(f"logs/ladas_system_{int(time.time())}.log")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
            # Console Handler (optional, for non-rich debug/info)
            # ch = logging.StreamHandler()
            # ch.setFormatter(formatter)
            # logger.addHandler(ch)
        
        # 3. Initialize Memory
        # (Using a single DB for testing. In prod, you might scope by session)
        self.db = Database("memory.db")
        self.task_store = TaskStore(self.db)
        self.action_log = ActionLog(self.db)
        
        # 4. Initialize State Tracker
        self.state = StateTracker()
        
        # 5. Initialize Hardware Interfaces
        console.print("[yellow]Initializing Screen Capture & Failsafe...[/yellow]")
        self.capture = CaptureManager(self.config)
        failsafe.start()
        
        # 6. Initialize Models (These take time/memory)
        console.print("[yellow]Initializing Models (LLM, Vision, OCR)...[/yellow]")
        # Mock LLM for now if path is missing to avoid crashing
        llm_path = self.config.get("reasoning", {}).get("model_path", "")
        # For a full implementation, proper paths must be set
        try:
            self.llm = LLMClient(model_path=llm_path) 
        except Exception as e:
            logger.exception("Failed to initialize LLMClient")
            allow_mock = self.config.get("system", {}).get("allow_mock_on_startup_failure", False)
            if allow_mock:
                self.llm = MockLLMClient()
                logger.warning("Using MockLLMClient due to initialization failure.")
                console.print("[yellow]Warning: Using MockLLMClient due to failure.[/yellow]")
            else:
                console.print("[bold red]Failed to initialize LLM. Configuration forbids mock fallback. See logs for details.[/bold red]")
                sys.exit(1)
        
        self.parser = InstructionParser(self.llm, self.config)
        self.planner = TaskPlanner(self.llm, self.config)
        self.decision = DecisionEngine(self.llm, self.config)
        
        self.ocr = OCREngine(self.config)
        self.vision = VisionDetector(self.config)
        
        self.executor = ActionExecutor(self.config)
        
        self.session_id = uuid.uuid4().hex[:8]
        
        self._validate_startup()
        console.print(f"[green]Initialization Complete. Session: {self.session_id}[/green]")

    def _validate_startup(self):
        """Validates critical dependencies before starting."""
        allow_mock = self.config.get("system", {}).get("allow_mock_on_startup_failure", False)
        
        # 1. LLM backend note
        # This repo currently uses Ollama via HTTP (see reasoning/llm_client.py). The config key
        # 'reasoning.model_path' is kept for future local-gguf support, but we don't hard-fail on it.
        llm_path = self.config.get("reasoning", {}).get("model_path")
        if llm_path and not os.path.exists(llm_path):
            logger.error(
                "Config reasoning.model_path points to a missing file: '%s'. "
                "Note: current LLMClient uses Ollama; this path is not used unless you switch to a local GGUF backend.",
                llm_path,
            )
            console.print(
                f"[yellow]Warning: reasoning.model_path not found at '{llm_path}'. "
                "(Current backend uses Ollama; safe to ignore unless you switch backends.)[/yellow]"
            )
            
        # 2. Check YOLO Model
        yolo_path = self.config.get("perception", {}).get("vision", {}).get("yolo_model_path", "yolov8n.pt")
        if not os.path.exists(yolo_path):
            error_msg = (
                f"YOLO model not found at '{yolo_path}'. "
                "Fix: download a YOLOv8 weights file (e.g. yolov8n.pt) and set perception.vision.yolo_model_path in config.yaml. "
                "The system will fall back to template matching if available."
            )
            logger.error(error_msg)
            if allow_mock:
                console.print(f"[yellow]{error_msg} (Continuing without YOLO)[/yellow]")
            else:
                console.print(f"[bold red]{error_msg}[/bold red]")
                sys.exit(1)
            
        # 3. Check Template Library (Warn only)
        template_path = self.config.get("perception", {}).get("vision", {}).get("template_library_path")
        if template_path and not os.path.exists(template_path):
            logger.warning(f"Template library path not found at {template_path}. Vision matching UI elements might be limited.")
            console.print(f"[yellow]Warning: Template library not found at {template_path}[/yellow]")
            
        logger.info("Startup validation passed successfully.")

    def run_terminal_loop(self):
        """The main interactive terminal loop."""
        console.print(r"""
╔══════════════════════════════════════════════════════════════╗
║          LOCAL AI DESKTOP AUTOMATION SYSTEM v1.0             ║
╚══════════════════════════════════════════════════════════════╝
        """, style="bold cyan")
        
        while True:
            try:
                self.state.reset()
                self.state.session_id = self.session_id
                console.print(f"\n[bold green]\[READY][/bold green] Enter command (type 'quit' to exit):")
                user_input = input("> ").strip()
                
                if not user_input:
                    continue
                if user_input.lower() in ["quit", "exit"]:
                    self._shutdown()
                    break
                    
                self._execute_task(user_input)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Keyboard interrupt. Shutting down...[/yellow]")
                self._shutdown()
                break

    def _execute_task(self, instruction: str):
        """The core execution lifecycle for a single task."""
        self.state.task_id = f"task_{int(time.time())}"
        
        # 1. Parsing
        self.state.transition_to(FSMState.PARSING)
        console.print("[dim cyan]\[PARSING][/dim cyan] Interpreting instruction...")
        
        # Try-catch blocks mock actual LLM call which might fail without model
        try:
             intent = self.parser.parse(instruction, self.state)
        except Exception:
             intent = {
                  "task_id": self.state.task_id, 
                  "raw_instruction": instruction,
                  "parsed_goal": instruction
             }
        
        self.task_store.create_task(self.session_id, self.state.task_id, instruction)
        
        # 2. Planning
        self.state.transition_to(FSMState.PLANNING)
        console.print("[dim cyan]\[PLANNING][/dim cyan] Generating step plan...")
        
        try:
             plan = self.planner.generate_plan(intent, self.state)
        except Exception:
             # Stub plan
             plan = {
                  "steps": [{
                       "step_id": "step_1",
                       "description": "Execute user command",
                       "max_retries": 1
                  }]
             }
             
        self.state.plan = plan
        self.task_store.update_task_plan(self.state.task_id, intent.get("parsed_goal", ""), plan)
        
        console.print("\n[bold]Task Plan:[/bold]")
        for i, step in enumerate(plan.get("steps", [])):
            console.print(f"  Step {i+1}: {step.get('description', 'Unknown')}")
        print()
            
        # 3. Execution Loop
        steps = plan.get("steps") or []
        if not steps:
            logger.info("Plan contains no steps. Transitioning to TASK_COMPLETE.")
            console.print("[yellow]Plan contains no steps. Ending task.[/yellow]")
            self.state.transition_to(FSMState.TASK_COMPLETE)
            self.task_store.update_task_status(self.state.task_id, self.state.fsm_state.name)
            return

        if not self.state.advance_step():
             return # Empty plan
             
        global_timeout = self.config.get("planning", {}).get("global_timeout_seconds", 1800)
        self.state.task_start_time = time.time()
        
        try:
            while self.state.fsm_state in [FSMState.EXECUTING, FSMState.VALIDATING, FSMState.RETRYING]:
                
                if time.time() - self.state.task_start_time > global_timeout:
                    self.state.transition_to(FSMState.TIMEOUT)
                    console.print(f"\n[bold red]\[TIMEOUT][/bold red] Global timeout exceeded.")
                    break
                    
                failsafe.check()
                
                step = self.state.get_current_step()
                if not step:
                    logger.info("No current step found. Transitioning to TASK_COMPLETE.")
                    self.state.transition_to(FSMState.TASK_COMPLETE)
                    break
                    
                console.print(f"[blue]\[EXECUTING][/blue] {step.get('description', 'Unknown Step')}")
                
                # Capture Screen
                capture_retries = 3
                capture_retry_sleep_s = 0.5
                cap_data = None
                dims = None
                for attempt in range(1, capture_retries + 1):
                    try:
                        cap_data = self.capture.capture_screen(self.session_id, self.state.current_step_id)
                        dims = self.capture.get_monitor_dimensions()
                        break
                    except Exception:
                        logger.exception("Screen capture failed (attempt %s/%s)", attempt, capture_retries)
                        if attempt < capture_retries:
                            time.sleep(capture_retry_sleep_s)

                if not cap_data or not dims:
                    console.print("[bold red]Screen capture failed repeatedly. Aborting task.[/bold red]")
                    self.state.transition_to(FSMState.FAILED)
                    break

                screen_path = cap_data["path"]
                screen_hash = cap_data["hash"]
                
                # Check for Infinite Loop
                history = self.action_log.get_recent_actions(self.state.task_id)
                last_action_type = history[-1]["action_type"] if history else None
                is_static_expected = last_action_type in ["wait", "run_command", "scroll"]
                
                if self.capture.check_loop(screen_hash):
                    if not is_static_expected:
                        self.state.repeated_state_count += 1
                        if self.state.repeated_state_count >= self.config.get("state", {}).get("repeated_state_limit", 5):
                            logger.warning(f"Infinite loop detected at step {self.state.current_step_id}. Aborting task.")
                            console.print("[yellow]\[WARNING][/yellow] Infinite loop detected. Aborting task safely.")
                            self.state.transition_to(FSMState.TASK_COMPLETE)
                            break
                else:
                    self.state.repeated_state_count = 0
                    
                # Perception Pipeline
                try:
                    ocr_data = self.ocr.process_image(screen_path, self.state.current_step_id)
                except Exception:
                    logger.exception("OCR failed; continuing without OCR")
                    ocr_data = []
                    
                try:
                    vis_data = self.vision.detect_elements(screen_path, self.state.current_step_id)
                except Exception:
                    logger.exception("Vision detection failed; continuing without detections")
                    vis_data = []
                
                try:
                    screen_state = StateBuilder.build_screen_state(
                        self.session_id, self.state.current_step_id,
                        self.config.get("capture", {}).get("monitor_index", 0),
                        self.config.get("capture", {}).get("capture_region", None),
                        dims, screen_hash, ocr_data, vis_data
                    )
                except Exception:
                    logger.exception("Failed to build screen_state; using minimal fallback state")
                    screen_state = {"resolution": dims, "elements": [], "text_regions": [], "screenshot_path": screen_path}
                
                # Action Decision
                history = self.action_log.get_recent_actions(self.state.task_id)
                try:
                     action_cmd = self.decision.get_next_action(
                         intent, step, self.state.current_step_idx, len(self.state.plan["steps"]), screen_state, history, self.state)
                except Exception:
                     # Mock decision for testing if LLM fails
                     action_cmd = {
                          "action_type": "wait",
                          "parameters": {"duration_ms": 1000},
                          "reasoning": "Mocking decision due to LLM unavailability."
                     }
                     # Example to break out of infinite wait
                     self.state.step_retry_count += 1
                     if self.state.step_retry_count > 1:
                          self.state.transition_to(FSMState.STEP_FAILED)
                          continue
                     
                console.print(f"  ├─ Action: {action_cmd.get('action_type')} | Reason: {action_cmd.get('reasoning')}")
                
                # Action Execution
                try:
                    self.executor.execute(action_cmd)
                except Exception as e:
                    logger.exception(f"Executor failed for action: {action_cmd}")
                    self.state.step_retry_count += 1
                    retry_limit = self.config.get("execution", {}).get("step_retry_limit", 3)
                    if self.state.step_retry_count > retry_limit:
                        console.print(f"[red]  └─ Status: ✗ Action failed repeatedly. Marking step failed.[/red]")
                        self.task_store.update_step_status(
                            self.state.task_id, self.state.current_step_id,
                            "FAILED", self.state.step_retry_count,
                        )
                        if not self.state.advance_step():
                            break
                    else:
                        console.print(f"[yellow]  └─ Status: ⚠ Action failed. Retrying ({self.state.step_retry_count}/{retry_limit})...[/yellow]")
                    continue
                
                # Log Action
                self.action_log.log_action(self.session_id, self.state.task_id, self.state.current_step_id, action_cmd, screen_hash)
                
                # --- Real result validation ---
                # Wait for the action to settle
                post_wait = max(action_cmd.get("post_action_wait_ms", 500) / 1000.0, 0.5)
                time.sleep(post_wait)
                
                # Re-capture the screen after the action
                self.state.transition_to(FSMState.VALIDATING)
                post_cap_data = None
                for attempt in range(1, 3):
                    try:
                        post_cap_data = self.capture.capture_screen(
                            self.session_id, f"{self.state.current_step_id}_post"
                        )
                        break
                    except Exception:
                        logger.warning("Post-action capture failed (attempt %s/2)", attempt)
                        time.sleep(0.3)
                
                # Determine whether the screen actually changed
                action_type = action_cmd.get("action_type", "")
                no_change_ok = action_type in ("wait", "scroll", "hover", "move", "press_key")
                
                if post_cap_data:
                    post_hash = post_cap_data["hash"]
                    # check_loop returns True when screen is unchanged
                    screen_unchanged = self.capture.check_loop(post_hash)
                
                    if screen_unchanged and not no_change_ok:
                        # Action had no visible effect — retry with backoff
                        self.state.step_retry_count += 1
                        retry_limit = self.config.get("execution", {}).get("step_retry_limit", 3)
                        if self.state.step_retry_count <= retry_limit:
                            delay = min(
                                self.config.get("state", {}).get("base_delay", 1.0)
                                * (2 ** self.state.step_retry_count),
                                self.config.get("state", {}).get("max_delay", 30.0)
                            )
                            console.print(
                                f"  [yellow][RETRY {self.state.step_retry_count}/{retry_limit}][/yellow] "
                                f"Screen unchanged. Retrying in {delay:.1f}s..."
                            )
                            self.state.transition_to(FSMState.RETRYING)
                            time.sleep(delay)
                            continue  # back to top of while loop — re-capture, re-decide, re-execute
                        else:
                            console.print(
                                f"  [red][STEP FAILED][/red] Screen unchanged after "
                                f"{retry_limit} retries. Moving to next step."
                            )
                            self.task_store.update_step_status(
                                self.state.task_id, self.state.current_step_id,
                                "FAILED", self.state.step_retry_count,
                            )
                    else:
                        console.print(
                            f"  └─ [green]✓ Step complete[/green] "
                            f"({'screen changed' if not no_change_ok else 'action acknowledged'})"
                        )
                        self.task_store.update_step_status(
                            self.state.task_id, self.state.current_step_id,
                            "COMPLETED", self.state.step_retry_count,
                        )
                else:
                    # Post-capture failed — assume step completed to avoid stalling
                    console.print("  └─ [yellow]✓ Step assumed complete (post-capture failed)[/yellow]")
                
                if not self.state.advance_step():
                    # Task completed
                    break
                # --- End real result validation ---
                    
                # Loop throttling
                idle_sleep = self.config.get("system", {}).get("loop_idle_sleep_ms", 100) / 1000.0
                time.sleep(idle_sleep)
                    
        except FailsafeTriggered:
            self.state.transition_to(FSMState.ABORTED)
            console.print("\n[bold red]\[ABORTED][/bold red] Failsafe triggered by user. Task stopped.")
        except Exception as e:
            logger.exception(f"Task execution failed: {e}")
            self.state.transition_to(FSMState.FAILED)
            console.print(f"\n[bold red]\[ERROR][/bold red] {e}")
            
        finally:
            self.task_store.update_task_status(self.state.task_id, self.state.fsm_state.name)
            self.capture.task_complete(self.session_id)
            if self.state.fsm_state == FSMState.TASK_COMPLETE:
                 console.print(f"\n[bold green]\[COMPLETE][/bold green] Task finished successfully.")

    def _shutdown(self):
        console.print("[yellow]Cleaning up processes...[/yellow]")
        failsafe.stop()
        self.capture.shutdown()
        console.print("Goodbye.")
        
if __name__ == "__main__":
    app = LADAS()
    app.run_terminal_loop()
