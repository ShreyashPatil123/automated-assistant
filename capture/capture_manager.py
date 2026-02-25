import os
import time
import imagehash
from PIL import Image
from capture.screen_capture import ScreenCapture
from capture.cleanup import CaptureCleanup
import logging
from collections import deque

class CaptureManager:
    """Orchestrates screen capture, cleanup, and loop detection via caching."""
    def __init__(self, config: dict):
        self.config = config.get("capture", {})
        self.temp_dir = os.path.join(os.getcwd(), "temp_screens")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        monitor_index = self.config.get("monitor_index", 0)
        self.screen_capture = ScreenCapture(monitor_index=monitor_index)
        
        self.cleanup = CaptureCleanup(
            temp_dir=self.temp_dir,
            max_count=self.config.get("max_screenshot_count", 200),
            max_age_seconds=self.config.get("max_retention_seconds", 3600)
        )
        
        # Start background cleanup every 60s
        self.cleanup.start_background_cleanup(interval_seconds=60)
        
        self.last_capture_path = None
        self.last_hash = None
        state_cfg = config.get("state", {})
        self.loop_repeat_limit = state_cfg.get("repeated_state_limit", 5)
        self.hash_window_size = max(int(self.loop_repeat_limit) * 2, int(self.loop_repeat_limit))
        self.hash_history = deque(maxlen=self.hash_window_size)
        self._consecutive_same_hash = 0
        
    def capture_screen(self, session_id: str, step_id: str) -> dict:
        """
        Capture the current screen. 
        Returns dict containing the file path and perceptual hash.
        """
        timestamp = int(time.time() * 1000)
        filename = f"{session_id}_{timestamp}_{step_id}.png"
        output_path = os.path.join(self.temp_dir, filename)
        
        region = self.config.get("capture_region", None)
        
        try:
            if region:
                self.screen_capture.capture_region(region, output_path)
            else:
                self.screen_capture.capture_full_screen(output_path)
        except Exception as e:
            raise RuntimeError(f"Screen capture failed (monitor_index={self.config.get('monitor_index', 0)}, region={region}): {e}") from e
            
        self.last_capture_path = output_path
        
        # Compute phash for loop detection
        try:
            img = Image.open(output_path)
            self.last_hash = str(imagehash.phash(img))
        except Exception:
            self.last_hash = None
            
        return {
            "path": output_path,
            "hash": self.last_hash,
            "timestamp": timestamp
        }
        
    def get_monitor_dimensions(self):
        return self.screen_capture.get_monitor_dimensions()
        
    def check_loop(self, new_hash: str) -> bool:
        """Return True if the screen looks stuck based on repeated hashes.

        More robust than a strict equality across the whole window:
        - Ignores missing hashes (e.g. if hashing fails).
        - Tracks consecutive repetition and also frequency inside a sliding window.
        """
        if not new_hash:
            self._consecutive_same_hash = 0
            return False

        if self.hash_history and self.hash_history[-1] == new_hash:
            self._consecutive_same_hash += 1
        else:
            self._consecutive_same_hash = 1

        self.hash_history.append(new_hash)

        if self._consecutive_same_hash >= self.loop_repeat_limit:
            logging.info(
                "Loop detected by consecutive hash repetition: hash=%s repeats=%s window=%s",
                new_hash,
                self._consecutive_same_hash,
                list(self.hash_history),
            )
            return True

        # Frequency-based guard to catch A/B/A/B toggles etc.
        freq = sum(1 for h in self.hash_history if h == new_hash)
        if freq >= self.loop_repeat_limit and len(self.hash_history) >= self.loop_repeat_limit:
            logging.info(
                "Loop suspected by sliding-window repetition: hash=%s freq=%s/%s window=%s",
                new_hash,
                freq,
                len(self.hash_history),
                list(self.hash_history),
            )
            return True

        return False
        
    def task_complete(self, session_id: str):
        """Called when a task is finished to clean up all its screens immediately."""
        self.cleanup.clean_session(session_id)
        self.cleanup.enforce_policy()
        
    def shutdown(self):
        self.cleanup.stop_background_cleanup()
        self.screen_capture.close()
