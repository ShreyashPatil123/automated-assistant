import mss
import logging
import os
import time
from typing import Tuple, Optional

class ScreenCapture:
    def __init__(self, monitor_index: int = 0):
        self.sct = mss.mss()
        self.monitor_index = monitor_index
        
        # Verify monitor index is valid
        if self.monitor_index >= len(self.sct.monitors):
            logging.warning(f"Monitor index {self.monitor_index} out of bounds. Using default monitor 1.")
            self.monitor_index = 1
        elif self.monitor_index == 0:
             # mss index 0 is "all monitors combined". If we want a specific one, it starts at 1.
             # but we'll allow 0 to mean "all monitors"
             pass
        else:
            self.monitor_index = self.monitor_index
            

    def capture_full_screen(self, output_path: str) -> str:
        """Capture the full screen of the selected monitor."""
        monitor = self.sct.monitors[self.monitor_index]
        return self._capture_region(monitor, output_path)

    def capture_region(self, region: Tuple[int, int, int, int], output_path: str) -> str:
        """
        Capture a specific region of the screen.
        region: (x, y, width, height)
        """
        monitor = {
            "left": region[0],
            "top": region[1],
            "width": region[2],
            "height": region[3]
        }
        return self._capture_region(monitor, output_path)
        
    def _capture_region(self, monitor_dict: dict, output_path: str) -> str:
        """Internal method to perform the capture and save it."""
        try:
            sct_img = self.sct.grab(monitor_dict)
            mss.tools.to_png(sct_img.rgb, sct_img.size, output=output_path)
            return output_path
        except Exception as e:
            logging.error(f"Failed to capture screen: {e}")
            raise
            
    def get_monitor_dimensions(self) -> Tuple[int, int]:
        """Returns (width, height) of the selected monitor."""
        monitor = self.sct.monitors[self.monitor_index]
        return (monitor["width"], monitor["height"])

    def get_monitor_offset(self) -> tuple[int, int]:
        """Returns the global (left, top) offset of the selected monitor."""
        monitor = self.sct.monitors[self.monitor_index]
        return (monitor["left"], monitor["top"])
        
    def close(self):
        """Clean up mss resources."""
        self.sct.close()
