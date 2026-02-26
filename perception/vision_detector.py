import logging
import cv2
import numpy as np
import os
from typing import List, Dict, Any

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class VisionDetector:
    def __init__(self, config: dict):
        self.config = config.get("perception", {}).get("vision", {})
        self.system_config = config.get("system", {})
        self.allow_mock = self.system_config.get("allow_mock_on_startup_failure", False)
        # Use the raw filename. Ultralytics will automatically download it to the root 
        # directory if it isn't found locally.
        self.yolo_model_path = self.config.get("yolo_model_path", "yolov8n.pt").split('/')[-1]
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        self.nms_threshold = self.config.get("nms_threshold", 0.45)
        self.device = self.config.get("detection_device", "cuda")
        self.template_dir = self.config.get("template_library_path", "perception/templates/")
        
        self.model = None
        
        if YOLO_AVAILABLE:
            if os.path.exists(self.yolo_model_path):
                try:
                    logging.info(f"Loading YOLOv8 model from {self.yolo_model_path} on {self.device}")
                    self.model = YOLO(self.yolo_model_path)
                    # Force model to device if specified
                    try:
                        self.model.to(self.device)
                    except Exception as e:
                        logging.warning(
                            "YOLO model loaded but moving to device '%s' failed (%s). Falling back to default device.",
                            self.device,
                            e,
                        )
                except Exception as e:
                    logging.error(
                        "Failed to load YOLOv8 model from '%s': %s. "
                        "If you intended to use YOLO, ensure 'ultralytics' is installed and the weights file exists.",
                        self.yolo_model_path,
                        e,
                    )
                    self.model = None
            else:
                logging.error(
                    "YOLO weights not found at '%s'. "
                    "Fix: set perception.vision.yolo_model_path to a valid .pt file (e.g. download yolov8n.pt) or disable YOLO by leaving ultralytics uninstalled.",
                    self.yolo_model_path,
                )
                if not self.allow_mock:
                    logging.warning("Continuing without YOLO (template matching only).")
        else:
             logging.error(
                 "ultralytics is not installed; YOLO vision is unavailable. Fix: pip install ultralytics (and torch) if you want detections."
             )

    def detect_elements(self, image_path: str, step_id: str) -> List[Dict[str, Any]]:
        """Run object detection on the screen image."""
        if self.model:
            try:
                results = self._detect_yolo(image_path, step_id)
                if results:
                    return results
            except Exception:
                logging.exception("YOLO detection crashed; continuing with template matching.")
        
        logging.info("Falling back to OpenCV template matching.")
        return self._detect_templates(image_path, step_id)

    def _detect_yolo(self, image_path: str, step_id: str) -> List[Dict[str, Any]]:
        """Run YOLO inference."""
        elements = []
        try:
             # Run inference
             # iou = nms threshold. conf = confidence threshold
             results = self.model(
                  image_path, 
                  conf=self.confidence_threshold, 
                  iou=self.nms_threshold, 
                  verbose=False
             )[0]
             
             # Extract bounding boxes, classes, and confidence
             boxes = results.boxes
             names = self.model.names
             
             for idx, box in enumerate(boxes):
                  x1, y1, x2, y2 = box.xyxy[0].tolist()
                  conf = box.conf[0].item()
                  cls_id = int(box.cls[0].item())
                  class_name = names[cls_id]
                  
                  x = int(x1)
                  y = int(y1)
                  w = int(x2 - x1)
                  h = int(y2 - y1)
                  
                  center_x = x + w // 2
                  center_y = y + h // 2
                  
                  elements.append({
                       "id": f"vis_{step_id}_{idx}",
                       "class": class_name,
                       "label": class_name, # Can be enriched later by OCR
                       "confidence": round(conf, 2),
                       "bounding_box": {"x": x, "y": y, "width": w, "height": h},
                       "center": {"x": center_x, "y": center_y}
                  })
        except Exception as e:
             logging.error(f"YOLO inference error: {e}")
             
        return elements

    def _detect_templates(self, image_path: str, step_id: str) -> List[Dict[str, Any]]:
        """Fallback: OpenCV template matching. Returns simple elements if templates exist."""
        elements = []
        if not os.path.exists(self.template_dir):
            return elements
            
        try:
             img_rgb = cv2.imread(image_path)
             if img_rgb is None:
                  return elements
                  
             img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
             
             # Very basic hardcoded example. A real implementation would loop through 
             # the template directory securely.
             templates = [f for f in os.listdir(self.template_dir) if f.endswith(('.png', '.jpg'))]
             idx = 0
             
             for template_file in templates:
                  template_path = os.path.join(self.template_dir, template_file)
                  template = cv2.imread(template_path, 0)
                  
                  if template is None:
                       continue
                       
                  w, h = template.shape[::-1]
                  class_name = os.path.splitext(template_file)[0] # e.g. "button", "checkbox"
                  
                  # Run matching
                  res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
                  
                  # Filter by threshold
                  loc = np.where(res >= self.confidence_threshold)
                  
                  for pt in zip(*loc[::-1]):
                       x, y = int(pt[0]), int(pt[1])
                       center_x = x + w // 2
                       center_y = y + h // 2
                       
                       elements.append({
                            "id": f"tmp_{step_id}_{idx}",
                            "class": class_name,
                            "label": class_name,
                            "confidence": round(float(res[pt[1]][pt[0]]), 2),
                            "bounding_box": {"x": x, "y": y, "width": int(w), "height": int(h)},
                            "center": {"x": center_x, "y": center_y}
                       })
                       idx += 1
                       
        except Exception as e:
             logging.error(f"Template matching error: {e}")
             
        return elements
