import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import yaml
import logging
from perception.ocr_engine import OCREngine
from perception.vision_detector import VisionDetector
from reasoning.llm_client import LLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def test_models():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    print("--- 1. Testing LLM ---")
    llm_path = config.get("reasoning", {}).get("model_path", "")
    try:
        llm = LLMClient(model_path=llm_path, n_gpu_layers=0)
        resp = llm.generate_text("Say 'Hello, World!'", max_tokens=10)
        print(f"LLM Response: {resp}")
    except Exception as e:
        print(f"LLM Test Failed: {e}")
        
    print("\n--- 2. Testing OCR ---")
    try:
        # We need a dummy image to test
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (200, 100), color = (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((10,10), "Test OCR", fill=(0,0,0))
        img.save("test_ocr.png")
        
        ocr = OCREngine(config)
        res = ocr.process_image("test_ocr.png", "test")
        print(f"OCR results: {[r['text'] for r in res]}")
        os.remove("test_ocr.png")
    except Exception as e:
        print(f"OCR Test Failed: {e}")
        
    print("\n--- 3. Testing Vision (YOLO) ---")
    try:
        vision = VisionDetector(config)
        # Using a blank image for a dry run just to see if model loads and predicts without crashing
        img = Image.new('RGB', (640, 640), color = (255, 255, 255))
        img.save("test_vis.png")
        res = vision.detect_elements("test_vis.png", "test")
        print(f"Vision loaded successfully. Detections on blank image: {len(res)}")
        os.remove("test_vis.png")
    except Exception as e:
        print(f"Vision Test Failed: {e}")

if __name__ == "__main__":
    test_models()
