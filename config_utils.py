import logging

logger = logging.getLogger("ladas.config")

def validate_config(config: dict) -> dict:
    """
    Validates the parsed configuration dictionary, enforcing types and bounds.
    Falls back to safe defaults if invalid values are found.
    """
    
    # helper for ints
    def get_int(section, key, default, min_val=None, max_val=None):
        val = config.get(section, {}).get(key, default)
        try:
            val = int(val)
            if min_val is not None and val < min_val:
                val = min_val
                logger.warning(f"Config {section}.{key} below minimum, clamping to {min_val}")
            if max_val is not None and val > max_val:
                val = max_val
                logger.warning(f"Config {section}.{key} above maximum, clamping to {max_val}")
            if section in config:
                config[section][key] = val
            return val
        except (ValueError, TypeError):
            logger.warning(f"Invalid config for {section}.{key}: {val}. Using default: {default}")
            if section in config:
                config[section][key] = default
            return default

    # helper for floats
    def get_float(section, key, default, min_val=None):
        val = config.get(section, {}).get(key, default)
        try:
            val = float(val)
            if min_val is not None and val < min_val:
                val = min_val
                logger.warning(f"Config {section}.{key} below minimum, clamping to {min_val}")
            if section in config:
                config[section][key] = val
            return val
        except (ValueError, TypeError):
            logger.warning(f"Invalid config for {section}.{key}: {val}. Using default: {default}")
            if section in config:
                config[section][key] = default
            return default

    # helper for bools
    def get_bool(section, key, default):
        val = config.get(section, {}).get(key, default)
        if not isinstance(val, bool):
            # Attempt to interpret strings like "true", "false" if needed, 
            # but standard yaml loader usually handles it.
            if str(val).lower() in ("true", "1", "yes"):
                val = True
            elif str(val).lower() in ("false", "0", "no"):
                val = False
            else:
                logger.warning(f"Invalid boolean config for {section}.{key}: {val}. Using default: {default}")
                val = default
        if section not in config:
            config[section] = {}
        config[section][key] = val
        return val

    # System Section
    if "system" not in config:
        config["system"] = {}
    get_int("system", "loop_idle_sleep_ms", 100, min_val=0)
    get_bool("system", "allow_mock_on_startup_failure", True)

    # Capture Section
    if "capture" not in config:
        config["capture"] = {}
    get_int("capture", "max_screenshot_count", 200, min_val=1)
    get_int("capture", "max_retention_seconds", 3600, min_val=60)

    # Planning Section
    if "planning" not in config:
        config["planning"] = {}
    get_int("planning", "max_steps", 50, min_val=1)
    get_int("planning", "max_replan_attempts", 3, min_val=0)
    get_int("planning", "global_timeout_seconds", 1800, min_val=1)

    # State Section
    if "state" not in config:
        config["state"] = {}
    get_int("state", "repeated_state_limit", 5, min_val=1)
    get_int("state", "element_wait_timeout", 30, min_val=1)
    get_int("state", "step_timeout_seconds", 120, min_val=1)
    get_int("state", "max_retries", 5, min_val=0)
    
    get_float("state", "base_delay", 1.0, min_val=0.0)
    get_float("state", "max_delay", 30.0, min_val=0.0)

    # Reasoning Section
    if "reasoning" not in config:
        config["reasoning"] = {}
    get_int("reasoning", "max_llm_calls_per_task", 20, min_val=1)
    model_path = config.get("reasoning", {}).get("model_path", "")
    if model_path is None:
        config["reasoning"]["model_path"] = ""
    elif not isinstance(model_path, str):
        logger.warning("Invalid config for reasoning.model_path: %r. Converting to string.", model_path)
        config["reasoning"]["model_path"] = str(model_path)

    # Perception Section
    if "perception" not in config:
        config["perception"] = {}
    if "vision" not in config["perception"]:
        config["perception"]["vision"] = {}
    if "ocr" not in config["perception"]:
        config["perception"]["ocr"] = {}
    yolo_path = config.get("perception", {}).get("vision", {}).get("yolo_model_path", "yolov8n.pt")
    if yolo_path is None:
        config["perception"]["vision"]["yolo_model_path"] = ""
    elif not isinstance(yolo_path, str):
        logger.warning("Invalid config for perception.vision.yolo_model_path: %r. Converting to string.", yolo_path)
        config["perception"]["vision"]["yolo_model_path"] = str(yolo_path)

    # Execution Section
    if "execution" not in config:
        config["execution"] = {}
    get_bool("execution", "dry_run", False)
    get_bool("execution", "unsafe_mode", False)
    get_int("execution", "step_retry_limit", 3, min_val=0)

    # Ensure allowed lists exist as lists
    for key in ["allowed_commands", "allowed_hotkeys"]:
        val = config.get("execution", {}).get(key, [])
        if not isinstance(val, list):
            logger.warning(f"Invalid list config for execution.{key}. Using empty list.")
            config["execution"][key] = []

    return config
