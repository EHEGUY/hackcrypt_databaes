"""
JSON Sanitizer - Simple and correct
Uses a proper library instead of manual type checking
"""
from typing import Any
import json


def sanitize_for_json(obj: Any) -> Any:
    """
    Convert object to JSON-serializable form.
    
    Uses json.dumps/loads round-trip to handle all cases correctly.
    This is simpler and more reliable than manual type checking.
    """
    try:
        # Custom encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as np
                
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                
                return super().default(obj)
        
        # Round-trip through JSON
        json_str = json.dumps(obj, cls=NumpyEncoder)
        return json.loads(json_str)
    
    except (TypeError, ValueError) as e:
        # Fallback: convert to string
        return str(obj)


# Keep the original functions for backward compatibility if needed
def ensure_float(x: Any) -> float:
    """Convert to Python float"""
    if x is None:
        return 0.0
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def ensure_int(x: Any) -> int:
    """Convert to Python int"""
    if x is None:
        return 0
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def ensure_bool(x: Any) -> bool:
    """Convert to Python bool"""
    if x is None:
        return False
    return bool(x)