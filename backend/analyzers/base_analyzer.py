"""
Base Analyzer Interface
Defines contract for all analyzer components
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseAnalyzer(ABC):
    """Abstract base class for all analyzers"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def analyze(self, frames: List[np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Analyze frames and return structured results
        
        Args:
            frames: List of video frames (BGR format)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with analysis results
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Get analyzer information and capabilities
        
        Returns:
            Dictionary with analyzer metadata
        """
        pass
    
    def _handle_error(self, error: Exception) -> Dict[str, Any]:
        """
        Standard error handling
        
        Args:
            error: Exception that occurred
            
        Returns:
            Error report dictionary
        """
        self.logger.error(f"Analysis error: {error}", exc_info=True)
        return {
            "error": str(error),
            "implemented": False,
            "available": False
        }
    
    def _validate_frames(self, frames: List[np.ndarray]) -> bool:
        """
        Validate frame input
        
        Args:
            frames: List of frames to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not frames:
            self.logger.warning("No frames provided")
            return False
        
        if not all(isinstance(f, np.ndarray) for f in frames):
            self.logger.warning("Invalid frame types")
            return False
        
        if not all(f.size > 0 for f in frames):
            self.logger.warning("Empty frames detected")
            return False
        
        return True
    
    def _sanitize_output(self, data: Any) -> Any:
        """
        Ensure output is JSON-serializable
        
        Args:
            data: Data to sanitize
            
        Returns:
            JSON-serializable data
        """
        if isinstance(data, dict):
            return {k: self._sanitize_output(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._sanitize_output(x) for x in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        elif data is None:
            return None
        else:
            return data