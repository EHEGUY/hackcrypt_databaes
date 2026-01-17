"""
Fixed Configuration with Xception Model
"""
import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ============================================================
    # APP SETTINGS
    # ============================================================
    APP_NAME: str = "Deepfake Detection API"
    VERSION: str = "2.1.0-with-xception"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # ============================================================
    # SERVER SETTINGS
    # ============================================================
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("PORT", "8010"))
    WORKERS: int = int(os.getenv("WORKERS", "1"))
    
    # ============================================================
    # CORS SETTINGS
    # ============================================================
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "*"  # Remove in production!
    ]
    
    # ============================================================
    # MODEL SETTINGS
    # ============================================================
    IMAGE_SIZE: int = 224
    USE_GPU: bool = os.getenv("USE_GPU", "true").lower() == "true"
    TORCH_DTYPE: str = "float16"  # float16 for GPU, float32 for CPU
    
    # Xception model
    XCEPTION_MODEL_PATH: str = os.getenv(
        "XCEPTION_MODEL_PATH", 
        "./model/xception_deepfake_image_5o.h5"
    )
    
    # Model ensemble
    USE_ENSEMBLE: bool = True
    MIN_MODELS_REQUIRED: int = 2
    
    # Batch processing
    BATCH_SIZE: int = 8
    
    # ============================================================
    # VIDEO PROCESSING
    # ============================================================
    MAX_VIDEO_FRAMES: int = 60
    FRAME_SAMPLING: str = "uniform"
    
    # ============================================================
    # DETECTION THRESHOLDS
    # ============================================================
    MIN_MODEL_AGREEMENT: float = 0.5
    
    # ============================================================
    # FILE UPLOAD LIMITS
    # ============================================================
    MAX_VIDEO_SIZE_MB: int = 150
    MAX_IMAGE_SIZE_MB: int = 15
    
    SUPPORTED_VIDEO_FORMATS: List[str] = [
        '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v'
    ]
    SUPPORTED_IMAGE_FORMATS: List[str] = [
        '.jpg', '.jpeg', '.png', '.bmp', '.webp'
    ]
    
    # ============================================================
    # TEMPORARY FILE SETTINGS
    # ============================================================
    TEMP_DIR: str = os.getenv("TMPDIR", os.getenv("TEMP", "/tmp"))
    TEMP_FILE_MAX_AGE: int = 3600
    
    # ============================================================
    # FACE DETECTION
    # ============================================================
    FACE_DETECTION_CONFIDENCE: float = 0.3
    FACE_PADDING_RATIO: float = 0.15
    
    # ============================================================
    # LOGGING
    # ============================================================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ENABLE_REQUEST_LOGGING: bool = True
    
    # ============================================================
    # RATE LIMITING
    # ============================================================
    ENABLE_RATE_LIMITING: bool = False
    RATE_LIMIT_PER_MINUTE: int = 10
    
    # ============================================================
    # SECURITY
    # ============================================================
    ENABLE_API_KEY_AUTH: bool = False
    API_KEYS: List[str] = os.getenv("API_KEYS", "").split(",")
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_settings()
    
    def _validate_settings(self):
        """Validate configuration values"""
        
        # Validate file sizes
        assert self.MAX_VIDEO_SIZE_MB > 0, "MAX_VIDEO_SIZE_MB must be positive"
        assert self.MAX_IMAGE_SIZE_MB > 0, "MAX_IMAGE_SIZE_MB must be positive"
        
        # Validate frame count
        assert self.MAX_VIDEO_FRAMES > 0, "MAX_VIDEO_FRAMES must be positive"
        assert self.MAX_VIDEO_FRAMES <= 60, "MAX_VIDEO_FRAMES should be <= 60 for performance"
        
        # Validate face detection
        assert 0.0 <= self.FACE_DETECTION_CONFIDENCE <= 1.0
        assert 0.0 <= self.FACE_PADDING_RATIO <= 0.5
        
        # Validate model agreement
        assert 0.0 <= self.MIN_MODEL_AGREEMENT <= 1.0
    
    def get_summary(self) -> dict:
        """Get configuration summary"""
        return {
            "app": {
                "name": self.APP_NAME,
                "version": self.VERSION,
                "environment": self.ENVIRONMENT
            },
            "models": {
                "xception": {
                    "enabled": os.path.exists(self.XCEPTION_MODEL_PATH) if self.XCEPTION_MODEL_PATH else False,
                    "path": self.XCEPTION_MODEL_PATH,
                    "accuracy": "~92%"
                },
                "ensemble_size": 5,  # Xception + 4 others
                "use_gpu": self.USE_GPU
            },
            "features": {
                "xception_model": "NEW - FaceForensics++ trained",
                "blink_detection": "FIXED - Sustained closure detection",
                "av_sync": "FIXED - Full lip contour",
                "frequency_analysis": "FIXED - Handles odd dimensions",
                "ensemble": "5 models total"
            }
        }


# Create global settings instance
settings = Settings()

# Print configuration summary
if __name__ == "__main__":
    import json
    print("="*60)
    print("UPDATED CONFIGURATION")
    print("="*60)
    print(json.dumps(settings.get_summary(), indent=2))
    print("\n" + "="*60)
    print("KEY UPDATES:")
    print("="*60)
    print("✓ Added Xception model (92% accuracy)")
    print("✓ Fixed frequency analyzer (DCT odd-size bug)")
    print("✓ Fixed blink detection (sustained closure)")
    print("✓ Fixed AV sync (full lip contour)")
    print("✓ 5-model ensemble now")
    print("="*60)