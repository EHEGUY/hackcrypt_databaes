from fastapi import APIRouter, UploadFile, File
import tempfile
import os
import cv2
import logging

from utils.video_processor import VideoProcessor
from utils.quality_agnostic_face_detector import QualityAgnosticFaceDetector
from models.comprehensive_detector import IntegratedDeepfakeDetector

logger = logging.getLogger(__name__)

router = APIRouter()

# ---- Global singletons ----
face_detector = QualityAgnosticFaceDetector(min_confidence=0.3)
video_processor = VideoProcessor(face_detector=face_detector)
deepfake_detector = IntegratedDeepfakeDetector()


def get_video_fps(video_path: str) -> float:
    """Extract FPS from video file"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return float(fps) if fps > 0 else 30.0
    except Exception as e:
        logger.warning(f"Could not extract FPS: {e}, using default 30.0")
        return 30.0


@router.post("/api/v1/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    """
    Analyze uploaded video for deepfake indicators
    
    Args:
        file: Video file to analyze
        
    Returns:
        Analysis results with all metrics
    """
    suffix = os.path.splitext(file.filename)[1]
    video_path = None

    try:
        # Write uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            video_path = tmp.name

        logger.info(f"Processing video: {file.filename}")

        # Extract frames from video
        frames = video_processor.extract_frames(video_path)
        
        if not frames or len(frames) == 0:
            return {
                "error": "No frames could be extracted from video",
                "filename": file.filename
            }

        # Get video FPS
        fps = get_video_fps(video_path)
        logger.info(f"Extracted {len(frames)} frames at {fps} FPS")

        # Run analysis
        result = deepfake_detector.analyze_video(
            frames=frames,
            fps=fps,
            video_path=video_path
        )

        logger.info(f"Analysis complete for {file.filename}")
        return result

    except Exception as e:
        logger.error(f"Error analyzing video: {e}", exc_info=True)
        return {
            "error": str(e),
            "filename": file.filename
        }

    finally:
        # Clean up temp file
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception as e:
                logger.warning(f"Could not delete temp file: {e}")