import cv2
from typing import List
import numpy as np


class VideoProcessor:
    """
    Handles video loading, frame extraction, and face cropping.
    """

    def __init__(self, face_detector):
        """
        face_detector: instance of QualityAgnosticFaceDetector
        """
        self.face_detector = face_detector

    def extract_frames(
        self,
        video_path: str,
        max_frames: int = 300
    ) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        frames = []

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        return frames

    def extract_face_frames(
        self,
        video_path: str,
        max_frames: int = 300
    ) -> List[np.ndarray]:
        frames = self.extract_frames(video_path, max_frames)
        face_frames = []

        for frame in frames:
            crops = self.face_detector.crop_faces(frame)
            if crops:
                face_frames.append(crops[0])  # strongest face only

        return face_frames
 