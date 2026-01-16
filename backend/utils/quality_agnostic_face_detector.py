import cv2
import numpy as np
from typing import List, Tuple
from insightface.app import FaceAnalysis


class QualityAgnosticFaceDetector:
    """
    Face detector using InsightFace (RetinaFace backend)
    Stable, fast, production-grade.
    """

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self.app = FaceAnalysis(
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        faces = self.app.get(frame)
        boxes = []

        for face in faces:
            if face.det_score < self.min_confidence:
                continue
            x1, y1, x2, y2 = map(int, face.bbox)
            boxes.append((x1, y1, x2, y2))

        return boxes

    def crop_faces(self, frame: np.ndarray) -> List[np.ndarray]:
        crops = []
        for x1, y1, x2, y2 in self.detect(frame):
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
        return crops
