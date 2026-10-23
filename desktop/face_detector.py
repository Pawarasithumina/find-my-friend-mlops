"""Lightweight classical face detection for the webcam front-end."""
from __future__ import annotations

import cv2


class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError("OpenCV frontal-face detector could not be loaded")

    def locate(self, frame):
        """Return faces as (x, y, w, h), ordered by descending area."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.13, minNeighbors=5,
            minSize=(65, 65), flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return sorted((tuple(int(v) for v in face) for face in faces),
                      key=lambda box: box[2] * box[3], reverse=True)

    @staticmethod
    def crop(frame, box, padding=0.18):
        x, y, w, h = box
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(frame.shape[1], x + w + pad_w)
        y2 = min(frame.shape[0], y + h + pad_h)
        return frame[y1:y2, x1:x2]
