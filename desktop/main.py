"""FaceOps TwinTrack: native Ubuntu webcam demonstration client.

Run: python -m desktop.main
Requires the local FastAPI server on 127.0.0.1:8000.
Images are kept in memory; no video or face crops are saved.
"""
from __future__ import annotations

import os
import sys
import time

import cv2
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QProgressBar, QSizePolicy, QVBoxLayout, QWidget,
)

from desktop.client import API_URL, compare, health
from desktop.face_detector import FaceDetector


STYLE = """
QWidget { background: #101827; color: #e9eff8; font-family: 'DejaVu Sans'; font-size: 12px; }
QLabel#subtitle { color: #9cacbf; }
QLabel#title { font-size: 24px; font-weight: 700; color: #f7fbff; }
QLabel#section { font-weight: 700; font-size: 14px; }
QLabel#prediction { font-weight: 700; font-size: 20px; color: #f6fbff; }
QLabel#muted { color: #a6b6c9; }
QFrame#card { background: #182539; border: 1px solid #30425c; border-radius: 12px; }
QLabel#camera { border: 1px solid #344863; border-radius: 10px; background: #0b1422; }
QPushButton { background: #2c6ee0; padding: 11px 18px; border: none; border-radius: 7px; font-weight: 650; }
QPushButton:hover { background: #3a80fa; }
QPushButton:disabled { background: #344154; color: #8291a6; }
QPushButton#secondary { background: #26364e; }
QPushButton#secondary:hover { background: #344761; }
QProgressBar { background: #26374e; border: none; border-radius: 5px; height: 10px; }
QProgressBar::chunk { background: #4c99e6; border-radius: 5px; }
"""


class InferenceThread(QThread):
    success = Signal(dict)
    failure = Signal(str)

    def __init__(self, jpeg_bytes: bytes, parent=None):
        super().__init__(parent)
        self.jpeg_bytes = jpeg_bytes

    def run(self):
        try:
            self.success.emit(compare(self.jpeg_bytes))
        except Exception as exc:
            self.failure.emit(f"{type(exc).__name__}: {exc}")


class ModelCard(QFrame):
    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(12)
        heading = QLabel(name)
        heading.setObjectName("section")
        self.version = QLabel("Waiting for inference")
        self.version.setObjectName("muted")
        self.prediction = QLabel("—")
        self.prediction.setObjectName("prediction")
        self.confidence = QProgressBar()
        self.confidence.setRange(0, 100)
        self.confidence.setValue(0)
        self.confidence.setTextVisible(True)
        self.detail = QLabel("No image submitted")
        self.detail.setObjectName("muted")
        self.detail.setWordWrap(True)
        for w in (heading, self.version, self.prediction, self.confidence, self.detail):
            lay.addWidget(w)

    def show_result(self, result):
        name = result.get("predicted_class", "unknown")
        confidence = max(0.0, min(1.0, float(result.get("confidence", 0))))
        low = bool(result.get("low_confidence", False))
        self.version.setText(
            f"{result.get('architecture', '')}  ·  Registry v{result.get('model_version', '?')}"
        )
        self.prediction.setText("Uncertain" if low else name.title())
        self.confidence.setValue(round(confidence * 100))
        note = " · Low confidence — verify manually" if low else ""
        self.detail.setText(f"Latency: {result.get('latency_ms', '?')} ms{note}")

    def clear_result(self):
        self.prediction.setText("—")
        self.confidence.setValue(0)
        self.detail.setText("No current face")


class FaceOpsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceOps TwinTrack · Live Desktop")
        self.resize(1160, 780)
        self.setMinimumSize(920, 660)
        self.camera = None
        self.detector = FaceDetector()
        self.worker = None
        self.last_infer = 0.0
        self.last_face = 0.0
        self.infer_interval = 1.0  # At most once per second
        self.api_online = False
        self.camera_index = int(os.getenv("FACEOPS_CAMERA_INDEX", "0"))

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(24, 22, 24, 20)
        outer.setSpacing(17)
        head = QHBoxLayout()
        titles = QVBoxLayout()
        title = QLabel("FaceOps TwinTrack")
        title.setObjectName("title")
        subtitle = QLabel("Live webcam demonstration  ·  MLflow production models")
        subtitle.setObjectName("subtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        head.addLayout(titles, 1)
        self.api_label = QLabel("API: checking...")
        self.api_label.setObjectName("muted")
        head.addWidget(self.api_label)
        outer.addLayout(head)

        body = QHBoxLayout()
        body.setSpacing(17)
        left = QVBoxLayout()
        left.setSpacing(12)
        self.video = QLabel("Camera is stopped\nClick Start Camera")
        self.video.setObjectName("camera")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(500, 370)
        self.video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left.addWidget(self.video, 1)
        actions = QHBoxLayout()
        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.setObjectName("secondary")
        self.stop_button.setEnabled(False)
        self.check_button = QPushButton("Check API")
        self.check_button.setObjectName("secondary")
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.check_button.clicked.connect(self.check_api)
        actions.addWidget(self.start_button)
        actions.addWidget(self.stop_button)
        actions.addWidget(self.check_button)
        actions.addStretch()
        left.addLayout(actions)
        self.camera_label = QLabel("No video captured or stored")
        self.camera_label.setObjectName("muted")
        left.addWidget(self.camera_label)
        body.addLayout(left, 3)

        right = QVBoxLayout()
        right.setSpacing(14)
        heading = QLabel("Live model comparison")
        heading.setObjectName("section")
        right.addWidget(heading)
        self.model_a = ModelCard("Model A  ·  MobileNetV3-Small")
        self.model_b = ModelCard("Model B  ·  ResNet18")
        right.addWidget(self.model_a)
        right.addWidget(self.model_b)
        self.agreement = QLabel("Agreement: waiting")
        self.agreement.setObjectName("section")
        right.addWidget(self.agreement)
        self.quality = QLabel("This is a classroom experiment, not a secure identity-verification tool.")
        self.quality.setObjectName("muted")
        self.quality.setWordWrap(True)
        right.addWidget(self.quality)
        right.addStretch()
        body.addLayout(right, 2)
        outer.addLayout(body, 1)

        self.footer = QLabel(
            "FaceOps | 209 source photos · dataset v2 · known classes were captured in single sessions. "
            "Unknown detection is not guaranteed. Use with consent."
        )
        self.footer.setWordWrap(True)
        self.footer.setObjectName("muted")
        outer.addWidget(self.footer)

        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self.tick)
        QTimer.singleShot(100, self.check_api)

    def check_api(self):
        try:
            health(API_URL)
            self.api_online = True
            self.api_label.setText("● API online · two production models")
            self.api_label.setStyleSheet("color: #68dba8;")
        except Exception as exc:
            self.api_online = False
            self.api_label.setText("● API offline")
            self.api_label.setStyleSheet("color: #fb8b8b;")
            self.camera_label.setText(f"Start the FastAPI backend first: {str(exc)[:110]}")

    def start_camera(self):
        if self.camera is not None:
            return
        if not self.api_online:
            self.check_api()
        if not self.api_online:
            QMessageBox.warning(self, "Backend offline", f"Start the FaceOps API at {API_URL} first.")
            return
        camera = cv2.VideoCapture(self.camera_index)
        if not camera.isOpened():
            camera.release()
            QMessageBox.warning(self, "Webcam unavailable", f"Could not open camera index {self.camera_index}.")
            return
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera = camera
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.camera_label.setText("Camera running · largest detected face is analyzed")
        self.timer.start()

    def stop_camera(self):
        self.timer.stop()
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.video.setPixmap(QPixmap())
        self.video.setText("Camera stopped\nNo video is stored")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.camera_label.setText("Camera stopped")
        self.model_a.clear_result()
        self.model_b.clear_result()
        self.agreement.setText("Agreement: waiting")

    def tick(self):
        if self.camera is None:
            return
        ok, frame = self.camera.read()
        if not ok:
            self.camera_label.setText("Cannot read webcam frame")
            return
        frame = cv2.flip(frame, 1)
        original_frame = frame.copy()  # Never send overlay boxes to the classifier
        boxes = self.detector.locate(frame)
        now = time.monotonic()
        if boxes:
            self.last_face = now
            for i, (x, y, w, h) in enumerate(boxes):
                color = (105, 220, 154) if i == 0 else (100, 130, 190)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if now - self.last_infer >= self.infer_interval and (
                self.worker is None or not self.worker.isRunning()
            ):
                crop = self.detector.crop(original_frame, boxes[0])
                if crop.size and min(crop.shape[:2]) >= 50:
                    success, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if success:
                        self.last_infer = now
                        self.start_inference(encoded.tobytes())
            self.camera_label.setText(
                f"Detected {len(boxes)} face(s) · comparing largest face" if len(boxes) > 1
                else "One face detected · live comparison active"
            )
        else:
            self.camera_label.setText("No frontal face detected")
            if now - self.last_face > 2:
                self.model_a.clear_result()
                self.model_b.clear_result()
                self.agreement.setText("Agreement: waiting for a face")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, channels = rgb.shape
        image = QImage(rgb.data, w, h, channels * w, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        self.video.setPixmap(pixmap.scaled(
            self.video.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def start_inference(self, jpeg_bytes):
        self.worker = InferenceThread(jpeg_bytes, self)
        self.worker.success.connect(self.on_inference)
        self.worker.failure.connect(self.on_inference_error)
        self.worker.start()

    def on_inference(self, result):
        if self.camera is None:
            return
        self.model_a.show_result(result["model_a"])
        self.model_b.show_result(result["model_b"])
        agreeing = bool(result.get("agreement", False))
        self.agreement.setText("Agreement: both models agree" if agreeing else "Agreement: models disagree")
        self.agreement.setStyleSheet("color: #75e0b0;" if agreeing else "color: #ffcf84;")

    def on_inference_error(self, message):
        if self.camera is not None:
            self.camera_label.setText(f"Inference error: {message[:130]}")
            self.agreement.setText("API request failed")

    def closeEvent(self, event):
        self.stop_camera()
        if self.worker is not None and self.worker.isRunning():
            # Requests time out; wait for the active worker before destroying QThread.
            self.worker.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    window = FaceOpsWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
