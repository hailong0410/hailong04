import sys
import cv2
import time
import queue
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QPushButton, QHBoxLayout, QVBoxLayout,
    QMessageBox
)
from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list

def recognize_faces(frame_rgb, mtcnn):
    boxes, probs = mtcnn.detect(frame_rgb, landmarks=False)
    if boxes is None or probs is None:
        return [], []

    boxes = boxes[probs > 0.9]
    faces = []
    valid_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        faces.append(frame_rgb[y1:y2, x1:x2, :])
        valid_boxes.append((x1, y1, x2, y2))
    return faces, valid_boxes

class RealTimeEmotionRecognizer:
    def __init__(self, model_name=None, device="cpu"):
        if model_name is None:
            model_name = get_model_list()[0]
        self.device = device
        self.recognizer = EmotiEffLibRecognizer(
            engine="onnx",
            model_name=model_name,
            device=device
        )
        self.mtcnn = MTCNN(keep_all=True, post_process=False, min_face_size=80, device=device)

    def process_frame(self, frame_bgr):
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces, boxes = recognize_faces(frame_rgb, self.mtcnn)

        if len(faces) == 0:
            return frame_bgr, None

        emotions = []
        for face in faces:
            emotion, _ = self.recognizer.predict_emotions([face], logits=False)
            emotions.append(emotion[0])  
        for (x1, y1, x2, y2), emo in zip(boxes, emotions):
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_bgr, emo, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

        return frame_bgr, emotions

class VideoThread(QThread):
    frame_processed = pyqtSignal(np.ndarray, object, float) 

    def __init__(self, source, model_name, device):
        super().__init__()
        self.source = source
        self.model_name = model_name
        self.device = device
        self.running = True
        self.frame_queue = queue.Queue(maxsize=1) 

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.frame_processed.emit(None, None, 0)
            return

        recognizer = RealTimeEmotionRecognizer(model_name=self.model_name, device=self.device)

        prev_time = time.time()
        frame_count = 0
        fps_ema = 0 
        alpha = 0.1  

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

            
            try:
                frame_to_process = self.frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            frame_processed, emotion = recognizer.process_frame(frame_to_process)
            frame_count += 1
            now = time.time()
            elapsed = now - prev_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            fps_ema = alpha * fps + (1 - alpha) * fps_ema  

            self.frame_processed.emit(frame_processed, emotion, fps_ema)

            prev_time = now
            self.msleep(5) 

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Emotion Recognition ")
        self.setGeometry(200, 200, 1200, 850)

        self.video_label = QLabel("Video sẽ hiển thị ở đây")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(1024, 768)
        self.video_label.setStyleSheet("""
            background-color: black; 
            border-radius: 10px; 
            border: 3px solid #4CAF50;
        """)

        # Controls in sidebar (only 3 buttons)
        self.btn_start = QPushButton(QIcon.fromTheme("media-playback-start"), "Bật Play")
        self.btn_start.setToolTip("Bắt đầu phát video hoặc webcam")
        self.btn_start.setStyleSheet(self.green_button_style())

        self.btn_stop = QPushButton(QIcon.fromTheme("media-playback-stop"), "Dừng")
        self.btn_stop.setToolTip("Dừng phát video")
        self.btn_stop.setStyleSheet(self.red_button_style())

        # Layout sidebar
        control_layout = QVBoxLayout()
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        control_layout.addStretch()

        control_widget = QWidget()
        control_widget.setLayout(control_layout)
        control_widget.setFixedWidth(180)
        control_widget.setStyleSheet("background-color: #f0f0f0; padding: 10px;")

        # Main layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(control_widget)
        main_layout.addWidget(self.video_label, stretch=1)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Variables
        self.video_thread = None

        # Connections
        self.btn_start.clicked.connect(self.start_capture)
        self.btn_stop.clicked.connect(self.stop_capture)

    def button_style(self):
        return """
            QPushButton {
                font-size: 15px;
                padding: 10px;
                border-radius: 8px;
                background-color: #2196F3;
                color: white;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """

    def green_button_style(self):
        return """
            QPushButton {
                font-size: 15px;
                padding: 10px;
                border-radius: 8px;
                background-color: #4CAF50;
                color: white;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
        """

    def red_button_style(self):
        return """
            QPushButton {
                font-size: 15px;
                padding: 10px;
                border-radius: 8px;
                background-color: #F44336;
                color: white;
            }
            QPushButton:hover {
                background-color: #D32F2F;
            }
        """

    def start_capture(self):
        if self.video_thread is not None and self.video_thread.isRunning():
            return

    
        source = 0
    
        model_name = get_model_list()[0]
        device = "cpu"

        self.video_thread = VideoThread(source, model_name, device)
        self.video_thread.frame_processed.connect(self.update_frame)
        self.video_thread.start()

    def update_frame(self, frame, emotion, fps):
        if frame is None:
            QMessageBox.critical(self, "Lỗi", "Không mở được nguồn video hoặc webcam!")
            self.stop_capture()
            return

        h, w, ch = frame.shape
        qt_img = QImage(frame.data, w, h, ch * w, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_img)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(scaled_pixmap)

    def stop_capture(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        self.video_label.clear()
        self.video_label.setText("Video sẽ hiển thị ở đây")

    def closeEvent(self, event):
        self.stop_capture()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
