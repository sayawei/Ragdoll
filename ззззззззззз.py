import sys
import os
import sqlite3
import threading
import logging
from pathlib import Path
import webbrowser
import time

import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import pyqtSignal, QThread

from tensorflow.keras.models import load_model
import pyttsx3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PetBreedApp")

DB_PATH = Path("pets.db")
MODEL_PATH = Path("cat_dog_breed_model.h5")
CLASS_INDICES_PATH = Path("cat_dog_class_indices.npy")

class VideoInferenceWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    prediction_ready = pyqtSignal(str, float)
    status_message = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, source=0, model_path=MODEL_PATH, class_inds_path=CLASS_INDICES_PATH, parent=None):
        super().__init__(parent)
        self.source = source
        self._running = True
        self._paused = False
        self.model_path = Path(model_path)
        self.class_inds_path = Path(class_inds_path)
        self.capture = None
        self.model = None
        self.class_names = []
        self.input_size = (128, 128)
        self.last_time = time.time()
        self.fps = 0.0
        self.conf_threshold = 0.4

    def run(self):
        try:
            self.status_message.emit("Loading model...")
            if not self.model_path.exists():
                self.status_message.emit(f"Model file not found: {self.model_path}")
                logger.error("Model not found")
                self.finished_signal.emit()
                return

            self.model = load_model(str(self.model_path))
            logger.info("Model loaded")

            try:
                shape = self.model.input_shape
                if len(shape) == 4:
                    _, h, w, c = shape
                    if h and w:
                        self.input_size = (w, h)
                logger.info(f"Model input shape deduced: {self.input_size}")
            except Exception:
                logger.exception("Can't deduce model input shape, using default.")

            if self.class_inds_path.exists():
                mapping = np.load(str(self.class_inds_path), allow_pickle=True).item()
                inv = {v: k for k, v in mapping.items()}
                self.class_names = [inv[i] for i in sorted(inv.keys())]
                logger.info(f"Loaded {len(self.class_names)} classes")
            else:
                self.status_message.emit("Class indices file not found.")
                logger.warning("class indices not found")

            self.capture = cv2.VideoCapture(self.source)
            if not self.capture.isOpened():
                self.status_message.emit(f"Cannot open video source {self.source}")
                logger.error("Cannot open video source")
                self.finished_signal.emit()
                return
            self.status_message.emit("Started video capture")

            while self._running:
                if self._paused:
                    time.sleep(0.05)
                    continue

                ret, frame = self.capture.read()
                if not ret or frame is None:
                    time.sleep(0.02)
                    continue

                now = time.time()
                dt = now - self.last_time
                if dt > 0:
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
                self.last_time = now

                self.frame_ready.emit(frame)

                try:
                    img = cv2.resize(frame, self.input_size, interpolation=cv2.INTER_AREA)
                    img = img.astype("float32") / 255.0
                    if img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    batch = np.expand_dims(img, axis=0)

                    preds = self.model.predict(batch)
                    class_idx = int(np.argmax(preds))
                    confidence = float(np.max(preds))

                    class_name = self.class_names[class_idx] if self.class_names else str(class_idx)

                    if confidence >= self.conf_threshold:
                        self.prediction_ready.emit(class_name, confidence)
                    else:
                        self.prediction_ready.emit("Unknown", confidence)
                except Exception as e:
                    logger.exception("Inference error: %s", e)
                    self.status_message.emit("Inference error")

                time.sleep(0.01)
        finally:
            if self.capture:
                self.capture.release()
            self.status_message.emit("Worker stopped")
            self.finished_signal.emit()

    def stop(self):
        self._running = False

    def pause(self, to=None):
        if to is None:
            self._paused = not self._paused
        else:
            self._paused = bool(to)

    def set_confidence_threshold(self, t: float):
        self.conf_threshold = float(t)

def speak_text_threadsafe(text):
    def _speak(txt):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            try:
                voices = engine.getProperty('voices')
                for v in voices:
                    if "english" in v.name.lower() or "en" in v.id.lower():
                        engine.setProperty('voice', v.id)
                        break
            except Exception:
                logger.debug("Can't set voice")
            engine.say(txt)
            engine.runAndWait()
        except Exception:
            logger.exception("TTS failed")

    th = threading.Thread(target=_speak, args=(text,), daemon=True)
    th.start()

class PetBreedApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🐾 Cat & Dog Breed Recognition")
        self.resize(1200, 700)

        # DB connection (main thread)
        self.conn = sqlite3.connect(str(DB_PATH))
        self._ensure_db()

        # UI
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # Video display
        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.video_label, 2)

        # Right panel
        right_panel = QtWidgets.QVBoxLayout()

        self.result_label = QtWidgets.QLabel("Detected: None")
        self.info_label = QtWidgets.QLabel("Breed Info: ...")
        self.quality_label = QtWidgets.QLabel("Image Quality: OK")
        self.fps_label = QtWidgets.QLabel("FPS: 0.0")
        for lbl in (self.result_label, self.info_label, self.quality_label, self.fps_label):
            lbl.setWordWrap(True)
            right_panel.addWidget(lbl)

        # Search
        self.search_input = QtWidgets.QLineEdit(placeholderText="Search breed...")
        self.search_input.textChanged.connect(self.search_breeds)
        self.search_results = QtWidgets.QListWidget()
        self.search_results.itemClicked.connect(self.select_breed_from_search)
        right_panel.addWidget(self.search_input)
        right_panel.addWidget(self.search_results)

        # Buttons row
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_speak = QtWidgets.QPushButton("🔊 Speak")
        self.btn_wiki = QtWidgets.QPushButton("🌐 Wikipedia")
        self.btn_add = QtWidgets.QPushButton("➕ Add Breed")
        self.btn_pause = QtWidgets.QPushButton("⏯ Pause")
        self.btn_snapshot = QtWidgets.QPushButton("📸 Snapshot")
        btns = [self.btn_speak, self.btn_wiki, self.btn_add, self.btn_pause, self.btn_snapshot]
        for b in btns:
            btn_layout.addWidget(b)
        right_panel.addLayout(btn_layout)

        # Confidence slider
        self.conf_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(40)
        self.conf_slider.valueChanged.connect(self.on_conf_change)
        right_panel.addWidget(QtWidgets.QLabel("Confidence threshold"))
        right_panel.addWidget(self.conf_slider)

        # Status
        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)

        right_panel.addStretch()
        main_layout.addLayout(right_panel, 1)

        # signals
        self.btn_speak.clicked.connect(self.speak_info)
        self.btn_wiki.clicked.connect(self.open_wikipedia)
        self.btn_add.clicked.connect(self.add_breed)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_snapshot.clicked.connect(self.take_snapshot)

        # Worker thread
        self.worker = VideoInferenceWorker(source=0)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.prediction_ready.connect(self.on_prediction_ready)
        self.worker.status_message.connect(self.on_status_message)
        self.worker.finished_signal.connect(self.on_worker_finished)
        self.worker.start()

        # state
        self.current_prediction = ("None", 0.0)
        self.latest_frame = None

    def _ensure_db(self):
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS breeds (
                name TEXT PRIMARY KEY,
                info TEXT
            )
        """)
        default_breed_info = {
            'Siamese': 'Elegant, social cats. Require attention and interactive play. Grooming minimal, but sensitive to cold.',
            'Persian': 'Calm and affectionate. Regular grooming required due to long fur. Ideal indoor cats.',
            'Maine Coon': 'Friendly giant cat. Requires moderate grooming. Loves play and social interaction.',
            'Bulldog': 'Calm and loyal dog. Moderate exercise. Regular skin fold cleaning required.',
            'Beagle': 'Active and curious. Needs daily walks and mental stimulation.',
            'German Shepherd': 'Intelligent, loyal, and trainable. Requires daily exercise and mental challenge.',
            'Golden Retriever': 'Friendly, gentle, highly trainable. Daily exercise and grooming needed.',
            'Sphynx': 'Hairless cat. Sensitive skin, needs regular bathing. Social and energetic.',
            'Poodle': 'Intelligent, hypoallergenic dog. Regular grooming and mental stimulation required.',
            'Abyssinian': 'Active, intelligent, curious, loves climbing high places. Minimal grooming needed. Origin: Considered one of the oldest breeds, possibly linked to Egypt. Fact: Abyssinians rarely stay still — real cat parkour specialists.',
            'Bombay': 'Very affectionate, social, enjoys being close to humans. Easy grooming. Origin: Developed in the USA as a “mini panther cat.” Fact: In good light, their eyes can shine as if lit from within.',
            'British Shorthair': 'Calm, independent, gentle. Coat requires occasional brushing. Origin: UK, one of the oldest European breeds. Fact: The famous “smiling cat” in Alice in Wonderland is believed to be a British Shorthair.',
            'Egyptian Mau': 'Fast, graceful, very loyal. Short coat, minimal grooming. Origin: Ancient Egypt. Fact: The only natural breed with a spotted coat.',
            'Ragdoll': 'Very soft, calm, loves being held. Soft coat, regular grooming needed. Origin: USA, 1960s. Fact: They go limp when picked up — hence the name “Ragdoll.”'
        }
        for name, info in default_breed_info.items():
            c.execute("INSERT OR IGNORE INTO breeds (name, info) VALUES (?, ?)", (name, info))
        self.conn.commit()

    def on_frame_ready(self, frame: np.ndarray):
        self.latest_frame = frame
        # Overlay FPS and label
        display = frame.copy()
        fps_text = f"FPS: {self.worker.fps:.1f}"
        cv2.putText(display, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # Convert to QImage
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio,
                                                   QtCore.Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)
        self.fps_label.setText(f"FPS: {self.worker.fps:.1f}")

    def on_prediction_ready(self, class_name: str, confidence: float):
        self.current_prediction = (class_name, confidence)
        self.result_label.setText(f"Detected: {class_name} ({confidence:.2f})")
        info = self.get_breed_info(class_name)
        self.info_label.setText(f"Breed Info: {info}")

    def on_status_message(self, msg: str):
        self.status_bar.showMessage(msg, 5000)

    def on_worker_finished(self):
        self.status_bar.showMessage("Worker finished", 3000)

    def speak_info(self):
        class_name, confidence = self.current_prediction
        text = f"{class_name}. Confidence {confidence:.2f}. " + self.get_breed_info(class_name)
        speak_text_threadsafe(text)

    def open_wikipedia(self):
        class_name, _ = self.current_prediction
        if not class_name or class_name == "Unknown" or class_name == "None":
            QtWidgets.QMessageBox.information(self, "No breed", "No detected breed to open on Wikipedia.")
            return
        webbrowser.open(f"https://en.wikipedia.org/wiki/{class_name.replace(' ', '_')}")

    def add_breed(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Add Breed", "Enter breed name:")
        if ok and name:
            info, ok2 = QtWidgets.QInputDialog.getMultiLineText(self, "Breed Info", f"Enter info for {name}:")
            if ok2 and info:
                c = self.conn.cursor()
                c.execute("INSERT OR REPLACE INTO breeds (name, info) VALUES (?, ?)", (name, info))
                self.conn.commit()
                QtWidgets.QMessageBox.information(self, "Success", f"{name} added to database.")
                self.search_breeds()

    def toggle_pause(self):
        self.worker.pause()
        self.btn_pause.setText("▶ Resume" if self.worker._paused else "⏯ Pause")

    def take_snapshot(self):
        if self.latest_frame is None:
            QtWidgets.QMessageBox.information(self, "No frame", "No frame available to save.")
            return
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save snapshot", "", "PNG Image (*.png);;JPEG (*.jpg)")
        if fname:
            cv2.imwrite(fname, self.latest_frame)
            QtWidgets.QMessageBox.information(self, "Saved", f"Snapshot saved to {fname}")

    def on_conf_change(self, val):
        thr = val / 100.0
        self.worker.set_confidence_threshold(thr)
        self.status_bar.showMessage(f"Confidence threshold set to {thr:.2f}", 2000)

    def search_breeds(self):
        q = self.search_input.text().strip().lower()
        self.search_results.clear()
        if not q:
            return
        c = self.conn.cursor()
        c.execute("SELECT name FROM breeds WHERE LOWER(name) LIKE ?", ('%' + q + '%',))
        rows = c.fetchall()
        self.search_results.addItems([r[0] for r in rows])

    def select_breed_from_search(self, item):
        name = item.text()
        info = self.get_breed_info(name)
        self.result_label.setText(f"Detected: {name} (Manual)")
        self.info_label.setText(f"Breed Info: {info}")

    def get_breed_info(self, breed_name):
        if not breed_name:
            return "No info available."
        c = self.conn.cursor()
        c.execute("SELECT info FROM breeds WHERE LOWER(name) = ?", (breed_name.lower(),))
        r = c.fetchone()
        return r[0] if r else "No info available."

    def closeEvent(self, event):
        self.worker.stop()
        self.worker.wait(timeout=2000)
        try:
            self.conn.close()
        except Exception:
            logger.exception("Error closing DB")
        event.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = PetBreedApp()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()