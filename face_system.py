# face_system.py

import cv2
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from config import (
    FACE_XML_PATH,
    DATA_DIR,
    MODEL_PATH,
    LOGS_PATH,
    MIN_FACE_SIZE,
    CONFIDENCE_THRESHOLD,
)
from face_db import FaceDatabase


class FaceSystem:
    def __init__(self):
        # Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(FACE_XML_PATH)

        # LBPH face recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        # Paths
        self.data_dir = Path(DATA_DIR)
        self.model_path = Path(MODEL_PATH)
        self.logs_path = Path(LOGS_PATH)

        # Database
        self.db = FaceDatabase()

        # Stats
        self.stats = {
            "total_detections": 0,
            "recognized_faces": 0,
            "unknown_faces": 0,
        }

    # ---------- LBPH train / load ----------

    def train_lbph(self) -> bool:
        # Reload labels to ensure we have latest data
        self.db.load_labels()
        
        faces = []
        labels = []

        for name in self.db.label_ids.keys():
            person_dir = self.db.get_person_dir(name)
            label_id = self.db.label_ids[name]

            for img_file in person_dir.glob("*.jpg"):
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                faces.append(img)
                labels.append(label_id)

        if not faces:
            print("❌ No training data found.")
            return False

        self.recognizer.train(faces, np.array(labels))
        self.recognizer.write(str(self.model_path))
        self.db.save_labels()
        print(f"✅ LBPH model trained with {len(faces)} images.")
        print(f"   Labels: {self.db.label_ids}")
        return True

    def load_lbph(self) -> bool:
        if not self.model_path.exists():
            print("❌ LBPH model not found. Train first.")
            return False
        self.recognizer.read(str(self.model_path))
        print("✅ LBPH model loaded.")
        return True

    # ---------- Add person ----------

    def add_person(self, name: str, num_samples: int = 50):
        label_id = self.db.get_or_create_label(name)
        person_dir = self.db.get_person_dir(name)

        cap = cv2.VideoCapture(0)
        count = 0

        print(f"\nAdding '{name}' (label {label_id}) ...")
        print(f"📁 Images will be saved to: {person_dir}")

        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (200, 200))

                img_path = person_dir / f"{name}_{count}.jpg"
                cv2.imwrite(str(img_path), roi)
                count += 1

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{count}/{num_samples}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Add Person (Q to stop)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.db.save_labels()
        print(f"✅ Captured {count} images for {name}.")
        print(f"⚠️  Don't forget to TRAIN the model (option 2) before testing recognition!")

    # ---------- Logging ----------

    def _log_event(self, name, confidence):
        entry = {
            "name": name,
            "confidence": float(confidence),
            "timestamp": datetime.now().timestamp(),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        logs = []
        if self.logs_path.exists():
            with open(self.logs_path, "r", encoding="utf-8") as f:
                logs = json.load(f)

        logs.append(entry)
        logs = logs[-1000:]

        with open(self.logs_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)

    # ---------- Live recognition (LBPH only) ----------

    def start(self):
        # Reload labels to ensure we have latest data
        self.db.load_labels()
        
        if not self.load_lbph():
            return

        try:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=MIN_FACE_SIZE,
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )

                self.stats["total_detections"] += len(faces)

                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_resized = cv2.resize(roi_gray, (200, 200))

                    label_id, raw_conf = self.recognizer.predict(roi_resized)

                    if raw_conf <= CONFIDENCE_THRESHOLD and label_id in self.db.names:
                        name = self.db.names[label_id]
                        color = (0, 255, 0)
                        confidence = 100 - raw_conf  # Convert for display
                        self.stats["recognized_faces"] += 1
                        self._log_event(name, confidence)
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)
                        confidence = raw_conf

                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(
                        frame,
                        f"{name} ({confidence:.1f})",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                cv2.imshow("LBPH Face Recognition", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

        except Exception as e:
            print(f"❌ Error during recognition: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Recognition stopped. Returning to menu...")
