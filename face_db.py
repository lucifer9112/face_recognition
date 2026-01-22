# face_db.py

import pickle
from pathlib import Path
from config import DATA_DIR, LABELS_PATH

class FaceDatabase:
    def __init__(self):
        self.data_dir = Path(DATA_DIR)
        self.data_dir.mkdir(exist_ok=True)
        self.labels_path = Path(LABELS_PATH)

        self.label_ids = {}
        self.names = {}
        self.load_labels()

    def load_labels(self):
        if self.labels_path.exists():
            with open(self.labels_path, "rb") as f:
                data = pickle.load(f)
                self.label_ids = data.get("label_ids", {})
                self.names = data.get("names", {})

    def save_labels(self):
        with open(self.labels_path, "wb") as f:
            pickle.dump(
                {"label_ids": self.label_ids, "names": self.names},
                f,
            )

    def get_or_create_label(self, name: str) -> int:
        if name not in self.label_ids:
            new_id = len(self.label_ids)
            self.label_ids[name] = new_id
            self.names[new_id] = name
        return self.label_ids[name]

    def get_person_dir(self, name: str) -> Path:
        person_dir = self.data_dir / name
        person_dir.mkdir(exist_ok=True)
        return person_dir
