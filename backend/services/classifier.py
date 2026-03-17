import json
from pathlib import Path

import numpy as np

try:
    import tensorflow as tf
except Exception:
    tf = None


class GestureClassifier:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[1]
        self.model_path = self.base_dir / "models" / "isl_bridge_lstm.keras"
        self.labels_path = self.base_dir / "models" / "labels.json"
        self.model = None
        self.labels = ["Hello", "Yes", "Stop", "Help", "Need Water"]
        self.model_type = "heuristic"
        self._load_model()

    def _load_model(self):
        if not tf or not self.model_path.exists():
            return

        self.model = tf.keras.models.load_model(self.model_path)
        if self.labels_path.exists():
            self.labels = json.loads(self.labels_path.read_text(encoding="utf-8"))
        self.model_type = "tensorflow"

    def predict(self, landmarks):
        if self.model is not None:
            return self._predict_tensorflow(landmarks)
        return self._predict_heuristic(landmarks)

    def _predict_tensorflow(self, landmarks):
        vector = []
        for point in landmarks:
            vector.extend(
                [
                    float(point.get("x", 0)),
                    float(point.get("y", 0)),
                    float(point.get("z", 0)),
                ]
            )

        data = np.array(vector, dtype=np.float32).reshape(1, 1, -1)
        scores = self.model.predict(data, verbose=0)[0]
        index = int(np.argmax(scores))
        return {
            "label": self.labels[index],
            "confidence": float(scores[index]),
            "source": "tensorflow",
        }

    def _predict_heuristic(self, landmarks):
        points = np.array(
            [
                [
                    float(point.get("x", 0)),
                    float(point.get("y", 0)),
                    float(point.get("z", 0)),
                ]
                for point in landmarks
            ],
            dtype=np.float32,
        )

        wrist = points[0]
        thumb_tip = points[4]
        index_tip = points[8]
        middle_tip = points[12]
        ring_tip = points[16]
        pinky_tip = points[20]

        index_extended = self._finger_extended(points, 8, 6)
        middle_extended = self._finger_extended(points, 12, 10)
        ring_extended = self._finger_extended(points, 16, 14)
        pinky_extended = self._finger_extended(points, 20, 18)
        thumb_extended = abs(thumb_tip[0] - points[3][0]) > 0.04

        finger_count = sum(
            [
                thumb_extended,
                index_extended,
                middle_extended,
                ring_extended,
                pinky_extended,
            ]
        )

        spread = np.mean(
            [
                np.linalg.norm(index_tip - middle_tip),
                np.linalg.norm(middle_tip - ring_tip),
                np.linalg.norm(ring_tip - pinky_tip),
            ]
        )
        palm_height = np.mean(points[[5, 9, 13, 17], 1])

        if finger_count >= 4 and spread > 0.06:
            return {"label": "Hello", "confidence": 0.93, "source": "heuristic"}

        if thumb_tip[1] < wrist[1] - 0.12 and not any(
            [index_extended, middle_extended, ring_extended, pinky_extended]
        ):
            return {"label": "Yes", "confidence": 0.9, "source": "heuristic"}

        if finger_count == 0 and np.mean(points[[8, 12, 16, 20], 1]) > palm_height:
            return {"label": "Stop", "confidence": 0.86, "source": "heuristic"}

        if index_extended and finger_count <= 2 and not middle_extended:
            return {"label": "Help", "confidence": 0.84, "source": "heuristic"}

        if index_extended and middle_extended and finger_count <= 3:
            return {"label": "Need Water", "confidence": 0.82, "source": "heuristic"}

        return {
            "label": "Gesture not recognised",
            "confidence": 0.45,
            "source": "heuristic",
        }

    @staticmethod
    def _finger_extended(points, tip_index, pip_index):
        return points[tip_index][1] < points[pip_index][1] - 0.02
