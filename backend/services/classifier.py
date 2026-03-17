import json
from pathlib import Path

import numpy as np

try:
    import tensorflow as tf
except Exception:
    tf = None


class GestureClassifier:
    DEFAULT_SEQUENCE_LENGTH = 12
    HAND_ORDER = ("Left", "Right")

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[1]
        self.general_model_path = self.base_dir / "models" / "isl_bridge_lstm.keras"
        self.general_labels_path = self.base_dir / "models" / "labels.json"
        self.alphabet_model_path = self.base_dir / "models" / "isl_alphabet.keras"
        self.alphabet_labels_path = self.base_dir / "models" / "isl_alphabet_labels.json"
        self.general_model = None
        self.alphabet_model = None
        self.general_labels = ["Hello", "Yes", "Stop", "Help", "Need Water"]
        self.alphabet_labels = []
        self.model_type = "heuristic"
        self.supported_targets = ["general"]
        self._load_models()

    def _load_models(self):
        if not tf:
            return

        if self.general_model_path.exists():
            self.general_model = tf.keras.models.load_model(
                self.general_model_path, compile=False
            )
            if self.general_labels_path.exists():
                self.general_labels = json.loads(
                    self.general_labels_path.read_text(encoding="utf-8")
                )
            self.model_type = "tensorflow"

        if self.alphabet_model_path.exists():
            self.alphabet_model = tf.keras.models.load_model(
                self.alphabet_model_path, compile=False
            )
            if self.alphabet_labels_path.exists():
                self.alphabet_labels = json.loads(
                    self.alphabet_labels_path.read_text(encoding="utf-8")
                )

        if self.alphabet_model is not None:
            self.supported_targets.append("alphabet")

    def predict(self, landmarks, target="general"):
        sequence = self._coerce_sequence(landmarks)
        if target == "alphabet" and self.alphabet_model is not None:
            return self._predict_alphabet(sequence[-1])
        if self.general_model is not None:
            return self._predict_tensorflow(sequence)
        return self._predict_heuristic(sequence[-1])

    def _predict_tensorflow(self, sequence):
        frames = np.stack([self._vectorize_frame(frame) for frame in sequence]).astype(
            np.float32
        )
        data = self._prepare_tensorflow_input(frames, self.general_model)
        scores = self.general_model.predict(data, verbose=0)[0]
        index = int(np.argmax(scores))
        return {
            "label": self.general_labels[index],
            "confidence": float(scores[index]),
            "source": "tensorflow",
        }

    def _predict_alphabet(self, frame):
        hand = self._primary_hand(frame)
        if hand is None:
            return {
                "label": "nothing",
                "confidence": 0.0,
                "source": "alphabet_model",
            }

        vector = self._landmarks_to_points(hand).reshape(-1).astype(np.float32)
        scores = self.alphabet_model.predict(vector.reshape(1, -1), verbose=0)[0]
        index = int(np.argmax(scores))
        return {
            "label": self.alphabet_labels[index],
            "confidence": float(scores[index]),
            "source": "alphabet_model",
        }

    def _predict_heuristic(self, frame):
        landmarks = self._primary_hand(frame)
        if landmarks is None:
            return {
                "label": "Gesture not recognised",
                "confidence": 0.0,
                "source": "heuristic",
            }

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

    def _coerce_sequence(self, landmarks):
        if not isinstance(landmarks, list) or not landmarks:
            raise ValueError("Expected at least one frame of landmarks.")

        first = landmarks[0]
        if isinstance(first, dict):
            if "x" in first:
                return [{"hands": [{"label": "Unknown", "landmarks": self._coerce_landmarks(landmarks)}]}]
            if "hands" in first:
                return [self._coerce_frame(frame) for frame in landmarks if frame]

        if isinstance(first, list):
            return [
                {"hands": [{"label": "Unknown", "landmarks": self._coerce_landmarks(frame)}]}
                for frame in landmarks
                if frame
            ]

        raise ValueError("Invalid landmarks payload.")

    def _coerce_frame(self, frame):
        if not isinstance(frame, dict):
            raise ValueError("Each frame must be an object with hand data.")

        hands = frame.get("hands", [])
        if not isinstance(hands, list) or not hands:
            raise ValueError("Each frame must contain at least one hand.")

        normalized_hands = []
        for hand in hands:
            if not isinstance(hand, dict):
                raise ValueError("Invalid hand payload.")
            normalized_hands.append(
                {
                    "label": str(hand.get("label", "Unknown")).strip() or "Unknown",
                    "landmarks": self._coerce_landmarks(hand.get("landmarks", [])),
                }
            )

        normalized_hands.sort(key=lambda hand: self._hand_sort_key(hand["label"]))
        return {"hands": normalized_hands}

    def _coerce_landmarks(self, frame):
        if not isinstance(frame, list) or len(frame) != 21:
            raise ValueError("Each hand must contain 21 landmarks.")
        return frame

    def _vectorize_frame(self, frame):
        points_by_hand = []
        for hand_name in self.HAND_ORDER:
            hand = self._get_hand(frame, hand_name)
            if hand is None:
                points_by_hand.append(np.zeros((21, 3), dtype=np.float32))
                continue
            points_by_hand.append(self._landmarks_to_points(hand["landmarks"]))

        return self._normalize_points(np.vstack(points_by_hand)).reshape(-1)

    def _prepare_tensorflow_input(self, frames, model):
        input_shape = getattr(model, "input_shape", None)
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if not input_shape:
            return frames.reshape(1, frames.shape[0], frames.shape[1])

        rank = len(input_shape)
        if rank == 2:
            return frames[-1].reshape(1, -1)

        if rank == 3:
            required_steps = input_shape[1]
            if required_steps and frames.shape[0] != required_steps:
                frames = self._pad_or_trim_sequence(frames, required_steps)
            return frames.reshape(1, frames.shape[0], frames.shape[1])

        return frames.reshape(1, frames.shape[0], frames.shape[1])

    def _pad_or_trim_sequence(self, frames, required_steps):
        if frames.shape[0] > required_steps:
            return frames[-required_steps:]

        if frames.shape[0] == required_steps:
            return frames

        pad_count = required_steps - frames.shape[0]
        pad_frame = np.repeat(frames[:1], pad_count, axis=0)
        return np.concatenate([pad_frame, frames], axis=0)

    def _normalize_points(self, points):
        non_zero_rows = np.any(np.abs(points) > 1e-6, axis=1)
        active_points = points[non_zero_rows]
        if not len(active_points):
            return points

        wrist = active_points[0]
        shifted = points - wrist
        active_shifted = shifted[non_zero_rows]
        scale = np.max(np.linalg.norm(active_shifted[:, :2], axis=1))
        if scale < 1e-6:
            scale = 1.0
        return shifted / scale

    def _primary_hand(self, frame):
        right_hand = self._get_hand(frame, "Right")
        if right_hand is not None:
            return right_hand["landmarks"]

        left_hand = self._get_hand(frame, "Left")
        if left_hand is not None:
            return left_hand["landmarks"]

        hands = frame.get("hands", [])
        return hands[0]["landmarks"] if hands else None

    def _get_hand(self, frame, name):
        for hand in frame.get("hands", []):
            if hand["label"].lower() == name.lower():
                return hand
        return None

    def _hand_sort_key(self, label):
        normalized = label.strip().lower()
        preferred = [name.lower() for name in self.HAND_ORDER]
        if normalized in preferred:
            return (preferred.index(normalized), normalized)
        return (len(preferred), normalized)

    def _landmarks_to_points(self, landmarks):
        return np.array(
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

    @staticmethod
    def _finger_extended(points, tip_index, pip_index):
        return points[tip_index][1] < points[pip_index][1] - 0.02
