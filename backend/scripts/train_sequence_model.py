import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "backend" / "data" / "video_landmarks" / "authorized_video_manifest.csv"
DEFAULT_MODEL_PATH = ROOT / "backend" / "models" / "isl_bridge_lstm.keras"
DEFAULT_LABELS_PATH = ROOT / "backend" / "models" / "labels.json"
DEFAULT_METRICS_PATH = ROOT / "backend" / "data" / "training_metrics.json"
HAND_ORDER = ("Left", "Right")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a two-hand ISL sequence model from extracted landmark JSON files."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-style", choices=["leaf", "path"], default="leaf")
    parser.add_argument(
        "--min-samples-per-label",
        type=int,
        default=1,
        help="Keep only labels with at least this many samples in the manifest.",
    )
    parser.add_argument(
        "--include-prefix",
        action="append",
        default=[],
        help="Only keep labels starting with one of these prefixes, e.g. words/ or phrases/.",
    )
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--labels-output", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    return parser.parse_args()


def read_manifest(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_label(raw_label: str, style: str):
    cleaned = raw_label.strip().replace("\\", "/")
    if style == "path":
        return cleaned
    return cleaned.split("/")[-1]


def load_json_frames(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def get_hand(frame, target_name):
    for hand in frame.get("hands", []):
        if str(hand.get("label", "")).lower() == target_name.lower():
            return hand
    return None


def landmarks_to_points(landmarks):
    return np.array(
        [
            [
                float(point.get("x", 0.0)),
                float(point.get("y", 0.0)),
                float(point.get("z", 0.0)),
            ]
            for point in landmarks
        ],
        dtype=np.float32,
    )


def normalize_points(points):
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


def vectorize_frame(frame):
    hand_points = []
    for hand_name in HAND_ORDER:
        hand = get_hand(frame, hand_name)
        if hand is None:
            hand_points.append(np.zeros((21, 3), dtype=np.float32))
        else:
            hand_points.append(landmarks_to_points(hand.get("landmarks", [])))

    stacked = np.vstack(hand_points)
    return normalize_points(stacked).reshape(-1)


def vectorize_sequence(frames, sequence_length):
    vectors = [vectorize_frame(frame) for frame in frames]
    if not vectors:
        return np.zeros((sequence_length, 126), dtype=np.float32)

    data = np.stack(vectors).astype(np.float32)
    if data.shape[0] > sequence_length:
        indices = np.linspace(0, data.shape[0] - 1, num=sequence_length, dtype=int)
        return data[indices]

    if data.shape[0] < sequence_length:
        pad = np.repeat(data[:1], sequence_length - data.shape[0], axis=0)
        return np.concatenate([pad, data], axis=0)

    return data


def build_samples(rows, sequence_length, label_style):
    samples = []
    for row in rows:
        landmarks_path = ROOT / row["landmarks_path"]
        if not landmarks_path.exists():
            continue
        frames = load_json_frames(landmarks_path)
        label = normalize_label(row["label"], label_style)
        samples.append(
            {
                "label": label,
                "features": vectorize_sequence(frames, sequence_length),
                "frame_count": len(frames),
                "landmarks_path": row["landmarks_path"],
            }
        )
    return samples


def filter_rows(rows, min_samples_per_label, include_prefixes):
    include_prefixes = tuple(include_prefixes or [])
    filtered_rows = []
    for row in rows:
        label = row["label"].strip().replace("\\", "/")
        if include_prefixes and not label.startswith(include_prefixes):
            continue
        filtered_rows.append(row)

    label_counts = defaultdict(int)
    for row in filtered_rows:
        label_counts[row["label"].strip().replace("\\", "/")] += 1

    if min_samples_per_label <= 1:
        return filtered_rows

    return [
        row
        for row in filtered_rows
        if label_counts[row["label"].strip().replace("\\", "/")] >= min_samples_per_label
    ]


def split_samples(samples, validation_split, seed):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["label"]].append(sample)

    train_samples = []
    val_samples = []
    random_generator = random.Random(seed)

    for label_samples in grouped.values():
        random_generator.shuffle(label_samples)
        if len(label_samples) < 2:
            train_samples.extend(label_samples)
            continue

        val_count = max(1, int(round(len(label_samples) * validation_split)))
        if val_count >= len(label_samples):
            val_count = len(label_samples) - 1

        val_samples.extend(label_samples[:val_count])
        train_samples.extend(label_samples[val_count:])

    random_generator.shuffle(train_samples)
    random_generator.shuffle(val_samples)
    return train_samples, val_samples


def encode_samples(samples, label_to_index):
    features = np.stack([sample["features"] for sample in samples]).astype(np.float32)
    labels = np.array([label_to_index[sample["label"]] for sample in samples], dtype=np.int32)
    return features, labels


def build_model(sequence_length, feature_dim, num_classes, learning_rate):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(sequence_length, feature_dim)),
            tf.keras.layers.Masking(mask_value=0.0),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(128, return_sequences=True)
            ),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="isl_sequence_classifier",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_outputs(model, labels, metrics, model_output, labels_output, metrics_output):
    model_output.parent.mkdir(parents=True, exist_ok=True)
    labels_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)

    model.save(model_output)
    labels_output.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    rows = filter_rows(
        read_manifest(args.manifest),
        min_samples_per_label=args.min_samples_per_label,
        include_prefixes=args.include_prefix,
    )
    samples = build_samples(rows, args.sequence_length, args.label_style)
    if len(samples) < 2:
        raise ValueError("Not enough extracted samples to train. Add more labeled videos first.")

    labels = sorted({sample["label"] for sample in samples})
    label_to_index = {label: index for index, label in enumerate(labels)}

    train_samples, val_samples = split_samples(
        samples, args.validation_split, args.seed
    )
    if not train_samples or not val_samples:
        raise ValueError(
            "Need at least two samples per class to create a training and validation split."
        )

    x_train, y_train = encode_samples(train_samples, label_to_index)
    x_val, y_val = encode_samples(val_samples, label_to_index)

    model = build_model(
        sequence_length=args.sequence_length,
        feature_dim=x_train.shape[-1],
        num_classes=len(labels),
        learning_rate=args.learning_rate,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            mode="max",
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)
    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)

    metrics = {
        "train_samples": len(train_samples),
        "validation_samples": len(val_samples),
        "labels": labels,
        "sequence_length": args.sequence_length,
        "feature_dim": int(x_train.shape[-1]),
        "train_loss": float(train_loss),
        "train_accuracy": float(train_accuracy),
        "validation_loss": float(val_loss),
        "validation_accuracy": float(val_accuracy),
        "history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
    }

    save_outputs(
        model,
        labels,
        metrics,
        args.model_output,
        args.labels_output,
        args.metrics_output,
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
