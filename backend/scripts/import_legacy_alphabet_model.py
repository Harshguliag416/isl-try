import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = Path(r"c:\Users\harsh\Downloads\weights.npz")
DEFAULT_LABEL_MAP = Path(r"c:\Users\harsh\Downloads\label_map.json")
DEFAULT_OUTPUT_MODEL = ROOT / "backend" / "models" / "isl_alphabet.keras"
DEFAULT_OUTPUT_LABELS = ROOT / "backend" / "models" / "isl_alphabet_labels.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild a compatible alphabet classifier from legacy NPZ weights."
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--output-model", type=Path, default=DEFAULT_OUTPUT_MODEL)
    parser.add_argument("--output-labels", type=Path, default=DEFAULT_OUTPUT_LABELS)
    return parser.parse_args()


def load_labels(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Expected label map JSON object.")
    return [label for _, label in sorted(raw.items(), key=lambda item: int(item[0]))]


def build_model(num_classes: int):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(63,), name="input_layer"),
            tf.keras.layers.Dense(256, activation="relu", name="dense"),
            tf.keras.layers.BatchNormalization(name="batch_normalization"),
            tf.keras.layers.Dropout(0.3, name="dropout"),
            tf.keras.layers.Dense(128, activation="relu", name="dense_1"),
            tf.keras.layers.BatchNormalization(name="batch_normalization_1"),
            tf.keras.layers.Dropout(0.3, name="dropout_1"),
            tf.keras.layers.Dense(64, activation="relu", name="dense_2"),
            tf.keras.layers.Dropout(0.2, name="dropout_2"),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="dense_3"),
        ],
        name="isl_alphabet_classifier",
    )
    model.build((None, 63))
    return model


def assign_weights(model, arrays):
    model.get_layer("dense").set_weights([arrays["arr_0"], arrays["arr_1"]])
    model.get_layer("batch_normalization").set_weights(
        [arrays["arr_2"], arrays["arr_3"], arrays["arr_4"], arrays["arr_5"]]
    )
    model.get_layer("dense_1").set_weights([arrays["arr_6"], arrays["arr_7"]])
    model.get_layer("batch_normalization_1").set_weights(
        [arrays["arr_8"], arrays["arr_9"], arrays["arr_10"], arrays["arr_11"]]
    )
    model.get_layer("dense_2").set_weights([arrays["arr_12"], arrays["arr_13"]])
    model.get_layer("dense_3").set_weights([arrays["arr_14"], arrays["arr_15"]])


def main():
    args = parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights file not found: {args.weights}")
    if not args.label_map.exists():
        raise FileNotFoundError(f"Label map file not found: {args.label_map}")

    labels = load_labels(args.label_map)
    arrays = np.load(args.weights)

    model = build_model(len(labels))
    assign_weights(model, arrays)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_labels.parent.mkdir(parents=True, exist_ok=True)

    model.save(args.output_model)
    args.output_labels.write_text(
        json.dumps(labels, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    reloaded = tf.keras.models.load_model(args.output_model, compile=False)
    print(
        json.dumps(
            {
                "saved_model": args.output_model.relative_to(ROOT).as_posix(),
                "saved_labels": args.output_labels.relative_to(ROOT).as_posix(),
                "input_shape": reloaded.input_shape,
                "output_shape": reloaded.output_shape,
                "labels": len(labels),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
