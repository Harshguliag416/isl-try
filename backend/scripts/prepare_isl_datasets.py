import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = ROOT / "datasets"
OUTPUT_ROOT = ROOT / "backend" / "data"

STATIC_DATASET_ROOT = (
    DATASETS_ROOT
    / "archive2"
    / "dataset"
    / "Adults ISL Images"
    / "Adults ISL images in Full Sleeves"
    / "English Alphabet"
)
SEQUENCE_DATASET_ROOT = (
    DATASETS_ROOT
    / "archive4"
    / "ISL_CSLRT_Corpus"
    / "ISL_CSLRT_Corpus"
    / "Frames_Sentence_Level"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_static_samples():
    samples = []
    if not STATIC_DATASET_ROOT.exists():
        return samples

    for class_dir in sorted(path for path in STATIC_DATASET_ROOT.iterdir() if path.is_dir()):
        image_paths = sorted(
            path for path in class_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image_path in image_paths:
            samples.append(
                {
                    "task": "static_sign",
                    "label": class_dir.name.strip(),
                    "sample_id": image_path.stem,
                    "path": image_path.relative_to(ROOT).as_posix(),
                }
            )
    return samples


def collect_sequence_samples():
    samples = []
    if not SEQUENCE_DATASET_ROOT.exists():
        return samples

    for phrase_dir in sorted(path for path in SEQUENCE_DATASET_ROOT.iterdir() if path.is_dir()):
        for sequence_dir in sorted(path for path in phrase_dir.iterdir() if path.is_dir()):
            frame_paths = sorted(
                path for path in sequence_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not frame_paths:
                continue

            samples.append(
                {
                    "task": "phrase_sequence",
                    "label": phrase_dir.name.strip(),
                    "sample_id": sequence_dir.name.strip(),
                    "num_frames": len(frame_paths),
                    "directory": sequence_dir.relative_to(ROOT).as_posix(),
                    "first_frame": frame_paths[0].relative_to(ROOT).as_posix(),
                    "last_frame": frame_paths[-1].relative_to(ROOT).as_posix(),
                }
            )
    return samples


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(static_samples, sequence_samples):
    static_counts = Counter(sample["label"] for sample in static_samples)
    sequence_counts = Counter(sample["label"] for sample in sequence_samples)
    return {
        "static_dataset_root": STATIC_DATASET_ROOT.relative_to(ROOT).as_posix(),
        "sequence_dataset_root": SEQUENCE_DATASET_ROOT.relative_to(ROOT).as_posix(),
        "static_total_images": len(static_samples),
        "static_labels": len(static_counts),
        "sequence_total_clips": len(sequence_samples),
        "sequence_labels": len(sequence_counts),
        "top_static_labels": static_counts.most_common(10),
        "top_sequence_labels": sequence_counts.most_common(10),
    }


def main():
    static_samples = collect_static_samples()
    sequence_samples = collect_sequence_samples()

    write_csv(
        OUTPUT_ROOT / "isl_static_manifest.csv",
        static_samples,
        ["task", "label", "sample_id", "path"],
    )
    write_csv(
        OUTPUT_ROOT / "isl_sequence_manifest.csv",
        sequence_samples,
        ["task", "label", "sample_id", "num_frames", "directory", "first_frame", "last_frame"],
    )

    summary = build_summary(static_samples, sequence_samples)
    summary_path = OUTPUT_ROOT / "isl_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
