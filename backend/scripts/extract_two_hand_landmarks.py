import argparse
import csv
import json
from pathlib import Path

import cv2

try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except Exception as exc:  # pragma: no cover
    python = None
    vision = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "datasets" / "authorized_videos"
DEFAULT_OUTPUT = ROOT / "backend" / "data" / "video_landmarks"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract two-hand MediaPipe landmarks from locally available, authorized videos."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-every", type=int, default=3)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--model-path", type=Path, default=None)
    return parser.parse_args()


def default_model_path():
    return ROOT / "backend" / "models" / "hand_landmarker.task"


def create_landmarker(model_path: Path):
    if python is None or vision is None:
        raise RuntimeError(
            "mediapipe is not installed. Run `pip install -r backend/requirements-ml.txt`."
        ) from IMPORT_ERROR

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    return vision.HandLandmarker.create_from_options(options)


def iter_videos(input_dir: Path):
    if not input_dir.exists():
        return []
    return sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)


def handedness_label(category):
    return category.display_name or category.category_name or "Unknown"


def normalize_hand_result(result):
    hands = []
    for index, landmarks in enumerate(result.hand_landmarks):
        handedness = result.handedness[index][0] if index < len(result.handedness) else None
        hands.append(
            {
                "label": handedness_label(handedness) if handedness else f"Hand {index + 1}",
                "score": float(handedness.score) if handedness else 0.0,
                "landmarks": [
                    {"x": float(point.x), "y": float(point.y), "z": float(point.z)}
                    for point in landmarks
                ],
            }
        )
    hands.sort(key=lambda hand: hand["label"])
    return hands


def extract_video_frames(video_path: Path, landmarker, sample_every: int):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []

    frames = []
    frame_index = 0
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % sample_every == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp_image_from_array(rgb_frame)
            timestamp_ms = int((frame_index / fps) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            hands = normalize_hand_result(result)
            if hands:
                frames.append({"frame_index": frame_index, "hands": hands})

        frame_index += 1

    capture.release()
    return frames


def mp_image_from_array(rgb_frame):
    return __import__("mediapipe").Image(
        image_format=__import__("mediapipe").ImageFormat.SRGB,
        data=rgb_frame,
    )


def infer_label(video_path: Path, input_dir: Path):
    relative_parent = video_path.relative_to(input_dir).parent
    return relative_parent.as_posix() if str(relative_parent) != "." else video_path.stem


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_path or default_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Hand landmarker model not found at {model_path}. Download it first."
        )

    videos = iter_videos(input_dir)
    if not videos:
        print(
            json.dumps(
                {
                    "message": "No videos found.",
                    "input_dir": str(input_dir),
                },
                indent=2,
            )
        )
        return

    manifest_rows = []
    landmarker = create_landmarker(model_path)

    for video_path in videos:
        frames = extract_video_frames(video_path, landmarker, args.sample_every)
        if len(frames) < args.min_frames:
            continue

        label = infer_label(video_path, input_dir)
        output_name = video_path.with_suffix(".json").name
        relative_dir = video_path.relative_to(input_dir).parent
        sample_output_dir = output_dir / relative_dir
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = sample_output_dir / output_name
        output_path.write_text(json.dumps(frames), encoding="utf-8")

        manifest_rows.append(
            {
                "label": label,
                "video_path": video_path.relative_to(ROOT).as_posix(),
                "landmarks_path": output_path.relative_to(ROOT).as_posix(),
                "num_frames": len(frames),
            }
        )

    manifest_path = output_dir / "authorized_video_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "video_path", "landmarks_path", "num_frames"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(
        json.dumps(
            {
                "videos_seen": len(videos),
                "samples_written": len(manifest_rows),
                "manifest": manifest_path.relative_to(ROOT).as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
