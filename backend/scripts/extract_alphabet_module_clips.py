import argparse
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import pytesseract
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = Path(
    r"C:\Users\harsh\Downloads\Video\Module 2 .1 Understanding the Indian sign language manual alphabet..mp4"
)
DEFAULT_OUTPUT = ROOT / "datasets" / "authorized_videos" / "alphabet"
TESSERACT_PATH = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract alphabet clips from a long labeled ISL alphabet teaching video."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--min-duration", type=float, default=0.7)
    parser.add_argument("--pad-seconds", type=float, default=0.25)
    return parser.parse_args()


def ocr_letter(frame):
    height, width = frame.shape[:2]
    crop = frame[int(height * 0.15):int(height * 0.75), 0:int(width * 0.45)]
    crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    image = Image.fromarray(thresh)
    config = "--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    text = pytesseract.image_to_string(image, config=config).strip().upper()
    text = re.sub(r"[^A-Z]", "", text)
    return text[:1] if len(text) == 1 else None


def collect_detections(video_path: Path, sample_every: int):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    detections = []

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % sample_every == 0:
            label = ocr_letter(frame)
            if label:
                detections.append(
                    {
                        "frame_index": frame_index,
                        "time_seconds": frame_index / fps,
                        "label": label,
                    }
                )
        frame_index += 1

    capture.release()
    return detections, fps, frame_count


def consolidate_segments(detections, min_duration: float):
    if not detections:
        return []

    segments = []
    current = {
        "label": detections[0]["label"],
        "start": detections[0]["time_seconds"],
        "end": detections[0]["time_seconds"],
    }

    for detection in detections[1:]:
        gap = detection["time_seconds"] - current["end"]
        if detection["label"] == current["label"] and gap <= 0.8:
            current["end"] = detection["time_seconds"]
            continue

        if current["end"] - current["start"] >= min_duration:
            segments.append(current)
        current = {
            "label": detection["label"],
            "start": detection["time_seconds"],
            "end": detection["time_seconds"],
        }

    if current["end"] - current["start"] >= min_duration:
        segments.append(current)
    return segments


def export_segments(video_path: Path, segments, output_dir: Path, pad_seconds: float):
    counters = defaultdict(int)
    written = []

    for segment in segments:
        label = segment["label"]
        counters[label] += 1
        destination_dir = output_dir / label
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"module_auto_{counters[label]:03d}.mp4"
        start = max(0.0, segment["start"] - pad_seconds)
        duration = (segment["end"] - segment["start"]) + (pad_seconds * 2)

        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{duration:.3f}",
            "-c",
            "copy",
            str(destination),
        ]
        subprocess.run(command, check=True, capture_output=True)
        written.append(
            {
                "label": label,
                "path": destination.relative_to(ROOT).as_posix(),
                "start": round(start, 3),
                "duration": round(duration, 3),
            }
        )
    return written


def main():
    args = parse_args()
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_PATH)

    detections, _, _ = collect_detections(args.video, args.sample_every)
    segments = consolidate_segments(detections, args.min_duration)
    written = export_segments(args.video, segments, args.output_dir, args.pad_seconds)

    print(
        {
            "detections": len(detections),
            "segments": len(segments),
            "written": len(written),
            "sample": written[:10],
        }
    )


if __name__ == "__main__":
    main()
