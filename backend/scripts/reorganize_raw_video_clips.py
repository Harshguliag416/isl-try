import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2


SOURCE_ROOT = Path(r"H:\HACKATHON\Video clips")
DEST_ROOT = Path(r"h:\HACKATHON\new\datasets\authorized_videos")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ""}
SKIP_EXTENSIONS = {".gvid", ".ini"}


def is_video_file(path: Path) -> bool:
    if path.suffix.lower() in SKIP_EXTENSIONS:
        return False

    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        return False

    capture = cv2.VideoCapture(str(path))
    opened = capture.isOpened()
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if opened else 0
    capture.release()
    return opened and frame_count > 0


def normalize_text(value: str) -> str:
    cleaned = value.replace("\u2019", "'").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.replace("&", " and ")
    cleaned = re.sub(r"[()]+", " ", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9' ]+", " ", cleaned)
    cleaned = cleaned.lower().strip()
    cleaned = cleaned.replace("'", "")
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")


def infer_letter_from_name(name: str):
    patterns = [
        r"^([A-Za-z])$",
        r"^letter[_ ]*\(?([A-Za-z])\)?$",
    ]
    for pattern in patterns:
        match = re.match(pattern, name, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def categorize_source(path: Path):
    parts = path.relative_to(SOURCE_ROOT).parts
    person = parts[0]
    parent = parts[-2] if len(parts) > 1 else ""
    stem = path.stem if path.suffix else path.name
    normalized = normalize_text(stem)

    if person.lower() == "arjun" and parent.lower() in {"letter", "sign languages alphabets"}:
        letter = infer_letter_from_name(stem)
        if letter:
            return "alphabet", letter, person.lower()
        return None

    if person.lower() == "tushar":
        category = "phrases" if "_" in normalized else "words"
        return category, normalized, person.lower()

    if person.lower() == "vikas":
        if normalized.startswith("copy_"):
            return None
        category = "phrases" if "_" in normalized else "words"
        return category, normalized, person.lower()

    return None


def destination_extension(path: Path):
    return path.suffix.lower() if path.suffix else ".mp4"


def move_files():
    counters = defaultdict(int)
    moved = []
    skipped = []

    for path in sorted(SOURCE_ROOT.rglob("*")):
        if not path.is_file():
            continue

        if not is_video_file(path):
            skipped.append({"path": str(path), "reason": "not_usable_video"})
            continue

        categorization = categorize_source(path)
        if categorization is None:
            skipped.append({"path": str(path), "reason": "unclassified"})
            continue

        category, label, person = categorization
        counters[(category, label, person)] += 1
        clip_id = counters[(category, label, person)]
        extension = destination_extension(path)
        destination_dir = DEST_ROOT / category / label
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / f"{person}_{clip_id:03d}{extension}"

        shutil.move(str(path), str(destination_path))
        moved.append(
            {
                "source": str(path),
                "destination": str(destination_path),
                "category": category,
                "label": label,
            }
        )

    return moved, skipped


def main():
    moved, skipped = move_files()
    summary = {
        "moved_count": len(moved),
        "skipped_count": len(skipped),
        "moved_sample": moved[:10],
        "skipped_sample": skipped[:10],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
