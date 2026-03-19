import argparse
import json
import re
import subprocess
import unicodedata
from collections import defaultdict
from pathlib import Path

import whisper


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO_DIR = Path(r"C:\Users\harsh\Downloads\Video")
DEFAULT_TRANSCRIPT_DIR = ROOT / "tmp_ocr"
DEFAULT_OUTPUT_DIR = ROOT / "datasets" / "authorized_videos"


MODULE_CONFIGS = {
    "Module 1.2 Greeting and salutations in Indian Sign Language.mp4": {
        "transcript_name": "module_12_whisper.json",
        "targets": [
            {"label": "hello", "category": "words", "patterns": ["hello"]},
            {"label": "hi", "category": "words", "patterns": ["hi"]},
            {
                "label": "good_morning",
                "category": "phrases",
                "patterns": ["good morning"],
            },
            {
                "label": "good_afternoon",
                "category": "phrases",
                "patterns": ["good afternoon"],
            },
            {
                "label": "good_evening",
                "category": "phrases",
                "patterns": ["good evening"],
            },
            {
                "label": "good_night",
                "category": "phrases",
                "patterns": ["good night"],
            },
            {"label": "thank_you", "category": "words", "patterns": ["thank you"]},
            {"label": "welcome", "category": "words", "patterns": ["welcome"]},
            {"label": "bye", "category": "words", "patterns": ["bye"]},
            {
                "label": "see_you_again",
                "category": "phrases",
                "patterns": ["see you again"],
            },
            {
                "label": "see_you_tomorrow",
                "category": "phrases",
                "patterns": ["see you tomorrow"],
            },
        ],
    },
    "Module 1.3 Some polite useful phrases.mp4": {
        "transcript_name": "module_13_whisper.json",
        "targets": [
            {
                "label": "how_are_you",
                "category": "phrases",
                "patterns": ["how are you"],
            },
            {
                "label": "i_am_fine",
                "category": "phrases",
                "patterns": ["i'm fine", "im fine"],
            },
            {"label": "thank_you", "category": "words", "patterns": ["thank you"]},
            {"label": "please", "category": "words", "patterns": ["please"]},
            {
                "label": "nice_to_meet_you",
                "category": "phrases",
                "patterns": ["nice to meet you"],
            },
            {
                "label": "nice_to_see_you",
                "category": "phrases",
                "patterns": ["nice to see you"],
            },
            {"label": "sorry", "category": "words", "patterns": ["sorry"]},
            {"label": "welcome", "category": "words", "patterns": ["welcome"]},
            {
                "label": "happy_birthday",
                "category": "phrases",
                "patterns": ["happy birthday", "wish you a very happy birthday"],
            },
            {
                "label": "good_to_see_you_again",
                "category": "phrases",
                "patterns": ["good to see you again"],
            },
            {
                "label": "take_care",
                "category": "phrases",
                "patterns": ["take care"],
            },
            {
                "label": "have_a_good_day",
                "category": "phrases",
                "patterns": ["have a good day"],
            },
            {
                "label": "well_done",
                "category": "phrases",
                "patterns": ["well done"],
            },
            {
                "label": "excuse_me",
                "category": "phrases",
                "patterns": ["excuse me"],
            },
            {
                "label": "congratulations",
                "category": "words",
                "patterns": ["congratulations"],
            },
            {
                "label": "please_give_me_your_pen",
                "category": "phrases",
                "patterns": ["please give me your pen"],
            },
            {
                "label": "thank_you_for_your_help",
                "category": "phrases",
                "patterns": ["thank you for your help"],
            },
            {
                "label": "please_go_ahead",
                "category": "phrases",
                "patterns": ["please go ahead"],
            },
            {
                "label": "congratulations_on_your_success",
                "category": "phrases",
                "patterns": ["congratulations on your success"],
            },
            {
                "label": "i_am_sorry_for_this_mistake",
                "category": "phrases",
                "patterns": ["i'm sorry for this mistake", "im sorry for this mistake"],
            },
            {
                "label": "your_signing_skill_is_excellent",
                "category": "phrases",
                "patterns": ["your signing skill is excellent"],
            },
            {
                "label": "take_care_and_get_well_soon",
                "category": "phrases",
                "patterns": ["take care and get well soon"],
            },
            {
                "label": "please_take_care_of_your_health",
                "category": "phrases",
                "patterns": ["please take care of your health"],
            },
        ],
    },
    "Module 2.2 Introducing oneself (name, work, place, etc.).mp4": {
        "transcript_name": "module_22_whisper.json",
        "targets": [
            {
                "label": "hi",
                "category": "words",
                "patterns": ["hi"],
            },
            {
                "label": "hello",
                "category": "words",
                "patterns": ["hello"],
            },
            {
                "label": "good_morning",
                "category": "phrases",
                "patterns": ["good morning"],
            },
            {
                "label": "what_is_your_name",
                "category": "phrases",
                "patterns": ["what is your name"],
            },
            {
                "label": "what_do_you_do_for_your_life",
                "category": "phrases",
                "patterns": ["what do you do for your life"],
            },
            {
                "label": "i_am_a_teacher",
                "category": "phrases",
                "patterns": ["i am a teacher"],
            },
            {
                "label": "i_am_an_engineer",
                "category": "phrases",
                "patterns": ["i am an engineer"],
            },
            {
                "label": "i_am_a_doctor",
                "category": "phrases",
                "patterns": ["i am a doctor"],
            },
            {
                "label": "where_do_you_work",
                "category": "phrases",
                "patterns": ["where do you work"],
            },
            {
                "label": "where_do_you_live",
                "category": "phrases",
                "patterns": ["where do you live"],
            },
            {
                "label": "i_am_from_mumbai",
                "category": "phrases",
                "patterns": ["i am a mumbai", "i am from mumbai"],
            },
            {
                "label": "i_am_from_delhi",
                "category": "phrases",
                "patterns": ["i am from delhi"],
            },
            {
                "label": "my_native_place_is_tamil_nadu",
                "category": "phrases",
                "patterns": ["my native place is tamil nadu"],
            },
            {
                "label": "thank_you",
                "category": "words",
                "patterns": ["thank you"],
            },
        ],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract word/phrase clips from long ISL teaching videos using Whisper timestamps."
    )
    parser.add_argument(
        "--videos",
        nargs="*",
        type=Path,
        help="Specific video files to process. Defaults to all configured modules.",
    )
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--transcript-dir", type=Path, default=DEFAULT_TRANSCRIPT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="base")
    parser.add_argument("--pad-seconds", type=float, default=0.35)
    parser.add_argument("--force-transcribe", action="store_true")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\\s']", " ", text)
    return re.sub(r"\\s+", " ", text).strip()


def get_videos(args):
    if args.videos:
        return args.videos
    return [args.video_dir / name for name in MODULE_CONFIGS]


def load_or_create_transcript(video_path: Path, transcript_path: Path, model_name: str, force: bool):
    if transcript_path.exists() and not force:
        return json.loads(transcript_path.read_text(encoding="utf-8"))

    model = whisper.load_model(model_name)
    result = model.transcribe(str(video_path), fp16=False)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def match_segments(segments, targets):
    matches = []
    last_seen = {}

    for segment in segments:
        normalized = normalize_text(segment["text"])
        if not normalized:
            continue

        for target in targets:
            if not any(
                re.search(rf"(?<![a-z0-9]){re.escape(pattern)}(?![a-z0-9])", normalized)
                for pattern in target["patterns"]
            ):
                continue

            dedupe_key = (target["label"], round(segment["start"], 1))
            if dedupe_key in last_seen:
                continue

            matches.append(
                {
                    "category": target["category"],
                    "label": target["label"],
                    "start": float(segment["start"]),
                    "end": float(segment["end"]),
                    "text": normalized,
                }
            )
            last_seen[dedupe_key] = True
            break

    return matches


def export_matches(video_path: Path, matches, output_dir: Path, pad_seconds: float):
    counters = defaultdict(int)
    written = []

    for match in matches:
        category_dir = output_dir / match["category"] / match["label"]
        category_dir.mkdir(parents=True, exist_ok=True)
        if counters[(match["category"], match["label"])] == 0:
            existing = sorted(category_dir.glob("module_auto_*.mp4"))
            if existing:
                last_index = max(int(path.stem.split("_")[-1]) for path in existing)
                counters[(match["category"], match["label"])] = last_index
        counters[(match["category"], match["label"])] += 1
        destination = category_dir / f"module_auto_{counters[(match['category'], match['label'])]:03d}.mp4"

        start = max(0.0, match["start"] - pad_seconds)
        duration = (match["end"] - match["start"]) + (pad_seconds * 2)
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
                "category": match["category"],
                "label": match["label"],
                "path": destination.relative_to(ROOT).as_posix(),
                "start": round(start, 3),
                "duration": round(duration, 3),
                "text": match["text"],
            }
        )

    return written


def process_video(video_path: Path, args):
    config = MODULE_CONFIGS.get(video_path.name)
    if not config:
        raise ValueError(f"No extraction config found for {video_path.name}")

    transcript_path = args.transcript_dir / config["transcript_name"]
    transcript = load_or_create_transcript(
        video_path,
        transcript_path,
        model_name=args.model,
        force=args.force_transcribe,
    )
    matches = match_segments(transcript["segments"], config["targets"])
    written = export_matches(video_path, matches, args.output_dir, args.pad_seconds)

    return {
        "video": video_path.name,
        "matches": len(matches),
        "written": len(written),
        "sample": written[:10],
    }


def main():
    args = parse_args()
    summaries = [process_video(video, args) for video in get_videos(args)]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
