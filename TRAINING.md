# ISL Model Training Guide

This project is set up so you can train your own ISL model from labeled videos.

## 1. Record The Videos

Use this folder layout:

```text
datasets/authorized_videos/
  alphabet/
    A/
      clip_001.mp4
      clip_002.mp4
    B/
      clip_001.mp4
  words/
    hello/
      clip_001.mp4
      clip_002.mp4
    help/
      clip_001.mp4
  phrases/
    i_need_water/
      clip_001.mp4
    can_you_repeat_that/
      clip_001.mp4
```

Guidelines:

- Record 20 to 50 clips per label to start.
- Use at least 3 different people.
- Use different backgrounds, clothes, and lighting.
- Keep both hands fully visible for word and phrase clips.
- Record from chest to above the head when possible.
- Keep the camera steady.
- Start each clip with hands visible and still for 1 second.
- Perform one sign per clip.
- Stop recording 1 second after the sign finishes.

Recommended clip lengths:

- Alphabet: 1 to 2 seconds
- Words: 2 to 4 seconds
- Phrases: 3 to 6 seconds

## 2. Extract Two-Hand Landmarks

Install ML dependencies:

```bash
pip install -r backend/requirements-ml.txt
```

Download the MediaPipe hand landmarker task file and place it at:

```text
backend/models/hand_landmarker.task
```

Run:

```bash
python backend/scripts/extract_two_hand_landmarks.py
```

This creates:

- `backend/data/video_landmarks/authorized_video_manifest.csv`
- one JSON landmark sequence per video

## 3. Train The Model

Run:

```bash
python backend/scripts/train_sequence_model.py
```

By default, this writes:

- `backend/models/isl_bridge_lstm.keras`
- `backend/models/labels.json`
- `backend/data/training_metrics.json`

The backend automatically loads this model for general word and phrase recognition.

## 4. Deploy The Model

Push the repo to GitHub and redeploy the backend on Render.

Then check:

```text
/api/health
```

You should see:

- `model_type: "tensorflow"`

## 5. Practical Training Plan

Start small instead of trying to train every possible sign at once.

Suggested order:

1. Alphabet only
2. 10 common words
3. 10 to 20 common phrases
4. Expand gradually after validation

For your project, a good first production set is:

- `hello`
- `help`
- `stop`
- `yes`
- `no`
- `thank_you`
- `water`
- `need_water`
- `hospital`
- `emergency`

Then add phrases like:

- `i_need_water`
- `please_help_me`
- `where_is_the_hospital`
- `can_you_repeat_that`
- `i_am_hungry`

## 6. Can You Train From A Folder Of Videos?

Yes.

If you give me a folder where each class has its own subfolder and the clips are labeled correctly, that is exactly the right format for training.

Example:

```text
datasets/authorized_videos/words/hello/clip_001.mp4
datasets/authorized_videos/words/hello/clip_002.mp4
datasets/authorized_videos/words/help/clip_001.mp4
datasets/authorized_videos/phrases/i_need_water/clip_001.mp4
```

That is trainable.

Accuracy depends on:

- number of clips per class
- diversity of people
- lighting and background variety
- whether both hands stay visible
- whether labels are clean and consistent
