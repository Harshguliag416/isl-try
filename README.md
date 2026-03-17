# ISL Bridge

ISL Bridge is a browser-based Indian Sign Language communication app built for hackathon demos and public deployment.

It includes:

- `Mute -> Deaf`: sign to on-screen text
- `Mute -> Hearing`: sign to text and spoken audio
- `Hearing -> Deaf`: speech to large readable text
- Hindi and English UI toggle saved in `localStorage`
- MediaPipe hand tracking in the browser
- Flask prediction API with a working heuristic classifier and optional TensorFlow model loading
- Optional MongoDB Atlas conversation logging

## Deploy-Friendly Architecture

- `frontend/`: React + Vite + Tailwind static app
- `backend/`: Flask API for prediction and history

Recommended free deployment:

- Frontend: Vercel
- Backend: Render
- Database: MongoDB Atlas free tier

This keeps the app accessible from any phone, PC, or mobile browser using a public URL.

## Live Demo Recommendation

For the smoothest college demo:

- Use Chrome or Edge on laptop for all three modes
- Use Chrome on Android for the best camera + speech support
- iPhone Safari may support camera, but speech recognition support is limited
- Open the Render backend once before the presentation so a free instance can wake up

## Local Run

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python app.py
```

### Frontend

```bash
cd frontend
npm install
copy .env.example .env
npm run dev
```

## Production Deploy

### Backend on Render

1. Create a new Web Service from the `backend` folder.
2. Build command: `pip install -r requirements-ml.txt`
3. Start command: `gunicorn app:app`
4. Add environment variables from `.env.example`

If you want deployed TensorFlow or alphabet recognition, make sure the model files under `backend/models/` are present in the repo or otherwise available to the Render service at build/runtime.

### Frontend on Vercel

1. Import the `frontend` folder as a project.
2. Add `VITE_API_URL` pointing to your Render backend URL.
3. Deploy.

Example:

```env
VITE_API_URL=https://your-render-service.onrender.com
```

## Model Upgrade Path

The backend works immediately using a geometric heuristic classifier.

To switch to your trained ISL model later:

1. Install ML dependencies with `pip install -r requirements-ml.txt`
2. Place the Keras model at `backend/models/isl_bridge_lstm.keras`
3. Add `backend/models/labels.json` as a JSON array of class labels
4. Restart the backend

The app will automatically use TensorFlow predictions when the model is present.

## Dataset Preparation

If you have extracted ISL datasets under `datasets/archive2` and `datasets/archive4`, generate training manifests with:

```bash
python backend/scripts/prepare_isl_datasets.py
```

This writes:

- `backend/data/isl_static_manifest.csv`
- `backend/data/isl_sequence_manifest.csv`
- `backend/data/isl_dataset_summary.json`

Use the static manifest for alphabet or single-sign image classification and the sequence manifest for phrase-level video/frame training.

## Training From Videos You Have Rights To Use

You can also train from local videos that you recorded yourself or have permission to use from creators or permissive sources.

Recommended folder layout:

```text
datasets/authorized_videos/
  hello/
    clip1.mp4
    clip2.mp4
  need_water/
    clip1.mp4
```

Install ML dependencies:

```bash
pip install -r backend/requirements-ml.txt
```

Download MediaPipe's hand landmarker task file and place it at:

```text
backend/models/hand_landmarker.task
```

Then extract two-hand landmark sequences from your local videos:

```bash
python backend/scripts/extract_two_hand_landmarks.py
```

This writes JSON landmark sequences and:

- `backend/data/video_landmarks/authorized_video_manifest.csv`

Use only videos you own, recorded yourself, or are explicitly licensed to reuse. Avoid bulk-downloading random YouTube videos unless you have permission from the creator.

## Importing The Legacy Alphabet Model

If you have the legacy alphabet classifier files:

- `weights.npz`
- `label_map.json`

you can rebuild a compatible Keras model with:

```bash
python backend/scripts/import_legacy_alphabet_model.py
```

This writes:

- `backend/models/isl_alphabet.keras`
- `backend/models/isl_alphabet_labels.json`

That recovered model is useful for single-hand alphabet spelling. It is separate from the future two-hand word and sentence model.
