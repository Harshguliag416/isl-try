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
2. Build command: `pip install -r requirements.txt`
3. Start command: `gunicorn app:app`
4. Add environment variables from `.env.example`

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
