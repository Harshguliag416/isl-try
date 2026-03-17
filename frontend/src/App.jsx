import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import { useEffect, useMemo, useRef, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";
const MAX_SEQUENCE_FRAMES = 12;
const MIN_SEQUENCE_FRAMES = 6;
const STABLE_PREDICTION_WINDOW = 4;

const MODES = {
  signToText: "signToText",
  signToSpeech: "signToSpeech",
  speechToText: "speechToText",
};

const COPY = {
  en: {
    badge: "Real-Time Sign Language Translator",
    title: "ISL Bridge",
    subtitle:
      "One browser app for sign-to-text, sign-to-speech, and speech-to-text communication.",
    chooseMode: "Choose a mode",
    signToText: "Mute -> Deaf",
    signToSpeech: "Mute -> Hearing",
    speechToText: "Hearing -> Deaf",
    startCamera: "Start Signing",
    stopCamera: "Stop Camera",
    startListening: "Speak Now",
    stopListening: "Stop Listening",
    clear: "Clear",
    translation: "Translation",
    confidence: "Confidence",
    backend: "Backend Status",
    online: "Online",
    offline: "Offline",
    heuristic: "Heuristic demo model",
    trained: "TensorFlow model",
    output: "Live Output",
    history: "Recent Conversations",
    noHistory: "No conversations yet.",
    noHand: "Show one hand clearly inside the camera frame.",
    listening: "Listening...",
    ready: "Ready",
    cameraHint: "MediaPipe hand tracking runs directly in the browser.",
    micHint: "Chrome or Edge gives the best speech recognition support.",
  },
  hi: {
    badge: "रियल-टाइम साइन लैंग्वेज ट्रांसलेटर",
    title: "आईएसएल ब्रिज",
    subtitle:
      "एक ही ब्राउज़र ऐप में sign-to-text, sign-to-speech और speech-to-text communication.",
    chooseMode: "मोड चुनें",
    signToText: "मूक -> बधिर",
    signToSpeech: "मूक -> सुनने वाला",
    speechToText: "सुनने वाला -> बधिर",
    startCamera: "साइन करें",
    stopCamera: "कैमरा बंद करें",
    startListening: "बोलें",
    stopListening: "सुनना बंद करें",
    clear: "साफ़ करें",
    translation: "अनुवाद",
    confidence: "विश्वास",
    backend: "बैकएंड स्टेटस",
    online: "ऑनलाइन",
    offline: "ऑफलाइन",
    heuristic: "ह्यूरिस्टिक डेमो मॉडल",
    trained: "टेंसरफ्लो मॉडल",
    output: "लाइव आउटपुट",
    history: "हाल की बातचीत",
    noHistory: "अभी तक कोई बातचीत नहीं है।",
    noHand: "कैमरे के फ्रेम में एक हाथ साफ़ दिखाएँ।",
    listening: "सुन रहा है...",
    ready: "तैयार",
    cameraHint: "MediaPipe hand tracking सीधे ब्राउज़र में चलती है।",
    micHint: "सबसे अच्छा speech recognition Chrome या Edge में मिलेगा।",
  },
};

function formatTranscript(text) {
  return text
    .replace(/\s+/g, " ")
    .trim()
    .replace(/(^\w)/, (match) => match.toUpperCase());
}

function cloneLandmarks(landmarks) {
  return landmarks.map((point) => ({
    x: point.x,
    y: point.y,
    z: point.z,
  }));
}

function cloneHands(landmarksList = [], handednesses = []) {
  return landmarksList
    .map((landmarks, index) => {
      const category = handednesses[index]?.[0];
      return {
        label: category?.displayName || category?.categoryName || `Hand ${index + 1}`,
        score: category?.score || 0,
        landmarks: cloneLandmarks(landmarks),
      };
    })
    .sort((left, right) => left.label.localeCompare(right.label));
}

function getStablePrediction(predictions) {
  if (!predictions.length) {
    return null;
  }

  const counts = predictions.reduce((accumulator, prediction) => {
    accumulator[prediction] = (accumulator[prediction] || 0) + 1;
    return accumulator;
  }, {});

  const [label, count] = Object.entries(counts).sort((left, right) => right[1] - left[1])[0];
  return count >= Math.ceil(predictions.length / 2) ? label : null;
}

function useSpeechRecognition(language, onTranscript) {
  const recognitionRef = useRef(null);
  const [supported, setSupported] = useState(true);
  const [listening, setListening] = useState(false);

  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setSupported(false);
      return undefined;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = language === "hi" ? "hi-IN" : "en-IN";
    recognition.onstart = () => setListening(true);
    recognition.onend = () => setListening(false);
    recognition.onerror = () => setListening(false);
    recognition.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map((result) => result[0]?.transcript || "")
        .join(" ");
      onTranscript(formatTranscript(transcript));
    };

    recognitionRef.current = recognition;

    return () => {
      recognition.stop();
      recognitionRef.current = null;
    };
  }, [language, onTranscript]);

  return {
    supported,
    listening,
    start: () => recognitionRef.current?.start(),
    stop: () => recognitionRef.current?.stop(),
  };
}

function App() {
  const [language, setLanguage] = useState(
    () => window.localStorage.getItem("isl-bridge-language") || "en",
  );
  const [mode, setMode] = useState(MODES.signToText);
  const [translation, setTranslation] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [status, setStatus] = useState("Ready");
  const [cameraActive, setCameraActive] = useState(false);
  const [history, setHistory] = useState([]);
  const [backendHealth, setBackendHealth] = useState({ ok: false, modelType: "heuristic" });
  const [error, setError] = useState("");

  const t = COPY[language];
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const animationRef = useRef(null);
  const inFlightRef = useRef(false);
  const lastPredictionRef = useRef(0);
  const lastPhraseRef = useRef("");
  const lastSpokenRef = useRef("");
  const lastSpeechSavedRef = useRef("");
  const sequenceBufferRef = useRef([]);
  const predictionWindowRef = useRef([]);

  const speechRecognition = useSpeechRecognition(language, (text) => {
    setTranslation(text);
    setConfidence(text ? 0.99 : 0);
    if (text) {
      setStatus(t.listening);
      if (text !== lastSpeechSavedRef.current) {
        lastSpeechSavedRef.current = text;
        persistConversation("speech_to_text", text, 0.99);
      }
    }
  });

  const modeCards = useMemo(
    () => [
      { id: MODES.signToText, label: t.signToText, accent: "from-cyan to-emerald-500" },
      { id: MODES.signToSpeech, label: t.signToSpeech, accent: "from-gold to-coral" },
      { id: MODES.speechToText, label: t.speechToText, accent: "from-indigo-500 to-cyan" },
    ],
    [t],
  );

  useEffect(() => {
    window.localStorage.setItem("isl-bridge-language", language);
  }, [language]);

  useEffect(() => {
    let ignore = false;

    fetch(`${API_URL}/api/health`)
      .then((response) => response.json())
      .then((data) => {
        if (!ignore) {
          setBackendHealth({ ok: true, modelType: data.model_type || "heuristic" });
        }
      })
      .catch(() => {
        if (!ignore) {
          setBackendHealth({ ok: false, modelType: "heuristic" });
        }
      });

    fetch(`${API_URL}/api/conversations`)
      .then((response) => response.json())
      .then((data) => {
        if (!ignore) {
          setHistory(data.items || []);
        }
      })
      .catch(() => undefined);

    return () => {
      ignore = true;
    };
  }, []);

  useEffect(() => {
    if (mode !== MODES.speechToText) {
      speechRecognition.stop();
    }
  }, [mode]);

  useEffect(() => {
    if (
      mode === MODES.signToSpeech &&
      translation &&
      translation !== lastSpokenRef.current
    ) {
      const utterance = new SpeechSynthesisUtterance(translation);
      utterance.lang = language === "hi" ? "hi-IN" : "en-IN";
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(utterance);
      lastSpokenRef.current = translation;
    }
  }, [language, mode, translation]);

  useEffect(() => {
    return () => stopCamera();
  }, []);

  async function loadHandLandmarker() {
    if (handLandmarkerRef.current) {
      return handLandmarkerRef.current;
    }

    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm",
    );

    const landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      },
      runningMode: "VIDEO",
      numHands: 2,
    });

    handLandmarkerRef.current = landmarker;
    return landmarker;
  }

  async function startCamera() {
    try {
      setError("");
      const landmarker = await loadHandLandmarker();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 960, height: 720 },
      });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setCameraActive(true);
      setStatus("Tracking hands");
      runDetectionLoop(landmarker);
    } catch (cameraError) {
      setError(cameraError.message || "Camera access failed.");
      setCameraActive(false);
    }
  }

  function stopCamera() {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraActive(false);
    setStatus(t.ready);
    sequenceBufferRef.current = [];
    predictionWindowRef.current = [];
  }

  function clearAll() {
    setTranslation("");
    setConfidence(0);
    setStatus(t.ready);
    lastPhraseRef.current = "";
    lastSpokenRef.current = "";
    lastSpeechSavedRef.current = "";
    sequenceBufferRef.current = [];
    predictionWindowRef.current = [];
    window.speechSynthesis.cancel();
  }

  function drawLandmarks(hands) {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext("2d");

    canvas.width = video.videoWidth || 960;
    canvas.height = video.videoHeight || 720;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!hands?.length) {
      return;
    }

    const palette = ["#22D3EE", "#F97316"];
    hands.forEach((hand, handIndex) => {
      ctx.fillStyle = palette[handIndex % palette.length];
      hand.landmarks.forEach((point) => {
        ctx.beginPath();
        ctx.arc(point.x * canvas.width, point.y * canvas.height, 6, 0, Math.PI * 2);
        ctx.fill();
      });
    });
  }

  function runDetectionLoop(handLandmarker) {
    const detect = async () => {
      const video = videoRef.current;
      if (!video || video.readyState < 2) {
        animationRef.current = requestAnimationFrame(detect);
        return;
      }

      const result = handLandmarker.detectForVideo(video, performance.now());
      const hands = cloneHands(result.landmarks || [], result.handednesses || []);
      drawLandmarks(hands);

      if (hands.length) {
        setStatus(hands.length === 2 ? "Tracking both hands" : "Tracking one hand");
        sequenceBufferRef.current = [
          ...sequenceBufferRef.current.slice(-(MAX_SEQUENCE_FRAMES - 1)),
          { hands },
        ];
        const now = Date.now();
        if (
          !inFlightRef.current &&
          sequenceBufferRef.current.length >= MIN_SEQUENCE_FRAMES &&
          now - lastPredictionRef.current > 650
        ) {
          inFlightRef.current = true;
          lastPredictionRef.current = now;
          await sendLandmarks(sequenceBufferRef.current);
          inFlightRef.current = false;
        }
      } else {
        setStatus(t.noHand);
        sequenceBufferRef.current = [];
        predictionWindowRef.current = [];
      }

      animationRef.current = requestAnimationFrame(detect);
    };

    animationRef.current = requestAnimationFrame(detect);
  }

  async function sendLandmarks(frames) {
    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          landmarks: frames,
          language,
          mode: mode === MODES.signToSpeech ? "sign_to_speech" : "sign_to_text",
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Prediction failed.");
      }

      const nextWindow = [
        ...predictionWindowRef.current.slice(-(STABLE_PREDICTION_WINDOW - 1)),
        data.prediction,
      ];
      predictionWindowRef.current = nextWindow;
      const stablePrediction = getStablePrediction(nextWindow);

      if (
        stablePrediction &&
        stablePrediction !== "Gesture not recognised" &&
        stablePrediction !== lastPhraseRef.current
      ) {
        lastPhraseRef.current = stablePrediction;
        setTranslation(stablePrediction);
        setConfidence(data.confidence || 0);
        persistConversation(data.mode, stablePrediction, data.confidence || 0);
      } else if (stablePrediction) {
        setConfidence(data.confidence || 0);
      }
    } catch (requestError) {
      setError(requestError.message || "Prediction failed.");
    }
  }

  async function persistConversation(entryMode, text, score) {
    if (!text) {
      return;
    }

    const item = {
      mode: entryMode,
      text,
      confidence: score,
      created_at: new Date().toISOString(),
    };

    setHistory((current) => [item, ...current].slice(0, 8));

    try {
      await fetch(`${API_URL}/api/conversations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(item),
      });
    } catch {
      // Local history keeps the UI usable even if storage is offline.
    }
  }

  const showCamera = mode !== MODES.speechToText;

  return (
    <div className="min-h-screen px-4 py-6 text-ink sm:px-6 lg:px-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-6">
        <header className="glass overflow-hidden rounded-[32px] border border-white/60 shadow-glow">
          <div className="grid gap-6 bg-[radial-gradient(circle_at_top_left,_rgba(14,165,167,0.18),_transparent_25%),linear-gradient(135deg,_rgba(255,255,255,0.96),_rgba(239,246,248,0.86))] p-6 lg:grid-cols-[1.2fr_0.8fr] lg:p-10">
            <div className="space-y-4">
              <span className="inline-flex rounded-full bg-white/80 px-4 py-2 text-sm font-semibold text-cyan shadow-sm">
                {t.badge}
              </span>
              <h1 className="font-display text-4xl font-extrabold tracking-tight sm:text-5xl">
                {t.title}
              </h1>
              <p className="max-w-2xl text-base text-slate-600 sm:text-lg">{t.subtitle}</p>
              <div className="flex gap-3">
                <button
                  type="button"
                  onClick={() => setLanguage("en")}
                  className={`rounded-full px-4 py-2 text-sm font-bold ${
                    language === "en" ? "bg-ink text-white" : "bg-white text-slate-600"
                  }`}
                >
                  EN
                </button>
                <button
                  type="button"
                  onClick={() => setLanguage("hi")}
                  className={`rounded-full px-4 py-2 text-sm font-bold ${
                    language === "hi" ? "bg-ink text-white" : "bg-white text-slate-600"
                  }`}
                >
                  HI
                </button>
              </div>
            </div>

            <div className="grid gap-3 rounded-[28px] bg-ink p-5 text-white shadow-glow">
              <div className="flex items-center justify-between text-sm">
                <span>{t.backend}</span>
                <span
                  className={`rounded-full px-3 py-1 font-semibold ${
                    backendHealth.ok ? "bg-emerald-500/20 text-emerald-200" : "bg-red-500/20 text-red-200"
                  }`}
                >
                  {backendHealth.ok ? t.online : t.offline}
                </span>
              </div>
              <div className="rounded-2xl bg-white/10 p-4">
                <p className="text-sm text-slate-300">
                  {backendHealth.modelType === "tensorflow" ? t.trained : t.heuristic}
                </p>
                <p className="mt-2 text-2xl font-bold">{status}</p>
              </div>
              <p className="text-sm text-slate-300">{showCamera ? t.cameraHint : t.micHint}</p>
            </div>
          </div>
        </header>

        <section className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
          <div className="glass rounded-[30px] border border-white/60 p-5 shadow-glow sm:p-6">
            <h2 className="mb-5 font-display text-2xl font-bold">{t.chooseMode}</h2>

            <div className="grid gap-3 sm:grid-cols-3">
              {modeCards.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setMode(item.id)}
                  className={`rounded-[24px] border p-4 text-left transition ${
                    mode === item.id
                      ? "border-transparent bg-ink text-white shadow-glow"
                      : "border-slate-200 bg-white/80 text-slate-700"
                  }`}
                >
                  <div className={`mb-3 h-2 rounded-full bg-gradient-to-r ${item.accent}`} />
                  <p className="font-semibold">{item.label}</p>
                </button>
              ))}
            </div>

            <div className="mt-6 flex flex-wrap gap-3">
              {showCamera ? (
                <button
                  type="button"
                  onClick={cameraActive ? stopCamera : startCamera}
                  className="rounded-full bg-cyan px-5 py-3 font-bold text-white hover:bg-teal-600"
                >
                  {cameraActive ? t.stopCamera : t.startCamera}
                </button>
              ) : (
                <button
                  type="button"
                  onClick={speechRecognition.listening ? speechRecognition.stop : speechRecognition.start}
                  disabled={!speechRecognition.supported}
                  className="rounded-full bg-coral px-5 py-3 font-bold text-white disabled:cursor-not-allowed disabled:bg-slate-300"
                >
                  {speechRecognition.listening ? t.stopListening : t.startListening}
                </button>
              )}

              <button
                type="button"
                onClick={clearAll}
                className="rounded-full border border-slate-300 bg-white px-5 py-3 font-bold text-slate-700"
              >
                {t.clear}
              </button>
            </div>

            {error ? (
              <p className="mt-4 rounded-2xl bg-red-50 px-4 py-3 text-sm font-medium text-red-700">
                {error}
              </p>
            ) : null}

            <div className="mt-6 rounded-[28px] bg-slate-950 p-4 text-white">
              {showCamera ? (
                <div className="relative overflow-hidden rounded-[24px]">
                  <video
                    ref={videoRef}
                    muted
                    playsInline
                    className="aspect-[4/3] w-full rounded-[24px] bg-slate-900 object-cover"
                  />
                  <canvas
                    ref={canvasRef}
                    className="pointer-events-none absolute inset-0 h-full w-full"
                  />
                  {!cameraActive ? (
                    <div className="absolute inset-0 flex items-center justify-center bg-slate-950/65 text-center">
                      <div>
                        <div className="mx-auto mb-3 h-4 w-4 rounded-full bg-cyan signal-pulse" />
                        <p className="text-sm text-slate-200">{t.noHand}</p>
                      </div>
                    </div>
                  ) : null}
                </div>
              ) : (
                <div className="flex aspect-[4/3] items-center justify-center rounded-[24px] border border-white/10 bg-[radial-gradient(circle_at_center,_rgba(14,165,167,0.22),_transparent_35%),linear-gradient(135deg,_rgba(15,23,42,0.96),_rgba(2,6,23,0.92))] p-6 text-center">
                  <div className="space-y-4">
                    <div className="mx-auto h-16 w-16 rounded-full bg-coral/25 p-5">
                      <div
                        className={`h-full w-full rounded-full ${
                          speechRecognition.listening ? "bg-coral signal-pulse" : "bg-white/70"
                        }`}
                      />
                    </div>
                    <p className="text-lg font-semibold">
                      {speechRecognition.listening ? t.listening : t.startListening}
                    </p>
                    <p className="text-sm text-slate-300">{t.micHint}</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="grid gap-6">
            <section className="glass rounded-[30px] border border-white/60 p-6 shadow-glow">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="font-display text-2xl font-bold">{t.output}</h2>
                <span className="rounded-full bg-ink px-3 py-1 text-sm font-semibold text-white">
                  {t.confidence}: {(confidence * 100).toFixed(0)}%
                </span>
              </div>
              <div className="min-h-[220px] rounded-[26px] bg-[linear-gradient(135deg,_rgba(8,18,28,0.96),_rgba(15,23,42,0.92))] p-6 text-white">
                <p className="mb-3 text-sm uppercase tracking-[0.3em] text-slate-400">
                  {t.translation}
                </p>
                <p className="text-3xl font-extrabold leading-tight sm:text-4xl">
                  {translation || "..."}
                </p>
              </div>
            </section>

            <section className="glass rounded-[30px] border border-white/60 p-6 shadow-glow">
              <h2 className="mb-4 font-display text-2xl font-bold">{t.history}</h2>
              <div className="space-y-3">
                {history.length ? (
                  history.map((item, index) => (
                    <div
                      key={`${item.created_at}-${index}`}
                      className="rounded-[22px] border border-slate-200 bg-white/80 p-4"
                    >
                      <div className="mb-1 flex items-center justify-between text-xs uppercase tracking-[0.18em] text-slate-400">
                        <span>{item.mode.replaceAll("_", " ")}</span>
                        <span>{new Date(item.created_at).toLocaleTimeString()}</span>
                      </div>
                      <p className="font-semibold text-slate-800">{item.text}</p>
                    </div>
                  ))
                ) : (
                  <p className="rounded-[22px] bg-white/70 p-4 text-slate-500">{t.noHistory}</p>
                )}
              </div>
            </section>
          </div>
        </section>
      </div>
    </div>
  );
}

export default App;
