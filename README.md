# AI Squat Coach

Real-time squat-form analyser. Points your webcam at you, counts reps,
scores your form, and speaks corrections out loud as you lift.

Pure OpenCV + MediaPipe. No web framework, no WebRTC, no async event loops —
one main thread drives the camera and rendering, one daemon thread drives
voice. That's it.

---

## What it does

| Capability | How |
|---|---|
| **Pose tracking** | MediaPipe Pose (model_complexity=0) on every 2nd frame; cached skeleton is replayed on skipped frames for smooth visuals at full fps |
| **Rep counting** | 4-state FSM — `Standing → Descending → Bottom → Ascending` — with hysteresis thresholds and a 30-frame calibration phase so it won't miscount noise |
| **Form checks** | Depth (hip below knee), knee-over-toe, knee valgus (collapse inward), back angle (forward lean) |
| **Scoring** | Per-rep score starts at 100, deducts per fault, rolling average across the session |
| **Voice coaching** | `say` / `espeak` / PowerShell via `subprocess` — truly non-blocking, thread-safe across macOS / Linux / Windows |
| **Live overlay** | Rep counter, phase badge, form-score bar, coaching text, sparkline of recent rep scores, FPS |
| **Session log** | Appended to `session_log.json` on exit — reps, scores, corrections by type, duration |

---

## Folder structure

```
squatAIcoach/
├── main.py              Entry point — OpenCV capture + render loop
├── config.py            Thresholds, cooldowns, coaching messages
├── pose_detector.py     MediaPipe wrapper + landmark smoothing + draw_cached
├── rep_counter.py       Squat FSM + calibration
├── squat_analyzer.py    Depth / knee-toe / valgus / back-angle checks + scoring
├── voice_coach.py       Queue-based subprocess TTS worker
├── session_logger.py    Per-session JSON log
├── utils.py             Math helpers (angle, moving average, colours)
├── requirements.txt
├── run.sh               Launcher (handles Python path + dep check)
└── session_log.json     Appended on every run
```

---

## How to run

### 1. Install dependencies (once)

```bash
cd ~/Desktop/squatAIcoach
pip install -r requirements.txt
```

Only three packages: `opencv-python`, `mediapipe`, `numpy`.

### 2. Launch

```bash
bash run.sh
```

…or directly:

```bash
python3 main.py
```

On first run MediaPipe downloads its pose model (~10 MB, cached for future runs).

### 3. Set up the shot

- Stand **1.5–2 m** from the camera
- **Side-on** to the camera for best depth / knee-travel detection
- **Full body in frame**, head to feet
- Stand still for ~1 second — the coach will finish calibrating, then you're good

---

## Controls

| Key | Action |
|---|---|
| `q` / `ESC` | Quit and save session to `session_log.json` |
| `p` | Pause / resume |
| `r` | Reset reps, scores, calibration (keeps window open) |
| `v` | Toggle voice feedback |
| `d` | Toggle debug overlay (raw back/knee angles) |

> Focus the OpenCV window before pressing keys — `cv2.waitKey()` only captures input when the window is in focus.

---

## Tuning

All thresholds live in **`config.py`**. Adjust to taste:

- `DEPTH_HIP_BELOW_KNEE_PX` — how strict depth-below-parallel is
- `KNEE_TOE_SLACK_FRACTION` — how much forward knee travel is allowed
- `VALGUS_SLACK_FRACTION` — how much inward knee collapse is allowed
- `BACK_LEAN_MAX_FROM_VERTICAL` — max torso lean before "chest up" fires
- `VOICE_COOLDOWN_SEC` — minimum seconds before re-speaking a cue
- `CONSECUTIVE_FRAMES` — how many bad frames in a row trigger an alert
- `ISSUE_MESSAGES` — edit the coaching lines directly

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Cannot open camera" | Close any app using the webcam (Zoom, Photo Booth, etc.) and grant camera permission in System Settings → Privacy → Camera |
| Skeleton lags / low fps | Raise `INFERENCE_EVERY` from 2 to 3 in `config.py`, or drop `FRAME_WIDTH`/`FRAME_HEIGHT` |
| No voice on macOS | `say` should always work; check volume. On Linux install `espeak`: `sudo apt install espeak` |
| Reps mis-count | Stand still longer during calibration, or tune `DESCENT_THRESHOLD` / `COMPLETE_THRESHOLD` in `config.py` |
| "No pose detected" | Step back so your full body is in frame; avoid busy backgrounds |

---

## Requirements

- Python 3.10+
- Webcam
- macOS, Linux, or Windows
# ai-fitness-coach
