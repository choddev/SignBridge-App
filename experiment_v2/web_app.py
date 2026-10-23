"""
experiment_v2 Web Arayuzu
Flask + WebSocket ile gercek zamanli isaret dili cevirisi

Baslat: python experiment_v2/web_app.py
Tarayici: http://localhost:5050
"""

import cv2
import mediapipe as mp
import numpy as np
import base64
import time
import json
import threading
from pathlib import Path
from collections import deque
from flask import Flask, render_template_string, Response, request, jsonify
from flask_socketio import SocketIO, emit
import onnxruntime as ort

BASE           = Path(__file__).parent
CKPT           = BASE / "checkpoints" / "best_model.pt"
ONNX_PATH      = BASE / "checkpoints" / "model_v2.onnx"
LABEL_MAP_PATH = BASE / "checkpoints" / "label_map.json"
SEQ_LEN        = 30
PROB_AVG_WINDOW = 3
INTER_SIGN_PAUSE_SEC = 0.50

# ---- Boyutlar (V2 modeli icin) ----
FACE_END   = 468 * 3          # 1404
POSE_END   = FACE_END + 33*3  # 1503
LEFT_END   = POSE_END + 21*3  # 1566
RIGHT_END  = LEFT_END + 21*3  # 1629
HAND_VEL   = 21*3 + 21*3      # 126
INPUT_SIZE = RIGHT_END + HAND_VEL  # 1755

def _normalize_gloss_key(k: str) -> str:
    return (k.replace("İ", "I").replace("ı", "I")
             .replace("Ş", "S").replace("ş", "S")
             .replace("Ç", "C").replace("ç", "C")
             .replace("Ğ", "G").replace("ğ", "G")
             .replace("Ü", "U").replace("ü", "U")
             .replace("Ö", "O").replace("ö", "O").upper())

# ---- 2 Katmanli Normalizasyon (eğitim ile aynı) ----
def normalize_landmarks(lm):
    result = lm.copy()
    # KATMAN 1: Omuz-merkezli
    pose = result[FACE_END:POSE_END].reshape(33, 3)
    ls, rs = pose[11].copy(), pose[12].copy()
    if not (np.allclose(ls, 0) and np.allclose(rs, 0)):
        center = rs.copy() if np.allclose(ls, 0) else (
                 ls.copy() if np.allclose(rs, 0) else (ls + rs) / 2.0)
        scale = float(np.linalg.norm(rs - ls))
        if scale >= 1e-4:
            all_lm = result.reshape(-1, 3)
            all_lm = (all_lm - center) / scale
            result = all_lm.flatten()
    # KATMAN 2: Sol el bilek-merkezli
    lh = result[POSE_END:LEFT_END].reshape(21, 3)
    if not np.allclose(lh[0], 0):
        wrist = lh[0].copy()
        palm_scale = np.linalg.norm(lh[9] - lh[0])
        if palm_scale > 1e-4:
            lh[1:] = (lh[1:] - wrist) / palm_scale
            result[POSE_END:LEFT_END] = lh.flatten()
    # KATMAN 2: Sag el bilek-merkezli
    rh = result[LEFT_END:RIGHT_END].reshape(21, 3)
    if not np.allclose(rh[0], 0):
        wrist = rh[0].copy()
        palm_scale = np.linalg.norm(rh[9] - rh[0])
        if palm_scale > 1e-4:
            rh[1:] = (rh[1:] - wrist) / palm_scale
            result[LEFT_END:RIGHT_END] = rh.flatten()
    return result

def add_velocity(seq):
    """(T, 1629) -> (T, 1755)"""
    hands = seq[:, POSE_END:]
    vel   = np.zeros_like(hands)
    vel[1:] = hands[1:] - hands[:-1]
    return np.concatenate([seq, vel], axis=1).astype(np.float32)

# ---- Model ----
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as _f:
    _raw_label_map = json.load(_f)
label_map    = {_normalize_gloss_key(k): int(v) for k, v in _raw_label_map.items()}
num_classes  = len(label_map)
idx_to_label = {v: k for k, v in label_map.items()}

ort_session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
device = "onnx-cpu"
print(f"Model yuklendi | {num_classes} kelime | {device}")

def run_model_probs(seq_30x1755: np.ndarray) -> np.ndarray:
    inp = np.ascontiguousarray(seq_30x1755[None, :, :], dtype=np.float32)
    logits = ort_session.run(None, {"input": inp})[0][0].astype(np.float64)
    logits -= np.max(logits)
    exp_l = np.exp(logits)
    return (exp_l / np.sum(exp_l)).astype(np.float32)

# ---- MediaPipe ----
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

def _is_valid_signing_hand(hand_lms) -> bool:
    if hand_lms is None:
        return False
    pts = hand_lms.landmark
    w, m = pts[0], pts[9]
    dx, dy, dz = (m.x - w.x), (m.y - w.y), (m.z - w.z)
    palm_scale = (dx*dx + dy*dy + dz*dz) ** 0.5
    if palm_scale < 0.022:
        return False
    if w.y > 0.94 and m.y > 0.94:
        return False
    return True

def detect_active_hands(results) -> bool:
    return _is_valid_signing_hand(results.left_hand_landmarks) or _is_valid_signing_hand(results.right_hand_landmarks)

def extract_landmarks(results):
    lm = []
    if results.face_landmarks:
        for p in results.face_landmarks.landmark: lm.extend([p.x, p.y, p.z])
    else:
        lm.extend([0.0] * 468 * 3)
    if results.pose_landmarks:
        for p in results.pose_landmarks.landmark: lm.extend([p.x, p.y, p.z])
    else:
        lm.extend([0.0] * 33 * 3)
    if _is_valid_signing_hand(results.left_hand_landmarks):
        for p in results.left_hand_landmarks.landmark: lm.extend([p.x, p.y, p.z])
    else:
        lm.extend([0.0] * 21 * 3)
    if _is_valid_signing_hand(results.right_hand_landmarks):
        for p in results.right_hand_landmarks.landmark: lm.extend([p.x, p.y, p.z])
    else:
        lm.extend([0.0] * 21 * 3)
    return np.array(lm, dtype=np.float32)

# ---- Uygulama durumu ----
state = {
    "buffer":           deque(maxlen=SEQ_LEN),
    "prob_history":     deque(maxlen=PROB_AVG_WINDOW),
    "smooth_counts":    {k: 0 for k in label_map},
    "sentence":         [],
    "current_word":     "",
    "current_conf":     0.0,
    "last_add_time":    0.0,
    "last_hand_time":   0.0,
    "pause_reset_done": False,
    "running":          False,
    "cap":              None,
    "holistic":         None,
    "demo_mode":        False,   # Demo modu: sadece 10 cumlede gecen kelimeleri tahmin et
}

# 10 demo cumlesinin 29 kelimesi (122 sinif yerine sadece bunlar arasinda secim)
DEMO_WORDS = {
    "MERHABA","NASILSIN","IYI","SEN","NE","YAPMAK",
    "BEN","CALISMAK","COK","YORULMAK",
    "SAAT","KAC","BITMEK",
    "AKSAM","BES","SONRA","BOS",
    "BERABER","KAHVE","ICMEK",
    "TAMAM","NEREDE","BULUSMAK",
    "PARK","YAN","KAFE",
    "GORUSMEK","KENDI","BAKMAK",
}

CONF_THRESHOLD  = 0.88
CONF_FAST_TRACK = 0.95
MARGIN_MIN      = 0.20
MARGIN_FAST     = 0.50
SMOOTH_NEEDED   = 3
SMOOTH_FAST     = 2
COOLDOWN_SEC    = 0.80   # Gecis gurultusunu onlemek icin stabil bekleme suresi
HAND_GONE_SEC   = 2.0    # el cekilince cumle bitis suresi

# ---- Flask ----
app = Flask(__name__)
app.config["SECRET_KEY"] = "signbridge_v2"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

HTML = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SignBridge - İşaret Dili Çevirisi</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background: #0f172a;  /* Soft Midnight: slate-900 (mavi tonlu, pure black degil) */
    color: #e2e8f0;
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
    position: relative;
    overflow-x: hidden;
  }
  /* Subtle background glow orbs for depth */
  body::before, body::after {
    content: ""; position: fixed; border-radius: 50%; pointer-events: none;
    z-index: 0; filter: blur(40px);
  }
  body::before {
    width: 540px; height: 540px;
    background: radial-gradient(circle, rgba(108,99,255,0.18), transparent 70%);
    top: -200px; left: -140px;
  }
  body::after {
    width: 460px; height: 460px;
    background: radial-gradient(circle, rgba(34,211,238,0.13), transparent 70%);
    bottom: -160px; right: -110px;
  }
  header, .main { position: relative; z-index: 1; }

  header {
    background: rgba(30,41,59,0.6);
    backdrop-filter: blur(24px) saturate(180%);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    padding: 14px 32px;
    display: flex; align-items: center; gap: 16px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    position: sticky; top: 0;
  }
  header h1 {
    font-size: 1.4rem; font-weight: 800; letter-spacing: -0.02em;
    background: linear-gradient(135deg, #a78bfa, #22d3ee);
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }
  header span { font-size: 0.82rem; color: #6b7280; font-weight: 500; }
  .pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 999px;
    padding: 5px 12px;
    font-size: 0.72rem; font-weight: 600;
    color: #a8aec1;
  }

  .main {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 360px;
    gap: 20px;
    padding: 18px;
    max-width: 1600px;       /* 1340 -> 1600 (camera daha buyuk) */
    margin: 0 auto;
  }

  /* Sol: kamera */
  .camera-section { display: flex; flex-direction: column; gap: 16px; }
  .dual-cam { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  @media(max-width:980px){ .dual-cam{ grid-template-columns: 1fr; } }
  .camera-wrapper {
    position: relative;
    background: #0a0b14;
    border-radius: 20px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.07);
    aspect-ratio: 4/3;
    min-height: 380px;       /* daha buyuk gorunum */
    box-shadow: 0 20px 60px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04);
  }
  #cameraFeed, #cameraFeedClean { width: 100%; height: 100%; object-fit: cover; display: block; }
  .cam-label {
    position: absolute; bottom: 8px; left: 8px;
    padding: 4px 10px; border-radius: 999px;
    background: rgba(0,0,0,0.6); color: #c4b5fd;
    font-size: 0.72rem; font-weight: 700;
    backdrop-filter: blur(8px);
    pointer-events: none;
  }
  .camera-placeholder {
    position: absolute; inset: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 14px; color: #4b5563;
  }
  .camera-placeholder svg { width: 72px; opacity: 0.4; }

  /* Anlik tahmin overlay */
  .pred-overlay {
    position: absolute; top: 0; left: 0; right: 0;
    background: linear-gradient(180deg, rgba(0,0,0,0.78) 0%, transparent 100%);
    padding: 18px 22px;
    display: flex; align-items: center; justify-content: space-between;
    pointer-events: none;
  }
  #predWord {
    font-size: 2.2rem; font-weight: 900; letter-spacing: -0.02em;
    background: linear-gradient(135deg, #fff, #c4b5fd);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    filter: drop-shadow(0 4px 18px rgba(108,99,255,0.7));
  }
  #predConf {
    font-size: 1.0rem; font-weight: 700;
    color: #6ee7b7;
    padding: 4px 12px; border-radius: 999px;
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.3);
  }
  .conf-bar-wrap {
    position: absolute; bottom: 0; left: 0; right: 0;
    height: 4px; background: rgba(255,255,255,0.06);
  }
  #confBar {
    height: 100%; width: 0%;
    background: linear-gradient(90deg, #7c3aed, #06b6d4, #10b981);
    transition: width 0.15s ease;
    box-shadow: 0 0 14px rgba(124,58,237,0.6);
  }

  /* Kontrol butonlari */
  .controls { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
  button {
    padding: 11px 22px;
    border: none; border-radius: 12px;
    font-size: 0.9rem; font-weight: 700;
    cursor: pointer;
    transition: all 0.2s;
    font-family: inherit;
  }
  button:active { transform: scale(0.97); }
  #btnStart {
    background: linear-gradient(135deg, #6c63ff, #a78bfa);
    color: #fff; flex: 1;
    box-shadow: 0 8px 22px rgba(108,99,255,0.4), inset 0 1px 0 rgba(255,255,255,0.15);
  }
  #btnStart:hover { transform: translateY(-1px); box-shadow: 0 12px 28px rgba(108,99,255,0.55); }
  #btnStart.active {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    box-shadow: 0 8px 22px rgba(239,68,68,0.42), inset 0 1px 0 rgba(255,255,255,0.15);
  }
  #btnClear {
    background: rgba(255,255,255,0.05);
    color: #94a3b8;
    border: 1px solid rgba(255,255,255,0.08);
  }
  #btnClear:hover { background: rgba(255,255,255,0.09); color: #e2e8f0; border-color: rgba(255,255,255,0.14); }

  /* Sag: panel */
  .side-panel { display: flex; flex-direction: column; gap: 16px; }
  .card {
    background: rgba(30,41,59,0.6);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 12px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04);
  }
  .card-title {
    font-size: 0.7rem; font-weight: 800;
    text-transform: uppercase; letter-spacing: 0.08em;
    background: linear-gradient(135deg, #a78bfa, #22d3ee);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    margin-bottom: 14px;
  }

  /* Cumle kutusu */
  #sentenceBox {
    min-height: 90px;
    background: rgba(0,0,0,0.25);
    border-radius: 12px;
    padding: 14px;
    font-size: 1.1rem;
    line-height: 1.55;
    color: #e2e8f0;
    border: 1px solid rgba(108,99,255,0.15);
    word-break: break-word;
  }
  #sentenceBox.empty { color: #4b5563; font-style: italic; font-weight: 500; }

  /* Gecmis cumle */
  #historyList {
    display: flex; flex-direction: column; gap: 8px;
    max-height: 220px; overflow-y: auto;
    padding-right: 4px;
  }
  #historyList::-webkit-scrollbar { width: 5px; }
  #historyList::-webkit-scrollbar-track { background: transparent; }
  #historyList::-webkit-scrollbar-thumb { background: rgba(108,99,255,0.3); border-radius: 99px; }
  .history-item {
    background: rgba(0,0,0,0.22);
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.84rem;
    color: #cbd5e1;
    border-left: 3px solid #a78bfa;
    transition: all 0.2s;
  }
  .history-item:hover {
    background: rgba(108,99,255,0.08);
    border-left-color: #22d3ee;
    transform: translateX(2px);
  }

  /* Kelime listesi (search + grid) */
  .word-card-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;gap:8px}
  .word-count-badge{
    padding:4px 11px;border-radius:999px;
    background:linear-gradient(135deg,rgba(108,99,255,0.18),rgba(34,211,238,0.12));
    border:1px solid rgba(108,99,255,0.25);
    font-size:0.7rem;font-weight:800;
    color:#c4b5fd;
    font-variant-numeric:tabular-nums;
    box-shadow:inset 0 1px 0 rgba(255,255,255,0.04);
  }
  .word-search-wrap{position:relative;margin-bottom:12px}
  .word-search-icon{
    position:absolute;left:12px;top:50%;transform:translateY(-50%);
    width:15px;height:15px;color:#64748b;pointer-events:none;
  }
  #wordSearch{
    width:100%;padding:9px 12px 9px 36px;
    border-radius:10px;border:1px solid rgba(255,255,255,0.08);
    background:rgba(255,255,255,0.04);
    color:#e2e8f0;font-size:0.84rem;font-weight:500;outline:none;
    transition:all 0.2s;font-family:inherit;
  }
  #wordSearch::placeholder{color:#64748b}
  #wordSearch:focus{
    border-color:rgba(108,99,255,0.5);
    background:rgba(108,99,255,0.06);
    box-shadow:0 0 0 3px rgba(108,99,255,0.12);
  }
  #wordList{
    display:grid;
    grid-template-columns:repeat(auto-fill,minmax(86px,1fr));
    gap:6px;
    max-height:280px;overflow-y:auto;
    padding-right:4px;
  }
  #wordList::-webkit-scrollbar{width:5px}
  #wordList::-webkit-scrollbar-track{background:transparent}
  #wordList::-webkit-scrollbar-thumb{background:rgba(108,99,255,0.3);border-radius:99px}
  #wordList::-webkit-scrollbar-thumb:hover{background:rgba(108,99,255,0.5)}
  .word-chip{
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.06);
    border-radius:8px;
    padding:7px 6px;
    font-size:0.72rem;font-weight:600;
    color:#94a3b8;
    text-align:center;
    transition:all 0.18s;
    cursor:default;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  }
  .word-chip:hover{
    background:rgba(108,99,255,0.08);
    border-color:rgba(108,99,255,0.25);
    color:#cbd5e1;
    transform:translateY(-1px);
  }
  .word-chip.active{
    background:linear-gradient(135deg,#6c63ff,#a78bfa);
    border-color:transparent;
    color:#fff;
    transform:scale(1.04);
    box-shadow:0 8px 22px rgba(108,99,255,0.45),inset 0 1px 0 rgba(255,255,255,0.18);
    font-weight:800;
  }
  /* Hizli Cumleler — word-chip ile ayni gorsel dile sahip */
  .qa-grid{
    display:grid;
    grid-template-columns:repeat(auto-fill,minmax(140px,1fr));
    gap:6px;
  }
  .qa-chip{
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.06);
    border-radius:8px;
    padding:9px 10px;
    font-size:0.78rem;font-weight:600;
    color:#cbd5e1;
    text-align:center;
    cursor:pointer;
    font-family:inherit;
    transition:all 0.18s;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  }
  .qa-chip:hover{
    background:linear-gradient(135deg,rgba(108,99,255,0.18),rgba(167,139,250,0.10));
    border-color:rgba(108,99,255,0.4);
    color:#fff;
    transform:translateY(-1px);
    box-shadow:0 6px 18px rgba(108,99,255,0.25);
  }
  .qa-chip:active{transform:scale(0.97)}
  .word-empty{
    display:none;
    text-align:center;padding:18px 8px;
    color:#475569;font-size:0.82rem;font-weight:500;font-style:italic;
  }
  .word-empty.show{display:block}

  /* Durum indikatoru */
  .status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #475569;
    display: inline-block; margin-right: 6px;
    transition: all 0.3s;
  }
  .status-dot.active {
    background: #10b981;
    box-shadow: 0 0 12px #10b981;
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  /* Cooldown bar */
  #cooldownBar {
    height: 3px;
    background: linear-gradient(90deg, #06b6d4, #22d3ee);
    border-radius: 2px;
    margin-top: 8px;
    width: 0%;
    transition: width 0.1s linear;
    box-shadow: 0 0 10px rgba(6,182,212,0.5);
  }
</style>
</head>
<body>

<header>
  <h1>SignBridge</h1>
  <span>Türk İşaret Dili Çevirisi</span>
  <div class="pill" id="modelInfo">Yükleniyor...</div>
  <div style="margin-left:auto; display:flex; align-items:center; font-size:0.85rem; color:#6b7280;">
    <span class="status-dot" id="statusDot"></span>
    <span id="statusText">Hazır</span>
  </div>
</header>

<div class="main">
  <!-- Sol: Iki kamera yan yana (Landmark + Raw) -->
  <div class="camera-section">
    <div class="dual-cam">
      <div class="camera-wrapper">
        <div class="camera-placeholder" id="placeholder">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9A2.25 2.25 0 0013.5 5.25h-9A2.25 2.25 0 002.25 9.75v9A2.25 2.25 0 004.5 18.75z"/>
          </svg>
          <p>Kamerayı başlatmak için aşağıdaki butona bas</p>
        </div>
        <img id="cameraFeed" style="display:none;" alt="Landmark">
        <div class="pred-overlay" style="display:none;" id="predOverlay">
          <span id="predWord">—</span>
          <span id="predConf">%0</span>
        </div>
        <div class="conf-bar-wrap"><div id="confBar"></div></div>
        <div class="cam-label">📍 Landmark</div>
      </div>
      <div class="camera-wrapper">
        <img id="cameraFeedClean" style="display:none;" alt="Ham">
        <div class="cam-label">🎥 Ham</div>
      </div>
    </div>

    <div class="controls">
      <button id="btnStart" onclick="toggleCamera()">Kamerayı Başlat</button>
      <button id="btnAdd" onclick="manualAdd()" style="background:#22c55e;color:#fff;font-size:1.1rem;padding:12px 28px;">
        ✋ Kelime Ekle (Space)
      </button>
      <button id="btnClear" onclick="clearSentence()">Temizle</button>
      <button onclick="saveSentence()" style="background:#1e293b;color:#94a3b8;border:1px solid #334155;">
        Kaydet
      </button>
      <button id="btnSpeak" onclick="speakNow()" style="background:#0f172a;color:#67e8f9;border:1px solid #155e75;">
        🔊 Sesli Oku
      </button>
      <label style="display:flex;align-items:center;gap:6px;color:#94a3b8;font-size:0.8rem;cursor:pointer;">
        <input type="checkbox" id="autoMode" checked> Otomatik Ekle
      </label>
      <label style="display:flex;align-items:center;gap:6px;color:#94a3b8;font-size:0.8rem;cursor:pointer;">
        <input type="checkbox" id="autoSpeak" checked> Otomatik Sesli Oku
      </label>
      <label id="demoModeLabel" style="display:flex;align-items:center;gap:6px;color:#22d3ee;font-size:0.8rem;cursor:pointer;border:1px solid #155e75;padding:6px 10px;border-radius:8px;background:rgba(6,182,212,0.08);">
        <input type="checkbox" id="demoMode" onchange="toggleDemoMode()"> 🎯 Demo Modu (29 kelime)
      </label>
    </div>
  </div>

  <!-- Sag: Panel -->
  <div class="side-panel">

    <!-- Cumle -->
    <div class="card">
      <div class="card-title">İşaret Kelimeleri</div>
      <div id="sentenceBox" class="empty">Henüz kelime algılanmadı...</div>
      <div id="cooldownBar"></div>
    </div>
    <div class="card">
      <div class="card-title">Türkçe Cümle</div>
      <div id="nlpBox" style="min-height:38px;font-size:1.1rem;font-weight:600;color:#a78bfa;padding:4px 0;">—</div>
    </div>

    <!-- Gecmis -->
    <div class="card">
      <div class="card-title">Geçmiş Cümleler</div>
      <div id="historyList">
        <div style="color:#4b5563; font-size:0.85rem;">Henüz cümle yok</div>
      </div>
    </div>

    <!-- Hizli Cumleler (10 demo cumlesi — avatar ile ayni) -->
    <div class="card qa-card">
      <div class="card-title">⚡ Hızlı Cümleler</div>
      <div class="qa-grid qa-phrases">
        <button class="qa-chip phrase" onclick="addPhrase(['MERHABA','NASILSIN'])">Merhaba nasılsın</button>
        <button class="qa-chip phrase" onclick="addPhrase(['IYI','SEN','NE','YAPMAK'])">İyiyim sen ne yapıyorsun</button>
        <button class="qa-chip phrase" onclick="addPhrase(['BEN','CALISMAK','COK','YORULMAK'])">Ben çalışıyorum çok yoruldum</button>
        <button class="qa-chip phrase" onclick="addPhrase(['SAAT','KAC','BITMEK'])">Saat kaçta bitiyor</button>
        <button class="qa-chip phrase" onclick="addPhrase(['AKSAM','BES','SONRA','BOS'])">Akşam beşten sonra boşum</button>
        <button class="qa-chip phrase" onclick="addPhrase(['BERABER','KAHVE','ICMEK'])">Beraber kahve içelim mi</button>
        <button class="qa-chip phrase" onclick="addPhrase(['TAMAM','NEREDE','BULUSMAK'])">Tamam nerede buluşalım</button>
        <button class="qa-chip phrase" onclick="addPhrase(['PARK','YAN','KAFE'])">Parkın yanındaki kafede</button>
        <button class="qa-chip phrase" onclick="addPhrase(['TAMAM','GORUSMEK'])">Tamam görüşürüz</button>
        <button class="qa-chip phrase" onclick="addPhrase(['KENDI','IYI','BAKMAK'])">Kendine iyi bak</button>
      </div>
    </div>

    <!-- Kelime listesi -->
    <div class="card word-card">
      <div class="word-card-head">
        <div class="card-title" style="margin-bottom:0">Bilinen Kelimeler</div>
        <span class="word-count-badge"><span id="wordCount">0</span></span>
      </div>
      <div class="word-search-wrap">
        <svg class="word-search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
        </svg>
        <input id="wordSearch" placeholder="Kelime ara..." oninput="filterWords()">
      </div>
      <div id="wordList"></div>
      <div id="wordEmpty" class="word-empty">Eşleşen kelime yok</div>
    </div>

  </div>
</div>

<script>
const socket = io();
let isRunning = false;
let sentence = [];
let history = [];
let lastPredWord = '';
let lastPredConf = 0;

function filterWords(){
  const q = (document.getElementById('wordSearch').value || '').toUpperCase().trim();
  let visible = 0;
  document.querySelectorAll('.word-chip').forEach(c => {
    const w = c.textContent.toUpperCase();
    const match = !q || w.includes(q);
    c.style.display = match ? '' : 'none';
    if(match) visible++;
  });
  const empty = document.getElementById('wordEmpty');
  if(empty) empty.classList.toggle('show', visible === 0 && q.length > 0);
}

function manualAdd() {
  if (lastPredWord && lastPredConf >= 0.70) {
    // Ayni kelimeyi art arda ekleme
    if (sentence.length > 0 && sentence[sentence.length-1] === lastPredWord) return;
    sentence.push(lastPredWord);
    updateSentenceBox();
    socket.emit("force_add");
    // Flash efekti
    const btn = document.getElementById("btnAdd");
    btn.style.background = "#a78bfa";
    setTimeout(() => btn.style.background = "#22c55e", 300);
  }
}

window.addPhrase = function(words) {
  if (!Array.isArray(words) || words.length === 0) return;
  socket.emit("clear_sentence");
  sentence = [];
  words.forEach(w => {
    socket.emit("add_word_manual", { word: w });
  });
};

socket.on("sentence_update", (data) => {
  if (data && Array.isArray(data.sentence)) {
    sentence = data.sentence;
    updateSentenceBox();
  }
});

// Model bilgisi al
socket.emit("get_info");
socket.on("info", (data) => {
  document.getElementById("modelInfo").textContent = data.num_classes + " kelime";
  document.getElementById("wordCount").textContent = data.num_classes;
  const wl = document.getElementById("wordList");
  wl.innerHTML = "";
  data.words.forEach(w => {
    const chip = document.createElement("div");
    chip.className = "word-chip";
    chip.id = "chip_" + w;
    chip.style.cursor = "pointer";
    chip.title = "Cümleye eklemek için tıkla";
    chip.onclick = () => socket.emit("add_word_manual", { word: w });
    // Turkce karakterli goster (GLOSS_TO_TURKISH'ten al)
    const display = GLOSS_TO_TURKISH[w] || w.toLowerCase();
    chip.textContent = display.charAt(0).toUpperCase() + display.slice(1);
    wl.appendChild(chip);
  });
});

// Tahmin guncelleme
socket.on("prediction", (data) => {
  const word = data.word;
  const conf = data.conf;
  const added = data.added;
  const sentenceData = data.sentence;

  lastPredWord = word;
  lastPredConf = conf;

  // Anlik tahmin goster — sadece emin tahminleri (gecis gurultusunu filtrele)
  if (conf >= 0.65) {
    document.getElementById("predWord").textContent = word || "—";
    document.getElementById("predConf").textContent = "%" + Math.round(conf * 100);
    document.getElementById("confBar").style.width = (conf * 100) + "%";
  } else {
    document.getElementById("predWord").textContent = "—";
    document.getElementById("predConf").textContent = "%" + Math.round(conf * 100);
    document.getElementById("confBar").style.width = (conf * 100) + "%";
  }

  // Aktif kelime chipini vurgula
  document.querySelectorAll(".word-chip").forEach(c => c.classList.remove("active"));
  const chip = document.getElementById("chip_" + word);
  if (chip && conf >= 0.80) chip.classList.add("active");

  // Cumle guncelle
  sentence = sentenceData;
  updateSentenceBox();

  // Kelime eklendiyse cooldown animasyonu
  if (added) {
    const bar = document.getElementById("cooldownBar");
    bar.style.width = "100%";
    setTimeout(() => { bar.style.width = "0%"; }, 1200);
  }

  // Eller cekildi, cumle bitti
  if (data.done && sentence.length > 0) {
    const gloss = sentence.join(" ");
    const turk = glossToSentence(sentence);
    history.unshift({gloss, turk});
    if (history.length > 10) history.pop();

    const hl = document.getElementById("historyList");
    hl.innerHTML = history.map(h =>
      `<div class="history-item"><div style="color:#a78bfa;font-weight:600">${h.turk}</div><div style="font-size:0.75rem;color:#4b5563;margin-top:2px">${h.gloss}</div></div>`
    ).join("");

    // Otomatik sesli oku (cumle silinmeden once)
    const autoSp = document.getElementById('autoSpeak');
    if (autoSp && autoSp.checked && typeof window.speakNow === 'function') {
      window.speakNow();
    }

    sentence = [];
    updateSentenceBox();
  }
});

// ═══════════════════════════════════════════════
// NLP: Gloss kelimeleri → Düzgün Türkçe cümle
// ═══════════════════════════════════════════════
const GLOSS_TO_TURKISH = {
  'MERHABA':'merhaba', 'TESEKKUR':'teşekkür ederim', 'EVET':'evet', 'HAYIR':'hayır',
  'IYI':'iyi', 'KOTU':'kötü', 'BUYUK':'büyük', 'KUCUK':'küçük',
  'BEN':'ben', 'SEN':'sen', 'BABA':'baba', 'ANNE':'anne', 'ABLA':'abla',
  'ABI':'abi', 'AMCA':'amca', 'ARKADAS':'arkadaş', 'OGRETMEN':'öğretmen',
  'DOKTOR':'doktor', 'HEMSIRE':'hemşire',
  'OKUL':'okul', 'EV':'ev', 'TUVALET':'tuvalet', 'SU':'su',
  'CAY':'çay', 'KAHVE':'kahve', 'EKMEK':'ekmek', 'YEMEK':'yemek',
  'CIKOLATA':'çikolata', 'BAKLAVA':'baklava', 'AYRAN':'ayran', 'ERIK':'erik',
  'SABAH':'sabah', 'AKSAM':'akşam', 'BUGUN':'bugün', 'DUN':'dün', 'GUN':'gün',
  'KIS':'kış',
  'BIR':'bir', 'IKI':'iki', 'UC':'üç', 'DORT':'dört', 'BES':'beş',
  'ALTI':'altı', 'YEDI':'yedi', 'SEKIZ':'sekiz', 'DOKUZ':'dokuz', 'SIFIR':'sıfır',
  'OCAK':'ocak', 'SUBAT':'şubat', 'MART':'mart', 'NISAN':'nisan',
  'MAYIS':'mayıs', 'HAZIRAN':'haziran', 'TEMMUZ':'temmuz', 'AGUSTOS':'ağustos',
  'EYLUL':'eylül', 'EKIM':'ekim', 'KASIM':'kasım', 'ARALIK':'aralık',
  'PAZARTESI':'pazartesi', 'SALI':'salı', 'CARSAMBA':'çarşamba',
  'PERSEMBE':'perşembe', 'CUMA':'cuma', 'CUMARTESI':'cumartesi', 'PAZAR':'pazar',
  'ASK':'aşk', 'YARDIM':'yardım', 'DIKKAT':'dikkat', 'CEZA':'ceza',
  'FUTBOL':'futbol', 'VOLEYBOL':'voleybol', 'HENTBOL':'hentbol',
  'TV':'TV', 'CD':'CD', 'BILGISAYAR':'bilgisayar',
  'AKILLI':'akıllı', 'APTAL':'aptal', 'ZENGIN':'zengin',
  'ALERJI':'alerji', 'AYIP':'ayıp', 'ORUC':'oruç', 'FINAL':'final',
  'ALLAH':'Allah', 'AMIN':'amin', 'ELHAMDULILLAH':'elhamdülillah',
  'ATATURK':'Atatürk', 'ARAP':'Arap', 'BARIS':'barış',
  'EVLENMEK':'evlenmek', 'YATMAK':'yatmak', 'DOYMAK':'doymak',
  'HANGI':'hangi', 'CUNKU':'çünkü', 'YETER':'yeter',
  'HOSCA KAL':'hoşça kal', 'BOS VERMEK':'boş vermek', 'BU KADAR':'bu kadar',
  'CEP TELEFONU':'cep telefonu', 'LUTFEN':'lütfen',
  // ─── Yeni 22 kelime (122 sinifa upgrade) ───
  'NASILSIN':'nasılsın', 'NE':'ne', 'KAC':'kaç', 'NEREDE':'nerede',
  'YAPMAK':'yapmak', 'CALISMAK':'çalışmak', 'YORULMAK':'yorulmak',
  'BITMEK':'bitmek', 'ICMEK':'içmek', 'BULUSMAK':'buluşmak',
  'GORUSMEK':'görüşmek', 'BAKMAK':'bakmak',
  'COK':'çok', 'SAAT':'saat', 'SONRA':'sonra', 'BOS':'boş',
  'BERABER':'beraber', 'TAMAM':'tamam',
  'PARK':'park', 'YAN':'yan', 'KAFE':'kafe', 'KENDI':'kendi',
};

// Cümle kalıpları — gloss sırasına göre Türkçe cümle üret
// AKILLI ÇEKİM MODÜLÜ — gloss dizisini doğal Türkçe cümleye çevirir
// Özne (BEN/SEN) takibi, fiil çekimi, soru tespiti, özel kalıplar.
const VERB_CONJ = {
  'YAPMAK':   { ben:'yapıyorum',   sen:'yapıyorsun',   _:'yapıyor' },
  'CALISMAK': { ben:'çalışıyorum', sen:'çalışıyorsun', _:'çalışıyor' },
  'YORULMAK': { ben:'yoruldum',    sen:'yoruldun',     _:'yoruldu' },
  'BITMEK':   { ben:'bitiyor',     sen:'bitiyor',      _:'bitiyor' },
  'ICMEK':    { ben:'içiyorum',    sen:'içiyorsun',    _:'içelim' },
  'BULUSMAK': { ben:'buluşalım',   sen:'buluşalım',    _:'buluşalım' },
  'GORUSMEK': { ben:'görüşürüz',   sen:'görüşürüz',    _:'görüşürüz' },
  'BAKMAK':   { ben:'bakıyorum',   sen:'bak',          _:'bak' },
  'EVLENMEK': { ben:'evleniyorum', sen:'evleniyorsun', _:'evleniyor' },
  'YATMAK':   { ben:'yatıyorum',   sen:'yatıyorsun',   _:'yatıyor' },
  'DOYMAK':   { ben:'doydum',      sen:'doydun',       _:'doydu' },
  'BOS VERMEK':{ ben:'boş veriyorum', sen:'boş ver',   _:'boş ver' },
};
const PHRASE_OVERRIDE = {
  'TESEKKUR': 'teşekkür ederim',
  'NASILSIN': 'nasılsın',
  'HOSCA KAL': 'hoşça kal',
  'BU KADAR': 'bu kadar',
  'CEP TELEFONU': 'cep telefonu',
  'AFERIN': 'aferin',
  'ELHAMDULILLAH': 'elhamdülillah',
  'AMIN': 'amin',
  'EVET': 'evet',
  'YETER': 'yeter',
  'DIKKAT': 'dikkat',
  'YARDIM': 'yardım',
  'KENDI': 'kendine',  // "kendine iyi bak"
};
const QUESTION_GLOSS = ['NE','KAC','NEREDE','HANGI','NASILSIN','SORU'];

function glossToSentence(glossWords) {
  if (!glossWords || glossWords.length === 0) return '—';

  const out = [];
  let lastSubject = null;        // KORU: yeni özne gelmedikçe persistent
  let lastWasVerb = false;       // virgül için

  // Lookahead için: subject context tüm cümleyi tarar
  const hasBen = glossWords.includes('BEN');
  const hasSen = glossWords.includes('SEN');
  // BERABER var ise cohortative: "icelim, gidelim, bulusalim" + sona "mi?"
  const hasBeraber = glossWords.includes('BERABER');
  // KENDI var ise imperative: "kendine iyi bak", fiiller mastar/emir hali, IYI plain
  const hasKendi = glossWords.includes('KENDI');
  // Tekrarlanan kelimeyi tek seferine indir (YAPMAK YAPMAK -> tek YAPMAK)
  glossWords = glossWords.filter((w, i) => i === 0 || w !== glossWords[i-1]);
  // Cumle basinda IYI ozne yokken "iyiyim" varsayalim (BEN IYI yoksa)
  const startsWithIyi = glossWords[0] === 'IYI' && !hasSen;
  // Virgul sonrasi gelen ozel sozler
  const COMMA_AFTER = new Set(['TAMAM','EVET','HOSCA KAL','AFERIN','MERHABA']);

  // Sayı + SONRA -> ablativ ek (beşten sonra)
  const NUM_ABL = {'BIR':'birden','IKI':'ikiden','UC':'üçten','DORT':'dörtten',
                   'BES':'beşten','ALTI':'altıdan','YEDI':'yediden','SEKIZ':'sekizden',
                   'DOKUZ':'dokuzdan','ON':'ondan'};
  // Locative ek üreten basit fonksiyon
  function locSuf(w) {
    const last = w.slice(-1);
    const vowels = w.match(/[aeıioöuü]/gi) || [];
    const lv = vowels.length ? vowels[vowels.length-1].toLowerCase() : 'a';
    const voiceless = 'çfhkpsşt'.includes(last);
    const front = 'eiöü'.includes(lv);
    return (voiceless ? 't' : 'd') + (front ? 'e' : 'a');
  }

  for (let i = 0; i < glossWords.length; i++) {
    const w = glossWords[i];
    const prev = i > 0 ? glossWords[i-1] : null;
    const next = i < glossWords.length - 1 ? glossWords[i+1] : null;
    const next2 = i < glossWords.length - 2 ? glossWords[i+2] : null;

    // Sayı + SONRA → "{sayı}-ten sonra"
    if (NUM_ABL[w] && next === 'SONRA') {
      out.push(NUM_ABL[w]);
      out.push('sonra');
      i++; // SONRA'yı da geç
      lastWasVerb = false;
      continue;
    }

    // PARK + YAN + isim → "parkın yanındaki {isim}-de"
    if (w === 'PARK' && next === 'YAN' && next2 && GLOSS_TO_TURKISH[next2]) {
      const nounTr = GLOSS_TO_TURKISH[next2];
      out.push('parkın');
      out.push('yanındaki');
      out.push(nounTr + locSuf(nounTr));
      i += 2; // YAN + nesne'yi atla
      lastWasVerb = false;
      continue;
    }
    // YAN tek başına: "yan" değil "yanı"
    if (w === 'YAN') {
      out.push('yanı');
      lastWasVerb = false;
      continue;
    }

    // BOS — özne yoksa 1. tekil varsay (kendinden bahis)
    if (w === 'BOS') {
      const s = lastSubject || (hasSen && !hasBen ? 'sen' : 'ben');
      if (s === 'sen') out.push('boşsun');
      else out.push('boşum');
      lastWasVerb = false;
      continue;
    }

    // SAAT + KAC -> "saat kaçta"
    if (w === 'KAC' && prev === 'SAAT') {
      out.push('kaçta'); lastWasVerb = false; continue;
    }
    // SAAT + sayı -> "saat beşte/altıda"
    if (['BES','ALTI','YEDI','SEKIZ','DOKUZ','UC','DORT','IKI','BIR','ON'].includes(w) && prev === 'SAAT') {
      const numMap = {'BIR':'birde','IKI':'ikide','UC':'üçte','DORT':'dörtte','BES':'beşte',
                      'ALTI':'altıda','YEDI':'yedide','SEKIZ':'sekizde','DOKUZ':'dokuzda','ON':'onda'};
      out.push(numMap[w] || (GLOSS_TO_TURKISH[w]+'te'));
      lastWasVerb = false; continue;
    }

    // Özne tespiti
    if (w === 'BEN') { lastSubject = 'ben'; out.push('ben'); lastWasVerb = false; continue; }
    if (w === 'SEN') { lastSubject = 'sen'; out.push('sen'); lastWasVerb = false; continue; }

    // IYI - KENDI baglamda plain "iyi", yoksa ozne'ye gore
    if (w === 'IYI') {
      if (hasKendi) out.push('iyi');
      else if (lastSubject === 'ben' || (i === 0 && hasBen) || (i === 0 && !hasSen)) out.push('iyiyim');
      else if (lastSubject === 'sen' || (i === 0 && hasSen)) out.push('iyisin');
      else out.push('iyi');
      lastWasVerb = false; continue;
    }

    // Virgul-sonrasi sozler (TAMAM, EVET, MERHABA, HOSCA KAL, AFERIN)
    if (COMMA_AFTER.has(w)) {
      const tr = (GLOSS_TO_TURKISH[w] || w).toLowerCase();
      // Sonraki kelime varsa virgul ekle
      const tail = (i < glossWords.length - 1) ? ',' : '';
      out.push(tr + tail);
      lastWasVerb = false;
      continue;
    }

    // Fiil çekimi
    if (VERB_CONJ[w]) {
      let subj;
      // KENDI varsa imperative (BAKMAK -> "bak"), BERABER varsa cohortative ("icelim")
      if (hasKendi && !hasBen && !hasSen) subj = 'sen';   // 2. tekil emir formu (sen subj BAKMAK -> "bak")
      else if (hasBeraber && !hasBen && !hasSen) subj = '_';
      else subj = lastSubject || (hasSen && !hasBen ? 'sen' : 'ben');
      if (lastWasVerb && out.length > 0) {
        out[out.length - 1] += ',';
      }
      out.push(VERB_CONJ[w][subj] || VERB_CONJ[w]['_']);
      lastWasVerb = true;
      continue;
    }

    // Özel kalıp
    if (PHRASE_OVERRIDE[w]) { out.push(PHRASE_OVERRIDE[w]); lastWasVerb = false; continue; }

    // Sözlük (GLOSS_TO_TURKISH) veya küçük harf
    out.push((GLOSS_TO_TURKISH[w] || w).toLowerCase());
    lastWasVerb = false;
  }

  let s = out.join(' ');
  s = s.replace(/\s+([?!.,])/g, '$1');  // boşluk + noktalama → noktalama
  s = s.replace(/,\s*$/, '');           // sondaki virgülü temizle

  // İlk harf büyük
  s = s.charAt(0).toUpperCase() + s.slice(1);

  // Soru mu cümle mi? (BERABER ile cohortative de soru hissi verir: "icelim mi?")
  const hasQ = glossWords.some(w => QUESTION_GLOSS.includes(w));
  if (hasQ || (hasBeraber && !hasBen && !hasSen)) {
    // BERABER ile fiil cohortative bittiyse "mi" ekle
    if (hasBeraber && !hasQ && !s.endsWith(' mi') && !s.endsWith(' mı') && !s.endsWith(' mu') && !s.endsWith(' mü')) {
      s = s.replace(/[.!?]?$/, '') + ' mi';
    }
    if (!/[?!]$/.test(s)) s += '?';
  } else {
    if (!/[.!?]$/.test(s)) s += '.';
  }
  return s;
}

// Otomatik sesli okuma zamanlayicisi — son kelime ekleninceye kadar bekler
let _autoSpeakTimer = null;
function _scheduleAutoSpeak() {
  const cb = document.getElementById('autoSpeak');
  if (!cb || !cb.checked) return;
  if (_autoSpeakTimer) clearTimeout(_autoSpeakTimer);
  _autoSpeakTimer = setTimeout(() => {
    if (sentence && sentence.length > 0 && typeof window.speakNow === 'function') {
      window.speakNow();
    }
  }, 1500);  // son ek-len-mesinden 1.5 sn sonra
}

function updateSentenceBox() {
  const box = document.getElementById("sentenceBox");
  const nlpBox = document.getElementById("nlpBox");
  if (sentence.length === 0) {
    box.textContent = "Henüz kelime algılanmadı...";
    box.classList.add("empty");
    nlpBox.textContent = "—";
  } else {
    box.textContent = sentence.join(" ");
    box.classList.remove("empty");
    nlpBox.textContent = glossToSentence(sentence);
    _scheduleAutoSpeak();
  }
}

function toggleCamera() {
  if (!isRunning) {
    socket.emit("start_camera");
    isRunning = true;
    document.getElementById("btnStart").textContent = "Kamerayı Durdur";
    document.getElementById("btnStart").classList.add("active");
    document.getElementById("placeholder").style.display = "none";
    document.getElementById("cameraFeed").style.display = "block";
    const cf2 = document.getElementById("cameraFeedClean"); if (cf2) cf2.style.display = "block";
    document.getElementById("predOverlay").style.display = "flex";
    document.getElementById("statusDot").classList.add("active");
    document.getElementById("statusText").textContent = "Canlı";
  } else {
    socket.emit("stop_camera");
    isRunning = false;
    document.getElementById("btnStart").textContent = "Kamerayı Başlat";
    document.getElementById("btnStart").classList.remove("active");
    document.getElementById("placeholder").style.display = "flex";
    document.getElementById("cameraFeed").style.display = "none";
    const cf2b = document.getElementById("cameraFeedClean"); if (cf2b) cf2b.style.display = "none";
    document.getElementById("predOverlay").style.display = "none";
    document.getElementById("statusDot").classList.remove("active");
    document.getElementById("statusText").textContent = "Durduruldu";
  }
}

function clearSentence() {
  socket.emit("clear_sentence");
  sentence = [];
  updateSentenceBox();
}

// ─── TTS (Sesli Okuma) ───
// ─── Demo Modu (sadece 10 cumlede gecen kelimeleri tahmin et) ───
window.toggleDemoMode = function() {
  const cb = document.getElementById('demoMode');
  const enabled = cb && cb.checked;
  socket.emit('set_demo_mode', { enabled });
  const lbl = document.getElementById('demoModeLabel');
  if (lbl) {
    lbl.style.background = enabled ? 'rgba(6,182,212,0.25)' : 'rgba(6,182,212,0.08)';
    lbl.style.color      = enabled ? '#a5f3fc' : '#22d3ee';
  }
};
socket.on('demo_mode_status', (d) => {
  console.log('[DEMO]', d.enabled ? 'AKTIF' : 'KAPALI', '— ' + d.words.length + ' kelime');
});

window.speakNow = function() {
  // Once sentence array, sonra UI'daki nlpBox/sentenceBox'tan al
  let text = '';
  if (sentence && sentence.length > 0) {
    text = glossToSentence(sentence);
  } else {
    const nlp = document.getElementById('nlpBox');
    if (nlp && nlp.textContent && nlp.textContent.trim() !== '—') {
      text = nlp.textContent.trim();
    }
  }
  console.log('[TTS] speakNow text=', JSON.stringify(text));
  if (!text || text === '—') {
    console.warn('[TTS] Bos cumle, okunmayacak');
    return;
  }
  if (!('speechSynthesis' in window)) {
    alert('Bu tarayıcı sesli okumayı desteklemiyor.');
    return;
  }
  try { speechSynthesis.cancel(); } catch(_) {}
  const utt = new SpeechSynthesisUtterance(text);
  utt.lang = 'tr-TR';
  utt.rate = 0.95;
  utt.volume = 1.0;
  const voices = speechSynthesis.getVoices();
  console.log('[TTS] available voices:', voices.length, voices.filter(v=>v.lang.startsWith('tr')).map(v=>v.name));
  const trVoice = voices.find(v => v.lang.toLowerCase().startsWith('tr'));
  if (trVoice) {
    utt.voice = trVoice;
    console.log('[TTS] using TR voice:', trVoice.name);
  } else {
    console.warn('[TTS] TR ses yok, default kullanilacak');
  }
  utt.onstart = () => console.log('[TTS] start');
  utt.onend   = () => console.log('[TTS] end');
  utt.onerror = (e) => console.error('[TTS] error:', e);
  speechSynthesis.speak(utt);
};
if ('speechSynthesis' in window) {
  speechSynthesis.getVoices();
  speechSynthesis.onvoiceschanged = () => {
    const v = speechSynthesis.getVoices();
    console.log('[TTS] voices loaded:', v.length, 'TR:', v.filter(x=>x.lang.startsWith('tr')).length);
  };
}

function saveSentence() {
  if (sentence.length === 0) return;
  const gloss = sentence.join(" ");
  const turk = glossToSentence(sentence);
  history.unshift({gloss, turk});
  if (history.length > 10) history.pop();
  // Auto sesli oku
  if (document.getElementById('autoSpeak') && document.getElementById('autoSpeak').checked) {
    speakNow();
  }

  const hl = document.getElementById("historyList");
  hl.innerHTML = history.map(h =>
    `<div class="history-item"><div style="color:#a78bfa;font-weight:600">${h.turk}</div><div style="font-size:0.75rem;color:#4b5563;margin-top:2px">${h.gloss}</div></div>`
  ).join("");

  clearSentence();
}

// Video frame al
socket.on("frame", (data) => {
  document.getElementById("cameraFeed").src = "data:image/jpeg;base64," + data.img;
  if (data.clean) {
    const cf2 = document.getElementById("cameraFeedClean");
    if (cf2) cf2.src = "data:image/jpeg;base64," + data.clean;
  }
});

// Klavye kisayollari
document.addEventListener("keydown", (e) => {
  if (e.key === "c" || e.key === "C") clearSentence();
  if (e.key === "s" || e.key === "S") saveSentence();
  if (e.key === " ") { e.preventDefault(); manualAdd(); }
});
</script>
</body>
</html>
"""

# ---- SocketIO olaylar ----
@socketio.on("connect")
def handle_connect():
    # Sayfa yenilenince HER SEYI sifirla
    state["running"] = False
    if state["cap"] is not None:
        try: state["cap"].release()
        except: pass
        state["cap"] = None
    state["sentence"] = []
    state["buffer"] = deque(maxlen=SEQ_LEN)
    state["smooth_counts"] = {k: 0 for k in label_map}
    state["current_word"] = ""
    state["current_conf"] = 0.0
    state["last_add_time"] = 0.0

@socketio.on("get_info")
def handle_info():
    emit("info", {
        "num_classes": num_classes,
        "words": list(label_map.keys()),
    })

@socketio.on("set_demo_mode")
def handle_set_demo_mode(data):
    enabled = bool((data or {}).get("enabled", False))
    state["demo_mode"] = enabled
    state["smooth_counts"] = {k: 0 for k in label_map}
    print(f"[DEMO MODE] {'AKTIF' if enabled else 'KAPALI'} ({len(DEMO_WORDS)} kelime)", flush=True)
    emit("demo_mode_status", {"enabled": enabled, "words": sorted(DEMO_WORDS)})

@socketio.on("clear_sentence")
def handle_clear():
    state["sentence"] = []
    state["smooth_counts"] = {k: 0 for k in label_map}

@socketio.on("force_add")
def handle_force_add():
    word = state["current_word"]
    conf = state["current_conf"]
    if word and conf >= CONF_THRESHOLD:
        state["sentence"].append(word)
        state["last_add_time"] = time.time()
        state["smooth_counts"] = {k: 0 for k in label_map}

@socketio.on("add_word_manual")
def handle_add_word_manual(data):
    """Client tarafindan dogrudan bir gloss ekleme.
    Hizli Cumleler / Hizli Kelimeler butonlari icin.
    Server state.sentence'i guncellenir + tum baglilara sentence_update yayinlanir.
    """
    word = (data or {}).get("word", "")
    if isinstance(word, str):
        word = word.strip().upper()
    if not word or word not in label_map:
        socketio.emit("sentence_update", {"sentence": list(state["sentence"])})
        return
    # Ayni kelimeyi pespese ekleme
    if state["sentence"] and state["sentence"][-1] == word:
        socketio.emit("sentence_update", {"sentence": list(state["sentence"])})
        return
    if len(state["sentence"]) >= 20:
        state["sentence"].pop(0)
    state["sentence"].append(word)
    state["last_add_time"] = time.time()
    state["smooth_counts"] = {k: 0 for k in label_map}
    socketio.emit("sentence_update", {"sentence": list(state["sentence"])})

@socketio.on("start_camera")
def handle_start():
    if state["running"]:
        return
    state["running"] = True
    state["buffer"]  = deque(maxlen=SEQ_LEN)
    state["smooth_counts"] = {k: 0 for k in label_map}

    cap = cv2.VideoCapture(0)
    state["cap"] = cap
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,            # collect.py ile uyumlu, 2x hizli
        min_detection_confidence=0.3,  # ilk tespit
        min_tracking_confidence=0.5,   # tracking guveni dustugunde yeni tespit yap (drift azalir)
        smooth_landmarks=True,
        refine_face_landmarks=False
    )
    state["holistic"] = holistic

    import threading
    def camera_loop():
        mp_draw = mp.solutions.drawing_utils
        try:
            while state["running"]:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                # Raw frame'in kopyasi (landmark CIZILMEDEN, yan kameraya gonderilecek)
                frame_clean = frame.copy()

                # Landmark ciz (sol gosterim)
                mp_draw.draw_landmarks(frame, results.left_hand_landmarks,
                                        mp.solutions.hands.HAND_CONNECTIONS)
                mp_draw.draw_landmarks(frame, results.right_hand_landmarks,
                                        mp.solutions.hands.HAND_CONNECTIONS)
                mp_draw.draw_landmarks(frame, results.pose_landmarks,
                                        mp.solutions.pose.POSE_CONNECTIONS)

                hand_visible = detect_active_hands(results)
                now_t = time.time()
                sentence_done = False

                if hand_visible:
                    state["last_hand_time"] = now_t
                    state["pause_reset_done"] = False
                    lm = extract_landmarks(results)
                    lm_norm = normalize_landmarks(lm)      # V2: 2-katman normalize
                    state["buffer"].append(lm_norm)
                else:
                    # Inter-sign pause: eller 500ms+ kaybolunca buffer ve olasilik gecmisini sifirla
                    if (not state.get("pause_reset_done") and state["last_hand_time"] > 0 and
                        (now_t - state["last_hand_time"]) >= INTER_SIGN_PAUSE_SEC):
                        state["buffer"].clear()
                        state["prob_history"].clear()
                        state["smooth_counts"] = {k: 0 for k in label_map}
                        state["current_word"] = ""
                        state["current_conf"] = 0.0
                        state["pause_reset_done"] = True

                    # El cekileli HAND_GONE_SEC gecti ve cumle varsa → cumle bitti
                    if (state["sentence"] and state["last_hand_time"] > 0 and
                        (now_t - state["last_hand_time"]) >= HAND_GONE_SEC):
                        sentence_done = True
                        state["last_hand_time"] = 0.0  # tekrar tetikleme

                pred_word = ""
                pred_conf = 0.0
                added     = False

                if hand_visible and len(state["buffer"]) == SEQ_LEN:
                    seq = np.array(state["buffer"], dtype=np.float32)   # (30, 1629) normalize
                    seq = add_velocity(seq)                              # (30, 1755)
                    seq = np.clip(seq, -5, 5)

                    raw_prob = run_model_probs(seq)
                    state["prob_history"].append(raw_prob)
                    prob = np.mean(state["prob_history"], axis=0)

                    top_idx = np.argsort(prob)[::-1][:15]
                    top_vals = prob[top_idx]
                    pred_conf   = float(top_vals[0])
                    top2_margin = float(top_vals[0] - top_vals[1])
                    pred_word   = idx_to_label[int(top_idx[0])]

                    # Demo modu: top-1 whitelist disinda ise, top-15 icinde whitelist ara
                    if state.get("demo_mode"):
                        demo_indices = [label_map[w] for w in DEMO_WORDS if w in label_map]
                        demo_probs = prob[demo_indices]
                        demo_sum = float(np.sum(demo_probs))
                        if demo_sum >= 0.35:
                            order = np.argsort(demo_probs)[::-1]
                            best_i = demo_indices[int(order[0])]
                            second_p = float(demo_probs[int(order[1])]) if len(order) > 1 else 0.0
                            pred_word = idx_to_label[best_i]
                            pred_conf = float(demo_probs[int(order[0])] / max(demo_sum, 1e-6)) * min(1.0, demo_sum + 0.35)
                            top2_margin = float(demo_probs[int(order[0])] - second_p)
                        else:
                            pred_word = ""
                            pred_conf = 0.0
                            top2_margin = 0.0

                    # TESHIS LOG (her 5 framede 1, server konsola yazar)
                    if hand_visible and (int(time.time()*5) % 5 == 0):
                        top5_str = " | ".join(
                            f"{idx_to_label[int(i)]}:{float(v)*100:.0f}%"
                            for v, i in list(zip(top_vals[:5].tolist(), top_idx[:5].tolist()))
                        )
                        print(f"[PRED] {top5_str}{'  [DEMO]' if state.get('demo_mode') else ''}", flush=True)

                    state["current_word"] = pred_word
                    state["current_conf"] = pred_conf

                    now = time.time()
                    in_cooldown = (now - state["last_add_time"]) < COOLDOWN_SEC

                    # Fast-track: yuksek guven + net margin -> hizli ekle
                    is_fast = (pred_conf >= CONF_FAST_TRACK and top2_margin >= MARGIN_FAST)
                    needed  = SMOOTH_FAST if is_fast else SMOOTH_NEEDED

                    # pred_word bos ise (demo modu reddetmis) smoothing'i SIFIRLAMA
                    if (pred_word and pred_conf >= CONF_THRESHOLD and
                        top2_margin >= MARGIN_MIN and hand_visible and not in_cooldown):
                        state["smooth_counts"][pred_word] += 1
                        for k in state["smooth_counts"]:
                            if k != pred_word:
                                state["smooth_counts"][k] = max(0, state["smooth_counts"][k] - 1)

                        if state["smooth_counts"][pred_word] >= needed:
                            # ── BAGLAM TEMELLI DUZELTMELER ──
                            if pred_word in ("KAHVE", "KAFE"):
                                last = state["sentence"][-1] if state["sentence"] else None
                                if last == "BERABER":
                                    pred_word = "KAHVE"   # "Beraber kahve icelim"
                                elif last in ("YAN", "PARK"):
                                    pred_word = "KAFE"    # "yanindaki kafede"
                            # Ayni kelimeyi art arda ekleme
                            if not (state["sentence"] and state["sentence"][-1] == pred_word):
                                if len(state["sentence"]) >= 20:
                                    state["sentence"].pop(0)
                                state["sentence"].append(pred_word)
                                added = True
                            state["last_add_time"] = now
                            state["smooth_counts"] = {k: 0 for k in label_map}
                            # FULL CLEAR: temiz baslangic
                            state["buffer"].clear()
                            state["prob_history"].clear()
                    else:
                        if not hand_visible or in_cooldown:
                            state["smooth_counts"] = {k: 0 for k in label_map}

                # Frame'i encode et (landmark'li)
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                img_b64 = base64.b64encode(buf).decode("utf-8")
                # Raw frame de encode et (landmark'siz, yan panel icin)
                _, buf_clean = cv2.imencode(".jpg", frame_clean, [cv2.IMWRITE_JPEG_QUALITY, 70])
                img_clean_b64 = base64.b64encode(buf_clean).decode("utf-8")

                socketio.emit("frame", {"img": img_b64, "clean": img_clean_b64})
                socketio.emit("prediction", {
                    "word":     pred_word if hand_visible else "",
                    "conf":     pred_conf if hand_visible else 0.0,
                    "added":    added,
                    "sentence": list(state["sentence"]),
                    "hand":     hand_visible,
                    "done":     sentence_done,
                })

                # Cumle bittiyse sifirla
                if sentence_done:
                    state["sentence"] = []
                    state["smooth_counts"] = {k: 0 for k in label_map}
        finally:
            cap.release()
            holistic.close()

    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

@socketio.on("stop_camera")
def handle_stop():
    state["running"] = False

# Separate state for HTTP predict_frame (mobile app)
http_state = {
    "buffer": deque(maxlen=SEQ_LEN),
    "prob_history": deque(maxlen=PROB_AVG_WINDOW),
    "last_hand_time": 0.0,
    "pause_reset_done": False,
    "holistic": None,
    "lock": threading.Lock(),
}

@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    """HTTP endpoint for mobile app - accepts base64 image, returns prediction"""
    try:
        data = request.json or {}
        image_base64 = data.get("image_base64", "")
        if not image_base64:
            return jsonify({"success": False, "error": "No image"})

        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"success": False, "error": "Cannot decode image"})

        # Pad portrait frame to 4:3 (640x480) to match training distribution
        ih, iw = img.shape[:2]
        if iw < ih:
            target_w = int(ih * 4 / 3)
            if target_w > iw:
                pad_total = target_w - iw
                pad_left  = pad_total // 2
                pad_right = pad_total - pad_left
                img = cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right,
                                         cv2.BORDER_CONSTANT, value=(0, 0, 0))
            img = cv2.resize(img, (640, 480))

        with http_state["lock"]:
            if http_state["holistic"] is None:
                http_state["holistic"] = mp_holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.35,
                    min_tracking_confidence=0.45,
                    smooth_landmarks=True,
                    refine_face_landmarks=False
                )
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = http_state["holistic"].process(rgb)

        hand_visible = detect_active_hands(results)
        now_t = time.time()

        if hand_visible:
            http_state["last_hand_time"] = now_t
            http_state["pause_reset_done"] = False
            lm = extract_landmarks(results)
            lm = normalize_landmarks(lm)
            http_state["buffer"].append(lm)
        else:
            if (not http_state["pause_reset_done"] and http_state["last_hand_time"] > 0 and
                (now_t - http_state["last_hand_time"]) >= INTER_SIGN_PAUSE_SEC):
                http_state["buffer"].clear()
                http_state["prob_history"].clear()
                http_state["pause_reset_done"] = True

        buf_len = len(http_state["buffer"])

        if not hand_visible:
            return jsonify({
                "success": True, "prediction_text": "", "confidence": 0.0,
                "buffer_size": buf_len, "hands_detected": False,
                "message": "Ellerinizi gösterin..."
            })

        if buf_len < SEQ_LEN:
            return jsonify({
                "success": True, "prediction_text": "", "confidence": 0.0,
                "buffer_size": buf_len, "hands_detected": True,
                "message": f"İşaret okunuyor ({buf_len}/{SEQ_LEN})"
            })

        seq = np.array(http_state["buffer"], dtype=np.float32)   # (30, 1629) normalize
        seq = add_velocity(seq)                                   # (30, 1755)
        seq = np.clip(seq, -5, 5)

        raw_prob = run_model_probs(seq)
        http_state["prob_history"].append(raw_prob)
        prob = np.mean(http_state["prob_history"], axis=0)

        if state.get("demo_mode"):
            demo_indices = [label_map[w] for w in DEMO_WORDS if w in label_map]
            demo_probs = prob[demo_indices]
            demo_sum = float(np.sum(demo_probs))
            order = np.argsort(demo_probs)[::-1]
            best_i = demo_indices[int(order[0])]
            second_p = float(demo_probs[int(order[1])]) if len(order) > 1 else 0.0
            pred_word = idx_to_label[best_i]
            pred_conf = float(demo_probs[int(order[0])])
            margin = float(pred_conf - second_p)
        else:
            top_idx = np.argsort(prob)[::-1][:2]
            pred_conf = float(prob[int(top_idx[0])])
            margin = float(prob[int(top_idx[0])] - prob[int(top_idx[1])])
            pred_word = idx_to_label[int(top_idx[0])]

        if pred_conf < 0.70 or margin < 0.15:
            pred_word = ""
            pred_conf = 0.0

        return jsonify({
            "success": True, "prediction_text": pred_word, "confidence": pred_conf,
            "buffer_size": buf_len, "hands_detected": hand_visible,
            "message": f"El algılandı • {pred_word} (%{int(pred_conf*100)})" if pred_word else "El algılandı"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/predict_landmarks", methods=["POST"])
def predict_landmarks():
    """Accepts raw 1629 landmarks from client-side MediaPipe Holistic, normalizes, buffers, and runs V2 GRU."""
    try:
        data = request.json or {}
        hand_visible = bool(data.get("hand_visible", False))
        demo_mode = bool(data.get("demo_mode", state.get("demo_mode", False)))
        now_t = time.time()

        if not hand_visible:
            if (not http_state["pause_reset_done"] and http_state["last_hand_time"] > 0 and
                (now_t - http_state["last_hand_time"]) >= INTER_SIGN_PAUSE_SEC):
                http_state["buffer"].clear()
                http_state["prob_history"].clear()
                http_state["pause_reset_done"] = True
            return jsonify({
                "success": True, "hands_detected": False,
                "prediction_text": "", "confidence": 0.0,
                "buffer_size": len(http_state["buffer"]),
                "message": "Ellerinizi gösterin..."
            })

        lm_raw = data.get("landmarks")
        if not lm_raw or len(lm_raw) != RIGHT_END:
            return jsonify({"success": False, "error": "Invalid landmarks length"})

        http_state["last_hand_time"] = now_t
        http_state["pause_reset_done"] = False
        lm = np.array(lm_raw, dtype=np.float32)
        lm = normalize_landmarks(lm)
        http_state["buffer"].append(lm)
        buf_len = len(http_state["buffer"])

        if buf_len < SEQ_LEN:
            return jsonify({
                "success": True, "hands_detected": True,
                "prediction_text": "", "confidence": 0.0,
                "buffer_size": buf_len,
                "message": f"İşaret okunuyor ({buf_len}/{SEQ_LEN})"
            })

        seq = np.array(http_state["buffer"], dtype=np.float32)
        seq = add_velocity(seq)
        seq = np.clip(seq, -5, 5)
        raw_prob = run_model_probs(seq)
        http_state["prob_history"].append(raw_prob)
        prob = np.mean(http_state["prob_history"], axis=0)

        if demo_mode:
            demo_indices = [label_map[w] for w in DEMO_WORDS if w in label_map]
            demo_probs = prob[demo_indices]
            demo_sum = float(np.sum(demo_probs))
            if demo_sum >= 0.35:
                order = np.argsort(demo_probs)[::-1]
                best_i = demo_indices[int(order[0])]
                second_p = float(demo_probs[int(order[1])]) if len(order) > 1 else 0.0
                pred_word = idx_to_label[best_i]
                pred_conf = float(demo_probs[int(order[0])] / max(demo_sum, 1e-6)) * min(1.0, demo_sum + 0.35)
                margin = float(demo_probs[int(order[0])] - second_p)
            else:
                pred_word, pred_conf, margin = "", 0.0, 0.0
        else:
            top_idx = np.argsort(prob)[::-1][:2]
            pred_conf = float(prob[int(top_idx[0])])
            margin = float(prob[int(top_idx[0])] - prob[int(top_idx[1])])
            pred_word = idx_to_label[int(top_idx[0])]

        if pred_conf < 0.72 or margin < 0.18:
            pred_word = ""
            pred_conf = 0.0

        return jsonify({
            "success": True, "hands_detected": True,
            "prediction_text": pred_word, "confidence": pred_conf, "margin": margin,
            "buffer_size": buf_len,
            "message": f"El algılandı • {pred_word} (%{int(pred_conf*100)})" if pred_word else "El algılandı"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/predict_sequence", methods=["POST"])
def predict_sequence():
    """Accepts normalized (30, 1755) sequence or raw 1629 landmark frame and runs V2 GRU model."""
    try:
        data = request.json or {}
        seq_flat = data.get("sequence")
        if not seq_flat or len(seq_flat) != SEQ_LEN * INPUT_SIZE:
            return jsonify({"success": False, "error": "Invalid sequence length"})
        seq = np.array(seq_flat, dtype=np.float32).reshape(SEQ_LEN, INPUT_SIZE)
        prob = run_model_probs(seq)
        top_idx = np.argsort(prob)[::-1][:3]
        pred_conf = float(prob[int(top_idx[0])])
        margin = float(prob[int(top_idx[0])] - prob[int(top_idx[1])])
        pred_word = idx_to_label[int(top_idx[0])]
        return jsonify({
            "success": True,
            "prediction_text": pred_word,
            "confidence": pred_conf,
            "margin": margin,
            "top3": [{"word": idx_to_label[int(i)], "conf": float(prob[int(i)])} for i in top_idx]
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/reset_buffer", methods=["POST"])
def reset_buffer():
    http_state["buffer"] = deque(maxlen=SEQ_LEN)
    http_state["prob_history"] = deque(maxlen=PROB_AVG_WINDOW)
    state["buffer"] = deque(maxlen=SEQ_LEN)
    state["prob_history"] = deque(maxlen=PROB_AVG_WINDOW)
    state["smooth_counts"] = {k: 0 for k in label_map}
    state["sentence"] = []
    # Tum WS oturumlarinin buffer'ini da sifirla (mobil istek yapan icin)
    for sess in list(ws_sessions.values()):
        sess["buffer"] = deque(maxlen=SEQ_LEN)
        sess["prob_history"] = deque(maxlen=PROB_AVG_WINDOW)
    return jsonify({"success": True})

# =========================================================================
# WebSocket streaming endpoint (mobil icin HIZLI yol)
# =========================================================================
ws_sessions = {}

def _get_ws_session(sid):
    sess = ws_sessions.get(sid)
    if sess is None:
        sess = {
            "buffer": deque(maxlen=SEQ_LEN),
            "prob_history": deque(maxlen=PROB_AVG_WINDOW),
            "last_hand_time": 0.0,
            "pause_reset_done": False,
            "holistic": mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.35,
                min_tracking_confidence=0.45,
                smooth_landmarks=True,
                refine_face_landmarks=False,
            ),
            "lock": threading.Lock(),
        }
        ws_sessions[sid] = sess
    return sess

@socketio.on("ws_frame")
def on_ws_frame(data):
    try:
        sid = request.sid
        if isinstance(data, dict):
            raw = data.get("jpeg") or data.get("image_base64") or b""
            flip = bool(data.get("flip", False))
        else:
            raw = data
            flip = False

        if isinstance(raw, str):
            jpeg = base64.b64decode(raw)
        elif isinstance(raw, (bytes, bytearray, memoryview)):
            jpeg = bytes(raw)
        else:
            try:
                jpeg = bytes(raw)
            except Exception:
                emit("ws_pred", {"success": False, "error": "bad payload type"})
                return

        if not jpeg:
            emit("ws_pred", {"success": False, "error": "empty jpeg"})
            return

        nparr = np.frombuffer(jpeg, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            emit("ws_pred", {"success": False, "error": "decode fail"})
            return

        if flip:
            img = cv2.flip(img, 1)

        ih, iw = img.shape[:2]
        if iw < ih:
            target_w = int(ih * 4 / 3)
            if target_w > iw:
                pad_total = target_w - iw
                pad_left  = pad_total // 2
                pad_right = pad_total - pad_left
                img = cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))
            img = cv2.resize(img, (640, 480))

        sess = _get_ws_session(sid)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with sess["lock"]:
            results = sess["holistic"].process(rgb)

        hand_visible = detect_active_hands(results)
        now_t = time.time()

        if hand_visible:
            sess["last_hand_time"] = now_t
            sess["pause_reset_done"] = False
            lm = extract_landmarks(results)
            lm = normalize_landmarks(lm)
            sess["buffer"].append(lm)
        else:
            if (not sess["pause_reset_done"] and sess["last_hand_time"] > 0 and
                (now_t - sess["last_hand_time"]) >= INTER_SIGN_PAUSE_SEC):
                sess["buffer"].clear()
                sess["prob_history"].clear()
                sess["pause_reset_done"] = True

        buf_len = len(sess["buffer"])

        if not hand_visible or buf_len < SEQ_LEN:
            emit("ws_pred", {
                "success": True, "prediction_text": "", "confidence": 0.0,
                "buffer_size": buf_len, "hands_detected": hand_visible,
                "message": f"Buffer: {buf_len}/{SEQ_LEN}" if hand_visible else "Ellerinizi gösterin..."
            })
            return

        seq = np.array(sess["buffer"], dtype=np.float32)
        seq = add_velocity(seq)
        seq = np.clip(seq, -5, 5)

        raw_prob = run_model_probs(seq)
        sess["prob_history"].append(raw_prob)
        prob = np.mean(sess["prob_history"], axis=0)

        if state.get("demo_mode"):
            wl_pairs = [(float(prob[label_map[w]]), w, label_map[w]) for w in DEMO_WORDS if w in label_map]
            wl_pairs.sort(reverse=True)
            conf_val = wl_pairs[0][0]
            label_str = wl_pairs[0][1]
        else:
            top3_idxs = np.argsort(prob)[::-1][:3]
            conf_val = float(prob[int(top3_idxs[0])])
            label_str = idx_to_label[int(top3_idxs[0])]

        emit("ws_pred", {
            "success": True,
            "prediction_text": label_str,
            "confidence": conf_val,
            "buffer_size": buf_len,
            "hands_detected": hand_visible,
        })
    except Exception as e:
        try:
            emit("ws_pred", {"success": False, "error": str(e)})
        except Exception:
            pass

@socketio.on("ws_reset")
def on_ws_reset():
    sid = request.sid
    sess = ws_sessions.get(sid)
    if sess is not None:
        sess["buffer"] = deque(maxlen=SEQ_LEN)
        sess["prob_history"] = deque(maxlen=PROB_AVG_WINDOW)
    emit("ws_pred", {"success": True, "reset": True})

# ---- Faz B: landmark-only endpoint ----
# Client MediaPipe'i sunucuda calistirir ama GRU'yu telefonda calistirir.
# Boylece telefon kendi normalize + velocity + inference yolunu yapar.
# Sunucu: JPEG -> MediaPipe -> 1629 ham landmark float32 + hand_visible.
@socketio.on("ws_frame_lm")
def on_ws_frame_lm(data):
    """
    Mobil app'ten binary JPEG frame.
    Cikis: ws_lm event'i ile {'lm': bytes(1629*4), 'hand': bool}
    GRU HIC calistirilmaz; istemci 30-frame buffer + velocity + clip + GRU yapar.
    """
    try:
        sid = request.sid

        if isinstance(data, dict):
            raw = data.get("jpeg") or data.get("image_base64") or b""
            flip = bool(data.get("flip", False))
        else:
            raw = data
            flip = False

        if isinstance(raw, str):
            jpeg = base64.b64decode(raw)
        elif isinstance(raw, (bytes, bytearray, memoryview)):
            jpeg = bytes(raw)
        else:
            try:
                jpeg = bytes(raw)
            except Exception:
                emit("ws_lm", {"success": False, "error": "bad payload type"})
                return

        if not jpeg:
            emit("ws_lm", {"success": False, "error": "empty jpeg"})
            return

        nparr = np.frombuffer(jpeg, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            emit("ws_lm", {"success": False, "error": "decode fail"})
            return

        if flip:
            img = cv2.flip(img, 1)

        # ─── KRİTİK FIX: Phase B icin de ayni 4:3 landscape pad ───
        ih, iw = img.shape[:2]
        if iw < ih:
            target_w = int(ih * 4 / 3)
            if target_w > iw:
                pad_total = target_w - iw
                pad_left  = pad_total // 2
                pad_right = pad_total - pad_left
                img = cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))
            img = cv2.resize(img, (640, 480))

        sess = _get_ws_session(sid)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with sess["lock"]:
            results = sess["holistic"].process(rgb)
        lm = extract_landmarks(results).astype(np.float32)   # (1629,)

        hand_visible = (results.left_hand_landmarks is not None or
                        results.right_hand_landmarks is not None)

        # Binary payload: 1629 float32 = 6516 bytes
        emit("ws_lm", {
            "success": True,
            "lm":      lm.tobytes(),
            "hand":    bool(hand_visible),
        })
    except Exception as e:
        try:
            emit("ws_lm", {"success": False, "error": str(e)})
        except Exception:
            pass

# WS oturumu kapandiginda MediaPipe kaynaklarini serbest birak
# (Asagidaki handler eski /914'teki disconnect handler'inin YERINE gecer;
# flask-socketio ayni event'a son kayit olan handler'i calistirir.)
@socketio.on("disconnect")
def _ws_cleanup_disconnect():
    try:
        sid = request.sid
        sess = ws_sessions.pop(sid, None)
        if sess is not None:
            try:
                sess["holistic"].close()
            except Exception:
                pass
    except Exception:
        pass
    # Eski davranis: kamera loop'u durdur
    state["running"] = False

# ---- Avatar & Mobile routes ----
import os, json
WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
AVATAR_DIR = os.path.join(WEB_DIR, "avatar")
POSES_FILE = os.path.join(AVATAR_DIR, "saved_poses.json")

# `/`  -> birlesik portal (web/index.html). Iki sayfayi iframe icinde sunar.
# `/camera-page` -> ESKI kameralı HTML (mevcut kodu hicbir sekilde bozmaz,
# wrapper'ın iframe src'si bunu kullanir).
@app.route("/")
def index():
    from flask import send_from_directory
    return send_from_directory(WEB_DIR, "index.html")

@app.route("/camera-page")
def camera_page():
    return render_template_string(HTML)

@app.route("/manifest.json")
def serve_manifest():
    from flask import send_from_directory
    return send_from_directory(WEB_DIR, "manifest.json")

@app.route("/favicon.ico")
def serve_favicon():
    return "", 204

@app.route("/avatar/<path:filename>")
def serve_avatar(filename):
    from flask import send_from_directory
    return send_from_directory(AVATAR_DIR, filename)

@app.route("/holistic/<path:filename>")
def serve_holistic(filename):
    from flask import send_from_directory
    return send_from_directory(os.path.join(AVATAR_DIR, "holistic"), filename)

@app.route("/model/<path:filename>")
def serve_model(filename):
    from flask import send_from_directory
    return send_from_directory(os.path.join(AVATAR_DIR, "model"), filename)

@app.route("/api/list-poses")
def list_poses():
    if os.path.exists(POSES_FILE):
        with open(POSES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return jsonify({"poses": data})
    return jsonify({"poses": {}})

@app.route("/api/save-pose", methods=["POST"])
def save_pose():
    data = request.json
    poses = {}
    if os.path.exists(POSES_FILE):
        with open(POSES_FILE, "r", encoding="utf-8") as f:
            poses = json.load(f)
    word = data.get("word", "")
    pose = data.get("pose", {})
    if word:
        poses[word] = pose
        with open(POSES_FILE, "w", encoding="utf-8") as f:
            json.dump(poses, f, ensure_ascii=False, indent=2)
    return jsonify({"success": True, "count": len(poses)})

if __name__ == "__main__":
    print("\n" + "="*50)
    print("SignBridge Web Arayuzu")
    print("="*50)
    print(f"Kelimeler ({num_classes}): {', '.join(label_map.keys())}")
    print(f"Adres: http://localhost:5050")
    print("="*50 + "\n")
    socketio.run(app, host="0.0.0.0", port=5050, debug=False,
                 allow_unsafe_werkzeug=True)
