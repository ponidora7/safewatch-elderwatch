"""
src/audio_detector.py
======================
Real-time distress voice detection using Vosk (offline speech recognition)
and amplitude-based scream detection.

Runs in a separate thread alongside the webcam monitor.
Detects Indonesian distress keywords: "tolong", "bantu", "aduh", "jatuh", "sakit"
Also detects screams via high audio amplitude (RMS).

Usage:
    detector = AudioDistressDetector()
    detector.start()
    ...
    alert = detector.get_alert()  # returns dict or None
    ...
    detector.stop()
"""

import json
import os
import sys
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

# Path resolving
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
VOSK_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "vosk-en")
# Auto-detect nested folders if the user extracted the zip into a subfolder
if os.path.isdir(VOSK_MODEL_PATH):
    subdirs = [os.path.join(VOSK_MODEL_PATH, d) for d in os.listdir(VOSK_MODEL_PATH) 
               if os.path.isdir(os.path.join(VOSK_MODEL_PATH, d))]
    if len(subdirs) == 1 and not os.path.exists(os.path.join(VOSK_MODEL_PATH, "am")):
        VOSK_MODEL_PATH = subdirs[0]


# ====================================
# CONFIGURATION
# ====================================
DISTRESS_KEYWORDS = [
    "help",
    "help me",
    "emergency",
    "falling",
    "fell down",
    "it hurts",
    "pain",
    "ambulance",
    "someone help",
]

SAMPLE_RATE = 16000          # Vosk optimal sample rate
BLOCK_SIZE = 4000            # Audio block size (~250ms at 16kHz)
ALERT_COOLDOWN = 10.0        # Seconds between alerts
SCREAM_RMS_THRESHOLD = 0.35  # RMS amplitude threshold for scream detection
SCREAM_DURATION_MIN = 0.5    # Minimum seconds of high amplitude to trigger scream


class AudioDistressDetector:
    """
    Threaded audio monitor that listens for distress keywords
    and scream-level audio via microphone.
    """

    def __init__(self, model_path: str = VOSK_MODEL_PATH, 
                 keywords: list = None,
                 sample_rate: int = SAMPLE_RATE):
        self.model_path = model_path
        self.keywords = keywords or DISTRESS_KEYWORDS
        self.sample_rate = sample_rate
        
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._alert_lock = threading.Lock()
        self._current_alert: Optional[dict] = None
        self._last_alert_time = 0.0
        self._is_ready = False
        self._error_message: Optional[str] = None
        
        # Scream detection state
        self._high_amplitude_start = 0.0
        self._high_amplitude_frames = deque(maxlen=10)

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def error_message(self) -> Optional[str]:
        return self._error_message

    def start(self):
        """Start the audio monitoring thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the audio monitoring thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def get_alert(self) -> Optional[dict]:
        """
        Get and consume the current alert (if any).
        
        Returns:
            dict with keys: 'type' ('keyword' or 'scream'), 'detail', 'timestamp'
            or None if no alert.
        """
        with self._alert_lock:
            alert = self._current_alert
            self._current_alert = None
            return alert

    def _set_alert(self, alert_type: str, detail: str):
        """Set an alert if cooldown has elapsed."""
        now = time.time()
        if now - self._last_alert_time < ALERT_COOLDOWN:
            return
        
        with self._alert_lock:
            self._current_alert = {
                "type": alert_type,
                "detail": detail,
                "timestamp": now,
            }
            self._last_alert_time = now

    def _check_scream(self, audio_data: np.ndarray):
        """Check if audio amplitude indicates a scream/loud cry."""
        # Calculate RMS (Root Mean Square) of the audio block
        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
        # Normalize to 0-1 range (int16 max = 32768)
        rms_normalized = rms / 32768.0
        
        self._high_amplitude_frames.append(rms_normalized)
        
        if rms_normalized > SCREAM_RMS_THRESHOLD:
            if self._high_amplitude_start == 0.0:
                self._high_amplitude_start = time.time()
            elif time.time() - self._high_amplitude_start >= SCREAM_DURATION_MIN:
                avg_rms = sum(self._high_amplitude_frames) / len(self._high_amplitude_frames)
                if avg_rms > SCREAM_RMS_THRESHOLD:
                    self._set_alert("scream", f"Teriakan terdeteksi (volume: {avg_rms:.2f})")
                    self._high_amplitude_start = 0.0
        else:
            self._high_amplitude_start = 0.0

    def _check_keywords(self, text: str):
        """Check if recognized text contains distress keywords."""
        text_lower = text.lower().strip()
        if not text_lower:
            return
        
        for keyword in self.keywords:
            if keyword in text_lower:
                self._set_alert("keyword", f'Kata darurat terdeteksi: "{keyword}" (teks: "{text_lower}")')
                return

    def _monitor_loop(self):
        """Main monitoring loop (runs in a thread)."""
        # Late imports to avoid blocking main thread if packages missing
        try:
            import sounddevice as sd
            from vosk import Model, KaldiRecognizer
        except ImportError as e:
            self._error_message = f"Dependensi audio tidak tersedia: {e.name}"
            print(f"⚠️ Audio Detector: {self._error_message}")
            return

        # Load Vosk model
        if not os.path.isdir(self.model_path):
            self._error_message = (
                f"Model Vosk tidak ditemukan di: {self.model_path}\n"
                f"Jalankan: python scripts/setup_vosk_model.py"
            )
            print(f"⚠️ Audio Detector: {self._error_message}")
            return

        try:
            model = Model(self.model_path)
            recognizer = KaldiRecognizer(model, self.sample_rate)
            recognizer.SetWords(True)
        except Exception as e:
            self._error_message = f"Gagal memuat model Vosk: {e}"
            print(f"❌ Audio Detector: {self._error_message}")
            return

        # Open microphone stream
        try:
            stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=BLOCK_SIZE,
                dtype="int16",
                channels=1,
            )
            stream.start()
            self._is_ready = True
            print("🎤 Audio Detector: Mikrofon aktif — memantau suara darurat...")
        except Exception as e:
            self._error_message = f"Gagal mengakses mikrofon: {e}"
            print(f"❌ Audio Detector: {self._error_message}")
            return

        try:
            while self._running:
                data, overflowed = stream.read(BLOCK_SIZE)
                if overflowed:
                    continue

                # Convert to numpy for scream detection
                audio_np = np.frombuffer(bytes(data), dtype=np.int16)
                self._check_scream(audio_np)

                # Feed to Vosk for speech recognition
                if recognizer.AcceptWaveform(bytes(data)):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        self._check_keywords(text)
                else:
                    # Partial results for faster keyword detection
                    partial = json.loads(recognizer.PartialResult())
                    partial_text = partial.get("partial", "")
                    if partial_text:
                        self._check_keywords(partial_text)
        except Exception as e:
            if self._running:
                print(f"❌ Audio Detector error: {e}")
        finally:
            stream.stop()
            stream.close()
            self._is_ready = False
            print("🎤 Audio Detector: Mikrofon dinonaktifkan.")
