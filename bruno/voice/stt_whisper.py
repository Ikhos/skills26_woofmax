import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import tempfile
import os

_MODEL_NAME = "base"
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = WhisperModel(_MODEL_NAME, device="cpu", compute_type="int8")
    return _model

def record_wav(seconds: float = 4.0, samplerate: int = 16000) -> str:
    seconds = float(seconds)
    samplerate = int(samplerate)

    print(f"BRUNO: Listening for {seconds:.1f}s... speak now.")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    write(tmp.name, samplerate, audio)
    return tmp.name

def transcribe_wav(path: str) -> str:
    model = _get_model()
    segments, _info = model.transcribe(path, beam_size=1)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text

def listen_and_transcribe(seconds: float = 4.0) -> str:
    wav = record_wav(seconds=seconds)
    try:
        return transcribe_wav(wav)
    finally:
        try:
            os.remove(wav)
        except Exception:
            pass
