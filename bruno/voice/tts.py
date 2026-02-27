import platform
import subprocess
import threading
import queue

_speaker_queue = queue.Queue()
_speaker_started = False
_speaker_lock = threading.Lock()


def _speak_blocking(text: str) -> None:
    """Run TTS and block until playback finishes. Never call from main loop."""
    try:
        if platform.system().lower() == "darwin":
            subprocess.run(["say", text], capture_output=True, timeout=120)
        else:
            subprocess.run(["espeak-ng", text], capture_output=True, timeout=120)
    except Exception:
        pass


def _speaker_thread() -> None:
    while True:
        text = _speaker_queue.get()
        if text:
            _speak_blocking(text)


def speak(text: str) -> None:
    """Enqueue a line; one thread speaks lines in order so they never overlap."""
    text = (text or "").strip()
    if not text:
        return
    with _speaker_lock:
        global _speaker_started
        if not _speaker_started:
            t = threading.Thread(target=_speaker_thread, daemon=True)
            t.start()
            _speaker_started = True
    _speaker_queue.put(text)
