import platform
import subprocess

def speak(text: str):
    text = (text or "").strip()
    if not text:
        return
    try:
        if platform.system().lower() == "darwin":
            subprocess.Popen(["say", text])
        else:
            subprocess.Popen(["espeak-ng", text])
    except Exception:
        pass
