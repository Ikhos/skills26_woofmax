import subprocess

SAY_BIN = "/usr/bin/say"

def speak(text: str):
    text = (text or "").strip()
    if not text:
        return
    subprocess.run(["/usr/bin/pkill", "-x", "say"], capture_output=True)
    subprocess.run([SAY_BIN, text], capture_output=True)
