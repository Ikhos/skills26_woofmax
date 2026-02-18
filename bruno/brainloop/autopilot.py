from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Set, Dict
import time
from .state import PerceptionState
from .risk import RiskResult

@dataclass
class AutoOutput:
    say: Optional[str] = None

class Autopilot:
    def __init__(self):
        # greeting is per recognized user, once per session
        self.greeted: Set[str] = set()

        # unlock prompt is per recognized user, once until unlock occurs
        self.unlock_prompted: Set[str] = set()

        # global rate-limit + anti-repeat
        self.last_say_ts: float = 0.0
        self.cooldown_s: float = 10.0  # slower = less annoying
        self.last_prompt_key: Optional[str] = None
        self.last_prompt_key_ts: float = 0.0
        self.repeat_same_prompt_after_s: float = 45.0  # if truly needed

    def _can_speak(self) -> bool:
        return (time.time() - self.last_say_ts) >= self.cooldown_s

    def _should_repeat_prompt(self, key: str) -> bool:
        # Prevent repeating same idea unless enough time passed
        if self.last_prompt_key != key:
            self.last_prompt_key = key
            self.last_prompt_key_ts = time.time()
            return True
        return (time.time() - self.last_prompt_key_ts) >= self.repeat_same_prompt_after_s

    def decide(self, state: PerceptionState, risk: RiskResult) -> AutoOutput:
        # Gather recognized names in frame
        recognized = [p.get("name") for p in state.people if p.get("recognized") and p.get("name")]
        recognized = [n for n in recognized if isinstance(n, str)]

        # 1) greet newly recognized users (once per session)
        for nm in recognized:
            if nm not in self.greeted:
                self.greeted.add(nm)
                if self._can_speak() and self._should_repeat_prompt(f"greet:{nm}"):
                    self.last_say_ts = time.time()
                    return AutoOutput(say=f"Hey {nm}.")
                return AutoOutput()

        # 2) if recognized but locked, prompt unlock ONCE per recognized user
        if recognized and not state.authorized_user:
            # pick first recognized name for messaging
            nm = recognized[0]
            if nm not in self.unlock_prompted:
                self.unlock_prompted.add(nm)
                if self._can_speak() and self._should_repeat_prompt(f"unlock:{nm}"):
                    self.last_say_ts = time.time()
                    return AutoOutput(say=f"I recognize you as {nm}, but I'm locked. Press U or enter your PIN to unlock.")
                return AutoOutput()

        # 3) if unlocked, clear unlock_prompted so it can prompt again in future sessions if needed
        if state.authorized_user:
            # once you unlock as someone, allow prompting for others later
            # (do not clear greeted)
            self.unlock_prompted.clear()

        # 4) unknown person nudge (rare)
        if "unknown person" in getattr(risk, "reasons", []):
            if self._can_speak() and self._should_repeat_prompt("unknown"):
                self.last_say_ts = time.time()
                return AutoOutput(say="I see someone I do not recognize yet.")
        return AutoOutput()
