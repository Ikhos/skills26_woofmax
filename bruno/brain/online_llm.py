import os
import json
from openai import OpenAI

class OnlineBrain:
    def __init__(self, model: str = "gpt-4.1-mini"):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def decide(self, scene_summary: dict, user_id: str | None = None) -> dict:
        system = (
            "You are BRUNO, a Baymax-like assistant in a robot dog. "
            "Be concise, calm, and helpful. "
            "Return ONLY valid JSON with keys: say, actions, safety."
        )

        user = {
            "scene_summary": scene_summary,
            "user_id": user_id
        }

        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user)}
            ],
        )

        text = resp.output_text.strip()
        return json.loads(text)
