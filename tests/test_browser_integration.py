import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from voice_agent.config import Settings
from voice_agent.main import create_app
from voice_agent.webhook import extract_perception_session_id


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_INDEX = PROJECT_ROOT / "static" / "index.html"


class BrowserIntegrationTests(unittest.TestCase):
    def test_static_index_does_not_use_forbidden_custom_llm_extra_body(self) -> None:
        html = STATIC_INDEX.read_text(encoding="utf-8")

        self.assertNotIn("customLlmExtraBody", html)
        self.assertNotIn("custom_llm_extra_body", html)

    def test_static_index_sends_perception_id_as_dynamic_variable(self) -> None:
        html = STATIC_INDEX.read_text(encoding="utf-8")

        self.assertIn("const dynamicVariables = { perception_session_id: perceptionSessionId };", html)
        self.assertIn("dynamicVariables,", html)
        self.assertIn("conversation_initiation_client_data", html)
        self.assertIn("dynamic_variables: dynamicVariables", html)

    def test_webhook_extracts_conversation_initiation_dynamic_variables(self) -> None:
        session_id, source = extract_perception_session_id(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "conversation_initiation_client_data": {
                    "dynamic_variables": {
                        "perception_session_id": "perception_123",
                    },
                },
            }
        )

        self.assertEqual(session_id, "perception_123")
        self.assertEqual(
            source,
            "conversation_initiation_client_data.dynamic_variables.perception_session_id",
        )

    def test_session_start_returns_same_origin_perception_proxy_urls(self) -> None:
        app = create_app(
            Settings(
                elevenlabs_agent_id="agent_123",
                voice_perception_url="http://127.0.0.1:8000",
                perception_language="uk",
            )
        )
        app.state.perception_client = FakePerceptionClient()

        with TestClient(app) as client:
            response = client.post("/session/start", json={})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["perception_session_id"], "perception_123")
        self.assertEqual(payload["perception_state_url"], "/perception/state/perception_123")
        self.assertEqual(payload["perception_audio_ws_url"], "/perception/audio/perception_123")
        self.assertEqual(payload["perception_language"], "uk")

    def test_perception_state_proxy_returns_state_envelope(self) -> None:
        app = create_app(Settings(voice_perception_url="http://127.0.0.1:8000"))
        app.state.perception_client = FakePerceptionClient()

        with TestClient(app) as client:
            response = client.get("/perception/state/perception_123")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "state": {
                    "emotion": "HAPPY",
                    "emotion_confidence": 0.9,
                    "stability": "stable",
                    "audio_events": [],
                    "hesitation_score": 0.1,
                }
            },
        )


class FakePerceptionClient:
    async def start_session(self, language: str) -> str:
        self.language = language
        return "perception_123"

    async def get_state(self, session_id: str) -> dict[str, object]:
        self.session_id = session_id
        return {
            "emotion": "HAPPY",
            "emotion_confidence": 0.9,
            "stability": "stable",
            "audio_events": [],
            "hesitation_score": 0.1,
        }

    async def is_reachable(self) -> bool:
        return True


if __name__ == "__main__":
    unittest.main()
