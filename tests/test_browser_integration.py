import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from voice_agent import clerk
from voice_agent.config import Settings
from voice_agent.main import create_app
from voice_agent.session import session_store
from voice_agent.webhook import extract_perception_session_id


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_INDEX = PROJECT_ROOT / "static" / "index.html"


class BrowserIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        session_store.clear()

    def tearDown(self) -> None:
        session_store.clear()

    def test_static_index_sends_perception_id_as_custom_llm_extra_body(self) -> None:
        html = STATIC_INDEX.read_text(encoding="utf-8")

        self.assertIn("const customLlmExtraBody = { perception_session_id: perceptionSessionId };", html)
        self.assertIn("customLlmExtraBody,", html)
        self.assertNotIn("custom_llm_extra_body", html)

    def test_static_index_sends_perception_id_as_dynamic_variable(self) -> None:
        html = STATIC_INDEX.read_text(encoding="utf-8")

        self.assertIn("const dynamicVariables = { perception_session_id: perceptionSessionId };", html)
        self.assertIn("dynamicVariables,", html)
        self.assertNotIn("conversation_initiation_client_data", html)

    def test_webhook_extracts_elevenlabs_extra_body(self) -> None:
        session_id, source = extract_perception_session_id(
            {
                "model": "custom",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "elevenlabs_extra_body": {
                    "perception_session_id": "perception_123",
                },
            }
        )

        self.assertEqual(session_id, "perception_123")
        self.assertEqual(source, "elevenlabs_extra_body.perception_session_id")

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

    def test_webhook_uses_single_active_session_when_body_lacks_id(self) -> None:
        app = create_app(
            Settings(
                elevenlabs_agent_id="agent_123",
                openai_api_key="test-key",
                voice_perception_url="http://127.0.0.1:8000",
            )
        )
        fake_client = FakePerceptionClient()
        app.state.perception_client = fake_client

        async def fake_run_turn(messages, perception_state):  # type: ignore[no-untyped-def]
            yield str(perception_state.get("emotion", "missing"))

        original_run_turn = clerk.run_turn
        clerk.run_turn = fake_run_turn
        try:
            with TestClient(app) as client:
                start_response = client.post("/session/start", json={})
                self.assertEqual(start_response.status_code, 200)

                webhook_response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "custom",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    },
                )
        finally:
            clerk.run_turn = original_run_turn

        self.assertEqual(webhook_response.status_code, 200)
        self.assertEqual(fake_client.session_id, "perception_123")
        self.assertIn("HAPPY", webhook_response.text)
        self.assertNotIn("NEUTRAL", webhook_response.text)


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
