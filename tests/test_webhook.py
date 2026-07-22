import logging
import unittest
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from voice_agent import clerk
from voice_agent.config import Settings
from voice_agent.main import create_app
from voice_agent.session import session_store
from voice_agent.webhook import _correlation_mode

WEBHOOK_LOGGER = "voice_agent.webhook"


class FakePerceptionClient:
    async def start_session(self, language: str) -> str:
        self.language = language
        return "perception_123"

    async def get_state(self, session_id: str) -> dict[str, object]:
        self.session_id = session_id
        return {
            "emotion": "FEARFUL",
            "emotion_confidence": 0.8,
            "stability": "shifting",
            "audio_events": ["Breath"],
            "hesitation_score": 0.75,
        }

    async def is_reachable(self) -> bool:
        return True


async def _fake_run_turn(messages, perception_state, language_state):  # type: ignore[no-untyped-def]
    yield str(perception_state.get("emotion", "missing"))


class CorrelationModeTests(unittest.TestCase):
    def test_direct_sources_map_to_direct_dynamic_variable(self) -> None:
        for source in (
            "top-level perception_session_id",
            "elevenlabs_extra_body.perception_session_id",
            "conversation_initiation_client_data.dynamic_variables.perception_session_id",
            "messages[0].system content perception_session_id marker",
            "last user message metadata.perception_session_id",
        ):
            self.assertEqual(_correlation_mode(source), "direct dynamic variable")

    def test_fallback_sources_map_to_demo_fallback(self) -> None:
        for source in (
            "single active local session fallback",
            "latest active demo session fallback",
        ):
            self.assertEqual(_correlation_mode(source), "latest active demo fallback")

    def test_missing_maps_to_none_resolved(self) -> None:
        self.assertEqual(_correlation_mode("missing"), "none resolved")


class WebhookInjectionLoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        session_store.clear()
        self._original_run_turn = clerk.run_turn
        clerk.run_turn = _fake_run_turn

    def tearDown(self) -> None:
        clerk.run_turn = self._original_run_turn
        session_store.clear()

    def _app(self) -> object:
        app = create_app(
            Settings(
                elevenlabs_agent_id="agent_123",
                openai_api_key="test-key",
                voice_perception_url="http://127.0.0.1:8000",
            )
        )
        app.state.perception_client = FakePerceptionClient()
        return app

    def test_explicit_id_logs_direct_correlation(self) -> None:
        app = self._app()
        with self.assertLogs(WEBHOOK_LOGGER, level=logging.INFO) as captured:
            with TestClient(app) as client:
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "custom",
                        "messages": [{"role": "user", "content": "hi"}],
                        "elevenlabs_extra_body": {
                            "perception_session_id": "perception_123"
                        },
                    },
                )
        self.assertEqual(response.status_code, 200)
        text = "\n".join(captured.output)
        self.assertIn(
            "LLM turn perception correlation: direct dynamic variable, "
            "session=perception_123",
            text,
        )

    def test_no_id_multiple_sessions_logs_demo_fallback(self) -> None:
        app = self._app()
        session_a = session_store.create("perception_a")
        session_b = session_store.create("perception_b")
        session_a.touch(datetime.now(timezone.utc) - timedelta(seconds=30))
        session_b.touch(datetime.now(timezone.utc) - timedelta(seconds=5))

        with self.assertLogs(WEBHOOK_LOGGER, level=logging.INFO) as captured:
            with TestClient(app) as client:
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "custom",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
        self.assertEqual(response.status_code, 200)
        text = "\n".join(captured.output)
        self.assertIn(
            "LLM turn perception correlation: latest active demo fallback, "
            "session=perception_b",
            text,
        )

    def test_injected_state_line_reports_named_fields(self) -> None:
        app = self._app()
        with self.assertLogs(WEBHOOK_LOGGER, level=logging.INFO) as captured:
            with TestClient(app) as client:
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "custom",
                        "messages": [{"role": "user", "content": "hi"}],
                        "elevenlabs_extra_body": {
                            "perception_session_id": "perception_123"
                        },
                    },
                )
        self.assertEqual(response.status_code, 200)
        text = "\n".join(captured.output)
        self.assertIn(
            "LLM turn injected perception state: emotion=FEARFUL, "
            "confidence=0.8, stability=shifting, hesitation=0.75, "
            "events=['Breath']",
            text,
        )

    def test_no_session_logs_neutral_fallback_warning(self) -> None:
        app = self._app()
        with self.assertLogs(WEBHOOK_LOGGER, level=logging.WARNING) as captured:
            with TestClient(app) as client:
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "custom",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
        self.assertEqual(response.status_code, 200)
        text = "\n".join(captured.output)
        self.assertIn(
            "LLM turn injected neutral fallback because no perception session "
            "was resolved",
            text,
        )


if __name__ == "__main__":
    unittest.main()
