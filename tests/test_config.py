import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from voice_agent.config import load_settings


class ConfigTests(unittest.TestCase):
    def test_placeholder_credentials_are_treated_as_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dotenv_path = Path(tmpdir) / ".env"
            dotenv_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=sk-your-openai-api-key",
                        "ELEVENLABS_API_KEY=xi-your-api-key",
                        "ELEVENLABS_AGENT_ID=agent_your-agent-id",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                settings = load_settings(dotenv_path)

        self.assertEqual(settings.openai_api_key, "")
        self.assertEqual(settings.elevenlabs_api_key, "")
        self.assertEqual(settings.elevenlabs_agent_id, "")

    def test_perception_language_is_environment_driven(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dotenv_path = Path(tmpdir) / ".env"
            dotenv_path.write_text("PERCEPTION_LANGUAGE=uk\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                settings = load_settings(dotenv_path)

        self.assertEqual(settings.perception_language, "uk")


if __name__ == "__main__":
    unittest.main()
