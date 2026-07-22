import unittest

from scripts.elevenlabs_agent import build_agent_payload, build_parser, normalize_custom_llm_url


class ElevenLabsAgentScriptTests(unittest.TestCase):
    def test_normalize_base_url(self) -> None:
        self.assertEqual(
            normalize_custom_llm_url("https://demo.ngrok-free.app/"),
            "https://demo.ngrok-free.app/v1/chat/completions",
        )

    def test_accepts_full_chat_completions_url(self) -> None:
        self.assertEqual(
            normalize_custom_llm_url("https://demo.ngrok-free.app/v1/chat/completions"),
            "https://demo.ngrok-free.app/v1/chat/completions",
        )

    def test_rejects_non_https_url(self) -> None:
        with self.assertRaises(ValueError):
            normalize_custom_llm_url("http://localhost:8001")

    def test_build_payload_contains_demo_settings(self) -> None:
        payload = build_agent_payload(public_base_url="https://demo.ngrok-free.app")
        self.assertEqual(payload["name"], "zollhof-clerk-demo")
        agent = payload["conversation_config"]["agent"]
        self.assertEqual(agent["language"], "de")
        self.assertEqual(agent["first_message"], "")
        self.assertEqual(agent["prompt"]["llm"], "custom-llm")
        custom_llm = agent["prompt"]["custom_llm"]
        self.assertEqual(
            custom_llm["url"],
            "https://demo.ngrok-free.app/v1/chat/completions",
        )
        self.assertNotIn("api_key", custom_llm)
        self.assertIn(
            "perception_session_id",
            agent["dynamic_variables"]["dynamic_variable_placeholders"],
        )

    def test_argparse_create(self) -> None:
        args = build_parser().parse_args(["create", "https://demo.ngrok-free.app"])
        self.assertEqual(args.command, "create")
        self.assertEqual(args.public_base_url, "https://demo.ngrok-free.app")

    def test_argparse_update_url(self) -> None:
        args = build_parser().parse_args(
            ["update-url", "https://demo.ngrok-free.app", "--agent-id", "agent_123"]
        )
        self.assertEqual(args.command, "update-url")
        self.assertEqual(args.agent_id, "agent_123")


if __name__ == "__main__":
    unittest.main()
