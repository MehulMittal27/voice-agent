import unittest
from datetime import datetime, timedelta, timezone

from voice_agent.session import SessionStore


class SessionStoreTests(unittest.TestCase):
    def test_get_latest_active_returns_freshest_session(self) -> None:
        current = datetime(2026, 1, 1, tzinfo=timezone.utc)
        store = SessionStore(clock=lambda: current)
        store.create("old_perception")
        current = current + timedelta(seconds=5)
        store.create("fresh_perception")

        session = store.get_latest_active(touch=False)

        self.assertIsNotNone(session)
        assert session is not None
        self.assertEqual(session.perception_session_id, "fresh_perception")

    def test_get_latest_active_respects_max_age(self) -> None:
        current = datetime(2026, 1, 1, tzinfo=timezone.utc)
        store = SessionStore(clock=lambda: current)
        store.create("stale_perception")
        current = current + timedelta(minutes=3)

        session = store.get_latest_active(
            touch=False,
            max_age=timedelta(minutes=2),
        )

        self.assertIsNone(session)


if __name__ == "__main__":
    unittest.main()
