import unittest
from datetime import datetime, timedelta, timezone

from voice_agent.session import SessionInfo, SessionStore, update_language_state


def _fresh_session() -> SessionInfo:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return SessionInfo(
        conversation_id="c",
        perception_session_id="p",
        created_at=now,
        last_seen=now,
    )


class UpdateLanguageStateTests(unittest.TestCase):
    def test_two_high_turns_trigger_offer(self) -> None:
        session = _fresh_session()
        first = update_language_state(session, {"hesitation_score": 0.8}, "hmm")
        self.assertEqual(first["mode"], "german")
        self.assertEqual(session.high_hesitation_streak, 1)
        second = update_language_state(session, {"hesitation_score": 0.8}, "uh")
        self.assertEqual(second["mode"], "offer_english")
        self.assertFalse(second["just_switched"])
        self.assertTrue(session.english_offered)

    def test_single_very_high_turn_triggers_offer(self) -> None:
        session = _fresh_session()
        result = update_language_state(session, {"hesitation_score": 0.95}, "was")
        self.assertEqual(result["mode"], "offer_english")
        self.assertTrue(session.english_offered)

    def test_accepting_after_offer_flips_english_mode(self) -> None:
        session = _fresh_session()
        update_language_state(session, {"hesitation_score": 0.95}, "was")
        result = update_language_state(session, {"hesitation_score": 0.3}, "yes please")
        self.assertEqual(result["mode"], "english_locked")
        self.assertTrue(result["just_switched"])
        self.assertTrue(session.english_mode)

    def test_declining_after_offer_prevents_reoffer(self) -> None:
        session = _fresh_session()
        update_language_state(session, {"hesitation_score": 0.95}, "was")
        declined = update_language_state(session, {"hesitation_score": 0.3}, "no keep going")
        self.assertEqual(declined["mode"], "german_locked")
        self.assertFalse(session.english_mode)
        # Even a fresh stress spike does not re-offer; it stays locked to German.
        again = update_language_state(session, {"hesitation_score": 0.95}, "hmm")
        self.assertEqual(again["mode"], "german_locked")
        self.assertFalse(session.english_mode)

    def test_explicit_request_flips_immediately(self) -> None:
        session = _fresh_session()
        result = update_language_state(session, {"hesitation_score": 0.1}, "can you speak english")
        self.assertEqual(result["mode"], "english_locked")
        self.assertTrue(result["just_switched"])
        self.assertTrue(session.english_mode)
        self.assertTrue(session.english_offered)

    def test_german_explicit_requests_flip_immediately(self) -> None:
        for phrase in (
            "sprechen sie englisch",
            "können wir englisch sprechen",
            "auf englisch bitte",
            "can we speak english",
            "continue in english",
        ):
            with self.subTest(phrase=phrase):
                session = _fresh_session()
                result = update_language_state(
                    session, {"hesitation_score": 0.1}, phrase
                )
                self.assertEqual(result["mode"], "english_locked")
                self.assertTrue(result["just_switched"])
                self.assertTrue(session.english_mode)
                self.assertTrue(session.english_offered)

    def test_explicit_request_overrides_prior_decline(self) -> None:
        session = _fresh_session()
        update_language_state(session, {"hesitation_score": 0.95}, "was")
        declined = update_language_state(
            session, {"hesitation_score": 0.3}, "no keep going"
        )
        self.assertEqual(declined["mode"], "german_locked")
        self.assertFalse(session.english_mode)
        # A later explicit request overrides the earlier decline.
        switched = update_language_state(
            session, {"hesitation_score": 0.3}, "in english please"
        )
        self.assertEqual(switched["mode"], "english_locked")
        self.assertTrue(switched["just_switched"])
        self.assertTrue(session.english_mode)

    def test_substring_eng_does_not_false_positive(self) -> None:
        session = _fresh_session()
        result = update_language_state(
            session,
            {"hesitation_score": 0.1},
            "ich war in england und der flur war sehr eng",
        )
        self.assertEqual(result["mode"], "german")
        self.assertFalse(session.english_mode)
        self.assertFalse(session.english_offered)

    def test_streak_resets_when_hesitation_drops(self) -> None:
        session = _fresh_session()
        update_language_state(session, {"hesitation_score": 0.8}, "hmm")
        self.assertEqual(session.high_hesitation_streak, 1)
        update_language_state(session, {"hesitation_score": 0.2}, "ok")
        self.assertEqual(session.high_hesitation_streak, 0)

    def test_established_english_stays_locked(self) -> None:
        session = _fresh_session()
        update_language_state(session, {"hesitation_score": 0.1}, "in english please")
        later = update_language_state(session, {"hesitation_score": 0.1}, "thank you")
        self.assertEqual(later["mode"], "english")
        self.assertFalse(later["just_switched"])


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
