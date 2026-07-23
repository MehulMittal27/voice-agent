import unittest

from voice_agent.data.mock import MockDataProvider, _load_authorities


class AuthorityCsvTests(unittest.TestCase):
    def test_load_authorities_returns_24_rows(self) -> None:
        self.assertEqual(len(_load_authorities()), 24)


class MockDataProviderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.provider = MockDataProvider()

    async def test_find_german_occupation_matches_english_nurse(self) -> None:
        occupations = await self.provider.find_german_occupation("nurse", "en")

        self.assertTrue(occupations)
        self.assertTrue(occupations[0].label_de.startswith("Pflegefachfrau"))

    async def test_find_german_occupation_matches_ukrainian_nurse(self) -> None:
        occupations = await self.provider.find_german_occupation("медсестра", "uk")

        self.assertTrue(occupations)
        self.assertTrue(occupations[0].label_de.startswith("Pflegefachfrau"))

    async def test_find_german_occupation_matches_ukrainian_doctor(self) -> None:
        occupations = await self.provider.find_german_occupation("лікар", "uk")

        self.assertTrue(occupations)
        self.assertEqual(occupations[0].label_de, "Arzt/Ärztin")

    async def test_get_recognition_authority_returns_lfp_for_nurse(self) -> None:
        authority = await self.provider.get_recognition_authority("nurse")

        self.assertIsNotNone(authority)
        assert authority is not None
        self.assertIn("Bayerisches Landesamt für Pflege", authority.name)

    async def test_get_recognition_authority_returns_kubb_fallback(self) -> None:
        authority = await self.provider.get_recognition_authority("space botanist")

        self.assertIsNotNone(authority)
        assert authority is not None
        self.assertIn("KuBB", authority.name)

    async def test_get_labour_market_status_returns_shortage_for_pflege(self) -> None:
        status = await self.provider.get_labour_market_status("Pflege")

        self.assertTrue(status.shortage)


if __name__ == "__main__":
    unittest.main()
