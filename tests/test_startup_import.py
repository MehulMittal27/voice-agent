import unittest

from fastapi import FastAPI
from uvicorn.importer import import_from_string


class StartupImportTests(unittest.TestCase):
    def test_uvicorn_import_string_loads_app_from_src_layout(self) -> None:
        app = import_from_string("voice_agent.main:app")

        self.assertIsInstance(app, FastAPI)
        self.assertEqual(app.title, "Voice Agent Service")


if __name__ == "__main__":
    unittest.main()
