from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import runtime_config
from runtime_config import PROJECT_ENV_OVERRIDE_NAMES, load_project_env


class RuntimeConfigTests(unittest.TestCase):
    def test_comfyui_root_discovers_portable_runtime_before_empty_external_stub(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "portable" / "python" / "python.exe"
            executable.parent.mkdir(parents=True)
            executable.touch()
            comfy = root / "portable" / "ComfyUI"
            comfy.mkdir()
            (comfy / "main.py").touch()
            previous = os.environ.pop("COMFYUI_ROOT", None)
            try:
                with mock.patch.object(runtime_config.sys, "executable", str(executable)):
                    self.assertEqual(runtime_config.comfyui_root(), comfy.resolve())
            finally:
                if previous is not None:
                    os.environ["COMFYUI_ROOT"] = previous

    def test_project_env_loads_first_equals_and_never_overrides_process(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "# comment\nAI_FACTORY_TEST_SECRET=abc=def==\n"
                "AI_FACTORY_TEST_EXISTING=from-file\n"
                "INVALID-NAME=ignored\n",
                encoding="utf-8",
            )
            old_secret = os.environ.pop("AI_FACTORY_TEST_SECRET", None)
            old_existing = os.environ.get("AI_FACTORY_TEST_EXISTING")
            os.environ["AI_FACTORY_TEST_EXISTING"] = "from-process"
            try:
                loaded = load_project_env(env_file)
                self.assertEqual(os.environ["AI_FACTORY_TEST_SECRET"], "abc=def==")
                self.assertEqual(os.environ["AI_FACTORY_TEST_EXISTING"], "from-process")
                self.assertEqual(loaded, ("AI_FACTORY_TEST_SECRET",))
                self.assertNotIn("INVALID-NAME", os.environ)
            finally:
                os.environ.pop("AI_FACTORY_TEST_SECRET", None)
                if old_secret is not None:
                    os.environ["AI_FACTORY_TEST_SECRET"] = old_secret
                if old_existing is None:
                    os.environ.pop("AI_FACTORY_TEST_EXISTING", None)
                else:
                    os.environ["AI_FACTORY_TEST_EXISTING"] = old_existing

    def test_existing_environment_wrapping_quotes_are_sanitized(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                'AI_FACTORY_TEST_URL="https://api.example.test/v1"\n',
                encoding="utf-8",
            )
            old_value = os.environ.get("AI_FACTORY_TEST_URL")
            os.environ["AI_FACTORY_TEST_URL"] = '"https://api.example.test/v1"'
            try:
                loaded = load_project_env(env_file)
                self.assertEqual(loaded, ())
                self.assertEqual(
                    os.environ["AI_FACTORY_TEST_URL"], "https://api.example.test/v1"
                )
            finally:
                if old_value is None:
                    os.environ.pop("AI_FACTORY_TEST_URL", None)
                else:
                    os.environ["AI_FACTORY_TEST_URL"] = old_value

    def test_project_nonsecret_minimax_routing_overrides_stale_user_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "MiniMax_PROTOCOL=anthropic\n"
                "MiniMax_BASE_URL=https://api.minimaxi.com/anthropic\n"
                "MiniMax_MODEL=MiniMax-M2.7\n"
                "MiniMax_API_KEY=project-secret-must-not-override\n",
                encoding="utf-8",
            )
            names = (*PROJECT_ENV_OVERRIDE_NAMES, "MiniMax_API_KEY")
            previous = {name: os.environ.get(name) for name in names}
            os.environ.update({
                "MiniMax_PROTOCOL": "openai",
                "MiniMax_BASE_URL": "https://api.minimax.chat/v1",
                "MiniMax_MODEL": "abab6.5s-chat",
                "MiniMax_API_KEY": "user-secret-wins",
            })
            try:
                loaded = load_project_env(env_file)
                self.assertEqual(os.environ["MiniMax_PROTOCOL"], "anthropic")
                self.assertEqual(
                    os.environ["MiniMax_BASE_URL"],
                    "https://api.minimaxi.com/anthropic",
                )
                self.assertEqual(os.environ["MiniMax_MODEL"], "MiniMax-M2.7")
                self.assertEqual(os.environ["MiniMax_API_KEY"], "user-secret-wins")
                self.assertEqual(set(loaded), set(PROJECT_ENV_OVERRIDE_NAMES))
                self.assertNotIn("MiniMax_API_KEY", loaded)
            finally:
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

    def test_windows_launcher_declares_safe_precedence_without_printing_secret(self):
        launcher = (
            Path(__file__).resolve().parents[1] / "启动.bat"
        ).read_text(encoding="utf-8")
        self.assertIn("project MiniMax routing will be loaded securely", launcher)
        self.assertIn("MiniMax_API_KEY remains process/user-first", launcher)
        self.assertNotIn("echo %MiniMax_API_KEY%", launcher)
        self.assertNotIn("echo !MiniMax_API_KEY!", launcher)


if __name__ == "__main__":
    unittest.main()
