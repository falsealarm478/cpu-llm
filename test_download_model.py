import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch
from download_model import DEFAULT_MODEL, DEFAULT_REVISION, ModelStateError, ensure_model
from start_server import server_args, configure_inference_env


class ModelTests(unittest.TestCase):
    def setUp(self):
        if os.environ.get("TEST_WORK_ROOT"):
            self.root = Path(os.environ["TEST_WORK_ROOT"]) / uuid.uuid4().hex
            self.root.mkdir(parents=True)
        else:
            self.temp = tempfile.TemporaryDirectory()
            self.addCleanup(self.temp.cleanup)
            self.root = Path(self.temp.name)

    def downloader(self, model_id, revision, directory, token):
        (directory / "config.json").write_text('{"model_type":"llama"}')
        (directory / "tokenizer_config.json").write_text('{}')
        (directory / "tokenizer.json").write_text('{}')
        (directory / "model.safetensors").write_bytes(b"synthetic-test-weights")

    def test_first_download_then_no_network(self):
        downloader = Mock(side_effect=self.downloader)
        first, reused = ensure_model(self.root, DEFAULT_MODEL, DEFAULT_REVISION, "TOKEN", downloader)
        self.assertFalse(reused)
        downloader.assert_called_once()
        blocked = Mock(side_effect=AssertionError("Network must not be used"))
        second, reused = ensure_model(self.root, DEFAULT_MODEL, DEFAULT_REVISION, downloader=blocked)
        self.assertTrue(reused)
        self.assertEqual(first, second)
        blocked.assert_not_called()
        self.assertNotIn("TOKEN", (first / ".ready.json").read_text())

    def test_interrupted_download_has_no_ready_marker_and_can_resume(self):
        def broken(*args):
            self.downloader(*args)
            raise ConnectionError("interrupted")
        with self.assertRaises(ConnectionError):
            ensure_model(self.root, DEFAULT_MODEL, DEFAULT_REVISION, downloader=broken)
        self.assertFalse(list(self.root.rglob(".ready.json")))
        _, reused = ensure_model(self.root, DEFAULT_MODEL, DEFAULT_REVISION, downloader=self.downloader)
        self.assertFalse(reused)

    def test_corrupt_warm_cache_fails_without_network(self):
        path, _ = ensure_model(self.root, DEFAULT_MODEL, DEFAULT_REVISION, downloader=self.downloader)
        (path / "model.safetensors").write_bytes(b"corruption")
        blocked = Mock(side_effect=AssertionError("No network"))
        with self.assertRaises(ModelStateError):
            ensure_model(self.root, DEFAULT_MODEL, DEFAULT_REVISION, downloader=blocked)
        blocked.assert_not_called()

    def test_missing_warm_file_fails_without_network(self):
        path, _ = ensure_model(self.root, DEFAULT_MODEL, DEFAULT_REVISION, downloader=self.downloader)
        (path / "tokenizer.json").unlink()
        blocked = Mock(side_effect=AssertionError("No network"))
        with self.assertRaises(ModelStateError):
            ensure_model(self.root, DEFAULT_MODEL, DEFAULT_REVISION, downloader=blocked)
        blocked.assert_not_called()

    def test_new_revision_uses_new_directory(self):
        first, _ = ensure_model(self.root, DEFAULT_MODEL, DEFAULT_REVISION, downloader=self.downloader)
        second, _ = ensure_model(self.root, DEFAULT_MODEL, "a" * 40, downloader=self.downloader)
        self.assertNotEqual(first, second)

    def test_unpinned_revision_is_rejected_before_download(self):
        downloader = Mock()
        with self.assertRaises(ModelStateError):
            ensure_model(self.root, DEFAULT_MODEL, "main", downloader=downloader)
        downloader.assert_not_called()

    def test_path_traversal_in_manifest_is_rejected(self):
        path, _ = ensure_model(self.root, DEFAULT_MODEL, DEFAULT_REVISION, downloader=self.downloader)
        marker = path / ".ready.json"
        manifest = json.loads(marker.read_text())
        manifest["files"]["../../outside"] = "bad"
        marker.write_text(json.dumps(manifest))
        with self.assertRaises(ModelStateError):
            ensure_model(self.root, DEFAULT_MODEL, DEFAULT_REVISION, downloader=Mock())

    def test_startup_uses_settings_and_privacy_flags(self):
        with patch.dict(os.environ, {"MAX_MODEL_LEN": "256", "MAX_NUM_SEQS": "1"}):
            args = server_args("/models/local")
        self.assertEqual(args[:3], ["vllm", "serve", "/models/local"])
        self.assertEqual(args[args.index("--max-model-len") + 1], "256")
        self.assertIn("--no-enable-log-requests", args)
        self.assertIn("privacy_logging.RequestAuditMiddleware", args)

    def test_cache_budget_default_and_override(self):
        with patch.dict(os.environ, {}, clear=True):
            args = server_args("/models/local")
        self.assertEqual(args[args.index("--kv-cache-memory-bytes") + 1], "134217728")
        with patch.dict(os.environ, {"KV_CACHE_MEMORY_BYTES": "268435456"}):
            args = server_args("/models/local")
        self.assertEqual(args[args.index("--kv-cache-memory-bytes") + 1], "268435456")

    def test_legacy_cache_variable_cannot_override_byte_budget(self):
        with patch.dict(os.environ, {"VLLM_CPU_KVCACHE_SPACE": "1", "HF_TOKEN": "SECRET",
                                    "VLLM_API_KEY": "API_SECRET"}, clear=True):
            configure_inference_env()
            self.assertNotIn("VLLM_CPU_KVCACHE_SPACE", os.environ)
            self.assertNotIn("HF_TOKEN", os.environ)
            self.assertEqual(os.environ["VLLM_API_KEY"], "API_SECRET")
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")


if __name__ == "__main__":
    unittest.main()
