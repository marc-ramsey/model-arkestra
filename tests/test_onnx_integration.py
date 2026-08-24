"""Real integration tests for onnx_server.py using actual ONNX Runtime."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
import pytest
import requests

BASE_DIR = Path(__file__).resolve().parent.parent.parent
# Use /tmp to avoid permission issues on shared home dirs
MODEL_CACHE = Path("/tmp/model-arkestra-cache/models")


def download_model(repo_id: str, pattern: str | None = None) -> str:
    """Download a model from HuggingFace, cache locally."""
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download
    path = snapshot_download(
        repo_id,
        allow_patterns=[pattern] if pattern else None,
        cache_dir=str(MODEL_CACHE / "hf_cache"),
    )
    return str(path)


def get_model_path(repo_id: str, filename: str) -> str:
    """Resolve a specific file inside a cached model."""
    base = download_model(repo_id)
    full = os.path.join(base, filename)
    if os.path.isfile(full):
        return full

    # HF hubs symlinks — resolve via realpath
    blobs_base = str(MODEL_CACHE / "hf_cache")
    for root, _, files in os.walk(blobs_base):
        target_basename = os.path.basename(filename)
        if any(target_basename == f for f in files):
            resolved = os.path.realpath(os.path.join(root, [f for f in files if target_basename in f][0]))
            return resolved

    # Last resort: walk the model dir and find by basename
    for root, _, files in os.walk(base):
        if os.path.basename(filename) in files:
            full_match = os.path.join(root, os.path.basename(filename))
            return os.path.realpath(full_match)

    raise FileNotFoundError(f"File {filename} not found in {repo_id}")


@pytest.fixture(scope="session")
def embedding_model_path():
    """Return path to bge-small-en-v1.5 model.onnx."""
    return get_model_path("Xenova/bge-small-en-v1.5", "onnx/model.onnx")


@pytest.fixture(scope="session")
def embedding_tokenizer_path():
    """Return path to bge-small-en-v1.5 snapshot dir (has tokenizer.json, vocab.txt)."""
    base = download_model("Xenova/bge-small-en-v1.5")
    return base  # the snapshot root has tokenizer files, not the onnx/ subdir


class TestOnnxEmbedding:
    """Test real embedding inference with ONNX Runtime via OnnxServer."""

    def test_load_model_and_run(self, embedding_model_path):
        """Verify ONNX Runtime can load the model and produce output."""
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(embedding_model_path, providers=["CPUExecutionProvider"])
        inputs = {inp.name: inp for inp in sess.get_inputs()}

        # Run with dummy input matching expected shape
        batch = 1
        seq_len = 5
        dtype = np.int64
        feed = {name: np.ones((batch, seq_len), dtype=dtype) for name in inputs}
        outputs = sess.run(None, feed)

        assert len(outputs) > 0
        output_shape = outputs[0].shape
        assert len(output_shape) == 3  # [batch, seq_len, hidden]
        assert output_shape[0] == 1

    def test_server_embed_endpoint(self, embedding_model_path, embedding_tokenizer_path, tmp_path):
        """Start the server and hit /v1/embeddings with real input."""
        port = 18765
        model_url = str(embedding_model_path)
        tokenizer_url = str(embedding_tokenizer_path)

        proc = subprocess.Popen(
            ["python", "-m", "model_arkestra.onnx_server",
             "--model", model_url,
             "--type", "embedding",
             "--port", str(port),
             "--tokenizer", tokenizer_url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to start
        deadline = time.time() + 15
        connected = False
        while time.time() < deadline:
            try:
                r = requests.get(f"http://localhost:{port}/health", timeout=2)
                if r.status_code == 200:
                    connected = True
                    break
            except requests.ConnectionError:
                time.sleep(0.5)

        assert connected, "Server failed to start within timeout"

        try:
            # Test embedding
            resp = requests.post(
                f"http://localhost:{port}/v1/embeddings",
                json={"input": "hello world"},
                headers={"Content-Type": "application/json"},
            )
            assert resp.status_code == 200, f"Embedding failed: {resp.text}"

            data = resp.json()
            assert "data" in data
            assert len(data["data"]) == 1
            embedding = data["data"][0]["embedding"]
            assert isinstance(embedding, list)
            # BGE-small has 384-dim embeddings
            assert len(embedding) == 384

            # Test semantic similarity: related texts should have higher cosine sim
            emb1 = np.array(data["data"][0]["embedding"])

            resp2 = requests.post(
                f"http://localhost:{port}/v1/embeddings",
                json={"input": "greetings earth"},  # similar meaning
                headers={"Content-Type": "application/json"},
            )
            emb2 = np.array(resp2.json()["data"][0]["embedding"])

            resp3 = requests.post(
                f"http://localhost:{port}/v1/embeddings",
                json={"input": "i love quantum computing"},  # different topic
                headers={"Content-Type": "application/json"},
            )
            emb3 = np.array(resp3.json()["data"][0]["embedding"])

            # Cosine similarity
            def cos(a, b):
                return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

            sim_close = cos(emb1, emb2)  # should be high
            sim_far = cos(emb1, emb3)    # should be lower

            assert sim_close > sim_far, (
                f"Semantic similarity failed: close={sim_close:.3f}, far={sim_far:.3f}"
            )

        finally:
            proc.terminate()
            proc.wait(timeout=5)


class TestOnnxHealth:
    """Test health endpoint."""

    def test_health(self):
        """Server starts and health check works."""
        # Use a minimal dummy ONNX model from onnxruntime's test data
        import onnxruntime as ort
        test_model = str(Path(ort.__file__).parent / "datasets" / "mul_1.onnx")

        proc = subprocess.Popen(
            ["python", "-m", "model_arkestra.onnx_server",
             "--model", test_model,
             "--type", "embedding",  # just need any type for health check
             "--port", "18766"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                r = requests.get("http://localhost:18766/health", timeout=2)
                if r.status_code == 200:
                    assert r.json()["status"] == "ok"
                    break
            except requests.ConnectionError:
                time.sleep(0.5)
        else:
            out, err = proc.communicate(timeout=1)
            raise RuntimeError(f"Server failed: {err.decode()}")

        proc.terminate()
        proc.wait(timeout=5)
