"""Bake the model and the corpus into the image.

Run at build time, not at run time: the image has to work offline, and the
corpus has to be the corpus -- the checksum is checked here, so a run never
has to trust a file it did not fetch.

This file is code the project wrote, not something downloaded, so it is
run from the image rather than from a download directory.
"""

import hashlib
import pathlib
import urllib.request

from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_DIR = pathlib.Path("/opt/models/Qwen2.5-1.5B-Instruct")
CORPUS_URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
CORPUS_SHA256 = "3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75"
CORPUS_PATH = pathlib.Path("/opt/data/pg1342.txt")


def bake_model() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        MODEL_ID,
        local_dir=str(MODEL_DIR),
        allow_patterns=["*.json", "*.safetensors", "*.txt"],
    )
    print("model baked at", MODEL_DIR)


def bake_corpus() -> None:
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(CORPUS_URL, CORPUS_PATH)
    digest = hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()
    if digest != CORPUS_SHA256:
        raise SystemExit(f"corpus sha256 {digest}, expected {CORPUS_SHA256}")
    print("corpus ok", digest)


if __name__ == "__main__":
    bake_model()
    bake_corpus()
