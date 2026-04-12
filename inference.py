from __future__ import annotations

import os

from openai import OpenAI  # required client for LLM-based inference path
from security_env.inference import main as _security_main

# Required for hosted inference
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct-1M")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional (used when running local image workflows)
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


def main() -> None:
    os.environ.setdefault("API_BASE_URL", API_BASE_URL)
    os.environ.setdefault("MODEL_NAME", MODEL_NAME)
    if OPENAI_API_KEY:
        os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)
    if HF_TOKEN:
        os.environ.setdefault("HF_TOKEN", HF_TOKEN)
    if LOCAL_IMAGE_NAME:
        os.environ.setdefault("LOCAL_IMAGE_NAME", LOCAL_IMAGE_NAME)

    _security_main()


if __name__ == "__main__":
    main()
