from __future__ import annotations

import argparse
import os
from typing import Any

from openai import OpenAI

from openai_compatible.clients.common import build_messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    parser.add_argument("--video")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    client = OpenAI(
        api_key=os.getenv("API_KEY", "I AM AN API KEY"),
        base_url=os.getenv("BASE_URL", "http://127.0.0.1:8000/v1"),
        timeout=120,
    )
    common: dict[str, Any] = {
        "model": os.getenv("MODEL_ID", "demo-multimodal-model"),
        "messages": build_messages(args.image, args.video),
        "max_completion_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": 42,
        "n": 1,
        "stream": args.stream,
        "extra_body": {
            "top_k": 40,
            "min_p": 0.05,
            "repetition_penalty": 1.05,
            "min_tokens": 1,
            "thinking_token_budget": 256,
        },
    }

    if not args.stream:
        response = client.chat.completions.create(**common)
        print(response.model_dump_json(indent=2))
        return

    stream = client.chat.completions.create(
        **common,
        stream_options={"include_usage": True},
    )
    reasoning_parts: list[str] = []
    answer_parts: list[str] = []
    for chunk in stream:
        if chunk.usage:
            print(f"\nusage: {chunk.usage}")
        for choice in chunk.choices:
            reasoning = getattr(choice.delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)
            if choice.delta.content:
                answer_parts.append(choice.delta.content)
                print(choice.delta.content, end="", flush=True)
    print(f"\n\nreasoning: {''.join(reasoning_parts)}")
    print(f"answer: {''.join(answer_parts)}")


if __name__ == "__main__":
    main()
