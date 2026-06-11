from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Any

import aiohttp
import httpx
import requests

from openai_compatible.clients.common import build_request


def iter_sse_lines(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        if payload:
            yield json.loads(payload)


async def iter_async_sse_lines(
    lines: AsyncIterator[str],
) -> AsyncIterator[dict[str, Any]]:
    async for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        if payload:
            yield json.loads(payload)


def requests_chat(
    url: str,
    data: dict[str, Any],
    headers: dict[str, str],
) -> Iterator[dict[str, Any]]:
    with requests.post(
        url, json=data, headers=headers, timeout=120, stream=data.get("stream", False)
    ) as response:
        response.raise_for_status()
        if not data.get("stream"):
            yield response.json()
            return
        yield from iter_sse_lines(response.iter_lines(decode_unicode=True))


def httpx_sync_chat(
    url: str,
    data: dict[str, Any],
    headers: dict[str, str],
) -> Iterator[dict[str, Any]]:
    with httpx.Client(timeout=120) as client:
        if not data.get("stream"):
            response = client.post(url, json=data, headers=headers)
            response.raise_for_status()
            yield response.json()
            return
        with client.stream("POST", url, json=data, headers=headers) as response:
            response.raise_for_status()
            yield from iter_sse_lines(response.iter_lines())


async def httpx_async_chat(
    url: str,
    data: dict[str, Any],
    headers: dict[str, str],
) -> AsyncIterator[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=120) as client:
        if not data.get("stream"):
            response = await client.post(url, json=data, headers=headers)
            response.raise_for_status()
            yield response.json()
            return
        async with client.stream("POST", url, json=data, headers=headers) as response:
            response.raise_for_status()
            async for event in iter_async_sse_lines(response.aiter_lines()):
                yield event


async def aiohttp_async_chat(
    url: str,
    data: dict[str, Any],
    headers: dict[str, str],
) -> AsyncIterator[dict[str, Any]]:
    timeout = aiohttp.ClientTimeout(total=120)
    async with (
        aiohttp.ClientSession(timeout=timeout, headers=headers) as session,
        session.post(url, json=data) as response,
    ):
        response.raise_for_status()
        if not data.get("stream"):
            yield await response.json()
            return
        buffer = ""
        async for chunk in response.content.iter_any():
            buffer += chunk.decode("utf-8")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                for event in iter_sse_lines([line]):
                    yield event
                if line.strip() == "data: [DONE]":
                    return


async def _run_async(
    implementation: str,
    url: str,
    data: dict[str, Any],
    headers: dict[str, str],
) -> None:
    function = aiohttp_async_chat if implementation == "aiohttp" else httpx_async_chat
    async for event in function(url, data, headers):
        print(json.dumps(event, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client",
        choices=["requests", "httpx", "httpx-async", "aiohttp"],
        default="httpx-async",
    )
    parser.add_argument("--image")
    parser.add_argument("--video")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("API_KEY", "I AM AN API KEY")
    url = os.getenv("CHAT_URL", "http://127.0.0.1:8000/v1/chat/completions")
    model = os.getenv("MODEL_ID", "demo-multimodal-model")
    headers = {"Accept": "application/json", "Authorization": f"Bearer {api_key}"}
    data = build_request(
        model=model,
        image=args.image,
        video=args.video,
        stream=args.stream,
    )

    if args.client == "requests":
        events = requests_chat(url, data, headers)
    elif args.client == "httpx":
        events = httpx_sync_chat(url, data, headers)
    else:
        asyncio.run(_run_async(args.client, url, data, headers))
        return
    for event in events:
        print(json.dumps(event, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
