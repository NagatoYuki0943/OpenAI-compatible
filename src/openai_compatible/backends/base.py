from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass(slots=True)
class GenerationRequest:
    model: str
    messages: list[dict[str, Any]]
    sampling_params: dict[str, Any]
    n: int = 1
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationResult:
    content: str = ""
    reasoning_content: str | None = None
    finish_reason: str = "stop"
    tool_calls: list[dict[str, Any]] | None = None
    logprobs: Any = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationChunk:
    index: int
    content: str | None = None
    reasoning_content: str | None = None
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    logprobs: Any = None
    usage: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class BaseModelBackend(ABC):
    def __init__(
        self,
        model_id: str,
        *,
        max_concurrency: int = 1,
        stream_chunk_size: int = 12,
    ) -> None:
        self.model_id = model_id
        self.stream_chunk_size = stream_chunk_size
        self.model: Any = None
        self._loaded = False
        self._lifecycle_lock = asyncio.Lock()
        self._inference_semaphore = asyncio.Semaphore(max_concurrency)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def load(self) -> None:
        async with self._lifecycle_lock:
            if self._loaded:
                return
            started_at = time.perf_counter()
            logger.info("Loading model | backend={} | model={}", type(self).__name__, self.model_id)
            try:
                self.model = await asyncio.to_thread(self.load_model)
                self._loaded = True
            except Exception:
                logger.exception(
                    "Model loading failed | backend={} | model={}",
                    type(self).__name__,
                    self.model_id,
                )
                raise
            logger.info(
                "Model loaded | backend={} | model={} | elapsed_ms={:.2f}",
                type(self).__name__,
                self.model_id,
                (time.perf_counter() - started_at) * 1000,
            )

    async def unload(self) -> None:
        async with self._lifecycle_lock:
            if not self._loaded:
                return
            started_at = time.perf_counter()
            try:
                await asyncio.to_thread(self.unload_model)
            except Exception:
                logger.exception(
                    "Model cleanup failed | backend={} | model={}",
                    type(self).__name__,
                    self.model_id,
                )
                raise
            finally:
                self.model = None
                self._loaded = False
            logger.info(
                "Model unloaded | backend={} | model={} | elapsed_ms={:.2f}",
                type(self).__name__,
                self.model_id,
                (time.perf_counter() - started_at) * 1000,
            )

    async def generate(self, request: GenerationRequest) -> list[GenerationResult]:
        self._ensure_loaded()
        started_at = time.perf_counter()
        logger.info(
            "Inference started | backend={} | model={} | choices={} | messages={}",
            type(self).__name__,
            request.model,
            request.n,
            len(request.messages),
        )
        try:
            async with self._inference_semaphore:
                results = await asyncio.to_thread(self.infer, request)
            if len(results) != request.n:
                raise RuntimeError(
                    f"Backend returned {len(results)} result(s), expected {request.n}"
                )
        except Exception:
            logger.exception(
                "Inference failed | backend={} | model={}",
                type(self).__name__,
                request.model,
            )
            raise
        logger.info(
            "Inference finished | backend={} | model={} | choices={} | elapsed_ms={:.2f}",
            type(self).__name__,
            request.model,
            len(results),
            (time.perf_counter() - started_at) * 1000,
        )
        return results

    async def stream_generate(self, request: GenerationRequest) -> AsyncIterator[GenerationChunk]:
        results = await self.generate(request)
        prompt_tokens = next(
            (item.prompt_tokens for item in results if item.prompt_tokens is not None),
            None,
        )
        completion_tokens = (
            sum(item.completion_tokens for item in results)
            if all(item.completion_tokens is not None for item in results)
            else None
        )
        reasoning_tokens = (
            sum(item.reasoning_tokens for item in results)
            if all(item.reasoning_tokens is not None for item in results)
            else None
        )
        for index, result in enumerate(results):
            if result.reasoning_content:
                for start in range(0, len(result.reasoning_content), self.stream_chunk_size):
                    yield GenerationChunk(
                        index=index,
                        reasoning_content=result.reasoning_content[
                            start : start + self.stream_chunk_size
                        ],
                    )
            for start in range(0, len(result.content), self.stream_chunk_size):
                yield GenerationChunk(
                    index=index,
                    content=result.content[start : start + self.stream_chunk_size],
                )
            usage = None
            if (
                index == len(results) - 1
                and prompt_tokens is not None
                and completion_tokens is not None
                and reasoning_tokens is not None
            ):
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
                }
            yield GenerationChunk(
                index=index,
                finish_reason=result.finish_reason,
                tool_calls=result.tool_calls,
                logprobs=result.logprobs,
                usage=usage,
            )

    @abstractmethod
    def load_model(self) -> Any:
        """Load tokenizer/model resources and return the model object."""

    @abstractmethod
    def infer(self, request: GenerationRequest) -> list[GenerationResult]:
        """Run synchronous inference in a worker thread."""

    def unload_model(self) -> None:
        """Release model resources. Override for GPU cleanup."""
        return None

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Model backend is not loaded")
