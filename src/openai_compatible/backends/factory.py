from __future__ import annotations

import importlib

from loguru import logger

from openai_compatible.backends.base import BaseModelBackend
from openai_compatible.backends.demo import DemoModelBackend
from openai_compatible.config import Settings


def create_model_backend(settings: Settings) -> BaseModelBackend:
    class_path = settings.model_backend_class
    if not class_path:
        backend_class: type[BaseModelBackend] = DemoModelBackend
    else:
        module_name, separator, class_name = class_path.partition(":")
        if not separator or not module_name or not class_name:
            raise ValueError("MODEL_BACKEND_CLASS must use the format 'package.module:ClassName'")
        module = importlib.import_module(module_name)
        backend_class = getattr(module, class_name)
        if not isinstance(backend_class, type) or not issubclass(backend_class, BaseModelBackend):
            raise TypeError(f"{class_path} is not a BaseModelBackend subclass")

    backend = backend_class(
        settings.model_id,
        max_concurrency=settings.model_max_concurrency,
        stream_chunk_size=settings.model_stream_chunk_size,
    )
    logger.info(
        "Using model backend | backend={} | model={}",
        type(backend).__name__,
        backend.model_id,
    )
    return backend
