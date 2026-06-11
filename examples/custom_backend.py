from typing import Any

from openai_compatible.backends import (
    BaseModelBackend,
    GenerationRequest,
    GenerationResult,
)


class CustomModelBackend(BaseModelBackend):
    def load_model(self) -> Any:
        # Return your tokenizer/model/pipeline object here.
        return {"ready": True}

    def infer(self, request: GenerationRequest) -> list[GenerationResult]:
        # Convert request.messages for your model and pass request.sampling_params.
        return [
            GenerationResult(
                content=f"Custom model response {index + 1}",
                reasoning_content="Optional reasoning output",
            )
            for index in range(request.n)
        ]

    def unload_model(self) -> None:
        # Release GPU memory or other external resources here.
        pass
