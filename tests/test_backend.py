from openai_compatible.backends import GenerationRequest


async def test_backend_lifecycle_is_idempotent(backend) -> None:
    await backend.load()
    await backend.load()
    assert backend.is_loaded
    assert backend.load_count == 1

    await backend.unload()
    await backend.unload()
    assert not backend.is_loaded
    assert backend.unload_count == 1


def test_model_card_contains_openai_and_capability_metadata(backend) -> None:
    card = backend.model_card()

    assert card["id"] == "test-model"
    assert card["object"] == "model"
    assert "reasoning" in card["capabilities"]
    assert card["input_modalities"] == ["text", "image"]
    assert card["inputModalities"] == ["text", "image"]
    assert card["max_output_tokens"] == 1024
    assert card["maxOutputTokens"] == 1024
    assert card["reasoning"]["supportedEfforts"] == ["low", "medium", "high"]


async def test_generate_and_default_stream_adapter(backend) -> None:
    await backend.load()
    request = GenerationRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        sampling_params={"top_k": 7},
        n=2,
        request_id="request-1",
    )

    results = await backend.generate(request)
    chunks = [chunk async for chunk in backend.stream_generate(request)]

    assert [result.content for result in results] == ["answer-0", "answer-1"]
    assert any(chunk.reasoning_content for chunk in chunks)
    assert any(chunk.finish_reason == "stop" for chunk in chunks)
    assert chunks[-1].usage["completion_tokens"] == 12
    await backend.unload()
