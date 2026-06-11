from openai_compatible.clients.common import build_request
from openai_compatible.clients.http import iter_sse_lines


def test_sse_parser_stops_at_done() -> None:
    lines = [
        'data: {"choices": [{"index": 0}]}',
        "",
        "data: [DONE]",
        'data: {"ignored": true}',
    ]

    assert list(iter_sse_lines(lines)) == [{"choices": [{"index": 0}]}]


def test_request_builder_supports_local_media(tmp_path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"png")

    request = build_request(
        model="test-model",
        image=str(image),
        video="https://example.com/video.mp4",
        stream=True,
    )
    content = request["messages"][1]["content"]

    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[2]["video_url"]["url"] == "https://example.com/video.mp4"
    assert request["stream_options"]["include_usage"] is True
