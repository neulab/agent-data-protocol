import codecs
import json
import signal
import urllib.request
from collections.abc import Iterator

DATASET_URL = "https://huggingface.co/datasets/Lite-Coder/LiteCoder-Terminal-SFT/resolve/main/litecoder-sft.json"


def stream_json_array(url: str) -> Iterator[dict]:
    """Stream objects from a large top-level JSON array without loading it all."""
    decoder = json.JSONDecoder()
    utf8_decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    started = False

    with urllib.request.urlopen(url, timeout=60) as response:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                buffer += utf8_decoder.decode(b"", final=True)
            else:
                buffer += utf8_decoder.decode(chunk)

            while True:
                stripped = buffer.lstrip()
                if not started:
                    if not stripped:
                        break
                    if stripped[0] != "[":
                        raise ValueError("Expected a JSON array")
                    buffer = stripped[1:]
                    started = True
                    continue

                stripped = buffer.lstrip()
                if not stripped:
                    buffer = stripped
                    break
                if stripped[0] == ",":
                    buffer = stripped[1:]
                    continue
                if stripped[0] == "]":
                    return

                try:
                    item, end = decoder.raw_decode(stripped)
                except json.JSONDecodeError:
                    if not chunk:
                        raise
                    buffer = stripped
                    break

                yield item
                buffer = stripped[end:]

            if not chunk:
                break


if __name__ == "__main__":
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    for sample in stream_json_array(DATASET_URL):
        print(json.dumps(sample, ensure_ascii=False))
