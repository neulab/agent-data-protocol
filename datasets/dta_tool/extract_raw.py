import json
import os
import sys
import urllib.request

DATASET_URL = "https://huggingface.co/datasets/dongsheng/DTA-Tool/resolve/main/DTA_tool.json"


def read_source():
    local_path = os.getenv("DTA_TOOL_JSON")
    if local_path:
        return open(local_path, "r", encoding="utf-8")
    return urllib.request.urlopen(DATASET_URL)


def iter_json_array(source, chunk_size=1024 * 1024):
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    eof = False

    while True:
        if not eof:
            chunk = source.read(chunk_size)
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8")
            if chunk:
                buffer += chunk
            else:
                eof = True

        if not started:
            buffer = buffer.lstrip()
            if not buffer:
                if eof:
                    break
                continue
            if buffer[0] != "[":
                raise ValueError("Expected a top-level JSON array")
            buffer = buffer[1:]
            started = True

        buffer = buffer.lstrip()
        if buffer.startswith(","):
            buffer = buffer[1:].lstrip()
        if buffer.startswith("]"):
            break

        try:
            item, idx = decoder.raw_decode(buffer)
        except json.JSONDecodeError:
            if eof:
                raise
            continue

        yield item
        buffer = buffer[idx:]

        if eof and not buffer.strip():
            break


def main():
    with read_source() as source:
        for item in iter_json_array(source):
            print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.stderr.close()
