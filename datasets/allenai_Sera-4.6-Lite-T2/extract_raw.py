import json
import signal
import urllib.request

DATA_URL = "https://huggingface.co/datasets/allenai/Sera-4.6-Lite-T2/resolve/main/sera-4.6-lite-t2_36083_string_enriched.jsonl"


if hasattr(signal, "SIGPIPE"):
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)


with urllib.request.urlopen(DATA_URL) as response:
    for line in response:
        item = json.loads(line)
        print(json.dumps(item, ensure_ascii=False))
