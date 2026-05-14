import sys
import urllib.request

DATA_URL = "https://huggingface.co/datasets/allenai/Sera-4.6-Lite-T2/resolve/main/sera-4.6-lite-t2_36083_string_enriched.jsonl"


if __name__ == "__main__":
    try:
        with urllib.request.urlopen(DATA_URL) as response:
            for raw_line in response:
                if raw_line.strip():
                    sys.stdout.write(raw_line.decode("utf-8"))
    except BrokenPipeError:
        pass
