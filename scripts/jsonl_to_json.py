import json
import sys

dictionaries = [json.loads(line) for line in sys.stdin]
indented_json = json.dumps(dictionaries, indent=2)
print(indented_json)
