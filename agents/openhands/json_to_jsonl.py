import json
import sys

data = json.load(sys.stdin)
for datum in data:
    print(json.dumps(datum))
