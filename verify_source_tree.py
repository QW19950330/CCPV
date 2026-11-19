"""Verify the public source snapshot using only the Python standard library."""
from pathlib import Path
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
if os.name == "nt" and not str(HERE).startswith(chr(92)*2+"?"+chr(92)):
    HERE = Path(chr(92)*2+"?"+chr(92)+str(HERE))

def main():
    manifest = json.loads((HERE / "PUBLIC_FILE_MANIFEST.json").read_text(encoding="utf-8"))
    checked = 0
    for row in manifest["files"]:
        path = HERE / row["path"]
        data = path.read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("Source snapshot mismatch: " + row["path"])
        checked += 1
    print(json.dumps({"status":"PASS_PUBLIC_SOURCE_SNAPSHOT","files":checked,
                      "training_updates":0,"model_forwards":0}))
if __name__ == "__main__":
    main()
