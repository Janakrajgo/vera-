"""
Helper to package static/index.html and dataset/expanded into Python modules:
  - ui_html.py
  - embedded_dataset.py
This ensures Render and other cloud platforms have the UI and data even if subfolders aren't committed to git.
"""
import base64
import gzip
import json
from pathlib import Path

# 1. Generate ui_html.py
static_index = Path("static/index.html")
if static_index.exists():
    with open(static_index, encoding="utf-8") as f:
        html_text = f.read()

    with open("ui_html.py", "w", encoding="utf-8") as f:
        f.write('"""Embedded UI HTML fallback for zero-configuration deployments."""\n\n')
        f.write('HTML_CONTENT = ' + repr(html_text) + '\n')
    print("Generated ui_html.py successfully!")

# 2. Generate embedded_dataset.py
all_data = {}
p = Path("dataset/expanded")
if p.exists():
    for folder, scope in [
        ("categories", "category"),
        ("merchants", "merchant"),
        ("customers", "customer"),
        ("triggers", "trigger"),
    ]:
        all_data[scope] = {}
        for f in (p / folder).glob("*.json"):
            with open(f, encoding="utf-8") as fp:
                d = json.load(fp)
                key = d.get("slug") or d.get("merchant_id") or d.get("customer_id") or d.get("id")
                all_data[scope][key] = d

    compressed_b64 = base64.b64encode(gzip.compress(json.dumps(all_data).encode("utf-8"))).decode("ascii")

    with open("embedded_dataset.py", "w", encoding="utf-8") as f:
        f.write('"""Embedded full dataset fallback for cloud deployments without subfolders."""\n')
        f.write("import gzip\nimport json\nimport base64\n\n")
        f.write(f'_COMPRESSED_DATA = "{compressed_b64}"\n\n')
        f.write("def get_embedded_contexts() -> dict[tuple[str, str], dict]:\n")
        f.write('    raw = gzip.decompress(base64.b64decode(_COMPRESSED_DATA.encode("ascii"))).decode("utf-8")\n')
        f.write("    parsed = json.loads(raw)\n")
        f.write("    res = {}\n")
        f.write("    for scope, items in parsed.items():\n")
        f.write("        for key, payload in items.items():\n")
        f.write('            res[(scope, key)] = {"version": 1, "payload": payload}\n')
        f.write("    return res\n")

    print("Generated embedded_dataset.py successfully!")
