"""Dump a notebook's readable code: every line under 300 chars, cell headers included."""
import json
import sys

nb = json.load(open(sys.argv[1], encoding="utf-8"))
with open(sys.argv[2], "w", encoding="utf-8") as out:
    for i, c in enumerate(nb["cells"]):
        s = "".join(c["source"])
        out.write(f"\n########## cell {i} [{c['cell_type']}] {len(s)} chars\n")
        for ln in s.split("\n"):
            out.write((ln if len(ln) < 300 else f"<<{len(ln)} chars>>") + "\n")
