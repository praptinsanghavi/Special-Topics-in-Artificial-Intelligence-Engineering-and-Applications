#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_files.py — File-based validator to match the current inline validate.py

This mirrors your current validate.py checks/output, but reads responses from
separate files in a responses/ directory (one file per test: Txx.txt).

Response file format (no escaping needed):
Line 1..N : JSON block (must start with '{' and be valid JSON)
Remainder : One short sentence summary (kept simple; word-capped)

CLI:
  python validate_files.py \
    --tests runs/specialized-ai-assistant.jsonl \
    --responses_dir responses \
    --summary_csv runs/specialized-ai-assistant_summary.csv \
    --artifacts_dir artifacts
"""

import argparse
import csv
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# --------- same hygiene rules as your validate.py ----------
WORD_RE = re.compile(r"\b[\w’'-]+\b", re.UNICODE)
CODE_FENCE_RE = re.compile(r"```")
NOISE = ["Searched project", "Analyzed data", "javascript", "console.log(", "await window.fs"]

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Strict JSONL loader (1 object per line)."""
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"{path}: JSON error at line {i}: {e}")
    return out

def extract_first_json(text: str) -> Tuple[Optional[str], str]:
    """Extract the first JSON object from text; return (json_str, remainder)."""
    depth, in_str, esc = 0, False, False
    start, i, n = None, 0, len(text)
    # optional: skip leading whitespace
    while i < n and text[i].isspace():
        i += 1
    while i < n:
        ch = text[i]
        if ch == "{" and not in_str:
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and not in_str:
            depth -= 1
            if depth == 0:
                return text[start:i+1], text[i+1:]
        elif ch == '"' and not esc:
            in_str = not in_str
        elif ch == "\\" and in_str:
            esc = not esc
        else:
            esc = False
        i += 1
    return None, text

def word_count(s: str) -> int:
    return len(WORD_RE.findall(s or ""))

def write_artifact(dirp: str, tid: str, content: str) -> None:
    os.makedirs(dirp, exist_ok=True)
    with open(os.path.join(dirp, f"failed_{tid}.txt"), "w", encoding="utf-8") as f:
        f.write(content or "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", required=True, help="JSONL with tests only (id, prompt, expects)")
    ap.add_argument("--responses_dir", required=True, help="Folder containing Txx.txt files (raw replies)")
    ap.add_argument("--summary_csv", required=True, help="Output CSV path")
    ap.add_argument("--artifacts_dir", required=True, help="Folder to dump failing raw replies")
    ap.add_argument("--max_prose_words", type=int, default=200)
    args = ap.parse_args()

    tests = load_jsonl(args.tests)
    rows: List[Dict[str, Any]] = []
    passes = 0
    fails = 0

    for rec in tests:
        tid = rec.get("id", "?")
        exp = rec.get("expects", {}) or {}

        # Read associated response file: responses/<id>.txt
        resp_path = os.path.join(args.responses_dir, f"{tid}.txt")
        if not os.path.exists(resp_path):
            fails += 1
            rows.append({"id": tid, "passed": False, "errors": "missing response file"})
            continue

        with open(resp_path, "r", encoding="utf-8") as f:
            reply = f.read()

        errors: List[str] = []
        json_ok = False

        # Extract and parse the JSON block
        jstr, rest = extract_first_json(reply)
        if not jstr:
            errors.append("no JSON found")
        else:
            try:
                jobj = json.loads(jstr)
                json_ok = True
            except Exception as e:
                errors.append(f"bad JSON: {e}")

        # Summary (remainder text) – check word cap
        summary = (rest or "").strip()
        cap = exp.get("prose_word_cap", args.max_prose_words)
        if word_count(summary) > cap:
            errors.append("too many words in prose")

        # Disallow logs/code fences/tool traces (same noisy markers as validate.py)
        if any(p in reply for p in NOISE) or CODE_FENCE_RE.search(reply):
            errors.append("log/code detected")

        # Require disclaimer if asked
        if exp.get("require_disclaimer") and not (json_ok and jobj.get("disclaimer")):
            errors.append("missing disclaimer")

        # Outcome
        passed = not errors
        if passed:
            passes += 1
        else:
            fails += 1
            write_artifact(args.artifacts_dir, tid, reply)

        rows.append({
            "id": tid,
            "passed": passed,
            "errors": "; ".join(errors)
        })

    # CSV summary (same simple shape as validate.py)
    os.makedirs(os.path.dirname(args.summary_csv) or ".", exist_ok=True)
    with open(args.summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["id","passed","errors"])
        w.writeheader()
        w.writerows(rows)

    print(f"Total={len(tests)}, Pass={passes}, Fail={fails}, Rate={passes/len(tests)*100:.1f}%")

if __name__ == "__main__":
    main()
