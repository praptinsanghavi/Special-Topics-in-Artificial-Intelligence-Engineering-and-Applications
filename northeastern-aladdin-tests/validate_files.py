#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_files.py — Unified, stricter file-based validator for JSONL tests + responses/*.txt

• Tests: JSONL with {"id","name","prompt","expects"} per line
• Responses: responses/<ID>.txt where the FIRST JSON object is the machine output,
  and the remainder of the file is a short prose summary (word-capped).

Why this version?
- Fixes a bug where per-check helper args (e.g., default_cap) were passed to checks
  that didn't accept them. All checks here now accept **ctx and read only what they need.
- Keeps the modular check registry so devs can add/remove checks easily.
- Adds friendly CLI flags and CI-friendly exit status.

Usage:
  python validate_files.py \
    --tests runs/specialized-ai-assistant.jsonl \
    --responses_dir responses \
    --summary_csv runs/specialized-ai-assistant_summary.csv \
    --artifacts_dir artifacts \
    --max_prose_words 200 \
    --require_json_first true \
    --strict_noise true \
    --verbose
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Utilities
# -----------------------------

WORD_RE = re.compile(r"\b[\w’'-]+\b", re.UNICODE)
CODE_FENCE_RE = re.compile(r"```")
NOISE = [
    "Searched project", "Analyzed data", "View analysis",
    "javascript", "console.log(", "import Papa",
    "await window.fs", "import * as"
]
FILENAME_RE = re.compile(r'[\w\-/ ]+\.(csv|pdf|xls|xlsx|tsv)', re.IGNORECASE)

def word_count(s: str) -> int:
    return len(WORD_RE.findall(s or ""))

def find_filenames(s: str) -> List[str]:
    return [m.group(0) for m in FILENAME_RE.finditer(s or "")]

def starts_with_json(s: str) -> bool:
    i = 0
    while i < len(s) and s[i].isspace():
        i += 1
    return i < len(s) and s[i] == "{"

def extract_first_json(text: str) -> Tuple[Optional[str], str]:
    """
    Extract FIRST JSON object from text (brace/quote/escape aware).
    Returns (json_str, remainder_after_json) or (None, original_text) if not found.
    """
    depth, in_str, esc = 0, False, False
    start, i, n = None, 0, len(text)

    # skip to the first '{'
    while i < n and text[i] != "{":
        i += 1
    if i >= n:
        return None, text

    start = i
    while i < n:
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i+1], text[i+1:]
        i += 1

    return None, text  # unclosed

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"{path}: JSONL parse error at line {i}: {e}")
    return out

def write_artifact(dirp: str, tid: str, content: str) -> None:
    os.makedirs(dirp, exist_ok=True)
    with open(os.path.join(dirp, f"failed_{tid}.txt"), "w", encoding="utf-8") as f:
        f.write(content or "")

# -----------------------------
# Expectation checks (each accepts **ctx)
# -----------------------------

def check_json_first(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    if ctx.get("require_json_first", True) and not starts_with_json(resp_text):
        errs.append("JSON not first")

def check_json_present(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    if jobj is None:
        errs.append("cannot extract/parse JSON block")

def check_prose_cap(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    cap = exp.get("prose_word_cap", ctx.get("default_cap", 200))
    wc = word_count(prose.strip())
    if wc > cap:
        errs.append(f"prose too long: {wc}>{cap}")

def check_hygiene(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    if not ctx.get("strict_noise", True):
        return
    if CODE_FENCE_RE.search(resp_text):
        errs.append("code fence detected")
    if any(p in resp_text for p in NOISE):
        errs.append("logs/tool output detected")

def check_style(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    want = exp.get("style")
    if want and jobj and jobj.get("style") != want:
        errs.append(f"style mismatch (expected {want}, got {jobj.get('style')})")

def check_disclaimer(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    if exp.get("require_disclaimer") and not (jobj and jobj.get("disclaimer")):
        errs.append("missing disclaimer")

def check_var_reasonable(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    if "VaR_reasonable" in exp and not (jobj and jobj.get("VaR_reasonable") is True):
        errs.append("VaR_reasonable not true")

def check_prompt_for_inputs(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    if exp.get("must_prompt_for_inputs"):
        if not re.search(r"please share|risk tolerance|sustainability|style|age", resp_text, re.I):
            errs.append("did not prompt for inputs")

def check_refuse_guarantee(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    if exp.get("must_refuse_guarantee"):
        if not re.search(r"\bno guarantees\b|\bcannot guarantee\b|\bnot guaranteed\b", resp_text, re.I):
            errs.append("did not refuse guarantees")

def check_concentration_flag(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    if exp.get("must_flag_concentration"):
        if not re.search(r"\bconcentration\b|\bdiversif", resp_text, re.I):
            errs.append("concentration risk not flagged")

def check_reflection_question(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    if exp.get("must_ask_question"):
        q = (jobj or {}).get("question", "")
        if not q or not str(q).strip().endswith(("?", "؟", "？")):
            errs.append("reflection question missing")

def check_msci(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    if exp.get("must_cite_msci") and not re.search(r"\bMSCI\b", resp_text):
        errs.append("MSCI not cited")
    if exp.get("no_fake_scores"):
        if re.search(r"\b(score|ESG score)\b\s*[:=]\s*\d+", resp_text, re.I):
            errs.append("appears to fabricate ESG scores")

def check_dataset_mismatch(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    if exp.get("must_correct_mismatch"):
        if not re.search(r"\bmismatch\b|\bnot appropriate\b|\bwrong dataset\b|\bshould use\b|\buse the correct\b", resp_text, re.I):
            errs.append("did not correct dataset mismatch")

def check_missing_file(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    if not exp.get("must_state_missing_file"):
        return
    p_files = set(find_filenames(prompt))
    r_files = set(find_filenames(resp_text))
    if p_files and not (p_files & r_files):
        if not re.search(r"(not found|unavailable|missing|cannot locate|no data)", resp_text, re.I):
            errs.append("missing-file disclosure not present")

def check_disambiguate(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    if exp.get("must_disambiguate"):
        if not re.search(r"\bdisambiguat|\bnot the other|\bthe specific file\b|\bchose\b", resp_text, re.I):
            errs.append("did not disambiguate between duplicate files")

def check_cov_fix(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    pat = exp.get("must_state_fix")
    if pat and not re.search(pat, resp_text, re.I):
        errs.append(f"covariance fix method not disclosed (need /{pat}/)")

def check_actions(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    want = exp.get("actions_include")
    if not want:
        return
    acts = (jobj or {}).get("actions", [])
    if not any(isinstance(a, str) and want.lower() in a.lower() for a in acts):
        errs.append(f"actions missing: {want}")

def check_risks(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    want = exp.get("risks_include")
    if not want:
        return
    risks = (jobj or {}).get("risks", [])
    if not any(isinstance(r, str) and want.lower() in r.lower() for r in risks):
        errs.append(f"risks missing: {want}")

def check_charts(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    if exp.get("no_charts") or exp.get("chart_only_if_explicit"):
        if re.search(r"\bchart|\bgraph|\bplot", resp_text, re.I):
            errs.append("mentioned charts/graphs when not allowed")

def check_climate(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    if exp.get("must_reference_climate"):
        if not re.search(r"\bclimate\b|\bSEC\b|\bstress\b", resp_text, re.I):
            errs.append("climate stress not referenced")

def check_overrides(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    if not exp.get("must_state_overrides"):
        return
    wanted_numbers: List[str] = []
    for key in ("mu", "sigma", "paths"):
        m = re.search(rf"{key}\s*=\s*([0-9.]+)", prompt, re.I)
        if m:
            wanted_numbers.append(m.group(1))
    txt = resp_text.lower()
    ok_tokens = ["mu", "σ", "sigma", "paths", "override"]
    ok = any(t in txt for t in ok_tokens) or any(x in resp_text for x in wanted_numbers)
    if not ok:
        errs.append("overrides not acknowledged")

def check_no_shorts(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    if exp.get("must_block_shorts"):
        if not re.search(r"\bno short|cannot short|long-only|shorting not allowed", resp_text, re.I):
            errs.append("shorting policy not enforced")

def check_missing_date(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    if exp.get("must_disclose_missing_date"):
        if not re.search(r"(no data|not available|missing|unavailable)", resp_text, re.I):
            errs.append("missing date disclosure not present")

def check_date_handled(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    if not exp.get("date_handled"):
        return
    m = re.search(r"\b(19|20)\d{2}-\d{2}-\d{2}\b", prompt)
    if m and (m.group(0) not in resp_text):
        errs.append(f"date not acknowledged: {m.group(0)}")

def check_tail_risk(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    if exp.get("must_flag_tail_risk"):
        if not re.search(r"\btail risk\b|\bfat[- ]?tail", resp_text, re.I):
            errs.append("tail risk not flagged")

def check_stochasticity(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    if exp.get("must_note_stochasticity"):
        if not re.search(r"\bstochastic|\brandom|\bruns may vary|\bnon[- ]?determin", resp_text, re.I):
            errs.append("stochastic behavior not acknowledged")

def check_avoid_single_asset(exp, prompt, resp_text, jobj, prose, errs, **ctx) -> None:
    if not exp.get("avoid_100_single_asset"):
        return
    alloc = (jobj or {}).get("allocation")
    if not isinstance(alloc, dict) or not alloc:
        return
    weights = [float(v) for v in alloc.values() if isinstance(v, (int, float))]
    total = sum(weights) if weights else 0.0
    if total <= 0:
        return
    if (max(weights) / total) > 0.99:
        errs.append("allocation concentrated ~100% in one asset")

CHECKS = [
    # order matters: format first
    check_json_first,
    check_json_present,
    check_prose_cap,
    check_hygiene,
    # content/policy
    check_style,
    check_disclaimer,
    check_var_reasonable,
    check_prompt_for_inputs,
    check_refuse_guarantee,
    check_concentration_flag,
    check_reflection_question,
    check_msci,
    check_dataset_mismatch,
    check_missing_file,
    check_disambiguate,
    check_cov_fix,
    check_actions,
    check_risks,
    check_charts,
    check_climate,
    check_overrides,
    check_no_shorts,
    check_missing_date,
    check_date_handled,
    check_tail_risk,
    check_stochasticity,
    check_avoid_single_asset,
]

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Validate JSONL tests against responses/<ID>.txt (JSON-first, prose after).")
    ap.add_argument("--tests", required=True, help="JSONL with id,name,prompt,expects (no responses inside)")
    ap.add_argument("--responses_dir", required=True, help="Directory containing <id>.txt response files")
    ap.add_argument("--summary_csv", required=True, help="Where to write the summary CSV")
    ap.add_argument("--artifacts_dir", required=True, help="Where to dump failing raw replies")
    ap.add_argument("--max_prose_words", type=int, default=200, help="Default prose word cap if a test doesn’t override")
    ap.add_argument("--require_json_first", type=str, default="true", choices=["true", "false"],
                    help="Fail if response doesn't start with a JSON object")
    ap.add_argument("--strict_noise", type=str, default="true", choices=["true", "false"],
                    help="Flag logs/tool traces/code fences in final output")
    ap.add_argument("--verbose", action="store_true", help="Print per-test status")
    args = ap.parse_args()

    require_json_first = (args.require_json_first.lower() == "true")
    strict_noise = (args.strict_noise.lower() == "true")

    tests = load_jsonl(args.tests)
    os.makedirs(os.path.dirname(args.summary_csv) or ".", exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    passes = 0
    fails = 0

    for t in tests:
        tid: str = t.get("id")
        name: str = t.get("name", "")
        exp: Dict[str, Any] = t.get("expects", {}) or {}
        prompt: str = t.get("prompt", "")

        resp_path = os.path.join(args.responses_dir, f"{tid}.txt")
        if not os.path.exists(resp_path):
            fails += 1
            rows.append({
                "id": tid, "name": name, "passed": False,
                "json_ok": False, "json_first_ok": False,
                "prose_words": "", "errors": "missing response file"
            })
            if args.verbose:
                print(f"[{tid}] ❌ missing response file")
            continue

        resp = open(resp_path, "r", encoding="utf-8").read()

        jstr, rest = extract_first_json(resp)
        jobj = None
        json_ok = False
        json_first_ok = starts_with_json(resp)

        errors: List[str] = []

        if jstr:
            try:
                jobj = json.loads(jstr)
                json_ok = True
            except Exception as e:
                errors.append(f"JSON parse error: {e}")
        else:
            # still allow checks to add their own messages (json_present will complain)
            pass

        prose = (rest or "").strip()

        ctx = {
            "default_cap": args.max_prose_words,
            "require_json_first": require_json_first,
            "strict_noise": strict_noise,
        }

        for chk in CHECKS:
            chk(exp=exp, prompt=prompt, resp_text=resp, jobj=jobj, prose=prose, errs=errors, **ctx)

        passed = (json_ok and not errors)
        if passed:
            passes += 1
            if args.verbose:
                print(f"[{tid}] ✅ pass")
        else:
            fails += 1
            write_artifact(args.artifacts_dir, tid, resp)
            if args.verbose:
                preview = ", ".join(errors[:3]) + ("..." if len(errors) > 3 else "")
                print(f"[{tid}] ❌ fail — {preview}")

        rows.append({
            "id": tid,
            "name": name,
            "passed": passed,
            "json_ok": json_ok,
            "json_first_ok": json_first_ok,
            "prose_words": word_count(prose),
            "errors": "; ".join(errors)
        })

    headers = ["id", "name", "passed", "json_ok", "json_first_ok", "prose_words", "errors"]
    with open(args.summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in headers}
            w.writerow(row)

    total = len(tests)
    rate = round(100 * passes / max(1, total), 1)
    print(f"Total={total}, Pass={passes}, Fail={fails}, Rate={rate}%")

    if fails > 0:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
