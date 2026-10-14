"""Expand the baseline search pools with extra phrasings from a local LLM (llama.cpp server).

Only adds PHRASINGS to existing canonical keys; never adds keys (keys map to the taxonomy and
to coverage balancing). Output is written to a separate file with provenance; review it, then
point `search_pools:` in your env config at it.

Start llama.cpp's server first (OpenAI-compatible API, default port 8080):
    scripts\\windows\\start_llm_server.bat          (laptop: paths set inside the .bat)
    llama-server -m path/to/model.gguf --port 8080 -c 4096 -ngl 99 --jinja

    python scripts/expand_search_pools.py --dry-run                # print, don't write
    python scripts/expand_search_pools.py                          # action, place, style
    python scripts/expand_search_pools.py --n 6 --slots action

Cleaning applied to every model suggestion:
  - lowercase, strip quotes/punctuation, 1-5 words, no duplicates (incl. existing phrasings)
  - drop phrasings containing words that belong to OTHER slots (e.g. an action phrasing that
    mentions a city or a place type), which would break the slot combinations
"""
import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from src.common.config import CONFIG_DIR  # noqa: E402

SLOT_HINTS = {
    "action": "what people are doing in the video",
    "place": "the type of place where it was filmed (not a named city)",
    "location": "the name of the city (alternative ways to write the city name only)",
    "style": "the kind of video (news report, raw footage, etc.)",
}
STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "and", "for", "to", "with", "by", "uk"}

PROMPT = """You help build YouTube search queries for a research dataset of real-world video footage.

Slot: {slot} ({hint})
Value: "{key}"
Existing phrasings: {existing}

Give {n} NEW short search phrasings (1-5 words) for this value that a person would realistically
type into YouTube to find real footage of it. Vary the wording; include common synonyms and
specific sub-types. Do NOT mention any city, country, or place name. Do NOT repeat existing phrasings.

Return JSON only: {{"phrasings": ["...", "..."]}}"""


SCHEMA = {"type": "object", "properties": {"phrasings": {"type": "array", "items": {"type": "string"}}},
          "required": ["phrasings"]}


def _post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def server_model(url: str) -> str:
    """Name/path of the GGUF loaded in llama-server, for provenance."""
    try:
        with urllib.request.urlopen(f"{url}/v1/models", timeout=10) as r:
            return json.loads(r.read())["data"][0]["id"]
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Cannot reach llama-server at {url} ({type(e).__name__}). Is it running?") from e


def chat(url: str, prompt: str, temperature: float) -> list:
    """Call llama-server's OpenAI-compatible endpoint, constraining output to the JSON schema."""
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": temperature,
            "response_format": {"type": "json_schema", "json_schema": {"name": "phrasings", "schema": SCHEMA}},
            "chat_template_kwargs": {"enable_thinking": False}}  # Qwen3: skip <think> (ignored by other models)
    text = _post(f"{url}/v1/chat/completions", body)["choices"][0]["message"]["content"] or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    try:
        return json.loads(text).get("phrasings", [])
    except json.JSONDecodeError:  # older builds may ignore the schema: pull the first {...} block
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)).get("phrasings", []) if m else []


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9']+", text.lower()) if t not in STOPWORDS}


def clean(raw: list, existing: list[str], banned: set[str], max_words: int = 5) -> tuple[list[str], list[str]]:
    seen = {e.lower() for e in existing}
    kept, dropped = [], []
    for p in raw:
        if not isinstance(p, str):
            continue
        p = re.sub(r"[\"“”'`.,!?;:()\[\]#]", "", p.lower()).strip()
        p = re.sub(r"\s+", " ", p)
        if len(p) < 3 or p in seen or not (1 <= len(p.split()) <= max_words):
            continue
        clash = tokens(p) & banned
        if clash:
            dropped.append(f"{p}  (mentions {', '.join(sorted(clash))})")
            continue
        seen.add(p)
        kept.append(p)
    return kept, dropped


def banned_tokens(pools: dict, slot: str) -> set[str]:
    """Words from other slots' keys/phrasings, minus words this slot already uses itself."""
    other, own = set(), set()
    for s, vals in pools["slots"].items():
        bag = own if s == slot else other
        for key, phr in vals.items():
            bag |= tokens(key) | set().union(*(tokens(p) for p in phr))
    return other - own


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="search_pools.yaml", help="baseline pools file in configs/")
    ap.add_argument("--out", default="search_pools.expanded.yaml", help="output file in configs/")
    ap.add_argument("--url", default="http://localhost:8080", help="llama-server URL")
    ap.add_argument("--n", type=int, default=8, help="phrasings to request per value")
    ap.add_argument("--slots", nargs="*", default=["action", "place", "style"],
                    help="slots to expand (location excluded by default: city-level only)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    model = server_model(args.url)
    print(f"llama-server at {args.url}, model: {model}")
    pools = yaml.safe_load((CONFIG_DIR / args.base).read_text(encoding="utf-8"))
    out = json.loads(json.dumps(pools))  # deep copy
    report = {}

    for slot in args.slots:
        if slot not in pools["slots"]:
            print(f"! unknown slot '{slot}', skipping")
            continue
        banned = banned_tokens(pools, slot)
        print(f"\n=== {slot} ===")
        for key, existing in pools["slots"][slot].items():
            prompt = PROMPT.format(slot=slot, hint=SLOT_HINTS.get(slot, slot), key=key,
                                   existing=", ".join(existing), n=args.n)
            try:
                raw = chat(args.url, prompt, args.temperature)
            except Exception as e:  # noqa: BLE001
                print(f"  ! {key}: model call failed ({type(e).__name__}: {e})")
                continue
            kept, dropped = clean(raw, existing, banned)
            out["slots"][slot][key] = existing + kept
            report[f"{slot}.{key}"] = {"added": len(kept), "dropped": len(dropped)}
            print(f"  {key:<14} +{len(kept):<2} {kept}")
            for d in dropped:
                print(f"  {'':<14} dropped: {d}")

    added = sum(r["added"] for r in report.values())
    print(f"\nTotal: +{added} phrasings across {len(report)} values")
    if args.dry_run or not report:
        print("Dry run / nothing generated: no file written.")
        return

    out["_provenance"] = {
        "generated_by": "scripts/expand_search_pools.py",
        "base_file": args.base, "backend": "llama.cpp llama-server", "model": model, "temperature": args.temperature,
        "n_requested_per_value": args.n, "slots_expanded": args.slots,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt_template": PROMPT, "reviewed": False,
    }
    header = ("# AUTO-GENERATED by scripts/expand_search_pools.py. Review the phrasings, delete bad ones,\n"
              "# set _provenance.reviewed: true, then set `search_pools: search_pools.expanded.yaml`\n"
              "# in your env config. The baseline file is never modified.\n")
    path = CONFIG_DIR / args.out
    path.write_text(header + yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
