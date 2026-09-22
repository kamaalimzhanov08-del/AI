"""Part 2 -- Gemini version.

Counts Gemini input tokens and, with --call, sends one real request
in English, Russian and Kazakh.

Run:
    python3 part2_measure.py
    python3 part2_measure.py --call
    python3 part2_measure.py --model gemini-2.5-flash --call
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from texts import CORPUS, LANGUAGES


load_dotenv()

OUTPUT_PATH = Path(__file__).with_name("measurements.json")
DEFAULT_MODEL = "gemini-3.6-flash"
MAX_TOKENS = 2048
API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def get_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print("GEMINI_API_KEY is not set in .env", file=sys.stderr)
        raise SystemExit(1)
    return key


def api_post(path: str, body: dict, api_key: str) -> dict:
    url = f"{API_BASE}/{path}"
    request = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        print(f"API error {exc.code}: {message}", file=sys.stderr)
        raise
    except URLError as exc:
        print(f"Could not reach Gemini API: {exc}", file=sys.stderr)
        raise


def count_tokens(api_key: str, model_id: str, text: str) -> int:
    result = api_post(
        f"models/{model_id}:countTokens",
        {"contents": [{"role": "user", "parts": [{"text": text}]}]},
        api_key,
    )
    return int(result.get("totalTokens", 0))


def count_request_tokens(api_key: str, model_id: str, lang: str) -> int:
    system_prompt = CORPUS["system_prompt"][lang]
    complaint = CORPUS["complaint"][lang]

    result = api_post(
        f"models/{model_id}:countTokens",
        {
            "generateContentRequest": {
                "model": f"models/{model_id}",
                "systemInstruction": {
                    "parts": [{"text": system_prompt}]
                },
                "contents": [
                    {"role": "user", "parts": [{"text": complaint}]}
                ],
            }
        },
        api_key,
    )
    return int(result.get("totalTokens", 0))


def one_real_request(
    api_key: str, model_id: str, lang: str
) -> Optional[Dict[str, int]]:
    system_prompt = CORPUS["system_prompt"][lang]
    complaint = CORPUS["complaint"][lang]

    response = api_post(
        f"models/{model_id}:generateContent",
        {
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {"role": "user", "parts": [{"text": complaint}]}
            ],
            "generationConfig": {
                "maxOutputTokens": MAX_TOKENS,
            },
        },
        api_key,
    )

    candidates = response.get("candidates", [])
    if not candidates:
        print("  model returned no candidates")
        return None

    candidate = candidates[0]
    finish_reason = candidate.get("finishReason", "UNKNOWN")
    print(f"  finish_reason: {finish_reason}")

    parts = candidate.get("content", {}).get("parts", [])
    for part in parts:
        if "text" in part:
            print("  --- answer ---")
            print("  " + part["text"].replace("\n", "\n  "))

    usage = response.get("usageMetadata", {})
    input_tokens = int(usage.get("promptTokenCount", 0))
    output_tokens = int(usage.get("candidatesTokenCount", 0))
    thinking_tokens = int(usage.get("thoughtsTokenCount", 0))

    # For a thinking model, thoughts are also output-side tokens.
    billed_output_tokens = output_tokens + thinking_tokens

    print(
        f"  billed: {input_tokens} in, "
        f"{billed_output_tokens} out"
        + (f" ({thinking_tokens} thinking)" if thinking_tokens else "")
    )

    if finish_reason == "MAX_TOKENS":
        print(f"  NOTE: answer was cut off at maxOutputTokens={MAX_TOKENS}.")

    return {
        "input_tokens": input_tokens,
        "output_tokens": billed_output_tokens,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure Gemini token counts and request cost."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--call",
        action="store_true",
        help="also answer the complaint in each language",
    )
    args = parser.parse_args()

    api_key = get_key()
    model_id = args.model

    counts: Dict[str, Dict[str, int]] = {}
    print(f"counting tokens on {model_id} (free, no model run)")

    try:
        for item_id, versions in CORPUS.items():
            counts[item_id] = {
                lang: count_tokens(api_key, model_id, versions[lang])
                for lang in LANGUAGES
            }
            row = "  ".join(
                f"{lang}={counts[item_id][lang]}" for lang in LANGUAGES
            )
            print(f"  {item_id:<14} {row}")

        request_tokens: Dict[str, int] = {
            lang: count_request_tokens(api_key, model_id, lang)
            for lang in LANGUAGES
        }
        row = "  ".join(
            f"{lang}={request_tokens[lang]}" for lang in LANGUAGES
        )
        print(
            f"  {'request':<14} {row} "
            "(system + complaint, one call)"
        )
    except (HTTPError, URLError):
        return 1

    billed: Dict[str, Dict[str, int]] = {}

    if args.call:
        print(
            f"\nanswering the same complaint on {model_id}, "
            "in each language:"
        )
        for lang in LANGUAGES:
            print(f"\n[{lang}]")
            try:
                result = one_real_request(api_key, model_id, lang)
            except (HTTPError, URLError):
                return 1
            if result is not None:
                billed[lang] = result

    payload = {
        "model": model_id,
        "model_id": model_id,
        "token_counts": counts,
        "request_tokens": request_tokens,
        "one_request_billed": billed or None,
    }

    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nwrote {OUTPUT_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
