"""Part 3 -- Gemini cost calculation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

from texts import LANGUAGES

DEFAULT_MEASUREMENTS = Path(__file__).with_name("measurements.json")

# Gemini 3.6 Flash Standard paid pricing, USD per 1M tokens.
INPUT_PRICE = 0.75
OUTPUT_PRICE = 3.75
MODEL_NAME = "gemini-3.6-flash"


def load_measurements(path: Path) -> Dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(f"{path.name} not found. Run part2_measure_gemini.py first.")
    except json.JSONDecodeError as exc:
        sys.exit(f"{path.name} is not valid JSON: {exc}")


def request_input_tokens(data: Dict[str, object], lang: str) -> int:
    request_tokens = data.get("request_tokens")
    if not request_tokens or lang not in request_tokens:
        sys.exit(
            f"measurements.json has no request_tokens for {lang!r}. "
            "Run part2_measure_gemini.py again."
        )
    return int(request_tokens[lang])


def resolve_output_tokens(
    billed: Optional[Dict[str, Dict[str, int]]],
    override: Optional[int],
) -> Tuple[Dict[str, int], str]:
    if override is not None:
        return {lang: override for lang in LANGUAGES}, "fixed by --output-tokens"

    if billed and all(lang in billed for lang in LANGUAGES):
        return (
            {lang: int(billed[lang]["output_tokens"]) for lang in LANGUAGES},
            "measured in Part 2, per language",
        )

    return (
        {lang: 300 for lang in LANGUAGES},
        "ASSUMED -- Part 2 ran without --call",
    )


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1_000_000 * INPUT_PRICE
        + output_tokens / 1_000_000 * OUTPUT_PRICE
    )


def header() -> str:
    return f"{'':<14}" + "".join(f"{lang.upper():>12}" for lang in LANGUAGES)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calculate Gemini 3.6 Flash request costs."
    )
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--requests-per-day", type=int, default=2000)
    parser.add_argument("--output-tokens", type=int, default=None)
    args = parser.parse_args()

    data = load_measurements(args.measurements)
    outputs, provenance = resolve_output_tokens(
        data.get("one_request_billed"), args.output_tokens
    )
    inputs = {lang: request_input_tokens(data, lang) for lang in LANGUAGES}

    print("prices: Gemini 3.6 Flash Standard")
    print(f"input:  ${INPUT_PRICE:.2f} / 1M tokens")
    print(f"output: ${OUTPUT_PRICE:.2f} / 1M tokens")
    print(f"tokens counted on {data['model_id']}")
    print(f"answer length: {provenance}\n")

    print("ONE SUPPORT REQUEST -- tokens, and cost in US cents")
    print("-" * 72)
    print(header())
    print(f"{'input tokens':<14}" + "".join(f"{inputs[l]:>12}" for l in LANGUAGES))
    print(f"{'output tokens':<14}" + "".join(f"{outputs[l]:>12}" for l in LANGUAGES))

    cents = [cost_usd(inputs[l], outputs[l]) * 100 for l in LANGUAGES]
    print(f"{MODEL_NAME:<14}" + "".join(f"{c:>12.4f}" for c in cents))

    per_year = args.requests_per_day * 365
    print(f"\nAT {args.requests_per_day:,} REQUESTS/DAY -- US dollars per year")
    print("-" * 72)
    print(header())

    yearly = [cost_usd(inputs[l], outputs[l]) * per_year for l in LANGUAGES]
    print(f"{MODEL_NAME:<14}" + "".join(f"{y:>12,.2f}" for y in yearly))

    print("\nTWO RATIOS THAT ARE NOT THE SAME NUMBER")
    print("-" * 72)
    print(header())
    print(f"{'input only':<14}" + "".join(
        f"{inputs[l] / inputs['en']:>11.2f}x" for l in LANGUAGES
    ))

    en_bill = cost_usd(inputs["en"], outputs["en"])
    print(f"{'total bill':<14}" + "".join(
        f"{cost_usd(inputs[l], outputs[l]) / en_bill:>11.2f}x"
        for l in LANGUAGES
    ))

    print("\nTHE NUMBER TO REMEMBER")
    print("-" * 72)
    en_year = en_bill * per_year
    for lang in ("ru", "kk"):
        lang_year = cost_usd(inputs[lang], outputs[lang]) * per_year
        print(
            f"{lang.upper()} instead of EN, same work, same volume: "
            f"${lang_year - en_year:,.2f}/year more "
            f"({lang_year / en_year:.2f}x)."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
