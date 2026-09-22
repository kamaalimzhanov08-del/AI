"""Part 0 -- the OpenAI pair, offline, no key.

Anthropic publishes no tokenizer for offline use, so :mod:`part2_measure`
cannot show a table across several vocabularies -- only Claude's own. This
script fills that gap with the two public tokenizers that bracket the whole
lecture-02 spread: ``o200k_base`` (kindest to Kazakh of the six tokenizers
compared in class) and ``cl100k_base`` (harshest). It does not attempt the
other four (Llama 3, Qwen2.5, Gemma, DeepSeek-V3) -- those live on
HuggingFace behind gated licenses or multi-hundred-megabyte downloads, which
does not work live in a classroom.

Run:
    python3 part0_tokenizers.py
"""

from __future__ import annotations

from typing import Dict, List

import tiktoken

from texts import CORPUS, LANGUAGES

#: The two OpenAI tokenizers that bracket the six compared in lecture 02:
#: o200k_base is the kindest to Kazakh there, cl100k_base the harshest.
TOKENIZER_NAMES = ("o200k_base", "cl100k_base")


def _row(label: str, cells: List[str]) -> str:
    """Format one fixed-width table row."""
    return f"{label:<14}" + "".join(f"{cell:>13}" for cell in cells)


def report(item_id: str, encodings: Dict[str, "tiktoken.Encoding"]) -> None:
    """Print token counts and EN-relative ratios for one corpus item.

    Args:
        item_id: A key of :data:`texts.CORPUS`, e.g. ``"complaint"``.
        encodings: Mapping of tokenizer name to a loaded ``tiktoken`` encoding.
    """
    versions = CORPUS[item_id]
    print(f"\n{item_id.upper()}")
    print("-" * 66)
    print(_row("", [lang.upper() for lang in LANGUAGES]))
    for name, enc in encodings.items():
        counts = {lang: len(enc.encode(versions[lang])) for lang in LANGUAGES}
        print(_row(name, [str(counts[lang]) for lang in LANGUAGES]))
        print(_row(
            "  ÷ EN",
            [f"{counts[lang] / counts['en']:.2f}x" for lang in LANGUAGES],
        ))


def main() -> None:
    """Print token counts and ratios for the whole corpus, both tokenizers."""
    print("PART 0 -- OpenAI tokenizer pair. No API key, no cost, no network.")
    encodings = {name: tiktoken.get_encoding(name) for name in TOKENIZER_NAMES}
    for item_id in CORPUS:
        report(item_id, encodings)

    print(
        "\nThese two bracket the six-tokenizer spread shown in lecture 02: "
        "o200k_base is the kindest to Kazakh there, cl100k_base the harshest. "
        "Llama 3, Qwen2.5, Gemma and DeepSeek-V3 are not reproduced here -- "
        "gated weights or large downloads, not a classroom fit."
    )


if __name__ == "__main__":
    main()
