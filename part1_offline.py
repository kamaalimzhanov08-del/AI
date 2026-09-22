"""Part 1 -- what you can measure without an API key, and what you cannot.

Counts characters, UTF-8 bytes and whitespace-separated words for the same
three texts in English, Russian and Kazakh, and prints the ratios against
English.

None of these is a token count. The point of this script is to make you
commit to a prediction before the measurement in Part 2 contradicts it.

Run:
    python3 part1_offline.py
"""

from __future__ import annotations

from typing import Dict, List

from texts import CORPUS, LANGUAGES


def measure(text: str) -> Dict[str, int]:
    """Return the three cheap, offline size measures of a string.

    Args:
        text: The string to measure.

    Returns:
        A mapping with ``chars`` (Unicode code points), ``bytes`` (length of
        the UTF-8 encoding) and ``words`` (whitespace-separated runs).
    """
    return {
        "chars": len(text),
        "bytes": len(text.encode("utf-8")),
        "words": len(text.split()),
    }


def _row(label: str, cells: List[str]) -> str:
    """Format one fixed-width table row."""
    return f"{label:<14}" + "".join(f"{cell:>13}" for cell in cells)


def report(item_id: str) -> None:
    """Print the offline measurements for one corpus item, all languages.

    Args:
        item_id: A key of :data:`texts.CORPUS`, e.g. ``"complaint"``.
    """
    versions = CORPUS[item_id]
    stats = {lang: measure(versions[lang]) for lang in LANGUAGES}
    base = stats["en"]

    print(f"\n{item_id.upper()}")
    print("-" * 66)
    print(_row("", [lang.upper() for lang in LANGUAGES]))
    for measure_name in ("chars", "bytes", "words"):
        print(_row(measure_name, [str(stats[lang][measure_name]) for lang in LANGUAGES]))
    print(_row(
        "bytes/EN",
        [f"{stats[lang]['bytes'] / base['bytes']:.2f}x" for lang in LANGUAGES],
    ))
    print(_row(
        "bytes/char",
        [f"{stats[lang]['bytes'] / stats[lang]['chars']:.2f}" for lang in LANGUAGES],
    ))


def main() -> None:
    """Print every corpus item and the questions to answer before Part 2."""
    print("PART 1 -- offline size measures. No API key, no cost, no network.")
    for item_id in CORPUS:
        report(item_id)

    print("\n" + "=" * 66)
    print("WRITE THESE DOWN BEFORE PART 2")
    print("=" * 66)
    print(
        "1. A Latin letter is 1 byte in UTF-8; Cyrillic is 2. Kazakh letters\n"
        "   outside the Russian alphabet (ә, ғ, қ, ң, ө, ұ, ү, һ, і) are also 2.\n"
        "   You can see that in the bytes/char row above.\n"
        "2. Predict, for the COMPLAINT item: how many TOKENS will the Russian\n"
        "   version cost, as a multiple of the English one?   RU / EN = ____\n"
        "3. Same question for Kazakh.                          KK / EN = ____\n"
        "4. On what did you base the prediction -- characters, bytes, or words?\n"
        "5. Which one, if any, would a tokenizer have any reason to follow?"
    )


if __name__ == "__main__":
    main()
