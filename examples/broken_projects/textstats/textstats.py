"""Text statistics utilities.

Every docstring here is the contract. The test suite pins it down.
"""


def word_count(text: str) -> int:
    """Number of whitespace-separated words."""
    return len(text.split(" "))


def average_word_length(text: str) -> float:
    """Mean word length in characters; 0.0 for empty input."""
    words = text.split()
    if not words:
        return 0.0
    return sum(len(word) for word in words) // len(words)


def is_palindrome(text: str) -> bool:
    """True if text reads the same backward, ignoring case, spaces, and
    punctuation ("A man, a plan, a canal: Panama!" -> True)."""
    cleaned = "".join(ch for ch in text if ch.isalnum())
    return cleaned == cleaned[::-1]


def top_words(text: str, n: int = 3) -> list[str]:
    """The n most frequent lowercase words, most common first;
    ties broken alphabetically."""
    from collections import Counter

    counts = Counter(text.lower().split())
    ranked = sorted(counts.items(), key=lambda kv: kv[1])
    return [word for word, _count in ranked[:n]]


def truncate(text: str, width: int) -> str:
    """Shorten to at most `width` characters, ending in "..." when cut.
    Strings already within the width pass through unchanged."""
    if len(text) <= width:
        return text
    return text[:width] + "..."
