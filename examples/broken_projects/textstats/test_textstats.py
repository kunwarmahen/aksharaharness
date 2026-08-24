"""Contract tests for textstats. These are CORRECT -- the implementation
is what's broken. Do not change this file to make tests pass.
"""

import unittest

from textstats import (
    average_word_length,
    is_palindrome,
    top_words,
    truncate,
    word_count,
)


class WordCount(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(word_count("one two three"), 3)

    def test_runs_of_whitespace(self):
        self.assertEqual(word_count("a  b   c"), 3)

    def test_empty(self):
        self.assertEqual(word_count(""), 0)


class AverageWordLength(unittest.TestCase):
    def test_fractional_mean(self):
        self.assertAlmostEqual(average_word_length("hi there"), 3.5)

    def test_single_word(self):
        self.assertAlmostEqual(average_word_length("abcd"), 4.0)

    def test_empty(self):
        self.assertEqual(average_word_length(""), 0.0)


class IsPalindrome(unittest.TestCase):
    def test_simple(self):
        self.assertTrue(is_palindrome("racecar"))

    def test_ignores_case_and_punctuation(self):
        self.assertTrue(is_palindrome("A man, a plan, a canal: Panama!"))

    def test_not_a_palindrome(self):
        self.assertFalse(is_palindrome("hello"))


class TopWords(unittest.TestCase):
    def test_order_and_alphabetical_ties(self):
        # b: 3 occurrences; a and c: 2 each (tie -> alphabetical)
        text = "b b b a c a c"
        self.assertEqual(top_words(text, 3), ["b", "a", "c"])

    def test_n_limits_output(self):
        self.assertEqual(len(top_words("x y z", 2)), 2)


class Truncate(unittest.TestCase):
    def test_passthrough(self):
        self.assertEqual(truncate("short", 10), "short")

    def test_exact_width_passes_through(self):
        self.assertEqual(truncate("exactlyten", 10), "exactlyten")

    def test_never_exceeds_width(self):
        out = truncate("this string is far too long", 12)
        self.assertLessEqual(len(out), 12)
        self.assertTrue(out.endswith("..."))
        self.assertEqual(out, "this stri...")


if __name__ == "__main__":
    unittest.main()
