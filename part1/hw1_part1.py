# part1_regex.py
# Spring 2026 NLP HW1 - Part 1: Regular Expressions
# Implements:
#  1) replace @username handles -> [MENTION]
#  2) replace URLs -> [URL]
#  3) replace hashtags -> [HASHTAG]
#
# Note: These regexes are intentionally practical (tweets + wiki-ish text),
# not perfect for every edge case on the internet.

import re

# Compiled patterns
# Mention: allow @ followed by:
# - letters/digits/underscore
# - allow internal spaces (e.g., "@angry barista" from the prompt)
# - stop at punctuation/newline
# Examples in prompt: @switchfoot, @Kenichan, @angry barista, @Alliana07
MENTION_RE = re.compile(r'(?<!\w)@[A-Za-z0-9_]+')

# URL: cover common forms:
# - http://... or https://...
# - www....
# Stops at whitespace or common closing punctuation.
URL_RE = re.compile(
    r'(?i)\b(?:https?://|www\.)'          # scheme or www
    r'[^\s<>()\[\]{}"\']+'                # run of non-space, not brackets/quotes
)

# Hashtag: # followed by letters/digits/underscore (common usage)
# Examples: #fb, #therapyfail, #AutomationAtaCost
HASHTAG_RE = re.compile(
    r'(?<!\w)#([A-Za-z0-9_]+)'
)


# Regex Functions
def replace_mentions(text: str) -> str:
    """Replace @handles with [MENTION]."""
    return MENTION_RE.sub("[MENTION]", text)


def replace_urls(text: str) -> str:
    """Replace URLs with [URL]."""
    return URL_RE.sub("[URL]", text)


def replace_hashtags(text: str) -> str:
    """Replace hashtags with [HASHTAG]."""
    return HASHTAG_RE.sub("[HASHTAG]", text)


def preprocess_part1(text: str) -> str:
    """
    Convenience pipeline for Part 1.
    Recommended order: URLs first (so @ in query params doesn't confuse mention),
    then mentions, then hashtags.
    """
    text = replace_urls(text)
    text = replace_mentions(text)
    text = replace_hashtags(text)
    return text
