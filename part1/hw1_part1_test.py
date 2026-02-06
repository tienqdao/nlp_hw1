# hw1_test.py
# Unit tests for Spring 2026 NLP HW1 - Parts 1 (regex)
#
# Run: python hw1_test.py
# Should print each test description and pass/fail + final summary.

from hw1_part1 import (
    replace_mentions, replace_urls, replace_hashtags, preprocess_part1
)

def assert_equal(name, got, expected):
    if got == expected:
        print(f"[PASS] {name}")
        return True
    else:
        print(f"[FAIL] {name}")
        print("  got     :", repr(got))
        print("  expected:", repr(expected))
        return False

def run_tests():
    total = 0
    passed = 0

    print("=== HW1 Part 1 Tests: Regex Preprocessing ===")

    # -------------------- Mentions --------------------
    total += 1
    passed += assert_equal(
        "Mention: simple handle",
        replace_mentions("hi @switchfoot!"),
        "hi [MENTION]!"
    )

    total += 1
    passed += assert_equal(
        "Mention: handle with digits",
        replace_mentions("thanks @Alliana07 for the info"),
        "thanks [MENTION] for the info"
    )

    total += 1
    passed += assert_equal(
        "Mention: handle with underscore",
        replace_mentions("cc @angry_barista please review"),
        "cc [MENTION] please review"
    )

    total += 1
    passed += assert_equal(
        "Mention: multiword example should only replace @angry",
        replace_mentions("met @angry_barista today"),
        "met [MENTION] today"
    )


    total += 1
    passed += assert_equal(
        "Mention: do not replace email-like '@' in emails",
        replace_mentions("contact me at bob@example.com please"),
        "contact me at bob@example.com please"
    )

    # -------------------- URLs --------------------
    total += 1
    passed += assert_equal(
        "URL: http scheme",
        replace_urls("pic http://twitpic.com/2y1zl wow"),
        "pic [URL] wow"
    )

    total += 1
    passed += assert_equal(
        "URL: https scheme with query",
        replace_urls("shop https://www.mycomicshop.com/search?TID=395031 now"),
        "shop [URL] now"
    )

    total += 1
    passed += assert_equal(
        "URL: www prefix",
        replace_urls("bookmark www.diigo.com/~tautao please"),
        "bookmark [URL] please"
    )

    total += 1
    passed += assert_equal(
        "URL: strips trailing punctuation reasonably",
        replace_urls("go to https://example.com/test). ok"),
        "go to [URL]). ok"
    )

    # -------------------- Hashtags --------------------
    total += 1
    passed += assert_equal(
        "Hashtag: simple",
        replace_hashtags("that was #fb"),
        "that was [HASHTAG]"
    )

    total += 1
    passed += assert_equal(
        "Hashtag: camelcase",
        replace_hashtags("new release #AutomationAtaCost today"),
        "new release [HASHTAG] today"
    )

    total += 1
    passed += assert_equal(
        "Hashtag: underscore + digits",
        replace_hashtags("topic #nlp_101 is fun"),
        "topic [HASHTAG] is fun"
    )

    total += 1
    passed += assert_equal(
        "Hashtag: do not replace inside words",
        replace_hashtags("abc#def should not change"),
        "abc#def should not change"
    )

    # -------------------- Pipeline --------------------
    total += 1
    passed += assert_equal(
        "Pipeline: URL then mention then hashtag",
        preprocess_part1("hey @Kenichan check https://t.co/xyz #fb"),
        "hey [MENTION] check [URL] [HASHTAG]"
    )

    print("\n=== Summary ===")
    print(f"Passed {passed}/{total} tests.")
    return passed == total

if __name__ == "__main__":
    ok = run_tests()
    raise SystemExit(0 if ok else 1)
