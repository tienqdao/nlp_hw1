import sys

# =============================================================================
# IMPORTS & SETUP
# =============================================================================
print("Loading test modules...")

# We wrap imports in try-except blocks to provide helpful error messages
# if a specific implementation file is missing.
try:
    from part1.hw1_part1 import replace_mentions, replace_urls, replace_hashtags, preprocess_part1
    HAS_PART1 = True
except ImportError:
    print("[!] Warning: Could not import 'hw1_part1.py'. Part 1 tests will be skipped.")
    HAS_PART1 = False

try:
    from part2.space_base import SpaceTokenizer
    HAS_SPACE = True
except ImportError:
    print("[!] Warning: Could not import 'space_base.py'. Space Tokenizer tests will be skipped.")
    HAS_SPACE = False

try:
    from part2.bpe import BPE_Tokenizer, get_gpt2_splits
    HAS_BPE = True
except ImportError:
    print("[!] Warning: Could not import 'bpe.py'. BPE tests will be skipped.")
    HAS_BPE = False

try:
    from sentencePiece_bpe import SentencePieceBPE
    HAS_SP = True
except ImportError:
    print("[!] Warning: Could not import 'sentencePiece_bpe.py'. SentencePiece tests will be skipped.")
    HAS_SP = False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def assert_test(name, condition):
    """
    Standardized assertion helper.
    Returns 1 if passed, 0 if failed.
    """
    print(f"Test: {name} ... ", end="")
    try:
        if condition:
            print("PASSED")
            return 1
        else:
            print("FAILED")
            return 0
    except Exception as e:
        print(f"FAILED (Exception: {e})")
        return 0

def assert_equal(name, got, expected):
    """
    Specific helper for Part 1 equality checks with verbose error printing.
    """
    if got == expected:
        print(f"[PASS] {name}")
        return 1
    else:
        print(f"[FAIL] {name}")
        print("  got     :", repr(got))
        print("  expected:", repr(expected))
        return 0

# =============================================================================
# PART 1: REGEX PREPROCESSING
# =============================================================================
def run_part1():
    if not HAS_PART1: return 0, 0
    
    print_header("PART 1: Regex Preprocessing")
    passed = 0
    total = 0

    # Mentions
    total += 1; passed += assert_equal("Mention: simple handle", replace_mentions("hi @switchfoot!"), "hi [MENTION]!")
    total += 1; passed += assert_equal("Mention: handle with digits", replace_mentions("thanks @Alliana07 for the info"), "thanks [MENTION] for the info")
    total += 1; passed += assert_equal("Mention: handle with underscore", replace_mentions("cc @angry_barista please review"), "cc [MENTION] please review")
    total += 1; passed += assert_equal("Mention: multiword example", replace_mentions("met @angry_barista today"), "met [MENTION] today")
    total += 1; passed += assert_equal("Mention: ignore email", replace_mentions("contact me at bob@example.com please"), "contact me at bob@example.com please")

    # URLs
    total += 1; passed += assert_equal("URL: http scheme", replace_urls("pic http://twitpic.com/2y1zl wow"), "pic [URL] wow")
    total += 1; passed += assert_equal("URL: https scheme with query", replace_urls("shop https://www.mycomicshop.com/search?TID=395031 now"), "shop [URL] now")
    total += 1; passed += assert_equal("URL: www prefix", replace_urls("bookmark www.diigo.com/~tautao please"), "bookmark [URL] please")
    total += 1; passed += assert_equal("URL: strips trailing punctuation", replace_urls("go to https://example.com/test). ok"), "go to [URL]). ok")

    # Hashtags
    total += 1; passed += assert_equal("Hashtag: simple", replace_hashtags("that was #fb"), "that was [HASHTAG]")
    total += 1; passed += assert_equal("Hashtag: camelcase", replace_hashtags("new release #AutomationAtaCost today"), "new release [HASHTAG] today")
    total += 1; passed += assert_equal("Hashtag: underscore + digits", replace_hashtags("topic #nlp_101 is fun"), "topic [HASHTAG] is fun")
    total += 1; passed += assert_equal("Hashtag: do not replace inside words", replace_hashtags("abc#def should not change"), "abc#def should not change")

    # Pipeline
    total += 1; passed += assert_equal("Pipeline: URL then mention then hashtag", preprocess_part1("hey @Kenichan check https://t.co/xyz #fb"), "hey [MENTION] check [URL] [HASHTAG]")

    return passed, total


# =============================================================================
# PART 2A: SPACE TOKENIZER
# =============================================================================
def run_part2a():
    if not HAS_SPACE: return 0, 0
    
    print_header("PART 2A: Space Tokenizer")
    passed = 0
    total = 0
    tokenizer = SpaceTokenizer()

    # Basic Functionality
    total += 1; passed += assert_test("Basic whitespace splitting", tokenizer.encode("Hello world") == ["Hello", "world"])  # "Hello world" -> ["Hello", "world"]
    total += 1; passed += assert_test("Basic decoding", tokenizer.decode(["Hello", "world"]) == "Hello world") # ["Hello", "world"] -> "Hello world"
    
    text = "The quick brown fox"
    total += 1; passed += assert_test("Round trip consistency", tokenizer.decode(tokenizer.encode(text)) == text)

    # Edge Cases:
    # 1. "a   b" should usually become ["a", "b"]
    # Decoding ["a", "b"] becomes "a b".
    total += 1; passed += assert_test("Multiple spaces", tokenizer.encode("a   b") == ["a", "b"])
    # 2. "Line1\nLine2\tTabbed" -> ["Line1", "Line2", "Tabbed"]
    total += 1; passed += assert_test("Newlines and tabs", tokenizer.encode("Line1\nLine2\tTabbed") == ["Line1", "Line2", "Tabbed"])
    total += 1; passed += assert_test("Empty string", tokenizer.encode("") == [])
    total += 1; passed += assert_test("Only whitespace", tokenizer.encode("   \n  \t ") == [])

    # Punctuation
    # A purely whitespace-based tokenizer does NOT split punctuation.
    total += 1; passed += assert_test("Punctuation stickiness", tokenizer.encode("Hello, world!") == ["Hello,", "world!"])

    return passed, total


# =============================================================================
# PART 2B: BPE TOKENIZER
# =============================================================================
def run_part2b():
    if not HAS_BPE: return 0, 0

    print_header("PART 2B: BPE Tokenizer (GPT-2 Style)")
    passed = 0
    total = 0

    # Group 1: Pre-tokenization Logic
    # 1. Expecting ["Hello", " world"] -> Space attaches to the next word
    total += 1; passed += assert_test("Basic word splitting (leading spaces)", get_gpt2_splits("Hello world") == ["Hello", " world"])
    
    # 2. Punctuation should be separate but respect spacing
    # "Hello", ",", " world", "!"
    splits = get_gpt2_splits("Hello, world!")
    total += 1; passed += assert_test("Splitting punctuation from words", " world" in splits and "!" in splits)
    
    # 3. GPT-2 regex usually splits "We're" -> "We", "'re"
    splits = get_gpt2_splits("We're going")
    total += 1; passed += assert_test("Handling English contractions ('re)", "'re" in splits)

    # Group 2: Basic Training & Encoding
    # 1. Ideally, encode -> decode should return the EXACT original string
    text = "The quick brown fox jumps."
    tokenizer = BPE_Tokenizer()
    tokenizer.train(text, num_merges=10)
    decoded = tokenizer.decode(tokenizer.encode(text))
    total += 1; passed += assert_test("Round-trip consistency", decoded == text)

    # 2. If we ask for 5 merges, vocab size should increase by exactly 5
    tokenizer = BPE_Tokenizer()
    initial_size = tokenizer.vocab_size
    tokenizer.train("ababababab", num_merges=5)
    total += 1; passed += assert_test("Vocabulary size growth", tokenizer.vocab_size == initial_size + 5)

    # Group 3: Byte-Level & Unicode
    # 1. This tests if the code crashes on non-ascii or handles them as bytes
    text = "I love 🍎 pie"
    tokenizer = BPE_Tokenizer()
    tokenizer.train(text, num_merges=5)
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    total += 1; passed += assert_test("Handling Unicode Emojis (🍎)", decoded == text and "🍎" in decoded)

    # 2. Japanese characters (multi-byte in UTF-8)
    text = "こんにちは"
    tokenizer = BPE_Tokenizer()
    tokenizer.train(text, num_merges=10)
    decoded = tokenizer.decode(tokenizer.encode(text))
    total += 1; passed += assert_test("Handling non-Latin scripts (Japanese)", decoded == text)

    # Group 4: Robustness
    tokenizer = BPE_Tokenizer()
    total += 1; passed += assert_test("Handling empty string", tokenizer.encode("") == [])

    # 1. Train on "apple", test on "apricot"
    # It should fall back to bytes for the unseen parts, not crash
    tokenizer = BPE_Tokenizer()
    tokenizer.train("apple", num_merges=2)
    decoded = tokenizer.decode(tokenizer.encode("apricot"))
    total += 1; passed += assert_test("Generalization to unseen words", decoded == "apricot")

    return passed, total


# =============================================================================
# PART 2c: SENTENCEPIECE BPE
# =============================================================================
def run_part2c():
    if not HAS_SP: return 0, 0

    print_header("PART 3: SentencePiece BPE")
    passed = 0
    total = 0

    # Core SentencePiece Behavior
    # 1. CRITICAL TEST: This distinguishes SP from GPT-2
    # Train on "the" repeatedly so it learns the token "the".
    # Then test on "ether".
    # GPT-2 would NOT use the "the" token because it splits "ether" into one word.
    # SentencePiece SHOULD use "the" inside "ether".
    text = "the " * 50
    sp = SentencePieceBPE()
    sp.train(text, num_merges=5)
    encoded = sp.encode("ether")
    decoded_tokens = [sp.id_to_bytes[i] for i in encoded]
    total += 1; passed += assert_test("Cross-boundary merging (finding 'the' inside 'ether')", b'the' in decoded_tokens)

    # 2. SentencePiece treats space (ASCII 32) as just another character.
    # It should be able to merge a letter with a space.
    # Train on "a a a a " -> frequent pair should be "a " (a + space)
    text = "a a a a " * 20
    sp = SentencePieceBPE()
    sp.train(text, num_merges=5)
    learned_tokens = sp.id_to_bytes.values()
    total += 1; passed += assert_test("Treating space as mergeable byte (b'a ')", b'a ' in learned_tokens)

    # Basic Mechanics
    # 1. Round trip: encode -> decode should return original string exactly
    text = "Hello world! This is a test."
    sp = SentencePieceBPE()
    sp.train(text, num_merges=20)
    decoded = sp.decode(sp.encode(text))
    total += 1; passed += assert_test("Round-trip consistency", decoded == text)

    # 2. Should add exactly 2 new tokens
    sp = SentencePieceBPE()
    initial = sp.vocab_size
    sp.train("abc", num_merges=2)
    total += 1; passed += assert_test("Vocabulary growth", sp.vocab_size == initial + 2)

    # Robustness
    # 1. Emojis are 4 bytes. SP should handle them as a stream of bytes.
    # Ideally, it learned the whole emoji as one token, or at least partials.
    # The key is that it decodes back correctly.
    text = "🙂" * 10
    sp = SentencePieceBPE()
    sp.train(text, num_merges=3)
    decoded = sp.decode(sp.encode("🙂"))
    total += 1; passed += assert_test("Emoji processing (Byte streaming)", decoded == "🙂")

    sp = SentencePieceBPE()
    total += 1; passed += assert_test("Empty string input", sp.encode("") == [])

     # 2. Ensure the split('\n') logic in train() works and doesn't crash
    text = "Line 1\nLine 2\nLine 3"
    sp = SentencePieceBPE()
    sp.train(text, num_merges=5)
    decoded = sp.decode(sp.encode("Line 2"))
    total += 1; passed += assert_test("Multiline string training", decoded == "Line 2")

    return passed, total


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    p1, t1 = run_part1()
    p2, t2 = run_part2a()
    p3, t3 = run_part2b()
    p4, t4 = run_part2c()

    total_passed = p1 + p2 + p3 + p4
    total_tests = t1 + t2 + t3 + t4

    print("\n" + "="*60)
    print("FINAL GRAND TOTAL")
    print("="*60)
    print(f"Part 1 (Regex)       : {p1}/{t1} passed")
    print(f"Part 2A (Space)      : {p2}/{t2} passed")
    print(f"Part 2B (BPE)        : {p3}/{t3} passed")
    print(f"Part 3 (SentencePiece): {p4}/{t4} passed")
    print("-" * 30)
    print(f"OVERALL              : {total_passed}/{total_tests} passed")

    # Exit with code 0 if all tests passed, 1 otherwise
    if total_passed == total_tests and total_tests > 0:
        print("\nSUCCESS: All tests passed!")
        sys.exit(0)
    else:
        print("\nFAILURE: Some tests failed or no tests were run.")
        sys.exit(1)