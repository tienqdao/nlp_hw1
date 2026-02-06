from space_base import SpaceTokenizer

class SpaceTokenizerTestSuite:
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0

    def run(self, name, assertion):
        self.total += 1
        print(f"Test {self.total}: {name} ... \n", end="")
        try:
            if assertion():
                print("PASSED")
                self.passed += 1
            else:
                print("FAILED")
                self.failed += 1
        except Exception as e:
            print(f"FAILED (Error: {e})")
            self.failed += 1

    def summary(self):
        print("\n" + "="*40)
        print(f"SUMMARY: {self.passed}/{self.total} passed.")

def run_tests():
    t = SpaceTokenizerTestSuite()
    tokenizer = SpaceTokenizer()

    print("\n--- Basic Functionality ---")
    
    def test_simple_split():
        # "Hello world" -> ["Hello", "world"]
        return tokenizer.encode("Hello world") == ["Hello", "world"]
    t.run("Basic whitespace splitting", test_simple_split)

    def test_decode_logic():
        # ["Hello", "world"] -> "Hello world"
        tokens = ["Hello", "world"]
        return tokenizer.decode(tokens) == "Hello world"
    t.run("Basic decoding (join with space)", test_decode_logic)

    def test_round_trip():
        text = "The quick brown fox"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        return decoded == text
    t.run("Round trip consistency (Encode -> Decode)", test_round_trip)

    print("\n--- Edge Cases & Whitespace Handling ---")

    def test_multiple_spaces():
        # "a   b" should usually become ["a", "b"]
        # Decoding ["a", "b"] becomes "a b".
        text = "a   b"
        encoded = tokenizer.encode(text)
        return encoded == ["a", "b"]
    t.run("Handling multiple spaces (collapse to single tokens)", test_multiple_spaces)

    def test_newlines_tabs():
        # "Line1\nLine2\tTabbed" -> ["Line1", "Line2", "Tabbed"]
        text = "Line1\nLine2\tTabbed"
        encoded = tokenizer.encode(text)
        return encoded == ["Line1", "Line2", "Tabbed"]
    t.run("Handling newlines and tabs", test_newlines_tabs)

    def test_empty_string():
        return tokenizer.encode("") == []
    t.run("Empty string input", test_empty_string)

    def test_only_spaces():
        return tokenizer.encode("   \n  \t ") == []
    t.run("String with only whitespace", test_only_spaces)

    print("\n--- Punctuation Attachment ---")
    
    def test_punctuation_stickiness():
        # A purely whitespace-based tokenizer does NOT split punctuation.
        text = "Hello, world!"
        encoded = tokenizer.encode(text)
        return encoded == ["Hello,", "world!"]
    t.run("Punctuation remains attached to words", test_punctuation_stickiness)

    t.summary()

if __name__ == "__main__":
    run_tests()