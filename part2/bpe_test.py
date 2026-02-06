from bpe import BPE_Tokenizer, get_gpt2_splits

class BPE_Test_Suite:
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

    def run_check(self, description, success_condition):
        self.total_tests += 1
        print(f"Test {self.total_tests}: {description} ... ", end="")
        
        try:
            if success_condition():
                print("PASSED")
                self.passed_tests += 1
            else:
                print("FAILED")
                self.failed_tests += 1
        except Exception as e:
            print(f"FAILED (Exception: {e})")
            self.failed_tests += 1

    def summary(self):
        print("\n" + "="*40)
        print(f"TEST SUMMARY: {self.passed_tests}/{self.total_tests} passed.")
        print("="*40)

def run_suite():
    tester = BPE_Test_Suite()
    
    print("\n--- Group 1: Pre-tokenization Logic (Regex) ---")
    
    def test_basic_split():
        text = "Hello world"
        splits = get_gpt2_splits(text)
        # Expecting ["Hello", " world"] -> Space attaches to the next word
        return splits == ["Hello", " world"]
    tester.run_check("Basic word splitting with leading spaces", test_basic_split)

    def test_punctuation_split():
        text = "Hello, world!"
        splits = get_gpt2_splits(text)
        # Punctuation should be separate but respect spacing
        # "Hello", ",", " world", "!"
        return " world" in splits and "!" in splits
    tester.run_check("Splitting punctuation from words", test_punctuation_split)

    def test_contraction_split():
        text = "We're going"
        splits = get_gpt2_splits(text)
        # GPT-2 regex usually splits "We're" -> "We", "'re"
        return "'re" in splits
    tester.run_check("Handling English contractions (e.g. 're)", test_contraction_split)


    print("\n--- Group 2: Basic BPE Training & Encoding ---")
    
    def test_idempotency():
        # Ideally, encode -> decode should return the EXACT original string
        text = "The quick brown fox jumps."
        tokenizer = BPE_Tokenizer()
        tokenizer.train(text, num_merges=10)
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        return decoded == text
    tester.run_check("Round-trip consistency (Encode -> Decode == Original)", test_idempotency)

    def test_vocab_growth():
        # If we ask for 5 merges, vocab size should increase by exactly 5
        tokenizer = BPE_Tokenizer()
        initial_size = tokenizer.vocab_size
        tokenizer.train("ababababab", num_merges=5)
        return tokenizer.vocab_size == initial_size + 5
    tester.run_check("Vocabulary size growth correctness", test_vocab_growth)


    print("\n--- Group 3: Byte-Level & Unicode Support (Edge Cases) ---")

    def test_emoji_support():
        # This tests if the code crashes on non-ascii or handles them as bytes
        text = "I love 🍎 pie"
        tokenizer = BPE_Tokenizer()
        tokenizer.train(text, num_merges=5)
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        return decoded == text and "🍎" in decoded
    tester.run_check("Handling Unicode Emojis (🍎)", test_emoji_support)

    def test_foreign_script():
        # Japanese characters (multi-byte in UTF-8)
        text = "こんにちは"
        tokenizer = BPE_Tokenizer()
        tokenizer.train(text, num_merges=10)
        decoded = tokenizer.decode(tokenizer.encode(text))
        return decoded == text
    tester.run_check("Handling non-Latin scripts (Japanese)", test_foreign_script)


    print("\n--- Group 4: Robustness & Boundary Cases ---")

    def test_empty_string():
        tokenizer = BPE_Tokenizer()
        encoded = tokenizer.encode("")
        return encoded == []
    tester.run_check("Handling empty string input", test_empty_string)

    def test_unknown_words():
        # Train on "apple", test on "apricot"
        # It should fall back to bytes for the unseen parts, not crash
        tokenizer = BPE_Tokenizer()
        tokenizer.train("apple", num_merges=2) 
        encoded = tokenizer.encode("apricot")
        decoded = tokenizer.decode(encoded)
        return decoded == "apricot"
    tester.run_check("Generalization to unseen words", test_unknown_words)

    tester.summary()

if __name__ == "__main__":
    run_suite()