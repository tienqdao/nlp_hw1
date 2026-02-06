from sentencePiece_bpe import SentencePieceBPE

class SP_BPE_Test_Suite:
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

    def run_check(self, description, success_condition):
        self.total_tests += 1
        print(f"Test {self.total_tests}: {description} ... \n", end="")
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
    tester = SP_BPE_Test_Suite()
    
    print("\n--- Group 1: Core SentencePiece Behavior ---")
    
    def test_cross_boundary_merge():
        # CRITICAL TEST: This distinguishes SP from GPT-2
        # Train on "the" repeatedly so it learns the token "the".
        # Then test on "ether".
        # GPT-2 would NOT use the "the" token because it splits "ether" into one word.
        # SentencePiece SHOULD use "the" inside "ether".
        text = "the " * 50
        sp = SentencePieceBPE()
        sp.train(text, num_merges=5) 
        
        encoded = sp.encode("ether")
        decoded_tokens = [sp.id_to_bytes[i] for i in encoded]
        
        # We expect one of the tokens to be exactly b'the'
        return b'the' in decoded_tokens
    tester.run_check("Cross-boundary merging (finding 'the' inside 'ether')", test_cross_boundary_merge)

    def test_space_as_byte():
        # SentencePiece treats space (ASCII 32) as just another character.
        # It should be able to merge a letter with a space.
        # Train on "a a a a " -> frequent pair should be "a " (a + space)
        text = "a a a a " * 20
        sp = SentencePieceBPE()
        sp.train(text, num_merges=5)
        
        # Check if we learned a token that ends in a space
        learned_tokens = sp.id_to_bytes.values()
        return b'a ' in learned_tokens
    tester.run_check("Treating space as a mergeable byte (e.g., b'a ')", test_space_as_byte)


    print("\n--- Group 2: Basic Mechanics ---")

    def test_idempotency():
        # Round trip: encode -> decode should return original string exactly
        text = "Hello world! This is a test."
        sp = SentencePieceBPE()
        sp.train(text, num_merges=20)
        encoded = sp.encode(text)
        decoded = sp.decode(encoded)
        return decoded == text
    tester.run_check("Round-trip consistency (Encode -> Decode)", test_idempotency)

    def test_vocab_growth():
        sp = SentencePieceBPE()
        initial = sp.vocab_size
        sp.train("abc", num_merges=2)
        # Should add exactly 2 new tokens
        return sp.vocab_size == initial + 2
    tester.run_check("Vocabulary growth matches num_merges", test_vocab_growth)


    print("\n--- Group 3: Robustness & Edge Cases ---")

    def test_emoji_streaming():
        # Emojis are 4 bytes. SP should handle them as a stream of bytes.
        text = "🙂" * 10
        sp = SentencePieceBPE()
        sp.train(text, num_merges=3) # enough to merge the 4 bytes of the emoji
        
        encoded = sp.encode("🙂")
        # Ideally, it learned the whole emoji as one token, or at least partials.
        # The key is that it decodes back correctly.
        decoded = sp.decode(encoded)
        return decoded == "🙂"
    tester.run_check("Emoji processing (Byte streaming)", test_emoji_streaming)

    def test_empty_string():
        sp = SentencePieceBPE()
        return sp.encode("") == []
    tester.run_check("Empty string input", test_empty_string)
    
    def test_multiline_training():
        # Ensure the split('\n') logic in train() works and doesn't crash
        text = "Line 1\nLine 2\nLine 3"
        sp = SentencePieceBPE()
        sp.train(text, num_merges=5)
        encoded = sp.encode("Line 2")
        return sp.decode(encoded) == "Line 2"
    tester.run_check("Multiline string training", test_multiline_training)

    tester.summary()

if __name__ == "__main__":
    run_suite()