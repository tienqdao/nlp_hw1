import regex as re
from collections import defaultdict, Counter

# -----------------------------------------------------------------------------
# 1. GPT-2 Pre-tokenization from text book (Figure 2.15)
# -----------------------------------------------------------------------------

def get_gpt2_splits(text):
    """
    Splits text using the exact regex pattern from the textbook.
    This ensures BPE does not cross word boundaries and handles spacing correctly.
    """
    pattern = re.compile(
        r"'s|'t|'re|'ve|'m|'ll|'d|"
        r" ?\p{L}+|"
        r" ?\p{N}+|"
        r" ?[^\s\p{L}\p{N}]+|"
        r"\s+(?!\S)|\s+"
    )
    return re.findall(pattern, text)

# -----------------------------------------------------------------------------
# 2. BPE Algorithm (Byte-Level)
# -----------------------------------------------------------------------------

def get_stats(vocab):
    """
    Compute the frequency of all adjacent byte/token pairs in the current vocabulary.
    vocab: A dictionary { (tuple_of_ids): frequency }
    """
    pairs = defaultdict(int)
    for word_ids, freq in vocab.items():
        # Iterate through the symbols in the word to find adjacent pairs
        for i in range(len(word_ids) - 1):
            pair = (word_ids[i], word_ids[i + 1])
            pairs[pair] += freq
    return pairs

def merge_vocab(pair, vocab, new_token_id):
    """
    Replace the pair (byte1, byte2) with (new_token_id) in all words in the vocabulary.
    """
    new_vocab = {}
    bigram = list(pair)
    
    for word_ids, freq in vocab.items():
        new_word = []
        i = 0
        while i < len(word_ids):
            # If we find the pair at current position
            if i < len(word_ids) - 1 and word_ids[i] == bigram[0] and word_ids[i+1] == bigram[1]:
                new_word.append(new_token_id)
                i += 2
            else:
                new_word.append(word_ids[i])
                i += 1
        new_vocab[tuple(new_word)] = new_vocab.get(tuple(new_word), 0) + freq # Accumulate frequency
    return new_vocab

# -----------------------------------------------------------------------------
# 3. Training Script
# -----------------------------------------------------------------------------

class BPE_Tokenizer:
    def __init__(self):
        self.merges = {} # (id1, id2) -> new_id
        # Initialize base byte vocabulary (0-255)
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        self.vocab_size = 256

    def train(self, text, num_merges=50):
        # Step 1: Pre-tokenize text into words using the textbook regex
        words = get_gpt2_splits(text)
        
        # Step 2: Convert words to bytes and count frequencies
        vocab = Counter()
        for w in words:
            vocab[tuple(w.encode('utf-8'))] += 1
            
        print(f"Start training with {len(vocab)} unique words...")

        for i in range(num_merges):
            # Count pairs
            pairs = get_stats(vocab)
            if not pairs:
                self.id_to_bytes[self.vocab_size] = b""  # dummy entry for empty token
                self.vocab_size += 1 # increase vocab size to account for the new token
                continue

                
            # Find the most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Create new token
            new_id = self.vocab_size
            self.merges[best_pair] = new_id
            
            # Update byte mapping for the new token (for visualization/decoding)
            self.id_to_bytes[new_id] = self.id_to_bytes[best_pair[0]] + self.id_to_bytes[best_pair[1]]
            
            # Apply merge to the vocabulary
            vocab = merge_vocab(best_pair, vocab, new_id)
            self.vocab_size += 1
            
            print(f"Merge {i+1}: {best_pair} -> {new_id} ({self.id_to_bytes[new_id]})")

    def encode(self, text):
        """Encodes new text using learned merges."""
        words = get_gpt2_splits(text)
        ids = []
        
        for word in words:
            # Start with raw bytes
            w_ids = list(word.encode('utf-8'))
            
            # Apply merges greedily in order of learning
            while len(w_ids) >= 2:
                # Iterate through all learned merges
                # If the pair exists in the word, merge it.
                changed = False
                for pair, new_id in self.merges.items():
                    new_w_ids = []
                    i = 0
                    while i < len(w_ids):
                        if i < len(w_ids) - 1 and w_ids[i] == pair[0] and w_ids[i+1] == pair[1]:
                            new_w_ids.append(new_id)
                            i += 2
                            changed = True
                        else:
                            new_w_ids.append(w_ids[i])
                            i += 1
                    w_ids = new_w_ids
                    if changed:
                        break # Restart scan after a merge
                
                if not changed:
                    break
            
            ids.extend(w_ids)
            
        return ids
    
    def decode(self, ids):
        """
        Converts a list of token IDs back into a string.
        """
        # 1. Concatenate the bytes for every token ID
        # self.id_to_bytes is the dictionary we built during training 
        byte_sequence = b"".join([self.id_to_bytes[idx] for idx in ids])
        
        # 2. Decode the byte sequence into a UTF-8 string
        return byte_sequence.decode('utf-8', errors='replace')