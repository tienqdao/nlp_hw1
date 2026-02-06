from collections import Counter, defaultdict

class SentencePieceBPE:
    def __init__(self):
        self.merges = {}  # (byte1, byte2) -> new_token_id
        # Initialize base vocab with all 256 UTF-8 bytes
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        self.vocab_size = 256

    def get_stats(self, sequences):
        """
        Count pair frequencies across all sequences.
        sequences: A dictionary { tuple_of_ids: count }
        """
        pairs = defaultdict(int)
        for ids, freq in sequences.items():
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i+1])
                pairs[pair] += freq
        return pairs

    def merge_ids(self, ids, pair, new_id):
        """
        Replaces all instances of `pair` with `new_id` in the sequence `ids`.
        """
        new_ids = []
        i = 0
        while i < len(ids):
            # Check for the pair at current position
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text, num_merges=50):
        """
        Train BPE without pre-tokenization.
        We treat the input as a list of sentences (split by newline for efficiency),
        but we DO NOT split by words/punctuation.
        """
        # 1. Initial Processing
        # We process line-by-line to use a Counter. 
        # This handles duplicates efficiently and keeps memory usage lower than one giant list.
        lines = text.split('\n')
        
        # Convert each line directly to raw UTF-8 bytes
        # No regex splitting happens here!
        vocab = Counter()
        for line in lines:
            if line: # skip empty lines
                vocab[tuple(line.encode('utf-8'))] += 1
                
        print(f"Training on {len(vocab)} unique sentences/lines...")

        # 2. Iterative Merging
        for i in range(num_merges):
            stats = self.get_stats(vocab)
            if not stats:
                break
            
            # Find most frequent pair
            pair = max(stats, key=stats.get)
            
            # Create new token
            new_id = self.vocab_size
            self.merges[pair] = new_id
            self.id_to_bytes[new_id] = self.id_to_bytes[pair[0]] + self.id_to_bytes[pair[1]]
            self.vocab_size += 1
            
            # Apply merge to all sequences in our vocab
            # We rebuild the dictionary because keys (sequences) change
            new_vocab = Counter()
            for ids, count in vocab.items():
                new_sequence = tuple(self.merge_ids(list(ids), pair, new_id))
                new_vocab[new_sequence] += count
            vocab = new_vocab
            
            # Visualization: repr() shows the byte string (e.g. b'e ')
            print(f"Merge {i+1}: {pair} -> {new_id} ({repr(self.id_to_bytes[new_id])})")

    def encode(self, text):
        """
        Encodes text by converting to bytes and applying learned merges.
        """
        # Convert entire text to bytes
        ids = list(text.encode('utf-8'))
        
        # Apply merges strictly in order of learning
        while True:
            stats = self.get_stats({tuple(ids): 1})
            
            # Find the earliest learned merge that applies to this sequence
            best_pair = None
            min_merge_id = float('inf')
            
            # Check which of the current adjacent pairs are in our merge list
            for pair in stats:
                if pair in self.merges:
                    # We want the merge that resulted in the smallest ID 
                    if self.merges[pair] < min_merge_id:
                        min_merge_id = self.merges[pair]
                        best_pair = pair
            
            if best_pair:
                ids = self.merge_ids(ids, best_pair, self.merges[best_pair])
            else:
                break
                
        return ids

    def decode(self, ids):
        b = b"".join([self.id_to_bytes[idx] for idx in ids])
        return b.decode('utf-8', errors='replace')