class SpaceTokenizer:
    def encode(self, text):
        """
        Tokenizes the input text based on spaces.

        Args:
            text (str): The input text to be tokenized.

        Returns:
            list: A list of tokens obtained by splitting the text on spaces.
        """
        return text.split()
    def decode(self, tokens):
        return " ".join(tokens)