import json
import os

# Default IAM-style character set
# Index 0 is ALWAYS reserved for the CTC blank token
DEFAULT_CHARS = (
    " !\"#&'()*+,-./0123456789:;?"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)


class Vocabulary:
    """
    Maps characters <-> integer indices for CTC-based recognition.
    Index 0 is reserved for the CTC blank token.
    """

    BLANK_TOKEN = "<BLANK>"
    BLANK_IDX = 0

    def __init__(self, chars=None):
        if chars is None:
            chars = DEFAULT_CHARS
        # Remove duplicates while preserving order
        seen = set()
        unique_chars = []
        for c in chars:
            if c not in seen:
                seen.add(c)
                unique_chars.append(c)
        self.chars = unique_chars
        self._build_maps()

    def _build_maps(self):
        # Index 0 = CTC blank; characters start at index 1
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.chars)}
        self.idx_to_char = {i + 1: c for i, c in enumerate(self.chars)}
        self.idx_to_char[self.BLANK_IDX] = self.BLANK_TOKEN

    @property
    def num_classes(self):
        """Number of real characters (excluding blank)."""
        return len(self.chars)

    @property
    def num_labels(self):
        """Total number of labels including blank (= num_classes + 1)."""
        return len(self.chars) + 1

    # ---- encode / decode ------------------------------------------------
    def encode(self, text):
        """Convert a string to a list of integer indices."""
        indices = []
        for c in text:
            if c in self.char_to_idx:
                indices.append(self.char_to_idx[c])
            # Characters not in vocab are silently skipped
        return indices

    def decode(self, indices, remove_blanks=True, collapse_repeats=True):
        """
        Convert a list of indices back to a string (greedy CTC decode).
        """
        chars = []
        prev_idx = None
        for idx in indices:
            if collapse_repeats and idx == prev_idx:
                prev_idx = idx
                continue
            if remove_blanks and idx == self.BLANK_IDX:
                prev_idx = idx
                continue
            if idx in self.idx_to_char and idx != self.BLANK_IDX:
                chars.append(self.idx_to_char[idx])
            prev_idx = idx
        return "".join(chars)

    # ---- persistence -----------------------------------------------------
    def save(self, path):
        data = {"chars": self.chars}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(chars=data["chars"])

    # ---- build from training transcriptions ------------------------------
    @classmethod
    def build_from_texts(cls, texts):
        """Create vocabulary from a list of training transcription strings."""
        char_set = set()
        for t in texts:
            char_set.update(t)
        chars = sorted(char_set)
        return cls(chars=chars)

    def __repr__(self):
        return f"Vocabulary(num_classes={self.num_classes}, chars={''.join(self.chars[:20])}...)"


if __name__ == "__main__":
    vocab = Vocabulary()
    print(vocab)
    print(f"  num_classes (excl blank): {vocab.num_classes}")
    print(f"  num_labels  (incl blank): {vocab.num_labels}")

    text = "Hello World!"
    encoded = vocab.encode(text)
    decoded = vocab.decode(encoded, collapse_repeats=False)
    print(f"  '{text}' -> {encoded} -> '{decoded}'")
