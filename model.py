"""
From Tokens to Bytes: Compute and Transfer Multipliers from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_split
import dml_data

SPLITS = {
    "train": [
        ("en", "train/en.txt"),
        ("multilingual", "train/multilingual.txt"),
        ("code", "train/code.txt"),
        ("math", "train/math.txt"),
    ],
    "val": [
        ("en", "val/en.txt"),
        ("multilingual", "val/multilingual.txt"),
        ("code", "val/code.txt"),
        ("math", "val/math.txt"),
    ],
    "heldout": [
        ("fi", "heldout/text_fi.txt"),
        ("hu", "heldout/text_hu.txt"),
        ("el", "heldout/text_el.txt"),
        ("ja", "heldout/text_ja.txt"),
        ("ocaml", "heldout/code_ocaml.txt"),
        ("prolog", "heldout/code_prolog.txt"),
        ("lean", "heldout/code_lean.txt"),
        ("fortran", "heldout/code_fortran.txt"),
        ("haskell", "heldout/code_haskell.txt"),
    ],
}

# Module-level cache: stores file bytes keyed by the resolved file path.
_FILES = {}


def load_split(split):
    """
    Load one corpus split and return a dictionary mapping
    dataset names to their raw file bytes.

    Each file is read at most once per process.
    """

    result = {}

    # Iterate through the files in the order defined in SPLITS.
    for name, path in SPLITS[split]:

        # Locate the file on the dataset volume.
        full_path = "t2b-mix/v1/" + path
        file_path = dml_data.find_file(full_path)

        # Raise an error if the requested file is unavailable.
        if file_path is None:
            raise FileNotFoundError(
                f"Dataset file not found: {full_path}"
            )

        # Read the file only if it is not already cached.
        if file_path not in _FILES:
            with open(file_path, "rb") as f:
                _FILES[file_path] = f.read()

        # Retrieve the cached bytes.
        result[name] = _FILES[file_path]

    return result

# Step 2 - bits_per_byte
import math


def bits_per_byte(total_nats, n_bytes):
    """
    Convert total negative log-likelihood from nats
    into bits per byte.

    Args:
        total_nats: Sum of negative log-likelihood values in nats.
        n_bytes: Number of UTF-8 bytes covered by predictions.

    Returns:
        Bits per byte as a float.

    Raises:
        ValueError: If n_bytes is not positive.
    """

    if n_bytes <= 0:
        raise ValueError("n_bytes must be positive")

    return float(total_nats / (math.log(2) * n_bytes))

# Step 3 - gpt2_pieces
import re

GPT2_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"
)


def gpt2_pieces(data):
    """
    Split a byte string into GPT-2-style pretokenized pieces.

    Args:
        data: Input text as raw bytes.

    Returns:
        A list of byte strings representing individual pretokenized pieces.

    Surrogate escaping ensures that even invalid or incomplete UTF-8
    sequences survive the decoding and re-encoding process unchanged.
    """

    # Decode bytes while preserving invalid UTF-8 bytes.
    text = data.decode("utf-8", errors="surrogateescape")

    # Find all GPT-2-style pretoken matches and convert each
    # match back into bytes using the same error-handling strategy.
    pieces = [
        match.group(0).encode("utf-8", errors="surrogateescape")
        for match in GPT2_PATTERN.finditer(text)
    ]

    return pieces

# Step 4 - to_stream
import numpy as np


def to_stream(blob, split_fn=None):
    """
    Convert a corpus blob into a byte stream with piece boundaries.

    Args:
        blob: Bytes containing documents separated by byte 0x1E.
        split_fn: Optional function that splits each document into
                  byte pieces. If None, only document boundaries
                  and separator boundaries are marked.

    Returns:
        ids: int32 NumPy array containing byte values.
        starts: Boolean NumPy array marking the beginning of pieces.

    A separator byte (0x1E) is treated as a piece of its own.
    """

    # Convert each byte into an int32 token ID.
    ids = np.frombuffer(blob, dtype=np.uint8).astype(np.int32)

    # Initialize the Boolean array for piece boundaries.
    starts = np.zeros(len(blob), dtype=bool)

    # An empty blob produces two empty arrays.
    if len(blob) == 0:
        return ids, starts

    # The first byte always begins a new piece.
    starts[0] = True

    separator = b"\x1e"
    doc_start = 0
    n = len(blob)

    # Process each document separated by the separator byte.
    while True:

        # Locate the next document separator.
        sep_pos = blob.find(separator, doc_start)

        # Determine the end of the current document.
        doc_end = sep_pos if sep_pos != -1 else n

        # Apply the optional pretokenizer to the current document.
        if split_fn is not None:
            document = blob[doc_start:doc_end]

            # Track the position of each piece in the original blob.
            offset = doc_start

            for piece in split_fn(document):

                piece_len = len(piece)

                # Mark the first byte of every non-empty piece.
                if piece_len > 0:
                    starts[offset] = True

                # Advance to the beginning of the next piece.
                offset += piece_len

        # Stop if there are no more separators.
        if sep_pos == -1:
            break

        # The separator byte must begin its own piece.
        starts[sep_pos] = True

        # The byte immediately after the separator starts a new document.
        next_start = sep_pos + 1

        if next_start < n:
            starts[next_start] = True

        # Continue with the next document.
        doc_start = next_start

    return ids, starts

# Step 5 - count_pairs
def count_pairs(ids, starts, vocab_size):
    """
    Count all adjacent token pairs that may be merged.

    Args:
        ids: Integer NumPy array containing token IDs.
        starts: Boolean array marking the beginning of pieces.
        vocab_size: Number of tokens in the vocabulary.

    Returns:
        A NumPy array of length vocab_size ** 2 containing
        the frequency of every possible adjacent token pair.
    """

    # Convert IDs to int64 to prevent integer overflow during
    # pair encoding.
    ids = np.asarray(ids, dtype=np.int64)

    # Ensure the piece-start mask is Boolean.
    starts = np.asarray(starts, dtype=bool)

    # A pair (ids[i], ids[i + 1]) is eligible only when
    # its right element does not begin a new piece.
    valid = ~starts[1:]

    # Encode each eligible pair as a unique integer.
    pair_codes = (
        ids[:-1][valid] * np.int64(vocab_size)
        + ids[1:][valid]
    )

    # Count the occurrences of every possible pair.
    counts = np.bincount(
        pair_codes,
        minlength=vocab_size * vocab_size
    )

    return counts

# Step 6 - merge_pair
def merge_pair(ids, starts, a, b, new_id):
    """
    Replace every allowed occurrence of the pair (a, b)
    with new_id, without overlapping merges.

    Args:
        ids: Integer NumPy array containing token IDs.
        starts: Boolean array marking the beginning of pieces.
        a: First token ID of the pair.
        b: Second token ID of the pair.
        new_id: Token ID assigned to the merged pair.

    Returns:
        A tuple (new_ids, new_starts) containing the
        updated token IDs and piece-start flags.
    """

    # Find all eligible adjacent pairs.
    # A pair is eligible only if its right element
    # does not begin a new piece.
    matches = np.flatnonzero(
        (ids[:-1] == a)
        & (ids[1:] == b)
        & (~starts[1:])
    )

    # If no eligible pair exists, return the original arrays.
    if len(matches) == 0:
        return ids, starts

    # Handle overlapping pairs when both token IDs are identical.
    if a == b:
        selected = []
        next_available = 0

        # Scan from left to right, avoiding overlapping merges.
        for i in matches:
            if i >= next_available:
                selected.append(i)

                # The next merge must begin after the merged pair.
                next_available = i + 2

        matches = np.asarray(selected, dtype=np.intp)

    # Copy the original token IDs before modifying them.
    new_ids = ids.copy()

    # Replace the first element of every selected pair.
    new_ids[matches] = new_id

    # Mark the second element of each merged pair for removal.
    keep = np.ones(len(ids), dtype=bool)
    keep[matches + 1] = False

    # Remove merged-away elements from both arrays.
    new_ids = new_ids[keep]
    new_starts = starts[keep]

    return new_ids, new_starts

# Step 7 - train_bpe
def train_bpe(ids, starts, n_merges):
    """
    Train a BPE tokenizer by learning up to n_merges pair merges.

    Args:
        ids: Integer NumPy array containing byte/token IDs.
        starts: Boolean array marking piece boundaries.
        n_merges: Maximum number of BPE merges to learn.

    Returns:
        A list of (a, b) tuples, where each tuple represents
        a pair of token IDs that should be merged.
    """

    # The final vocabulary contains the original 256 byte IDs
    # plus the maximum number of requested merges.
    vocab_size = 256 + n_merges

    # Store the learned BPE merges in their training order.
    merges = []

    # Perform the BPE merging process.
    for k in range(n_merges):

        # Count all eligible adjacent token pairs.
        counts = count_pairs(ids, starts, vocab_size)

        # Stop early if no eligible pair remains.
        if counts.max() == 0:
            break

        # Select the most frequent pair.
        # np.argmax automatically resolves ties by choosing
        # the smallest encoded pair ID.
        best_code = int(np.argmax(counts))

        # Decode the selected pair into its two token IDs.
        a = best_code // vocab_size
        b = best_code % vocab_size

        # Store the selected pair as Python integers.
        a = int(a)
        b = int(b)

        merges.append((a, b))

        # Apply the merge using the next available token ID.
        new_id = 256 + k

        ids, starts = merge_pair(
            ids,
            starts,
            a,
            b,
            new_id
        )

    return merges

# Step 8 - encode_stream
def encode_stream(ids, starts, merges):
    """
    Tokenize a byte stream by applying the learned BPE merges
    in their original training order.

    Args:
        ids: Integer NumPy array containing byte/token IDs.
        starts: Boolean array marking piece boundaries.
        merges: List of (a, b) tuples representing learned BPE merges.

    Returns:
        The final token ID NumPy array.
    """

    # Apply each learned merge exactly once, in list order.
    for k, (a, b) in enumerate(merges):

        # Each merge creates a new token ID starting from 256.
        new_id = 256 + k

        # Apply the merge and carry forward the updated arrays.
        ids, starts = merge_pair(
            ids,
            starts,
            a,
            b,
            new_id
        )

    # Return only the final token ID array.
    return ids

# Step 9 - make_tokenizer
def make_tokenizer(name, merges, pretok):
    """
    Create a tokenizer dictionary from a list of BPE merges.

    Args:
        name: Tokenizer name ("byte", "bpe", or "bpe_np").
        merges: List of (a, b) tuples representing BPE merges.
        pretok: Whether GPT-2-style pretokenization is enabled.

    Returns:
        A dictionary containing:
            name: Tokenizer name.
            merges: List of BPE merge tuples.
            pretok: Boolean indicating whether pretokenization is used.
            vocab: List of byte strings representing vocabulary tokens.
            ranks: Dictionary mapping each merge pair to its rank.
            cache: Empty dictionary for caching encoded pieces.
    """

    # Ensure all merge pairs contain ordinary Python integers.
    merges = [
        (int(a), int(b))
        for a, b in merges
    ]

    # Initialize the vocabulary with all 256 possible byte values.
    vocab = [
        bytes([i])
        for i in range(256)
    ]

    # Build the vocabulary by applying merges in their original order.
    for a, b in merges:

        # The new token is the concatenation of its two components.
        vocab.append(vocab[a] + vocab[b])

    # Map each merge pair to its position in the merge sequence.
    ranks = {
        pair: k
        for k, pair in enumerate(merges)
    }

    # Return the complete tokenizer configuration.
    return {
        "name": name,
        "merges": merges,
        "pretok": bool(pretok),
        "vocab": vocab,
        "ranks": ranks,
        "cache": {},
    }

# Step 10 - encode_many
def encode_many(tok, strings):
    """
    Encode multiple byte strings independently using a tokenizer.

    Args:
        tok: Tokenizer dictionary containing merges, pretok,
             and cache.
        strings: List of input byte strings.

    Returns:
        A list of Python lists containing token IDs,
        one list for each input string.

    Raises:
        ValueError: If any input string contains the separator byte 0x1E.
    """

    separator = b"\x1e"

    # Convert the input iterable into a list.
    strings = list(strings)

    # Reject strings containing the reserved separator byte.
    if any(separator in s for s in strings):
        raise ValueError("Input strings must not contain the separator byte 0x1E")

    # Return an empty list when there are no input strings.
    if not strings:
        return []

    # Helper function to split encoded token IDs at separator ID 30.
    # Empty strings and consecutive separators are preserved.
    def split_encoded(encoded):
        boundaries = np.flatnonzero(encoded == 30)

        results = []
        start = 0

        for pos in boundaries:
            results.append(encoded[start:pos].tolist())
            start = pos + 1

        results.append(encoded[start:].tolist())

        return results

    # ---------------------------------------------------------
    # Case 1: No GPT-2-style pretokenization (BPE-NP).
    # ---------------------------------------------------------
    if not tok["pretok"]:

        # Join all strings using the reserved separator.
        blob = separator.join(strings)

        # Convert the combined blob into a byte stream.
        ids, starts = to_stream(blob)

        # Apply all learned BPE merges in a single encoding call.
        encoded = encode_stream(ids, starts, tok["merges"])

        # Split the encoded stream back into individual strings.
        return split_encoded(encoded)

    # ---------------------------------------------------------
    # Case 2: GPT-2-style pretokenization (BPE).
    # ---------------------------------------------------------

    # Split each input string into GPT-2-style byte pieces.
    pieces_per_string = [
        gpt2_pieces(s)
        for s in strings
    ]

    # Collect pieces that are not already present in the cache.
    # dict.fromkeys() also prevents duplicate pieces from
    # being encoded multiple times in the same call.
    missing_pieces = list(dict.fromkeys(
        piece
        for pieces in pieces_per_string
        for piece in pieces
        if piece not in tok["cache"]
    ))

    # Encode all missing pieces together in a single call.
    if missing_pieces:

        # Separate independent pieces using the reserved byte.
        blob = separator.join(missing_pieces)

        # Build a stream with document boundaries.
        ids, starts = to_stream(blob)

        # Encode the combined stream.
        encoded = encode_stream(ids, starts, tok["merges"])

        # Recover the encoding of each individual piece.
        encoded_pieces = split_encoded(encoded)

        # Store the results in the tokenizer cache.
        for piece, piece_ids in zip(missing_pieces, encoded_pieces):
            tok["cache"][piece] = piece_ids

    # Reconstruct each original string's token sequence
    # using the cached encodings of its individual pieces.
    results = []

    for pieces in pieces_per_string:
        token_ids = []

        for piece in pieces:
            token_ids.extend(tok["cache"][piece])

        results.append(token_ids)

    return results

