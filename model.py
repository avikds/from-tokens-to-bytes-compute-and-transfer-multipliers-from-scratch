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

