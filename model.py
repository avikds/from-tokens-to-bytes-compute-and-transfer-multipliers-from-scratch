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

# Step 11 - bytes_per_token
def bytes_per_token(tok, blobs, max_bytes=None):
    """
    Measure the average number of bytes represented by each token
    for every domain in the supplied corpus.

    Args:
        tok: Tokenizer dictionary containing merges and pretok.
        blobs: Dictionary mapping domain names to raw byte strings.
        max_bytes: Optional maximum number of bytes to process
                   from each domain.

    Returns:
        A dictionary mapping each domain name to its average
        number of bytes per token, as a float.
    """

    results = {}

    # Select the appropriate pretokenization function.
    split_fn = gpt2_pieces if tok["pretok"] else None

    # Process each corpus domain independently.
    for name, blob in blobs.items():

        # Restrict the input to the requested number of bytes.
        if max_bytes is not None:
            blob = blob[:max_bytes]

        # Convert the byte blob into a stream with piece boundaries.
        ids, starts = to_stream(blob, split_fn=split_fn)

        # Apply the tokenizer's learned BPE merges.
        encoded = encode_stream(
            ids,
            starts,
            tok["merges"]
        )

        # Count the original input bytes and resulting tokens.
        n_bytes = len(blob)
        n_tokens = len(encoded)

        # Calculate average bytes per token.
        # An empty input has no tokens, so return 0.0.
        results[name] = (
            float(n_bytes / n_tokens)
            if n_tokens > 0
            else 0.0
        )

    return results

# Step 12 - toy_rung
def toy_rung(L, head_dim=32, aspect=32, multiple=16, ffn_ratio=3.6):
    """
    Build one rung of the scaled-down Transformer ladder.

    Args:
        L: Number of Transformer layers.
        head_dim: Dimension of each attention head.
        aspect: Controls the model width relative to depth.
        multiple: Required rounding multiple for model dimensions.
        ffn_ratio: Ratio used to determine the FFN width.

    Returns:
        Dictionary containing the Transformer architecture
        and backbone parameter count.

    Raises:
        ValueError: If L is less than 1.
    """

    # A Transformer must have at least one layer.
    if L < 1:
        raise ValueError("L must be at least 1")

    L = int(L)

    # Model width: nearest multiple of `multiple` to aspect * L.
    D = multiple * math.floor(
        (aspect * L) / multiple + 0.5
    )

    # Number of attention heads equals the number of layers.
    heads = L

    # Total attention projection width.
    d_attn = head_dim * L

    # Feed-forward network width: nearest multiple of `multiple`.
    F = multiple * math.floor(
        (ffn_ratio * D) / multiple + 0.5
    )

    # Backbone parameters:
    # 4 * D * d_attn for Q, K, V and output projections.
    # 3 * D * F for the three SwiGLU matrices.
    backbone_params = L * (
        4 * D * d_attn + 3 * D * F
    )

    # Return all architecture entries as Python integers.
    return {
        "layers": int(L),
        "d_model": int(D),
        "heads": int(heads),
        "head_dim": int(head_dim),
        "d_attn": int(d_attn),
        "ffn": int(F),
        "backbone_params": int(backbone_params),
    }

# Step 13 - train_flops_per_token
def train_flops_per_token(rung, vocab, seq_len):
    """
    Calculate training FLOPs per token for one Transformer rung
    using the project's causal-attention FLOP convention.

    Args:
        rung: Dictionary containing Transformer architecture
              and backbone parameter information.
        vocab: Vocabulary size.
        seq_len: Sequence length.

    Returns:
        Total training FLOPs per token as an integer.
    """

    # Extract the required Transformer architecture parameters.
    layers = rung["layers"]
    d_model = rung["d_model"]
    d_attn = rung["d_attn"]
    backbone_params = rung["backbone_params"]

    # Dense forward-pass FLOPs:
    # Backbone operations plus the vocabulary output head.
    dense_flops = 2 * (
        backbone_params + vocab * d_model
    )

    # Causal attention FLOPs averaged over sequence positions.
    attention_flops = (
        layers * 2 * d_attn * (seq_len + 1)
    )

    # Training requires three times the forward-pass FLOPs.
    total_flops = 3 * (
        dense_flops + attention_flops
    )

    return int(total_flops)

# Step 14 - plan_run
def plan_run(rung, vocab, seq_len, batch, budget):
    """
    Convert a compute budget into a number of training steps.

    Args:
        rung: Transformer architecture dictionary.
        vocab: Vocabulary size.
        seq_len: Number of tokens per training sequence.
        batch: Number of sequences per training step.
        budget: Total available compute budget in FLOPs.

    Returns:
        Dictionary containing:
            steps: Number of training steps.
            tokens: Total number of training tokens.
            flops: Actual total FLOPs spent.
            flops_per_token: Training FLOPs per token.
    """

    # Calculate the training FLOPs required per token.
    flops_per_token = train_flops_per_token(
        rung, vocab, seq_len
    )

    # Calculate the FLOPs required for one training step.
    per_step = flops_per_token * batch * seq_len

    # Convert the compute budget into training steps.
    # Use half-up rounding and enforce a minimum of one step.
    steps = max(
        1,
        math.floor(budget / per_step + 0.5)
    )

    # Calculate the total number of tokens processed.
    tokens = steps * batch * seq_len

    # Calculate the actual compute consumed.
    flops = steps * per_step

    return {
        "steps": int(steps),
        "tokens": int(tokens),
        "flops": int(flops),
        "flops_per_token": int(flops_per_token),
    }

# Step 15 - init_params
from tinygrad import Tensor


def init_params(rung, vocab, seq_len, seed=0):
    """
    Initialize all Transformer parameters as realized tinygrad tensors.

    Args:
        rung: Dictionary containing Transformer architecture details.
        vocab: Vocabulary size.
        seq_len: Maximum sequence length.
        seed: Random seed for reproducible initialization.

    Returns:
        Dictionary containing all initialized Transformer parameters.
    """

    # Extract architecture dimensions.
    D = rung["d_model"]
    A = rung["d_attn"]
    F = rung["ffn"]
    n = rung["layers"]

    # Create one random number generator for reproducibility.
    rng = np.random.default_rng(seed)

    # Helper function to create and realize a float32 tensor.
    def realize(array):
        return Tensor(array).contiguous().realize()

    # Helper function to initialize weights with a normal distribution.
    def randn(std, shape):
        array = rng.normal(0.0, std, shape).astype(np.float32)
        return realize(array)

    # Helper function to initialize normalization vectors to ones.
    def ones(shape):
        return realize(np.ones(shape, dtype=np.float32))

    # Store all model parameters in the required creation order.
    params = {}

    # 1. Token and positional embeddings.
    params["tok_emb"] = randn(0.02, (vocab, D))
    params["pos_emb"] = randn(0.02, (seq_len, D))

    # Scaling factor for the residual projection weights.
    residual_std = 0.02 / math.sqrt(2 * n)

    # 2. Initialize parameters for every Transformer layer.
    for i in range(n):

        # First pre-normalization layer.
        params[f"norm1.{i}"] = ones((D,))

        # Combined query, key and value projection.
        params[f"qkv.{i}"] = randn(0.02, (D, 3 * A))

        # Attention output projection.
        params[f"proj.{i}"] = randn(residual_std, (A, D))

        # Second pre-normalization layer.
        params[f"norm2.{i}"] = ones((D,))

        # SwiGLU feed-forward network weights.
        params[f"w1.{i}"] = randn(0.02, (D, F))
        params[f"w3.{i}"] = randn(0.02, (D, F))

        # Feed-forward output projection.
        params[f"w2.{i}"] = randn(residual_std, (F, D))

    # 3. Final normalization layer.
    params["norm_f"] = ones((D,))

    # 4. Output vocabulary projection head.
    params["head"] = randn(0.02, (D, vocab))

    return params

# Step 16 - forward
def forward(params, rung, x):
    """
    Compute next-token logits using a pre-norm causal Transformer.

    Args:
        params: Dictionary containing initialized Transformer parameters.
        rung: Dictionary containing Transformer architecture details.
        x: Integer tinygrad Tensor of shape (B, T), where T <= seq_len.

    Returns:
        Logits Tensor of shape (B, T, vocab).
    """

    # Extract input dimensions.
    B, T = x.shape

    # Extract Transformer architecture dimensions.
    D = rung["d_model"]
    A = rung["d_attn"]
    heads = rung["heads"]
    head_dim = rung["head_dim"]

    # RMSNorm helper.
    # Normalize over the last dimension and apply the learned weight.
    def rms(h, g):
        variance = (h * h).mean(axis=-1, keepdim=True)
        return h * (variance + 1e-5).rsqrt() * g

    # ---------------------------------------------------------
    # 1. Token and positional embeddings.
    # ---------------------------------------------------------

    h = params["tok_emb"][x] + params["pos_emb"][:T]

    # ---------------------------------------------------------
    # 2. Transformer layers.
    # ---------------------------------------------------------

    for i in range(rung["layers"]):

        # -----------------------------------------------------
        # Pre-normalization followed by QKV projection.
        # -----------------------------------------------------

        z = rms(h, params[f"norm1.{i}"])

        qkv = z @ params[f"qkv.{i}"]

        # Split the combined projection into query, key and value.
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape each tensor into (B, T, heads, head_dim).
        q = q.reshape(B, T, heads, head_dim)
        k = k.reshape(B, T, heads, head_dim)
        v = v.reshape(B, T, heads, head_dim)

        # Move the attention-head dimension forward:
        # (B, heads, T, head_dim).
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # -----------------------------------------------------
        # Causal self-attention.
        # -----------------------------------------------------

        # Each position can attend only to itself and
        # preceding positions.
        attn = q.scaled_dot_product_attention(
            k,
            v,
            is_causal=True
        )

        # Restore the original sequence layout:
        # (B, heads, T, head_dim)
        # -> (B, T, heads, head_dim)
        # -> (B, T, d_attn).
        attn = attn.transpose(1, 2).reshape(B, T, A)

        # Attention output projection and residual connection.
        h = h + attn @ params[f"proj.{i}"]

        # -----------------------------------------------------
        # Pre-normalization followed by SwiGLU feed-forward.
        # -----------------------------------------------------

        z = rms(h, params[f"norm2.{i}"])

        # SwiGLU activation:
        # silu(z @ w1) element-wise multiplied by (z @ w3).
        gate = (z @ params[f"w1.{i}"]).silu()
        value = z @ params[f"w3.{i}"]

        ffn = gate * value

        # Feed-forward output projection and residual connection.
        h = h + ffn @ params[f"w2.{i}"]

    # ---------------------------------------------------------
    # 3. Final RMS normalization and vocabulary projection.
    # ---------------------------------------------------------

    h = rms(h, params["norm_f"])

    logits = h @ params["head"]

    return logits

# Step 17 - wsd_lr
def wsd_lr(step, total, peak, warmup_frac=0.05, decay_frac=0.2):
    """
    Calculate the learning rate using a warmup-stable-decay schedule.

    Args:
        step: Current zero-based training step.
        total: Total number of training steps.
        peak: Maximum learning rate.
        warmup_frac: Fraction of training steps used for warmup.
        decay_frac: Fraction of training steps used for decay.

    Returns:
        Learning rate for the given training step.
    """

    # Calculate the warmup and decay durations.
    # Each phase must contain at least one step.
    warm = max(1, int(warmup_frac * total))
    decay = max(1, int(decay_frac * total))

    # Phase 1: Linear warmup.
    if step < warm:
        return peak * (step + 1) / warm

    # Phase 3: Linear decay during the final training steps.
    elif step >= total - decay:
        return peak * (total - step) / decay

    # Phase 2: Stable learning rate.
    else:
        return peak

# Step 18 - TinyLM
from tinygrad import TinyJit


class TinyLM:
    def __init__(
        self,
        rung,
        vocab,
        seq_len,
        batch,
        eval_batch=16,
        wd=0.1,
        b1=0.9,
        b2=0.95,
        eps=1e-8,
    ):
        """
        Initialize the Transformer model, AdamW optimizer state,
        and compiled training and scoring functions.
        """

        # Store the configuration and optimizer hyperparameters.
        self.rung = rung
        self.vocab = vocab
        self.seq_len = seq_len
        self.batch = batch
        self.eval_batch = eval_batch

        self.wd = wd
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

        # Initialize all Transformer parameters.
        self.params = init_params(
            rung, vocab, seq_len, 0
        )

        # Preserve the parameter order for gradient computation
        # and optimizer updates.
        param_values = list(self.params.values())

        # Initialize AdamW first-moment estimates.
        self.m = [
            Tensor(np.zeros(p.shape, dtype=np.float32)).realize()
            for p in param_values
        ]

        # Initialize AdamW second-moment estimates.
        self.v = [
            Tensor(np.zeros(p.shape, dtype=np.float32)).realize()
            for p in param_values
        ]

        # Initialize the AdamW timestep.
        self.t = Tensor([0.0]).realize()

        # -----------------------------------------------------
        # Compiled training step.
        # -----------------------------------------------------

        def training_step(x, y, lr):

            # Forward pass and next-token prediction loss.
            logits = forward(self.params, self.rung, x)

            loss = logits.sparse_categorical_crossentropy(y)

            # Calculate gradients for every model parameter.
            grads = loss.gradient(*param_values)

            # Increment the AdamW timestep.
            self.t.assign(self.t + 1.0)

            # Calculate AdamW bias-correction denominators.
            bias1 = 1.0 - self.b1 ** self.t
            bias2 = 1.0 - self.b2 ** self.t

            # -------------------------------------------------
            # AdamW parameter updates.
            # -------------------------------------------------

            for p, m, v, g in zip(
                param_values,
                self.m,
                self.v,
                grads
            ):

                # Update first and second gradient moments.
                m.assign(
                    self.b1 * m + (1.0 - self.b1) * g
                )

                v.assign(
                    self.b2 * v + (1.0 - self.b2) * g * g
                )

                # Bias-corrected Adam moments.
                m_hat = m / bias1
                v_hat = v / bias2

                # Adam update.
                update = m_hat / (v_hat + self.eps).sqrt()

                # Apply weight decay only to 2-D parameters.
                if len(p.shape) == 2:
                    update = update + self.wd * p

                # Update model parameters.
                p.assign(p - lr * update)

            # Realize the loss and all updated optimizer/model states.
            loss.realize(
                self.t,
                *param_values,
                *self.m,
                *self.v
            )

            return loss

        # Compile the training function.
        self._train_step = TinyJit(training_step)

        # -----------------------------------------------------
        # Compiled scoring function.
        # -----------------------------------------------------

        def scoring_step(x):

            # Compute log probabilities over the vocabulary.
            return forward(
                self.params,
                self.rung,
                x
            ).log_softmax(-1).realize()

        # Compile the scoring function.
        self._scoring_step = TinyJit(scoring_step)

    def reset(self, seed):
        """
        Reinitialize model parameters and reset AdamW state.

        Existing parameter and optimizer tensors are updated in place
        rather than replaced, preserving the TinyJit references.
        """

        # Generate fresh parameters using the requested seed.
        new_params = init_params(
            self.rung,
            self.vocab,
            self.seq_len,
            seed
        )

        # Copy the new parameter values into existing tensors.
        for name, p in self.params.items():
            p.assign(new_params[name])

        # Reset both AdamW moment estimates.
        for moment_list in (self.m, self.v):

            for moment in moment_list:
                moment.assign(
                    Tensor(
                        np.zeros(moment.shape, dtype=np.float32)
                    )
                )

        # Reset the optimizer timestep.
        self.t.assign(Tensor([0.0]))

        # Realize all parameter and optimizer-state updates.
        self.t.realize(
            *self.params.values(),
            *self.m,
            *self.v
        )

    def train(self, stream, steps, peak_lr, seed):
        """
        Train the Transformer using random windows from a token stream.

        Args:
            stream: NumPy array containing training token IDs.
            steps: Number of training steps.
            peak_lr: Peak learning rate for the WSD schedule.
            seed: Random seed for sampling training windows.

        Returns:
            A list of training losses as Python floats,
            recorded every 50 steps and after the final step.
        """

        # Create a reproducible NumPy random generator.
        rng = np.random.default_rng(seed)

        # Ensure the stream contains int32 token IDs.
        stream = np.asarray(stream, dtype=np.int32)

        if stream.ndim != 1:
            raise ValueError("stream must be a one-dimensional array")

        # Validate that the stream is long enough to sample windows.
        upper = len(stream) - self.seq_len - 1

        if upper <= 0 and steps > 0:
            raise ValueError("stream is too short for the requested sequence length")

        # Record losses at the required intervals.
        losses = []

        # Relative positions for each sampled window.
        offsets = np.arange(
            self.seq_len + 1,
            dtype=np.int64
        )

        # Run the training loop.
        for i in range(steps):

            # Draw random starting positions for the batch.
            starts = rng.integers(
                0,
                upper,
                self.batch
            )

            # Construct indices for windows of seq_len + 1 tokens.
            idx = starts[:, None] + offsets[None, :]

            # Extract the sampled windows.
            windows = stream[idx]

            # Inputs contain the first seq_len tokens.
            x = windows[:, :-1].astype(
                np.int32,
                copy=False
            )

            # Targets contain the next seq_len tokens.
            y = windows[:, 1:].astype(
                np.int32,
                copy=False
            )

            # Convert inputs and targets into tinygrad tensors.
            x = Tensor(x)
            y = Tensor(y)

            # Calculate the learning rate for the current step.
            lr = Tensor([
                wsd_lr(i, steps, peak_lr)
            ])

            # Execute the compiled training step.
            loss = self._train_step(x, y, lr)

            # Record the loss every 50 steps and after the last step.
            if (i + 1) % 50 == 0 or i == steps - 1:
                losses.append(float(loss.numpy()))

        return losses

    def log_probs(self, x):
        """
        Compute log probabilities for an evaluation batch.

        Args:
            x: NumPy int array of shape (eval_batch, seq_len).

        Returns:
            NumPy array of shape (eval_batch, seq_len, vocab).
        """

        # Ensure inputs use the required int32 representation.
        x = np.asarray(x, dtype=np.int32)

        # Validate the evaluation input shape.
        expected_shape = (
            self.eval_batch,
            self.seq_len
        )

        if x.shape != expected_shape:
            raise ValueError(
                f"Expected input shape {expected_shape}, got {x.shape}"
            )

        # Convert inputs to a tinygrad tensor.
        x = Tensor(x)

        # Run the compiled scoring function.
        result = self._scoring_step(x)

        # Return the realized result as a NumPy array.
        return result.numpy()

# Step 19 - score_sequences
def score_sequences(model, seqs):
    """
    Score a list of token sequences using TinyLM in fixed-size batches.

    Args:
        model: Initialized TinyLM instance.
        seqs: List of token sequences.

    Returns:
        A list of (cum, last) tuples, one per input sequence.

        cum:
            Float64 array of length n containing cumulative
            log probabilities, with cum[0] = 0.

        last:
            Float64 array containing the next-token
            log-probability distribution.

    Raises:
        ValueError: If a sequence is longer than model.seq_len.
        ValueError: If a sequence is empty.
    """

    results = []

    # Process sequences in chunks of model.eval_batch.
    seqs = iter(seqs)

    while True:

        # Collect the next batch of sequences.
        chunk = []

        for _ in range(model.eval_batch):
            try:
                chunk.append(next(seqs))
            except StopIteration:
                break

        # Stop when all sequences have been processed.
        if not chunk:
            break

        # Create a fixed-shape int32 batch filled with zeros.
        batch = np.zeros(
            (model.eval_batch, model.seq_len),
            dtype=np.int32
        )

        lengths = []

        # Copy each sequence into the beginning of its batch row.
        for row, seq in enumerate(chunk):

            q = np.asarray(seq, dtype=np.int32)

            n = len(q)

            # Reject sequences exceeding the model context length.
            if n > model.seq_len:
                raise ValueError(
                    f"Sequence length {n} exceeds "
                    f"model.seq_len={model.seq_len}"
                )

            # An empty sequence has no position n-1 to score.
            if n == 0:
                raise ValueError("Sequences must not be empty")

            batch[row, :n] = q
            lengths.append(n)

        # Obtain log probabilities with one model call per chunk.
        log_probs = model.log_probs(batch)

        # Calculate cumulative probabilities for each sequence.
        for row, n in enumerate(lengths):

            # Extract the log-probability matrix for this sequence.
            row_log_probs = log_probs[row]

            # Convert the relevant values to float64.
            row_log_probs = row_log_probs.astype(np.float64)

            # Initialize cumulative log probabilities.
            cum = np.zeros(n, dtype=np.float64)

            # Position i-1 predicts token q[i].
            if n > 1:
                q = np.asarray(chunk[row], dtype=np.int32)

                token_log_probs = row_log_probs[
                    np.arange(n - 1),
                    q[1:]
                ]

                # Calculate cumulative sums.
                cum[1:] = np.cumsum(token_log_probs)

            # Distribution of the next token after the full sequence.
            last = row_log_probs[n - 1].copy()

            # Store the result in the original input order.
            results.append((cum, last))

    return results

