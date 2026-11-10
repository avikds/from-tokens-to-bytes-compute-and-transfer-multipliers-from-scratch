# From Tokens to Bytes: Compute and Transfer Multipliers from Scratch

Cartesia's two posts "From Tokens to Bytes" (October 2026) compare ways of feeding text to a language model with two kinds of law: compute-to-loss, the validation bits per byte a training budget buys on the compute-optimal frontier, and loss-to-loss, how that validation loss turns into loss on text the model was not trained for. This project rebuilds the measuring method at toy scale, in tinygrad on one CPU, and applies it to three encodings of the same 14 MB corpus (English books, six other languages, Python, C, JavaScript, Go and Rust, LaTeX math): raw bytes, BPE with a GPT-2-style pretokenizer, and BPE with no pretokenization (BPE-NP), both with 1,024 tokens. You write a vectorized BPE trainer and encoder that serve both tokenizers, a scaled-down version of the posts' depth-indexed Transformer ladder with their causal FLOP count, a pre-norm Transformer with a TinyJit AdamW loop and a warmup-stable-decay schedule, validation bits per byte, prompt-and-answer scoring on nine held-out languages (Finnish, Hungarian, Greek, Japanese, OCaml, Prolog, Lean, Fortran, Haskell) done both canonically and by summing over the valid token sequences that cover the prompt boundary (ByteSampler's valid covering tree), an iso-FLOP sweep runner, power-law fits, compute multipliers, the implied loss-to-loss curves and the posts' loss-to-loss compute multiplier. What the full sweep found (four sizes, five budgets from 1e12 to 1.6e13 FLOPs, three seeds, 180 runs on the prod CPU image, all with the functions you write; held-out loss is the mean over Finnish, Hungarian and the five programming languages). On compute-to-validation, pretokenized BPE wins: at its largest-budget validation loss (2.30 bits per byte) BPE-NP needs 1.41 times its compute (1.37 to 1.46 across seeds) and the byte model 2.95 times (2.78 to 3.18), a gap that is 8.5 times at BPE's smallest-budget loss and narrows because the byte law is steeper. On transfer the byte model wins: at its own best validation loss (about 2.55), BPE and BPE-NP runs with the same validation loss score 0.04 to 0.09 bits per byte worse on held-out text, on every seed, and extrapolated with the fitted laws to BPE's 2.30 the gap grows to 0.12 to 0.19. BPE and BPE-NP transfer about equally: at matched validation loss BPE-NP's held-out loss ranges from 0.10 lower to 0.05 higher than BPE's, with no consistent sign. On compute-to-held-out the three come out close, within a factor of about 1.5 either way depending on the seed, because the token models' held-out loss barely improves past 2e12 FLOPs (fitted exponents 0.007 and 0.011, against 0.038 for bytes). Marginalizing over the prompt boundary lowers the token models' held-out scores by 0.14 to 0.18 bits per byte (3 to 4%), most on code. How that compares with the posts. Reproduced in direction: the compute ordering BPE, then BPE-NP, then bytes (the posts: 1.5 and almost 7.2 times; here 1.41, and 3 to 8.5 times depending on the loss, so the closeness of the first number means little at this scale), and bytes as the best transfer at equal validation loss. Not reproduced: pretokenization does not improve transfer here (the posts: 1.26 times on loss-to-loss); the prompt-boundary correction matters as much for BPE as for BPE-NP, while the posts call BPE-NP's uncorrected scores "not even sensible"; BPE-NP compresses no better than BPE at this vocabulary (2.24 bytes per token for both, against 32% more in the posts), so that part of their result is not tested; BPE's lead on compute-to-downstream (1.85 times over BPE-NP and about 4.7 over bytes in the posts) disappears; and off-frontier runs sit 2 to 5% from the implied loss-to-loss curves instead of within about 1%. The house loss-to-loss compute multiplier is ill-conditioned here: with held-out exponents near 0.01, a gap of a tenth of a bit becomes a factor of 24 to 775 for BPE depending on the seed, so the project reports the gap in bits next to it and explains why. Greek and Japanese, scripts the training text barely contains, score 6 to 23 bits per byte without a consistent improvement and are reported apart. Nothing here reproduces the posts' multipliers. The scaffold runs the whole method in miniature in about 30 seconds on the prod image: tokenizers with 256 merges, one model size with 48-position windows, three budgets up to 1.6e10 FLOPs (a few dozen training steps each) and one seed. It shows every piece working; its numbers are noisy and its ranking can differ from the sweep and from the posts.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_split
- [x] **2.** bits_per_byte
- [x] **3.** gpt2_pieces
- [x] **4.** to_stream
- [x] **5.** count_pairs
- [x] **6.** merge_pair
- [x] **7.** train_bpe
- [x] **8.** encode_stream
- [x] **9.** make_tokenizer
- [x] **10.** encode_many
- [x] **11.** bytes_per_token
- [x] **12.** toy_rung
- [x] **13.** train_flops_per_token
- [x] **14.** plan_run
- [x] **15.** init_params
- [x] **16.** forward
- [x] **17.** wsd_lr
- [x] **18.** TinyLM
- [x] **19.** score_sequences
- [x] **20.** val_bpb
- [x] **21.** make_excerpts
- [x] **22.** covering_leaves
- [x] **23.** answer_plan

---

Built on Deep-ML.
