"""
From Tokens to Bytes: Compute and Transfer Multipliers from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""From Tokens to Bytes, in miniature: the whole method end to end in about half a minute on one CPU.

This run is tiny next to the sweep the project description reports (four sizes, five budgets up to 1.6e13 FLOPs, three
seeds, 256-byte and 96-token windows): here the tokenizers get 256 merges learned on 200 KB, every model is L1 and reads
48-position windows (48 bytes for the byte model, about 90 bytes for the token models), the three budgets of 4e9 to
1.6e10 FLOPs buy a few dozen training steps each, and there is one seed and a handful of held-out excerpts. Its numbers are
noisy, and its ranking can differ from the sweep and from the posts. The printout reports what this run measured."""
from solution import *
import time
import numpy as np

T0 = time.time()
SEQ = {"byte": 48, "bpe": 48, "bpe_np": 48}   # one window length, so all three models share compiled kernels
BATCH = 32
N_MERGES = 256
SIZES = [1]
BUDGETS = [4e9, 8e9, 1.6e10]
SEED = 0
ENCODINGS = ("byte", "bpe", "bpe_np")
EVAL7 = ["fi", "hu", "ocaml", "prolog", "lean", "fortran", "haskell"]   # held-out scripts that occur in training


def lr_rule(rung):
    # the sweep's rule, calibrated at 4e12 FLOPs; at L1 it gives 6e-3
    return 6e-3 * (rung["d_model"] / 32) ** -1.5


def head(blobs, n_bytes):
    """The first n_bytes of every file, cut at a document boundary, joined into one blob."""
    parts = []
    for blob in blobs.values():
        cut = blob[:n_bytes]
        parts.append(cut[: cut.rfind(b"\x1e")] if b"\x1e" in cut else cut)
    return b"\x1e".join(parts)


train, val, held = load_split("train"), load_split("val"), load_split("heldout")
weights = {d: len(b) for d, b in train.items()}

print("1. Three encodings of the same bytes")
sample = head(train, 50_000)
toks = {"byte": make_tokenizer("byte", [], False)}
for name, split_fn in (("bpe", gpt2_pieces), ("bpe_np", None)):
    toks[name] = make_tokenizer(name, train_bpe(*to_stream(sample, split_fn), N_MERGES), split_fn is not None)
crossing = [v for v in toks["bpe_np"]["vocab"][256:] if b" " in v.strip(b" ")]
print(f"   {N_MERGES} merges each on {len(sample):,} bytes; BPE-NP tokens with a space inside: {len(crossing)}, e.g. {crossing[:4]}")
bpt = {n: {**bytes_per_token(toks[n], val, 20_000), **bytes_per_token(toks[n], held, 20_000)} for n in ("bpe", "bpe_np")}
print("   bytes per token, BPE / BPE-NP: " + ", ".join(f"{d} {bpt['bpe'][d]:.2f}/{bpt['bpe_np'][d]:.2f}" for d in list(val) + EVAL7))
streams = {}
for n in ENCODINGS:
    split_fn = gpt2_pieces if toks[n]["pretok"] else None
    parts = [encode_stream(*to_stream(b[:400_000], split_fn), toks[n]["merges"]) for b in train.values()]
    streams[n] = np.concatenate([np.concatenate([p, [30]]) for p in parts])[:-1].astype(np.int32)
print("   training streams (first 400 KB of each domain): " + ", ".join(f"{n} {len(streams[n]):,}" for n in ENCODINGS) + " ids")
val_chunks = {d: [p for p, _ in make_excerpts(b, 6, 48, 0, seed=1)] for d, b in val.items()}
held_ex = {d: make_excerpts(held[d], 3, 32, 16, seed=2) for d in EVAL7}
plans = {n: {d: answer_plan(tok, ex) for d, ex in held_ex.items()} for n, tok in toks.items()}
print(f"   evaluation: 6 chunks of 48 bytes per validation domain; 3 prompts of 32 bytes with 16-byte answers in each of "
      f"{len(EVAL7)} held-out languages  [{time.time() - T0:.0f} s]")

print(f"\n2. Training L1 models, batch {BATCH}, budgets {', '.join(f'{b:.1e}' for b in BUDGETS)} FLOPs, seed {SEED}")
print(f"   {'encoding':8s}{'budget':>9s}{'steps':>7s}{'val bpb':>9s}{'held-out':>10s}{'canonical':>11s}")
runs = {}
for n in ENCODINGS:
    runs[n] = iso_flop_sweep(toks[n], streams[n], SIZES, BUDGETS, SEED, lr_rule, val_chunks, plans[n], SEQ[n], BATCH,
                             val_weights=weights, eval_domains=EVAL7)
    for r in runs[n]:
        print(f"   {n:8s}{r['budget']:9.1e}{r['steps']:7d}{r['val']:9.3f}{r['eval']:10.3f}{r['eval_canonical']:11.3f}")
print("   'held-out' is the marginal answer score, 'canonical' the standard one (the same for bytes)")

print("\n3. Measured at equal compute, the largest budget")
last = {n: runs[n][-1] for n in ENCODINGS}
order = sorted(ENCODINGS, key=lambda n: last[n]["val"])
print("   validation, best first: " + ", ".join(f"{n} {last[n]['val']:.3f}" for n in order))
order = sorted(ENCODINGS, key=lambda n: last[n]["eval"])
print("   held-out, best first:   " + ", ".join(f"{n} {last[n]['eval']:.3f}" for n in order))
for n in ("bpe", "bpe_np"):
    g = sum(r["eval_canonical"] - r["eval"] for r in runs[n]) / len(runs[n])
    print(f"   marginal scoring lowers {n}'s held-out bits per byte by {g:.3f} on average over its runs")


def show(m):
    return f"{m:.2f}x" if 0.01 < m < 1000 else "n/a"   # a law with an exponent near zero makes inversions meaningless


print("\n4. The project's analysis on these runs: lines through three points, read at BPE's last validation loss")
try:
    s = summarize(runs)
except (OverflowError, ValueError) as e:
    s = None
    print(f"   these runs gave a law too flat to invert ({e})")
if s is not None:
    x0 = s["val_loss"]
    for n in ENCODINGS:
        f, r = s["families"][n], s["rows"][n]
        vals = [x["val"] for x in runs[n]]
        where = "inside" if min(vals) <= x0 <= max(vals) else "outside"
        print(f"   {n:8s} alpha val {f['val']['alpha']:.3f}, held-out {f['eval']['alpha']:.3f} | compute vs BPE {show(r['m_val']):>8s} | "
              f"implied held-out {implied_eval(f, x0):.3f} | loss-to-loss vs byte {show(r['l2l_vs_ref']):>8s} "
              f"(x = {x0:.3f} is {where} its measured range {min(vals):.3f}-{max(vals):.3f})")
    print("   Readings outside a family's measured range are extrapolations of a three-point line; at this size they can be")
    print(f"   off by orders of magnitude. By the fitted lines: least compute {s['best_compute']}, best transfer {s['best_transfer']}.")

print(f"\nTotal {time.time() - T0:.0f} s. One size, a few dozen steps per run, one seed: expect noise. The description reports the")
print("three-seed sweep (180 runs up to 1.6e13 FLOPs), where BPE wins on compute and bytes transfer best at equal validation")
print("loss; nothing in either reproduces the posts' multipliers.")

