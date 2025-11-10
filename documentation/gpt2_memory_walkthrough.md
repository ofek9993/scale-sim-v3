# GPT-2 memory walkthrough

The GPT-2 topology shipped with SCALE-Sim captures the dense GEMMs that dominate a single
transformer block. This walkthrough couples that topology with `configs/google.cfg`
(the 256×256 output-stationary TPU v1 array) and explains how SCALE-Sim turns the CSV
entries into operand address matrices, scratchpad traffic, and DRAM bursts.

## Why `google.cfg`?

`google.cfg` provisions a 256×256 array with multi-megabyte scratchpads and a
10-word-per-cycle DRAM interface. That matches the scale of GPT-2's 1024-token
sequence length and keeps the discussion focused on how the simulator schedules
activations and weights rather than on SRAM capacity limits.

## Script overview

`scripts/gpt2_memory_walkthrough.py` drives the simulator layer by layer. For each
GEMM it:

1. Derives GPT-2-specific facts (sequence length, head dimension, number of heads,
   hidden size, and intermediate size) directly from the CSV.
2. Prints the operand address matrices along with decoded samples so you can see
   which tokens and channels each word represents.
3. Runs the layer twice—once in CALC mode (ideal bandwidth) and once with the
   user-provided 10 word/cycle interface—to highlight how double-buffer prefetches
   stretch the timeline without changing compute cycles.
4. Annotates the first SRAM and DRAM trace entries so you can follow how the
   systolic array consumes activations, weights, and outputs.

Use the `--layers` flag to focus on a subset of rows. Increase `--matrix-cols` if
you want to see more columns of each operand window. For example, to examine the
MLP expansion:

```bash
python3 scripts/gpt2_memory_walkthrough.py \
  --layers PW-FF-L1 \
  --matrix-rows 3 \
  --trace-rows 4
```

## Sample output excerpt (`--layers QKT --matrix-rows 1 --matrix-cols 6 --trace-rows 1`)

```
=== Layer 0: QKT ===
GEMM layout: M=1024 rows, K=64 columns, N=1024 outputs per token.
IFMAP address matrix (showing first 1 row × 6 columns): 0 1 2 3 4 5 ...
Filter address matrix (showing first 1 row × 6 columns): 10000000 10000064 ...
Sample IFMAP addresses → decoded tokens/features:
  0: token 0, feature 0
  1: token 0, feature 1

-- CALC (ideal bandwidth) run --
CALC mode cycles: total 25861, compute 9183, stall 0, idle window 16678
  IFMAP DRAM warm-up bursts:
  Burst finishing at cycle -12583: pulled 10 word(s) → A[token=0, k=0], A[token=0, k=1], ...

-- USER (configured bandwidth) run --
USER mode cycles: total 120594, compute 9183, stall 0, idle window 111411
  IFMAP DRAM warm-up bursts:
  Burst finishing at cycle -6554: pulled 10 word(s) → A[token=0, k=0], A[token=0, k=1], ...
Difference summary: compute time stays at 9183 cycles, but USER mode introduces an additional 94733 warm-up/flush cycles...
```

The excerpt shows how the ideal CALC run grabs entire activation panels in wide
bursts, whereas the 10 word/cycle USER mode stretches those prefetches over a
longer warm-up window.

## Next steps

Run the script without `--layers` to narrate the entire block (attention score,
context accumulation, packed QKV projection, projection back to the model
dimension, and the two MLP passes). Increase `--matrix-rows` or `--trace-rows` to
inspect deeper portions of the operand matrices and trace timelines.
