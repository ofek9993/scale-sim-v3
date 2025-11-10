# GPT-2 SCALE-Sim memory walkthrough

=== Hardware backdrop: google.cfg ===
ArrayHeight × ArrayWidth = 256 × 256. This is the TPU v1-style systolic core with 65,536
multiply-accumulate cells, large enough to process GPT-2's 1024-token minibatch with high
utilization.

Scratchpad sizes: IFMAP 6144 kB, FILTER 6144 kB, OFMAP 2048 kB. These multi-megabyte buffers
easily hold the M×K and K×N panels we stream for each GEMM stage, so in this walkthrough we
focus on bandwidth rather than capacity limits.

Offsets carve global DRAM into three regions: IFMAP starts at 0, FILTER at 10000000, and OFMAP
at 20000000. SCALE-Sim tags every operand address relative to those anchors.

Dataflow = OS. The TPU config keeps outputs stationary, so partial sums stay resident in OFMAP
scratchpad banks while activations and weights stream in. The simulator will therefore highlight
OFMAP SRAM activity when accumulations roll across the array.

InterfaceBandwidth = USER with configured bandwidth(s) 10 words/cycle. We'll compare that
practical limit to CALC mode, which solves for the ideal bandwidth that would keep this array
100% busy.

=== GPT-2 block dimensions inferred from the CSV ===
 • Sequence length: 1024 tokens
 • Batch size: 1 (implicit in the M dimension)
 • Hidden size: 1600
 • Head dimension: 64
 • Number of attention heads: 25
 • Intermediate MLP size: 3072

=== Matching CSV rows to GPT-2 math ===
The CSV lists one transformer block's dense GEMMs. M corresponds to sequence length (batch ×
tokens), K captures the incoming feature width, and N is the number of outputs each layer emits
per token.

Layer 0: QKT
  Role: Attention score GEMM (per head): queries (M rows) × key vectors (K width) produce sequence-length attention logits (N columns).
  M = 1024 → processing 1024 token vectors in parallel (batch × sequence).
  K = 64 → each token contributes 64 input features. For this layer that equals the attention
  head dimension 64.
  N = 1024 → the layer emits 1024 features per token, i.e. sequence-length attention logits (N
  = 1024) for each query token.

Layer 1: QKTV
  Role: Context accumulation (per head): the attention probabilities multiply value vectors so each token receives a weighted context.
  M = 1024 → processing 1024 token vectors in parallel (batch × sequence).
  K = 1024 → each token contributes 1024 input features. For this layer that equals the
  sequence length 1024 captured by the score matrix.
  N = 64 → the layer emits 64 features per token, i.e. 64 value features per head (head_dim).

Layer 2: Linear1
  Role: c_attn projection: the model dimension (K) fans out to 3× hidden_size outputs that hold Q, K, and V packed together.
  M = 1024 → processing 1024 token vectors in parallel (batch × sequence).
  K = 1600 → each token contributes 1600 input features. For this layer that equals the hidden
  size 1600.
  N = 4800 → the layer emits 4800 features per token, i.e. 3 × hidden_size outputs (4800)
  packing Q, K, and V.

Layer 3: Linear2
  Role: c_proj projection: mixes the concatenated head outputs back into the model dimension.
  M = 1024 → processing 1024 token vectors in parallel (batch × sequence).
  K = 1600 → each token contributes 1600 input features. For this layer that equals the hidden
  size 1600.
  N = 1600 → the layer emits 1600 features per token, i.e. hidden-size outputs (1600).

Layer 4: PW-FF-L1
  Role: MLP expansion: grows each token from hidden_size to the intermediate GeLU width.
  M = 1024 → processing 1024 token vectors in parallel (batch × sequence).
  K = 1600 → each token contributes 1600 input features. For this layer that equals the hidden
  size 1600.
  N = 3072 → the layer emits 3072 features per token, i.e. the intermediate GeLU width (3072).

Layer 5: PW-FF-L2
  Role: MLP contraction: projects the GeLU activations back to the hidden_size output.
  M = 1024 → processing 1024 token vectors in parallel (batch × sequence).
  K = 3072 → each token contributes 3072 input features. For this layer that equals the
  intermediate MLP size 3072.
  N = 1600 → the layer emits 1600 features per token, i.e. the hidden size 1600.

=== Layer 0: QKT ===
GEMM layout: M=1024 rows, K=64 columns, N=1024 outputs per token.

IFMAP activations start at address 0 and occupy 65536 words. Row m uses addresses [0 + m×64, 0 +
m×64 + 63] for its K input features.

FILTER weights start at 10000000. Output channel n owns a contiguous block of K weights starting
at 10000000 + n×64.

OFMAP results begin at 20000000. Token m, output n lives at 20000000 + m×1024 + n_idx.

IFMAP address matrix (showing first 1 of 1024 rows and first 6 of 64 columns):
        0       1       2       3       4       5 ...
  ...

Filter address matrix (showing first 1 of 64 rows and first 6 of 1024 columns):
  10000000 10000064 10000128 10000192 10000256 10000320 ...
  ...

OFMAP address matrix (showing first 1 of 1024 rows and first 6 of 1024 columns):
  20000000 20000001 20000002 20000003 20000004 20000005 ...
  ...

Sample IFMAP addresses → decoded tokens/features:
  0: token 0, feature 0
  1: token 0, feature 1
  2: token 0, feature 2
  3: token 0, feature 3
  4: token 0, feature 4
  5: token 0, feature 5

Sample FILTER addresses → decoded output-channel weights:
  10000000: output channel 0, weight 0
  10000064: output channel 1, weight 0
  10000128: output channel 2, weight 0
  10000192: output channel 3, weight 0
  10000256: output channel 4, weight 0
  10000320: output channel 5, weight 0

Sample OFMAP addresses → decoded output slots:
  20000000: token 0, output channel 0
  20000001: token 0, output channel 1
  20000002: token 0, output channel 2
  20000003: token 0, output channel 3
  20000004: token 0, output channel 4
  20000005: token 0, output channel 5

-- CALC (ideal bandwidth) run --
CALC mode cycles: total 25861, compute 9183, stall 0, idle window 16678
  Array utilization 11.15% | Mapping efficiency 100.00% | Compute utilization 11.15%
  IFMAP SRAM timeline snippet:
  Cycle 1: array consumed 1 IFMAP word(s): A[token=0, k=0]
  IFMAP DRAM warm-up bursts:
  Burst finishing at cycle -12583: pulled 10 word(s) → A[token=0, k=0], A[token=0, k=1], A[token=0, k=2], A[token=0, k=3], A[token=0, k=4], A[token=0, k=5], A[token=0, k=6], A[token=0, k=7], ...

  FILTER SRAM timeline snippet:
  Cycle 1: array consumed 1 FILTER word(s): W[channel=0, k=0]
  FILTER DRAM warm-up bursts:
  Burst finishing at cycle -12583: pulled 10 word(s) → W[channel=0, k=0], W[channel=0, k=1], W[channel=0, k=2], W[channel=0, k=3], W[channel=0, k=4], W[channel=0, k=5], W[channel=0, k=6], W[channel=0, k=7], ...

  OFMAP SRAM timeline snippet:
  OFMAP DRAM warm-up bursts:
  Burst finishing at cycle 9183: pulled 256 word(s) → O[token=255, n=0], O[token=254, n=0], O[token=255, n=1], O[token=253, n=0], O[token=254, n=1], O[token=255, n=2], O[token=252, n=0], O[token=253, n=1], ...

-- USER (configured bandwidth) run --
USER mode cycles: total 120594, compute 9183, stall 0, idle window 111411
  Array utilization 11.15% | Mapping efficiency 100.00% | Compute utilization 11.15%
  IFMAP SRAM timeline snippet:
  Cycle 1: array consumed 1 IFMAP word(s): A[token=0, k=0]
  IFMAP DRAM warm-up bursts:
  Burst finishing at cycle -6554: pulled 10 word(s) → A[token=0, k=0], A[token=0, k=1], A[token=1, k=0], A[token=0, k=2], A[token=1, k=1], A[token=2, k=0], A[token=0, k=3], A[token=1, k=2], ...

  FILTER SRAM timeline snippet:
  Cycle 1: array consumed 1 FILTER word(s): W[channel=0, k=0]
  FILTER DRAM warm-up bursts:
  Burst finishing at cycle -6554: pulled 10 word(s) → W[channel=0, k=0], W[channel=0, k=1], W[channel=1, k=0], W[channel=0, k=2], W[channel=1, k=1], W[channel=2, k=0], W[channel=0, k=3], W[channel=1, k=2], ...

  OFMAP SRAM timeline snippet:
  OFMAP DRAM warm-up bursts:
  Burst finishing at cycle 9183: pulled 10 word(s) → O[token=255, n=0], O[token=254, n=0], O[token=255, n=1], O[token=253, n=0], O[token=254, n=1], O[token=255, n=2], O[token=252, n=0], O[token=253, n=1], ...

Difference summary: compute time stays at 9183 cycles, but USER mode introduces an additional
94733 warm-up/flush cycles because the configured bandwidth feeds the double buffers more slowly
than the ideal CALC analysis.

