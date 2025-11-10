# Demo memory walkthrough guide

This guide accompanies `scripts/demo_memory_walkthrough.py` and the toy
transformer assets in `configs/demo_memory_walkthrough.cfg` and
`topologies/GEMM_mnk/demo_tiny_transformer.csv`. It explains how the walkthrough
connects a Hugging Face style transformer block to SCALE-Sim's notion of
operands, addresses, and memory traffic.

## Demo model at a glance

| Fact | Value |
| --- | --- |
| Batch size | 2 |
| Sequence length | 4 tokens |
| Hidden size | 8 |
| Attention heads | 2 |
| Head dimension | 4 |
| Intermediate size | 16 |

The topology CSV expands these numbers into seven GEMM layers. Each row maps to
a familiar transformer sub-layer:

| Layer name | Role | MNK tuple |
| --- | --- | --- |
| `Toy_Q`, `Toy_K`, `Toy_V` | Linear projections that build queries, keys, and values | `M=8`, `N=8`, `K=8` |
| `Toy_AttnScores` | Attention score matmul (`QKᵀ`) | `M=16`, `N=4`, `K=4` |
| `Toy_AttnProj` | Output projection after concatenating heads | `M=8`, `N=8`, `K=8` |
| `Toy_MLP_Up` | Feed-forward expansion | `M=8`, `N=16`, `K=8` |
| `Toy_MLP_Down` | Feed-forward contraction | `M=8`, `N=8`, `K=16` |

SCALE-Sim flattens each `M × K` activation table starting at the IFMAP offset,
allocates `K` weights per output channel starting at the filter offset, and
keeps room for the `M × N` results starting at the OFMAP offset. No numerical
values are generated—only addresses that describe when the systolic array would
fetch or write a word.

## Configuration knobs

The demo configuration keeps the numbers small so the console prints stay
readable:

* `ArrayHeight` × `ArrayWidth` = `4 × 4`: a 16-PE weight-stationary array.
* `IfmapSramSzkB`, `FilterSramSzkB`, `OfmapSramSzkB` = `1 kB`: effectively
  infinite for the toy workload, guaranteeing that capacity is not the limiting
  factor.
* Offsets = `0`, `1000`, `2000`: carve global memory into three non-overlapping
  regions for IFMAPs, filters, and OFMAPs.
* `IfmapSRAMBankBandwidth` = `FilterSRAMBankBandwidth` = `OfmapSRAMBankBandwidth`
  = `1 word/cycle`: this is the bandwidth knob that USER mode constrains.
* `InterfaceBandwidth = CALC`: the walkthrough toggles between CALC and USER at
  runtime so both views are visible without editing the config file.

## What the walkthrough prints

Running `python3 scripts/demo_memory_walkthrough.py` now produces:

1. A configuration explainer that ties each knob to simulator behavior.
2. A topology explainer that translates every CSV row into plain-language
   transformer operations.
3. Operand address matrices for the selected layer (default `Toy_Q`). Each entry
   is a word address or `-1` for padding; you can pick a different layer with
   `--layer <name>`.
4. CALC vs. USER runs that save traces in `outputs_demo_memory/calc_mode/<layer>`
   and `outputs_demo_memory/user_mode/<layer>`, previewing SRAM and DRAM traffic
   for IFMAP, filter, and OFMAP streams.
5. Narrative decoding of the trace tables. The script converts addresses back to
   `(token, feature)` tuples and explains why CALC collapses the warm-up into one
   burst while USER mode stretches it out cycle-by-cycle.
6. A timing comparison that connects the extra idle cycles in USER mode to the
   limited scratchpad bandwidth and describes when Ramulator-driven stalls would
   appear.

Invoke `--layer Toy_MLP_Up` (or any other layer name or index) to inspect a
different GEMM. Combine this with `--matrix-rows` and `--trace-rows` to balance
verbosity against readability.
