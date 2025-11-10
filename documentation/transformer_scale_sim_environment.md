# SCALE-Sim Transformer Exploration Handbook

This handbook compiles everything you need to follow a transformer-style GEMM
from the topology CSV down to the SRAM/DRAM traces that SCALE-Sim emits. It
pairs model-centric intuition (where the M/N/K numbers come from) with the
exact Python entry points responsible for address generation, memory-system
modeling, and reporting. Use it as the “book” for MNK workloads: skim it before
building a new config, keep it handy while reading traces, and jump to the file
references when you want to inspect or modify simulator internals.

---

## 1. Assets that describe the workload and hardware

| Asset | Purpose | Where it lives | Notes |
| --- | --- | --- | --- |
| **Topology CSV** | Lists transformer layers as GEMMs (`Layer,M,N,K[,sparsity]`). | `topologies/GEMM_mnk/*.csv` | Only tensor shapes are needed; SCALE-Sim turns them into convolution-style metadata on load.【F:scalesim/topology_utils.py†L62-L117】 |
| **Configuration file** | Declares systolic array geometry, dataflow, scratchpad sizes, bandwidth assumptions, and layout/banking knobs. | `configs/*.cfg` | Parsed into a `scale_config` object that every layer reuses.【F:scalesim/scale_config.py†L22-L180】 |
| **Optional layout CSV** | Overrides default banking/layout per layer. | `layouts/*.csv` | Enables experiments with custom interleaving when `[layout]` flags are on.【F:scalesim/layout_utils.py†L16-L149】 |

When you run `python -m scalesim.scale -c <cfg> -t <csv> -i gemm`, the CLI feeds
these three assets into the orchestrator so the simulator knows *what* to run
and *which hardware* to emulate.【F:scalesim/scale.py†L1-L53】

---

## 2. From MNK triples to operand address matrices

1. **Topology parsing.** GEMM rows first enter `load_arrays_gemm`, which expands
   each `M,N,K` triple into SCALE-Sim’s internal convolution vocabulary (ifmap
   rows/cols, filter height/width, channel counts). This ensures the rest of the
   pipeline can treat GEMMs and convolutions uniformly.【F:scalesim/topology_utils.py†L62-L161】
2. **Operand matrix construction.** `operand_matrix.build_ifmap_matrix` and
   friends then compute address tables. Addresses are contiguous integers with
   operand-specific base offsets (`IFMAP_BASE=0`, `FILTER_BASE=1000`,
   `OFMAP_BASE=2000` in the walkthrough demos) so every lookup becomes “fetch
   address X” rather than “fetch value Y.”【F:scalesim/compute/operand_matrix.py†L42-L200】
3. **No real tensors required.** Because only addresses are generated, the
   simulator can reproduce transformer memory traffic using shapes from a model
   card or `config.json`. There is no need to download GPT-2 weights or token
   embeddings to evaluate hardware behavior.

> **Mental model.** Think of each operand matrix row as a *token position* and
> each column as a *tap or channel*. A `-1` entry means “padding / not used,”
> which happens when the physical array is larger than the logical tile.

---

## 3. Configuration deep dive (what to watch for transformers)

### 3.1 Array and dataflow

- `ArrayHeight` / `ArrayWidth` fix the physical systolic tile. Transformers with
  tall-but-skinny GEMMs (e.g., long sequences) benefit from a balanced footprint
  so utilization stays high.【F:configs/scale.cfg†L5-L17】
- `Dataflow` selects which operand stays stationary—`ws` keeps weights in place
  (great for large weight reuse), `os` keeps outputs, and `is` keeps inputs.
  The flag determines which compute engine class `single_layer_sim` instantiates
  for every layer.【F:scalesim/single_layer_sim.py†L186-L220】

### 3.2 Scratchpad sizing and partitioning

- `IfmapSramSzkB`, `FilterSramSzkB`, `OfmapSramSzkB` size the double-buffered
  scratchpads that sit between DRAM and the systolic core. Bigger scratchpads
  hide more latency and support wider prefetch bursts.【F:configs/scale.cfg†L5-L17】【F:scalesim/memory/double_buffered_scratchpad_mem.py†L18-L139】
- `IfmapOffset`, `FilterOffset`, `OfmapOffset` define non-overlapping base
  addresses. These are the numbers the operand matrices add to their element
  indices, producing the absolute addresses you see in traces.【F:scalesim/compute/operand_matrix.py†L42-L123】

### 3.3 Memory bandwidth and layout

- `[run_presets] InterfaceBandwidth = CALC|USER` flips between analytical and
  user-supplied bandwidth. `CALC` swaps in `ReadBufferEstimateBw` to report the
  bandwidth required to keep the array busy; `USER` keeps your GB/s values and
  accumulates stall cycles if demand outruns supply.【F:configs/scale.cfg†L33-L36】【F:scalesim/memory/double_buffered_scratchpad_mem.py†L70-L138】
- `[layout]` exposes SRAM banking knobs: bank counts, ports, and bandwidth per
  bank. These numbers directly initialize read/write buffers so you can evaluate
  conflict-induced bubbles or multi-bank scaling.【F:configs/scale.cfg†L19-L28】【F:scalesim/memory/read_buffer.py†L16-L190】

### 3.4 Optional sparsity plumbing

- Set `SparsitySupport` to `Y` and provide ratios (e.g., `4:8`) in the topology
  to activate metadata generation. `single_layer_sim` forwards these options to
  the compression estimators so reports include metadata footprint alongside data
  traffic.【F:configs/scale.cfg†L29-L32】【F:scalesim/single_layer_sim.py†L157-L204】

---

## 4. Execution timeline (what runs when)

1. **CLI bootstraps the run.** `scalesim.scale` loads the config/topology/layout,
   instantiates `scale_sim`, and hands off to `simulator.simulator` for the layer
   loop.【F:scalesim/scale.py†L1-L53】【F:scalesim/scale_sim.py†L1-L140】
2. **Per-layer setup.** For each topology entry, `single_layer_sim`:
   - Builds IFMAP/FILTER/OFMAP matrices (address generation).【F:scalesim/single_layer_sim.py†L186-L206】
   - Picks the right `systolic_compute_*` engine based on dataflow.
   - Requests demand/prefetch matrices that spell out which addresses must be
     present on which cycle.【F:scalesim/compute/systolic_compute_ws.py†L20-L210】
   - Configures double-buffered scratchpads and, if requested, Ramulator traces.
3. **Service loop.** The double buffers interleave compute consumption with
   prefetch requests. Each SRAM read buffer tracks what addresses it holds; when
   the compute engine asks for something missing, it triggers a DRAM burst and
   blocks until the data arrives, accumulating stall cycles if necessary.【F:scalesim/memory/double_buffered_scratchpad_mem.py†L18-L139】
4. **Reporting and teardown.** Once demand matrices are exhausted, the layer
   flushes OFMAP writes, emits SRAM/DRAM trace CSVs, and stores cycle/bandwidth
   statistics for the global reports.【F:scalesim/single_layer_sim.py†L236-L332】
5. **Run-level aggregation.** After the final layer, `simulator.generate_reports`
   writes `COMPUTE_REPORT.csv`, `BANDWIDTH_REPORT.csv`, `DETAILED_ACCESS_REPORT.csv`,
   and optional sparsity summaries into the run directory.【F:scalesim/simulator.py†L25-L178】

---

## 5. Understanding the memory system

### 5.1 What “double buffered” means

- Every operand gets two halves in scratchpad memory: **active** (feeding the
  array) and **prefetch** (loading the next tile). When the active half empties,
  the roles swap. This overlap hides DRAM latency as long as bandwidth is
  sufficient.【F:scalesim/memory/double_buffered_scratchpad_mem.py†L18-L139】
- The read buffer tracks `buffer_content` (addresses resident), launch/completion
  cycles for outstanding bursts, and metadata about bank occupancy to model
  conflicts.【F:scalesim/memory/read_buffer.py†L16-L190】

### 5.2 Where addresses originate

- Operand matrices deliver **absolute addresses** (base offset + index). They do
  not contain tensor values. The scratchpad compares these integers against its
  active set to decide whether a read hits or misses. This is why you can trace
  every DRAM transaction back to specific rows/columns in the operand matrices.【F:scalesim/compute/operand_matrix.py†L42-L200】

### 5.3 DRAM interaction and bandwidth modes

- **CALC mode:** Instead of obeying a fixed GB/s cap, read buffers solve for the
  minimum prefetch rate that would keep the array fed. Resulting bandwidths
  appear in the reports; stall cycles stay at zero because requests are assumed
  to arrive just in time.【F:scalesim/memory/double_buffered_scratchpad_mem.py†L70-L138】
- **USER mode:** Bandwidth caps come from the config. If the array requests data
  faster than DRAM can deliver it, `single_layer_sim` increments stall counters
  and extends the layer’s total cycle count accordingly.【F:scalesim/single_layer_sim.py†L236-L332】
- **Ramulator integration:** Provide Ramulator traces and the simulator will use
  their latency numbers instead of the analytical model, injecting realistic
  contention-induced stalls. Hooks live in the scratchpad class alongside the
  CALC/USER logic.【F:scalesim/memory/double_buffered_scratchpad_mem.py†L70-L138】

### 5.4 What the traces show

- **SRAM traces (`*_SRAM_TRACE.csv`).** Each row starts with the cycle number,
  followed by one column per array column (reads) or row (writes). Entries are
  the addresses consumed or produced in that cycle; `-1` denotes “no request,”
  which is common during array warm-up or diagonal wavefront fill.
- **DRAM traces (`*_DRAM_TRACE.csv`).** Column 0 is the completion cycle of a
  burst. Negative cycles indicate prefetches that completed before compute cycle
  0 to warm the active buffer. Remaining columns list the addresses delivered in
  that burst, grouped according to the configured bandwidth (e.g., how many
  words per cycle).【F:scalesim/single_layer_sim.py†L236-L300】

---

## 6. Following a transformer token through the system

1. **Token enters the layer.** A row in the topology CSV (say, `GPT2_FC1`) tells
   the simulator there are `M = batch×sequence` tokens, each with `K = hidden`
   features, producing `N = intermediate` outputs. Operand matrices map these to
   address grids so every token’s features correspond to specific IFMAP
   addresses.【F:scalesim/topology_utils.py†L62-L161】【F:scalesim/compute/operand_matrix.py†L86-L200】
2. **Weights are staged.** In weight-stationary mode (`ws`), the filter matrix
   addresses get loaded into scratchpad first. The prefetch half issues DRAM
   bursts until the active half contains the columns needed for the next tile.
3. **Activations stream in.** As the array begins compute, each cycle pulls IFMAP
   addresses from SRAM. If an address is missing, the read buffer pauses the
   column, triggers a DRAM fetch, and either inserts stall cycles (USER mode) or
   extends the idle gap before compute starts (CALC mode).
4. **Partial sums accumulate.** OFMAP write buffers collect output addresses and
   hold partial sums until a tile finishes. Once a buffer is full or the layer
   ends, the write buffer flushes to DRAM (recorded in `OFMAP_DRAM_TRACE.csv`).
5. **Reports record the story.** The compute report lists total cycles, stall
   cycles, idle time, and utilization. The bandwidth report confirms average
   SRAM/DRAM demand. Detailed access reports summarize how many reads/writes each
   operand generated. Together they tell you whether the layer was compute-bound
   or memory-bound and which operand caused trouble.【F:scalesim/single_layer_sim.py†L236-L332】【F:scalesim/simulator.py†L25-L178】

---

## 7. Essential source files (and what to inspect inside them)

| Topic | File | Highlights |
| --- | --- | --- |
| CLI orchestration | `scalesim/scale.py` | Entry point; parse arguments, instantiate simulator.【F:scalesim/scale.py†L1-L53】 |
| Topology translation | `scalesim/topology_utils.py` | Converts GEMM CSV rows into convolution-style parameters and sparsity annotations.【F:scalesim/topology_utils.py†L62-L161】 |
| Operand matrices | `scalesim/compute/operand_matrix.py` | Generates IFMAP/FILTER/OFMAP address grids; study how offsets and strides build absolute addresses.【F:scalesim/compute/operand_matrix.py†L42-L200】 |
| Dataflow-specific compute engines | `scalesim/compute/systolic_compute_ws.py`, `_os.py`, `_is.py` | Schedule operand arrivals for each dataflow; inspect diagonal wavefront logic and demand-matrix creation.【F:scalesim/compute/systolic_compute_ws.py†L20-L210】 |
| Memory model | `scalesim/memory/double_buffered_scratchpad_mem.py`, `read_buffer.py`, `write_buffer.py` | Implement double buffering, bandwidth throttling, banking conflicts, and optional Ramulator latency overrides.【F:scalesim/memory/double_buffered_scratchpad_mem.py†L18-L139】【F:scalesim/memory/read_buffer.py†L16-L190】 |
| Layer orchestration | `scalesim/single_layer_sim.py` | Glues topology, compute, and memory; emits traces and reports.【F:scalesim/single_layer_sim.py†L186-L332】 |
| Run aggregation | `scalesim/simulator.py` | Manages layer loop, summarizes metrics, writes final CSV reports.【F:scalesim/simulator.py†L25-L178】 |

---

## 8. Practical workflow for MNK transformer studies

1. **Extract MNK values** from the target model’s configuration (batch size,
   sequence length, hidden size, intermediate size, number of heads). Fill a
   GEMM CSV—no tensors required.
2. **Start with CALC bandwidth** to understand ideal requirements. Examine the
   bandwidth report to see IFMAP vs. filter pressure; verify stall cycles stay at
   zero.【F:scalesim/memory/double_buffered_scratchpad_mem.py†L70-L138】
3. **Switch to USER bandwidth** with realistic DRAM/SRAM limits. Compare total
   cycles and stall counts against the CALC run to identify bottlenecks.
4. **Dive into SRAM/DRAM traces** for layers exhibiting stalls or low
   utilization. Cross-reference addresses against operand matrices to see which
   tokens or channels caused thrashing.
5. **Iterate on the config**: adjust array geometry, scratchpad sizes, bank
   counts, or dataflow. Re-run and monitor utilization, bandwidth, and stall
   metrics until the architecture balances compute and memory demands.

With this handbook you can trace every step of the SCALE-Sim pipeline, know
exactly where addresses come from, understand how the memory system feeds the
systolic array, and pinpoint the Python modules responsible for each decision.
Whether you are validating an existing GPT-2 configuration or designing a new
transformer accelerator, the references above highlight the levers and code
paths that matter most.
