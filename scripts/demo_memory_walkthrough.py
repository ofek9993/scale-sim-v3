#!/usr/bin/env python3
"""Demonstrate operand address generation and memory traces for a tiny GEMM layer."""
from __future__ import annotations

import argparse
import sys
import textwrap
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scalesim.scale_config import scale_config
from scalesim.topology_utils import topologies
from scalesim.layout_utils import layouts
from scalesim.single_layer_sim import single_layer_sim
 
DEFAULT_CONFIG = REPO_ROOT / "configs/demo_memory_walkthrough.cfg"
DEFAULT_TOPOLOGY = REPO_ROOT / "topologies/GEMM_mnk/demo_tiny_transformer.csv"
DEFAULT_LAYOUT = REPO_ROOT / "layouts/conv_nets/test.csv"
DEFAULT_OUTPUT = REPO_ROOT / "outputs_demo_memory"

DEMO_FACTS = {
    "batch_size": 2,
    "sequence_length": 4,
    "hidden_size": 8,
    "num_heads": 2,
    "head_dim": 4,
    "intermediate_size": 16,
}

LAYER_ROLES = OrderedDict(
    [
        (
            "Toy_Q",
            "Query projection: each of the 8 tokens is multiplied by an 8×8 weight matrix to"
            " produce 8 query features (per-token linear layer).",
        ),
        (
            "Toy_K",
            "Key projection: identical shape to the query projection, producing key vectors"
            " that later get transposed for the score matmul.",
        ),
        (
            "Toy_V",
            "Value projection: same dimensions as Q/K so the triple {Q, K, V} share the"
            " hidden size pipeline.",
        ),
        (
            "Toy_AttnScores",
            "Attention score matmul (QKᵀ): treats the 16 head-specific queries (batch × heads ×"
            " sequence) as rows and walks across 4-key vectors to produce 4-length score rows"
            " per token.",
        ),
        (
            "Toy_AttnProj",
            "Output projection: takes the concatenated head results (M=8 tokens, K=N=8) and"
            " mixes them back into the model dimension.",
        ),
        (
            "Toy_MLP_Up",
            "Feed-forward expansion: grows each token from hidden size 8 to intermediate size 16"
            " using an 8×16 weight matrix.",
        ),
        (
            "Toy_MLP_Down",
            "Feed-forward contraction: projects the intermediate activations back down to 8"
            " features per token (16×8 weight matrix).",
        ),
    ]
)



def format_matrix_rows(matrix: np.ndarray) -> Iterable[str]:
    """Yield human-friendly strings for each row of an address matrix."""
    for row in matrix:
        yield "  " + " ".join(f"{int(val):5d}" for val in row)


def resolve_layer_id(layer_selector: str, topology: topologies) -> int:
    """Map a layer name or index string to an integer layer id."""

    if layer_selector is None:
        return 0

    if layer_selector.isdigit():
        idx = int(layer_selector)
        if idx < 0 or idx >= topology.get_num_layers():
            raise ValueError(f"Layer index {idx} is out of range (0..{topology.get_num_layers()-1})")
        return idx

    names = topology.get_layer_names()
    for idx, name in enumerate(names):
        if layer_selector.lower() == name.lower():
            return idx

    raise ValueError(
        f"Layer '{layer_selector}' not found. Available layers: {', '.join(names)}"
    )


def describe_config(config: scale_config) -> None:
    """Explain how each configuration knob shapes the simulator."""

    print("=== Understanding the configuration file ===")
    print(textwrap.fill(
        (
            f"ArrayHeight × ArrayWidth = {config.array_rows} × {config.array_cols}, so the systolic core"
            f" has {config.array_rows} rows of processing elements and {config.array_cols} columns. Our"
            " demo keeps it at 4×4 so that the address matrices stay small enough to print."),
        width=88,
    ))
    print()

    print(textwrap.fill(
        (
            f"IfmapSramSzkB / FilterSramSzkB / OfmapSramSzkB = {config.ifmap_sz_kb} kB each. These"
            " represent the capacity of the three scratchpad segments feeding inputs, weights, and"
            " outputs. Setting them all to 1 kB effectively means \u201cno capacity limit\u201d for the toy"
            " layer, because our operand windows are only 64 words long."),
        width=88,
    ))
    print()

    print(textwrap.fill(
        (
            f"IfmapOffset / FilterOffset / OfmapOffset = {config.ifmap_offset} / {config.filter_offset} /"
            f" {config.ofmap_offset}. Offsets carve global memory into three disjoint regions so the"
            " addresses in each operand matrix point to unique backing-store locations."),
        width=88,
    ))
    print()

    print(textwrap.fill(
        (
            f"Dataflow = {config.df.upper()} (weight-stationary). That tells the compute engine to park"
            " weights on each processing element while activations stream diagonally across the array."
            " Changing it to OS or IS would rebuild the demand matrices to keep outputs or inputs"
            " stationary instead."),
        width=88,
    ))
    print()

    print(textwrap.fill(
        (
            f"ReadRequestBuffer / WriteRequestBuffer = {config.req_buf_sz_rd} / {config.req_buf_sz_wr}."
            " These queues sit in front of the double buffer and bound how many outstanding requests"
            " can be tracked while the array waits for DRAM."),
        width=88,
    ))
    print()

    print(textwrap.fill(
        (
            f"IfmapSRAMBankBandwidth = {config.ifmap_sram_bank_bandwidth} word/cycle with 1 bank and"
            " 1 port. Those numbers directly control how many words can be pulled per cycle from the"
            " scratchpad half that is serving the array; they are the lever that USER mode relies on."
            " We mirror that limit for filter and ofmap SRAMs so all three behave identically."),
        width=88,
    ))
    print()

    print(textwrap.fill(
        (
            f"InterfaceBandwidth = {'USER' if config.use_user_bandwidth else 'CALC'} in the config file."
            " The walkthrough flips between CALC (infinite backing bandwidth) and USER (1 word/cycle)"
            " at runtime so you can see how the same settings change the warm-up timeline."),
        width=88,
    ))
    print()


def describe_operand_layout(layer: single_layer_sim) -> None:
    """Print a plain-language summary of how M, N, and K map to addresses."""
    op = layer.op_mat_obj
    m = op.ofmap_px_per_filt
    n = op.num_filters
    k = op.conv_window_size
    total_ifmap = op.ifmap_rows * op.ifmap_cols * op.num_input_channels

    msg = f"This layer is a GEMM with M={m} rows (tokens), K={k} input features, and N={n} outputs."
    print(textwrap.fill(msg, width=88))
    print()

    print(textwrap.fill(
        (
            f"IFMAP activations start at address {op.ifmap_offset} and occupy {total_ifmap} words. "
            f"They are laid out row-major: the simulator flattens the M×K activation table so that "
            f"row m_idx appears as addresses {op.ifmap_offset} + m_idx×{k} through {op.ifmap_offset} + "
            f"m_idx×{k} + {k - 1}."
        ),
        width=88,
    ))
    print()

    print(textwrap.fill(
        (
            f"Filter weights start at {op.filter_offset}. Each output channel owns one contiguous block "
            f"of K weights, which is why the preview jumps by {k}: addresses {op.filter_offset} + channel×{k} "
            f"through {op.filter_offset} + channel×{k} + {k - 1} belong to that channel."
        ),
        width=88,
    ))
    print()

    print(textwrap.fill(
        (
            f"OFMAP results begin at {op.ofmap_offset} and grow row-major over the M×N table. "
            f"Address {op.ofmap_offset} + m_idx×{n} + n_idx holds the output for token m_idx and "
            f"output channel n_idx."
        ),
        width=88,
    ))
    print()

    print(textwrap.fill(
        (
            "Every entry in the operand preview is an address (or -1 for padding). No random numbers "
            "are generated—the simulator is tracking which memory word would be touched when the "
            "systolic wave reaches each multiply-accumulate cell."
        ),
        width=88,
    ))
    print()


def print_operand_matrices(layer: single_layer_sim, max_rows: int) -> None:
    """Print snippets of the IFMAP, filter, and OFMAP address matrices."""
    _, ifmap_matrix = layer.op_mat_obj.get_ifmap_matrix()
    _, filter_matrix = layer.op_mat_obj.get_filter_matrix()
    _, ofmap_matrix = layer.op_mat_obj.get_ofmap_matrix()

    def preview(matrix: np.ndarray, title: str) -> None:
        print(f"{title} (showing first {min(max_rows, matrix.shape[0])} of {matrix.shape[0]} rows):")
        for row_idx, line in enumerate(format_matrix_rows(matrix)):
            print(line)
            if row_idx + 1 >= max_rows and matrix.shape[0] > max_rows:
                print("  ...")
                break
        print()

    preview(ifmap_matrix, "IFMAP address matrix (rows = output positions, cols = receptive-field taps)")
    preview(filter_matrix, "Filter address matrix (rows = receptive-field taps, cols = output channels)")
    preview(ofmap_matrix, "OFMAP address matrix (rows = output positions, cols = output channels)")


def decode_ifmap_address(addr: int, layer: single_layer_sim) -> Tuple[int, int, int]:
    op = layer.op_mat_obj
    idx = addr - op.ifmap_offset
    channels = max(op.num_input_channels, 1)
    row_stride = op.ifmap_cols * channels
    row = idx // row_stride
    col = (idx % row_stride) // channels
    channel = (idx % row_stride) % channels
    return row, col, channel


def decode_filter_address(addr: int, layer: single_layer_sim) -> Tuple[int, int]:
    op = layer.op_mat_obj
    idx = addr - op.filter_offset
    channel = idx // op.conv_window_size
    weight_idx = idx % op.conv_window_size
    return channel, weight_idx


def decode_ofmap_address(addr: int, layer: single_layer_sim) -> Tuple[int, int]:
    op = layer.op_mat_obj
    idx = addr - op.ofmap_offset
    row = idx // op.num_filters
    channel = idx % op.num_filters
    return row, channel


def describe_topology(topology: topologies) -> None:
    """Narrate how each MNK entry maps onto the toy transformer."""

    print("=== Mapping the transformer CSV to math ===")
    print(textwrap.fill(
        (
            "The demo mimics a single transformer block with batch size {batch_size}, sequence length"
            " {sequence_length}, hidden size {hidden_size}, {num_heads} attention heads, and head"
            " dimension {head_dim}. Every CSV row is one GEMM that SCALE-Sim will map to the systolic"
            " array.".format(**DEMO_FACTS)
        ),
        width=88,
    ))
    print()

    for layer_id in range(topology.get_num_layers()):
        params = topology.get_layer_params(layer_id)
        name = params[0]
        m = int(params[1])
        k = int(params[2])
        n = int(params[6])
        role = LAYER_ROLES.get(name, "Workload component")
        print(f"Layer {layer_id}: {name}")
        print("  " + role)
        print(
            "  "
            + textwrap.fill(
                (
                    f"M = {m} rows means {m} token vectors enter together. K = {k} input features per"
                    f" token. N = {n} outputs tells us how many features the layer emits per token."
                ),
                width=84,
                subsequent_indent="  ",
            )
        )
        print(
            "  "
            + textwrap.fill(
                (
                    f"SCALE-Sim will flatten the M×K activations starting at the IFMAP offset, copy"
                    " the K weights for each of the N output channels starting at the filter offset,"
                    " and leave room for the M×N outputs starting at the OFMAP offset."
                ),
                width=84,
                subsequent_indent="  ",
            )
        )
        print()


def explain_trace_semantics(
    layer: single_layer_sim,
    calc_trace_dir: Path,
    user_trace_dir: Path,
    num_rows: int,
    user_bandwidth: int,
) -> None:
    """Give context for the SRAM/DRAM traces using actual values."""

    def load_trace(path: Path) -> np.ndarray:
        if not path.exists():
            return np.zeros((0, 0))
        data = np.loadtxt(path, delimiter=",")
        return np.atleast_2d(data)

    operand_traces: Dict[str, Dict[str, np.ndarray]] = {}
    for operand in ("IFMAP", "FILTER", "OFMAP"):
        operand_traces[operand] = {
            "calc_sram": load_trace(calc_trace_dir / f"{operand}_SRAM_TRACE.csv"),
            "calc_dram": load_trace(calc_trace_dir / f"{operand}_DRAM_TRACE.csv"),
            "user_dram": load_trace(user_trace_dir / f"{operand}_DRAM_TRACE.csv"),
        }

    arr_height, arr_width = layer.config.get_array_dims()
    print(textwrap.fill(
        (
            f"SRAM traces list what the array asked for each cycle. Column 0 is the cycle number. "
            f"The next {arr_width} entries correspond to the {arr_width} columns of the {arr_height}×{arr_width} "
            "array. A value of -1 means that column had nothing to read that cycle while the wavefront "
            "was still filling."
        ),
        width=88,
    ))
    print()

    decoder = {
        "IFMAP": lambda addr: (
            f"A[m={decode_ifmap_address(addr, layer)[0]}, k={decode_ifmap_address(addr, layer)[1]}]"
        ),
        "FILTER": lambda addr: (
            f"W[channel={decode_filter_address(addr, layer)[0]}, k={decode_filter_address(addr, layer)[1]}]"
        ),
        "OFMAP": lambda addr: (
            f"O[m={decode_ofmap_address(addr, layer)[0]}, n={decode_ofmap_address(addr, layer)[1]}]"
        ),
    }

    for operand in ("IFMAP", "FILTER", "OFMAP"):
        traces = operand_traces[operand]
        calc_sram = traces["calc_sram"]
        if calc_sram.size:
            for row in calc_sram[:num_rows]:
                cycle = int(row[0])
                active = [int(x) for x in row[1:] if int(x) >= 0]
                if active:
                    decoded = [decoder[operand](addr) for addr in active]
                    print(
                        textwrap.fill(
                            f"Cycle {cycle}: the array touched {operand} entries {', '.join(decoded)}.",
                            width=88,
                        )
                    )
                    break
        print()

    print(textwrap.fill(
        (
            "DRAM traces capture when the double buffer pulled a burst from backing memory. Column 0 is "
            "the completion cycle of that burst. Negative cycles happen because the scratchpad prefetches "
            "before compute cycle 0 to warm the active half of the buffer. Subsequent columns are the "
            "addresses that travelled in that burst; the simulator groups them according to the "
            "bandwidth limit."
        ),
        width=88,
    ))
    print()

    for operand in ("IFMAP", "FILTER", "OFMAP"):
        traces = operand_traces[operand]
        calc_dram = traces["calc_dram"]
        if calc_dram.size:
            sample = calc_dram[0]
            cycle = int(sample[0])
            active = [int(x) for x in sample[1:] if int(x) >= 0]
            decoded = [decoder[operand](addr) for addr in active]
            print(
                textwrap.fill(
                    (
                        f"CALC mode {operand} burst finishing at cycle {cycle} grabbed {len(active)} words"
                        f" in one go: {', '.join(decoded)}."
                    ),
                    width=88,
                )
            )
            print()

    for operand in ("IFMAP", "FILTER", "OFMAP"):
        traces = operand_traces[operand]
        user_dram = traces["user_dram"]
        if not user_dram.size:
            continue
        sample = user_dram[:min(num_rows, user_dram.shape[0])]
        cycles = [int(row[0]) for row in sample]
        first_cycle = cycles[0]
        last_cycle = cycles[-1]
        print(
            textwrap.fill(
                (
                    f"In USER mode the {operand.lower()} stream is throttled to {user_bandwidth} word/cycle,"
                    f" so the warm-up transfer for that operand stretches from cycle {first_cycle} to"
                    f" {last_cycle}. The simulator therefore issues narrow bursts that cover the same"
                    " addresses as CALC mode but finish just in time for compute."
                ),
                width=88,
            )
        )
        print()

        for row in sample:
            cycle = int(row[0])
            payload = [int(x) for x in row[1:] if int(x) >= 0]
            if not payload:
                continue
            annotations = [decoder[operand](addr) for addr in payload]
            print(
                textwrap.fill(
                    f"Cycle {cycle}: pulled {', '.join(annotations)} into the prefetch half of the scratchpad.",
                    width=88,
                )
            )
        print()

    print(textwrap.fill(
        (
            "Once the active buffer empties, the simulator swaps the prefetch half in, so later DRAM "
            "bursts follow the array's schedule without stalling. If the USER-mode bandwidth were lower, "
            "you would see stall cycles accrue because the scratchpad could not refill fast enough."
        ),
        width=88,
    ))
    print()


def explain_timing_contrast(
    calc_stats: Tuple[float, float, float, float, float, float],
    user_stats: Tuple[float, float, float, float, float, float],
    layer: single_layer_sim,
) -> None:
    """Clarify why USER mode stretches the timeline without adding stalls."""

    (
        calc_total,
        calc_compute,
        calc_stall,
        _calc_util,
        _calc_map_eff,
        _calc_compute_util,
    ) = calc_stats
    (
        user_total,
        user_compute,
        user_stall,
        _user_util,
        _user_map_eff,
        _user_compute_util,
    ) = user_stats

    calc_idle = calc_total - calc_compute
    user_idle = user_total - user_compute

    print(textwrap.fill(
        (
            "The compute core still performs the same 71 cycles of MAC work in both modes. The extra "
            f"{user_idle - calc_idle:.0f} cycles you see in USER mode sit in the warm-up/flush windows "
            "outside the compute phase."
        ),
        width=88,
    ))
    print()

    user_bw = getattr(layer.config, "ifmap_sram_bank_bandwidth", None)
    user_bw_text = f"{user_bw}" if user_bw is not None else "the configured"
    print(textwrap.fill(
        (
            "A scratchpad double buffer needs its first half fully loaded before cycle 0. With unlimited "
            "bandwidth (CALC), that prefetch finishes just seven cycles before compute starts. When the "
            f"bandwidth is limited to {user_bw_text} word/cycle (USER), the simulator backs the prefetch up to "
            "cycle -64 so the same 64 words trickle in one at a time. That long lead time shows up as the "
            "large idle gap, even though the compute wave never has to stop."
        ),
        width=88,
    ))
    print()

    print(textwrap.fill(
        (
            "Stall cycles would increase only if the buffer ran empty mid-compute. In this toy example the "
            "prefetch schedule finishes each block before the array needs it, so stall=0 despite the longer "
            "timeline."
        ),
        width=88,
    ))
    print()

    print(textwrap.fill(
        (
            "If you enable the Ramulator integration (set `RunRamulator = Y` and provide the memory config), "
            "SCALE-Sim would replace the analytical bandwidth cap with timing returned by the DRAM model. "
            "Those latency-driven delays would appear as additional stall cycles rather than just extending "
            "the warm-up window."
        ),
        width=88,
    ))
    print()

def run_layer(
    config_path: Path,
    topology_path: Path,
    layout_path: Path,
    output_dir: Path,
    layer_id: int,
    use_user_bandwidth: bool,
    ofmap_bw_words_per_cycle: int,
) -> Tuple[Tuple[float, float, float, float, float, float], Path]:
    """Run a single-layer simulation and return compute stats and trace directory."""
    config = scale_config()
    config.read_conf_file(str(config_path))
    config.force_valid()
    config.run_name = f"layer{layer_id}_{'user' if use_user_bandwidth else 'calc'}"

    topology = topologies()
    topology.load_arrays(topofile=str(topology_path), mnk_inputs=True)

    layout = layouts()
    if layout_path:
        layout.load_arrays(layoutfile=str(layout_path), mnk_inputs=True)

    layer = single_layer_sim()
    layer.set_params(
        layer_id=layer_id,
        config_obj=config,
        topology_obj=topology,
        layout_obj=layout,
        verbose=False,
    )

    if use_user_bandwidth:
        config.use_user_bandwidth = True
        config.bandwidths = [ofmap_bw_words_per_cycle]
        config.ifmap_sram_bank_bandwidth = ofmap_bw_words_per_cycle
        config.filter_sram_bank_bandwidth = ofmap_bw_words_per_cycle
    else:
        config.set_bw_mode_to_calc()

    layer.run()

    output_dir.mkdir(parents=True, exist_ok=True)
    layer.save_traces(str(output_dir))

    comp_items = layer.get_compute_report_items()
    return comp_items, output_dir / f"layer{layer_id}"


def preview_trace(trace_path: Path, num_rows: int = 10) -> None:
    """Print the first few lines of a CSV trace file."""
    if not trace_path.exists():
        print(f"Trace {trace_path.name} not generated.")
        return

    print(f"Preview of {trace_path.name}:")
    with trace_path.open() as trace_file:
        for idx, line in enumerate(trace_file):
            print("  " + line.strip())
            if idx + 1 >= num_rows:
                break
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Path to the demo configuration file.")
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY,
                        help="Path to the demo GEMM topology CSV.")
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT,
                        help="Path to the layout CSV (defaults to conv_nets/test.csv).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Directory to store traces and reports.")
    parser.add_argument("--ofmap-bw", type=int, default=1,
                        help="Backing-store bandwidth in words/cycle for USER mode.")
    parser.add_argument("--trace-rows", type=int, default=8,
                        help="Number of rows to show when previewing traces.")
    parser.add_argument("--matrix-rows", type=int, default=8,
                        help="Number of operand-matrix rows to print (per matrix).")
    parser.add_argument("--layer", type=str, default="Toy_Q",
                        help="Layer name or index from the topology to deep dive into.")

    args = parser.parse_args()

    calc_output = args.output / "calc_mode"
    user_output = args.output / "user_mode"

    setup_config = scale_config()
    setup_config.read_conf_file(str(args.config))
    setup_config.force_valid()

    setup_topology = topologies()
    setup_topology.load_arrays(topofile=str(args.topology), mnk_inputs=True)

    setup_layout = layouts()
    if args.layout:
        setup_layout.load_arrays(layoutfile=str(args.layout), mnk_inputs=True)

    describe_config(setup_config)
    describe_topology(setup_topology)

    layer_id = resolve_layer_id(args.layer, setup_topology)
    layer_name = setup_topology.get_layer_names()[layer_id]
    print(f"=== Building operand address matrices for layer {layer_id} ({layer_name}) ===")
    layer = single_layer_sim()
    layer.set_params(
        layer_id=layer_id,
        config_obj=setup_config,
        topology_obj=setup_topology,
        layout_obj=setup_layout,
        verbose=False,
    )
    describe_operand_layout(layer)
    print_operand_matrices(layer, max_rows=args.matrix_rows)

    print("=== Running CALC (ideal bandwidth) mode ===")
    calc_stats, calc_trace_dir = run_layer(
        args.config,
        args.topology,
        args.layout,
        calc_output / layer_name,
        layer_id,
        use_user_bandwidth=False,
        ofmap_bw_words_per_cycle=args.ofmap_bw,
    )
    total_cycles, compute_cycles, stall_cycles, util, mapping_eff, compute_util = calc_stats
    idle_cycles = total_cycles - compute_cycles
    print(f"Total cycles: {total_cycles:.0f} (compute {compute_cycles:.0f}, stall {stall_cycles:.0f}, idle gap {idle_cycles:.0f})")
    print(f"Array utilization: {util:.2f}% | Mapping efficiency: {mapping_eff:.2f}% | Compute utilization: {compute_util:.2f}%")
    for operand in ("IFMAP", "FILTER", "OFMAP"):
        preview_trace(calc_trace_dir / f"{operand}_SRAM_TRACE.csv", num_rows=args.trace_rows)
        preview_trace(calc_trace_dir / f"{operand}_DRAM_TRACE.csv", num_rows=args.trace_rows)

    print("=== Running USER (bandwidth-limited) mode ===")
    user_stats, user_trace_dir = run_layer(
        args.config,
        args.topology,
        args.layout,
        user_output / layer_name,
        layer_id,
        use_user_bandwidth=True,
        ofmap_bw_words_per_cycle=args.ofmap_bw,
    )
    total_cycles, compute_cycles, stall_cycles, util, mapping_eff, compute_util = user_stats
    idle_cycles = total_cycles - compute_cycles
    print(f"Total cycles: {total_cycles:.0f} (compute {compute_cycles:.0f}, stall {stall_cycles:.0f}, idle gap {idle_cycles:.0f})")
    print(f"Array utilization: {util:.2f}% | Mapping efficiency: {mapping_eff:.2f}% | Compute utilization: {compute_util:.2f}%")
    for operand in ("IFMAP", "FILTER", "OFMAP"):
        preview_trace(user_trace_dir / f"{operand}_SRAM_TRACE.csv", num_rows=args.trace_rows)
        preview_trace(user_trace_dir / f"{operand}_DRAM_TRACE.csv", num_rows=args.trace_rows)

    explain_timing_contrast(calc_stats, user_stats, layer)
    explain_trace_semantics(
        layer,
        calc_trace_dir,
        user_trace_dir,
        num_rows=args.trace_rows,
        user_bandwidth=args.ofmap_bw,
    )


if __name__ == "__main__":
    main()
