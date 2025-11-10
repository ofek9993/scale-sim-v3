#!/usr/bin/env python3
"""Walk through how SCALE-Sim maps the GPT-2 GEMM topology onto the Google TPU config."""
from __future__ import annotations

import argparse
import sys
import textwrap
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scalesim.scale_config import scale_config
from scalesim.topology_utils import topologies
from scalesim.layout_utils import layouts
from scalesim.single_layer_sim import single_layer_sim

# Helper imports reused from the tiny-demo walkthrough so we decode addresses consistently.
from scripts.demo_memory_walkthrough import (
    decode_filter_address,
    decode_ifmap_address,
    decode_ofmap_address,
)

DEFAULT_CONFIG = REPO_ROOT / "configs/google.cfg"
DEFAULT_TOPOLOGY = REPO_ROOT / "topologies/GEMM_mnk/gpt2.csv"
DEFAULT_LAYOUT = REPO_ROOT / "layouts/conv_nets/test.csv"
DEFAULT_OUTPUT = REPO_ROOT / "outputs_gpt2_memory"


GPT2_LAYER_ROLES: Dict[str, str] = {
    "QKT": "Attention score GEMM (per head): queries (M rows) × key vectors (K width) produce sequence-length attention logits (N columns).",
    "QKTV": "Context accumulation (per head): the attention probabilities multiply value vectors so each token receives a weighted context.",
    "Linear1": "c_attn projection: the model dimension (K) fans out to 3× hidden_size outputs that hold Q, K, and V packed together.",
    "Linear2": "c_proj projection: mixes the concatenated head outputs back into the model dimension.",
    "PW-FF-L1": "MLP expansion: grows each token from hidden_size to the intermediate GeLU width.",
    "PW-FF-L2": "MLP contraction: projects the GeLU activations back to the hidden_size output." ,
}


@dataclass
class LayerFacts:
    name: str
    m: int
    n: int
    k: int


@dataclass
class GPT2Facts:
    sequence_length: int
    head_dim: int
    hidden_size: int
    num_heads: int
    intermediate_size: int
    batch_size: int = 1

    def to_bullets(self) -> Iterable[str]:
        yield f"Sequence length: {self.sequence_length} tokens"
        yield f"Batch size: {self.batch_size} (implicit in the M dimension)"
        yield f"Hidden size: {self.hidden_size}"
        yield f"Head dimension: {self.head_dim}"
        yield f"Number of attention heads: {self.num_heads}"
        yield f"Intermediate MLP size: {self.intermediate_size}"


def derive_layer_facts(topo: topologies) -> Dict[str, LayerFacts]:
    facts: "OrderedDict[str, LayerFacts]" = OrderedDict()
    for layer_id in range(topo.get_num_layers()):
        params = topo.get_layer_params(layer_id)
        name = params[0]
        m = int(params[1])
        k = int(params[2])
        n = int(params[6])
        facts[name] = LayerFacts(name=name, m=m, n=n, k=k)
    return facts


def derive_gpt2_facts(layer_facts: Dict[str, LayerFacts]) -> GPT2Facts:
    qkt = layer_facts.get("QKT")
    linear1 = layer_facts.get("Linear1")
    pw_ff = layer_facts.get("PW-FF-L1")
    if not (qkt and linear1 and pw_ff):
        raise ValueError("GPT-2 topology CSV is missing expected layer names.")

    sequence_length = qkt.m
    head_dim = qkt.k
    hidden_size = linear1.k
    num_heads = max(1, hidden_size // head_dim)
    intermediate_size = pw_ff.n
    return GPT2Facts(
        sequence_length=sequence_length,
        head_dim=head_dim,
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
    )


def describe_config(config: scale_config, facts: GPT2Facts) -> None:
    print("=== Hardware backdrop: google.cfg ===")
    print(textwrap.fill(
        (
            f"ArrayHeight × ArrayWidth = {config.array_rows} × {config.array_cols}. This is the TPU v1-style "
            "systolic core with 65,536 multiply-accumulate cells, large enough to process GPT-2's 1024-token"
            " minibatch with high utilization."
        ),
        width=96,
    ))
    print()

    print(textwrap.fill(
        (
            f"Scratchpad sizes: IFMAP {config.ifmap_sz_kb} kB, FILTER {config.filter_sz_kb} kB, OFMAP {config.ofmap_sz_kb} kB. "
            "These multi-megabyte buffers easily hold the M×K and K×N panels we stream for each GEMM stage, "
            "so in this walkthrough we focus on bandwidth rather than capacity limits."
        ),
        width=96,
    ))
    print()

    print(textwrap.fill(
        (
            f"Offsets carve global DRAM into three regions: IFMAP starts at {config.ifmap_offset}, FILTER at {config.filter_offset}, "
            f"and OFMAP at {config.ofmap_offset}. SCALE-Sim tags every operand address relative to those anchors."
        ),
        width=96,
    ))
    print()

    flow = config.df.upper()
    print(textwrap.fill(
        (
            f"Dataflow = {flow}. The TPU config keeps outputs stationary, so partial sums stay resident in OFMAP scratchpad "
            "banks while activations and weights stream in. The simulator will therefore highlight OFMAP SRAM activity "
            "when accumulations roll across the array."
        ),
        width=96,
    ))
    print()

    if config.use_user_dram_bandwidth():
        bw_desc = ", ".join(str(x) for x in config.bandwidths)
        print(textwrap.fill(
            (
                f"InterfaceBandwidth = USER with configured bandwidth(s) {bw_desc} words/cycle. We'll compare that practical limit to "
                "CALC mode, which solves for the ideal bandwidth that would keep this array 100% busy."
            ),
            width=96,
        ))
    else:
        print(textwrap.fill(
            (
                "InterfaceBandwidth = CALC. The simulator will analytically determine the minimum prefetch bandwidth for each operand."
            ),
            width=96,
        ))
    print()

    print("=== GPT-2 block dimensions inferred from the CSV ===")
    for bullet in facts.to_bullets():
        print(f" • {bullet}")
    print()


def describe_topology(layer_facts: Dict[str, LayerFacts], facts: GPT2Facts) -> None:
    print("=== Matching CSV rows to GPT-2 math ===")
    header = (
        "The CSV lists one transformer block's dense GEMMs. M corresponds to sequence length (batch × tokens), "
        "K captures the incoming feature width, and N is the number of outputs each layer emits per token."
    )
    print(textwrap.fill(header, width=96))
    print()

    for idx, (name, fact) in enumerate(layer_facts.items()):
        role = GPT2_LAYER_ROLES.get(name, "Transformer component")
        print(f"Layer {idx}: {name}")
        print(f"  Role: {role}")
        print(
            "  "
            + textwrap.fill(
                f"M = {fact.m} → processing {fact.m} token vectors in parallel (batch × sequence).",
                width=92,
                subsequent_indent="  ",
            )
        )

        if name == "QKT":
            k_note = f"the attention head dimension {facts.head_dim}"
        elif name == "QKTV":
            k_note = f"the sequence length {facts.sequence_length} captured by the score matrix"
        elif name == "PW-FF-L2":
            k_note = f"the intermediate MLP size {facts.intermediate_size}"
        else:
            k_note = f"the hidden size {facts.hidden_size}"
        print(
            "  "
            + textwrap.fill(
                f"K = {fact.k} → each token contributes {fact.k} input features. For this layer that equals {k_note}.",
                width=92,
                subsequent_indent="  ",
            )
        )

        if name == "QKT":
            n_note = f"sequence-length attention logits (N = {fact.n}) for each query token"
        elif name == "QKTV":
            n_note = f"{fact.n} value features per head (head_dim)"
        elif name == "Linear1":
            n_note = f"3 × hidden_size outputs ({3 * facts.hidden_size}) packing Q, K, and V"
        elif name == "Linear2":
            n_note = f"hidden-size outputs ({facts.hidden_size})"
        elif name == "PW-FF-L1":
            n_note = f"the intermediate GeLU width ({facts.intermediate_size})"
        elif name == "PW-FF-L2":
            n_note = f"the hidden size {facts.hidden_size}"
        else:
            n_note = f"{fact.n} outputs per token"
        print(
            "  "
            + textwrap.fill(
                f"N = {fact.n} → the layer emits {fact.n} features per token, i.e. {n_note}.",
                width=92,
                subsequent_indent="  ",
            )
        )
        print()


def describe_operand_layout(layer: single_layer_sim) -> None:
    op = layer.op_mat_obj
    m = op.ofmap_px_per_filt
    n = op.num_filters
    k = op.conv_window_size
    total_ifmap = op.ifmap_rows * op.ifmap_cols * op.num_input_channels

    msg = f"GEMM layout: M={m} rows, K={k} columns, N={n} outputs per token."
    print(textwrap.fill(msg, width=96))
    print()

    print(
        textwrap.fill(
            (
                f"IFMAP activations start at address {op.ifmap_offset} and occupy {total_ifmap} words. Row m uses addresses "
                f"[{op.ifmap_offset} + m×{k}, {op.ifmap_offset} + m×{k} + {k - 1}] for its K input features."
            ),
            width=96,
        )
    )
    print()

    print(
        textwrap.fill(
            (
                f"FILTER weights start at {op.filter_offset}. Output channel n owns a contiguous block of K weights starting at {op.filter_offset} + n×{k}."
            ),
            width=96,
        )
    )
    print()

    print(
        textwrap.fill(
            (
                f"OFMAP results begin at {op.ofmap_offset}. Token m, output n lives at {op.ofmap_offset} + m×{n} + n_idx."
            ),
            width=96,
        )
    )
    print()


def print_operand_matrices_compact(layer: single_layer_sim, max_rows: int, max_cols: int = 8) -> None:
    op = layer.op_mat_obj
    _, ifmap_matrix = op.get_ifmap_matrix()
    _, filter_matrix = op.get_filter_matrix()
    _, ofmap_matrix = op.get_ofmap_matrix()

    def preview(matrix: np.ndarray, title: str) -> None:
        rows = min(max_rows, matrix.shape[0])
        cols = min(max_cols, matrix.shape[1])
        print(
            f"{title} (showing first {rows} of {matrix.shape[0]} rows and first {cols} of {matrix.shape[1]} columns):"
        )
        for row_idx in range(rows):
            row_vals = [f"{int(val):7d}" for val in matrix[row_idx, :cols]]
            if matrix.shape[1] > cols:
                row_vals.append("...")
            print("  " + " ".join(row_vals))
        if matrix.shape[0] > rows:
            print("  ...")
        print()

    preview(ifmap_matrix, "IFMAP address matrix")
    preview(filter_matrix, "Filter address matrix")
    preview(ofmap_matrix, "OFMAP address matrix")


def sample_operand_addresses(layer: single_layer_sim, max_samples: int = 6) -> None:
    op = layer.op_mat_obj
    _, ifmap_matrix = op.get_ifmap_matrix()
    _, filter_matrix = op.get_filter_matrix()
    _, ofmap_matrix = op.get_ofmap_matrix()

    def collect_samples(matrix: np.ndarray) -> List[int]:
        samples: List[int] = []
        for row in matrix:
            for value in row:
                if int(value) >= 0:
                    samples.append(int(value))
                if len(samples) >= max_samples:
                    return samples
        return samples

    if_samples = collect_samples(ifmap_matrix)
    filt_samples = collect_samples(filter_matrix)
    ofmap_samples = collect_samples(ofmap_matrix)

    print("Sample IFMAP addresses → decoded tokens/features:")
    for addr in if_samples:
        row, col, _ = decode_ifmap_address(addr, layer)
        print(f"  {addr}: token {row}, feature {col}")
    print()

    print("Sample FILTER addresses → decoded output-channel weights:")
    for addr in filt_samples:
        channel, weight_idx = decode_filter_address(addr, layer)
        print(f"  {addr}: output channel {channel}, weight {weight_idx}")
    print()

    print("Sample OFMAP addresses → decoded output slots:")
    for addr in ofmap_samples:
        row, channel = decode_ofmap_address(addr, layer)
        print(f"  {addr}: token {row}, output channel {channel}")
    print()


def read_trace_rows(trace_path: Path, limit: int) -> List[List[float]]:
    rows: List[List[float]] = []
    if not trace_path.exists():
        return rows
    with trace_path.open() as handle:
        for idx, line in enumerate(handle):
            if idx >= limit:
                break
            parts = [float(x) for x in line.strip().split(",") if x]
            rows.append(parts)
    return rows


def summarize_sram_activity(layer: single_layer_sim, trace_dir: Path, operand: str, limit: int) -> None:
    trace_rows = read_trace_rows(trace_dir / f"{operand}_SRAM_TRACE.csv", limit)
    if not trace_rows:
        print(f"  No {operand} SRAM activity recorded.")
        return

    decoder = {
        "IFMAP": lambda addr: f"A[token={decode_ifmap_address(addr, layer)[0]}, k={decode_ifmap_address(addr, layer)[1]}]",
        "FILTER": lambda addr: f"W[channel={decode_filter_address(addr, layer)[0]}, k={decode_filter_address(addr, layer)[1]}]",
        "OFMAP": lambda addr: f"O[token={decode_ofmap_address(addr, layer)[0]}, n={decode_ofmap_address(addr, layer)[1]}]",
    }

    for row in trace_rows:
        cycle = int(row[0])
        payload = [int(x) for x in row[1:] if int(x) >= 0]
        if not payload:
            continue
        decoded = ", ".join(decoder[operand](addr) for addr in payload[:8])
        if len(payload) > 8:
            decoded += ", ..."
        print(f"  Cycle {cycle}: array consumed {len(payload)} {operand} word(s): {decoded}")
        break


def summarize_dram_activity(layer: single_layer_sim, trace_dir: Path, operand: str, limit: int, mode: str) -> None:
    trace_rows = read_trace_rows(trace_dir / f"{operand}_DRAM_TRACE.csv", limit)
    if not trace_rows:
        print(f"  No {operand} DRAM bursts captured in {mode} mode.")
        return

    decoder = {
        "IFMAP": lambda addr: f"A[token={decode_ifmap_address(addr, layer)[0]}, k={decode_ifmap_address(addr, layer)[1]}]",
        "FILTER": lambda addr: f"W[channel={decode_filter_address(addr, layer)[0]}, k={decode_filter_address(addr, layer)[1]}]",
        "OFMAP": lambda addr: f"O[token={decode_ofmap_address(addr, layer)[0]}, n={decode_ofmap_address(addr, layer)[1]}]",
    }

    for row in trace_rows:
        cycle = int(row[0])
        payload = [int(x) for x in row[1:] if int(x) >= 0]
        if not payload:
            continue
        decoded = ", ".join(decoder[operand](addr) for addr in payload[:8])
        if len(payload) > 8:
            decoded += ", ..."
        print(f"  Burst finishing at cycle {cycle}: pulled {len(payload)} word(s) → {decoded}")
    print()


def execute_layer(
    config_path: Path,
    topology_path: Path,
    layout_path: Path,
    output_dir: Path,
    layer_id: int,
    use_user_bandwidth: bool,
) -> Tuple[Tuple[float, float, float, float, float, float], Path]:
    config = scale_config()
    config.read_conf_file(str(config_path))
    config.force_valid()
    config.run_name = f"gpt2_layer{layer_id}_{'user' if use_user_bandwidth else 'calc'}"

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
    else:
        config.set_bw_mode_to_calc()

    layer.run()

    output_dir.mkdir(parents=True, exist_ok=True)
    layer.save_traces(str(output_dir))

    stats = layer.get_compute_report_items()
    return stats, output_dir / f"layer{layer_id}"


def print_stats(stats: Sequence[float], mode: str) -> None:
    total, compute, stall, util, map_eff, compute_util = stats
    idle = total - compute
    print(f"{mode} mode cycles: total {total:.0f}, compute {compute:.0f}, stall {stall:.0f}, idle window {idle:.0f}")
    print(f"  Array utilization {util:.2f}% | Mapping efficiency {map_eff:.2f}% | Compute utilization {compute_util:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Architecture config (google.cfg by default).")
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY, help="GPT-2 GEMM topology CSV.")
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT, help="Layout CSV (defaults to conv_nets/test.csv).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory for traces/reports.")
    parser.add_argument("--trace-rows", type=int, default=6, help="How many trace rows to decode per operand.")
    parser.add_argument("--matrix-rows", type=int, default=6, help="How many operand-matrix rows to preview.")
    parser.add_argument("--matrix-cols", type=int, default=8, help="How many columns to show from each operand matrix.")
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Comma-separated list of layer names (default: all layers in order).",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Optional path to tee the walkthrough output into a Markdown or log file.",
    )

    args = parser.parse_args()

    config = scale_config()
    config.read_conf_file(str(args.config))
    config.force_valid()

    topo = topologies()
    topo.load_arrays(topofile=str(args.topology), mnk_inputs=True)

    layout = layouts()
    if args.layout:
        layout.load_arrays(layoutfile=str(args.layout), mnk_inputs=True)

    layer_facts = derive_layer_facts(topo)
    gpt2_facts = derive_gpt2_facts(layer_facts)

    def run_walkthrough() -> None:
        describe_config(config, gpt2_facts)
        describe_topology(layer_facts, gpt2_facts)

        layer_order = list(layer_facts.keys())
        name_to_id = {name: idx for idx, name in enumerate(layer_order)}

        names = layer_order
        if args.layers.lower() != "all":
            requested = [name.strip() for name in args.layers.split(",") if name.strip()]
            missing = [name for name in requested if name not in layer_facts]
            if missing:
                raise ValueError(f"Unknown layers requested: {', '.join(missing)}")
            names = requested

        for name in names:
            layer_id = name_to_id[name]
            print(f"=== Layer {layer_id}: {name} ===")
            layer = single_layer_sim()
            layer.set_params(
                layer_id=layer_id,
                config_obj=config,
                topology_obj=topo,
                layout_obj=layout,
                verbose=False,
            )
            describe_operand_layout(layer)
            print_operand_matrices_compact(layer, max_rows=args.matrix_rows, max_cols=args.matrix_cols)
            sample_operand_addresses(layer)

            print("-- CALC (ideal bandwidth) run --")
            calc_stats, calc_dir = execute_layer(
                args.config,
                args.topology,
                args.layout,
                args.output / "calc_mode" / name,
                layer_id,
                use_user_bandwidth=False,
            )
            print_stats(calc_stats, "CALC")
            for operand in ("IFMAP", "FILTER", "OFMAP"):
                print(f"  {operand} SRAM timeline snippet:")
                summarize_sram_activity(layer, calc_dir, operand, args.trace_rows)
                print(f"  {operand} DRAM warm-up bursts:")
                summarize_dram_activity(layer, calc_dir, operand, args.trace_rows, mode="CALC")

            print("-- USER (configured bandwidth) run --")
            user_stats, user_dir = execute_layer(
                args.config,
                args.topology,
                args.layout,
                args.output / "user_mode" / name,
                layer_id,
                use_user_bandwidth=True,
            )
            print_stats(user_stats, "USER")
            for operand in ("IFMAP", "FILTER", "OFMAP"):
                print(f"  {operand} SRAM timeline snippet:")
                summarize_sram_activity(layer, user_dir, operand, args.trace_rows)
                print(f"  {operand} DRAM warm-up bursts:")
                summarize_dram_activity(layer, user_dir, operand, args.trace_rows, mode="USER")

            total_calc, calc_compute, _, _, _, _ = calc_stats
            total_user, user_compute, _, _, _, _ = user_stats
            extra_idle = (total_user - user_compute) - (total_calc - calc_compute)
            print(
                textwrap.fill(
                    (
                        f"Difference summary: compute time stays at {calc_compute:.0f} cycles, but USER mode introduces an"
                        f" additional {extra_idle:.0f} warm-up/flush cycles because the configured bandwidth feeds the double"
                        " buffers more slowly than the ideal CALC analysis."
                    ),
                    width=96,
                )
            )
            print()

    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("w", encoding="utf-8") as log_handle:
            class Tee:
                def __init__(self, *streams):
                    self.streams = streams

                def write(self, data: str) -> None:
                    for stream in self.streams:
                        stream.write(data)

                def flush(self) -> None:
                    for stream in self.streams:
                        stream.flush()

            tee = Tee(sys.stdout, log_handle)
            original_stdout = sys.stdout
            try:
                sys.stdout = tee
                print("# GPT-2 SCALE-Sim memory walkthrough")
                print()
                run_walkthrough()
            finally:
                sys.stdout = original_stdout
    else:
        run_walkthrough()


if __name__ == "__main__":
    main()
