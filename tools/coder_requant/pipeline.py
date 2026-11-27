#!/usr/bin/env python3
"""Requantize the Coder's experts from the base Q8_0, optionally with fewer experts (variants A, B, C).

Steps (each a subcommand; all paths are arguments):
  match        Coder router rows -> base expert indices (indices.json); range reads, no download needed
  slice-base   base Q8_0 (512 experts, 6 shards) -> Q8_0 with the Coder's 256 experts, shard by shard:
               download, verify SHA-256, copy, delete (--delete-shards); peak ~ intermediate + largest shard
  select       scores on coding traces -> keep.json (experts per layer, Coder numbering)
  slice        cut a GGUF (intermediate or imatrix) to keep.json
  slice-mtp    cut the MTP head (one expert layer) to a keep list: byte copy, expert_count set, checked
  types        tensor-type file and llama-quantize command of a variant
  assemble     Coder trunk with the requantized experts and the cut router -> final trunk file
  check        final file against Coder and keep.json: router rows byte-equal, shapes, expert_count
  variant      slice + imatrix slice + quantize + assemble + check for one variant
  repack       copy of a trunk file with its IQ3_XXS experts as TONA_IQ3_XXS_R, or with --tile its Q6_K layer
               matrices as TONA_Q6_K_T (Mac files; docs/use/tona-types.md)
  dense        copy of a 512-expert Flash-Next trunk with Q3_K as Q6_K and Q4_0, Q5_0 as Q8_0 (lossless, verified)

The Coder's second shard (PLE table) has no per-expert tensor and is used unchanged next to the new trunk.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coder_requant import (
    dense_repack,
    mtp,
    repack,
    repack_tile,
    router_match,
    select_experts,
    slicing,
    sources,
    variants,
)
from coder_requant.gguf_layout import arch_key, expert_count, scalar

REPO = Path(__file__).resolve().parents[2]
N_LAYER = 48


def pins(args):
    coder = sources.load_pins(REPO / "toolchain" / "models.toml", "coder-iq1_m")
    base = sources.load_pins(Path(args.sources), "base-q8_0")
    return coder, base


def reader(pin: sources.Pinned, local_dir: str | None) -> sources.Reader:
    return sources.Reader(Path(local_dir) / pin.name if local_dir else None, pin.url)


def cmd_match(args) -> None:
    coder_pins, base_pins = pins(args)
    cr = reader(coder_pins[0], args.coder_dir)
    cl = cr.layout()
    bases = [(r, r.layout()) for r in (reader(p, args.base_dir) for p in base_pins)]
    out = []
    for layer in range(N_LAYER):
        name = f"blk.{layer}.ffn_gate_inp.weight"
        br, bl = next((r, lay) for r, lay in bases if name in lay.by_name)
        t = cl.by_name[name]
        idx = router_match.match_layer(cr.tensor(cl, name), br.tensor(bl, name), t.dims[0])
        print(f"layer {layer}: {len(idx)} matched", file=sys.stderr)
        out.append(idx)
    Path(args.out).write_text(json.dumps({"n_expert_in": expert_count(bases[0][1]), "keep": out}))


def cmd_slice_base(args) -> None:
    _, base_pins = pins(args)
    keep = select_experts.read_keep(args.indices)
    shard_dir = Path(args.shards_dir)
    layouts = {p.name: reader(p, args.shards_dir).layout() for p in base_pins}
    tensors = [t for p in base_pins for t in slicing.plan_slice(layouts[p.name], p.name, keep)]
    first = layouts[base_pins[0].name]
    kvs = slicing.drop_split_kvs(slicing.model_kvs(first, slicing.equal_count(keep)))
    by_name = {p.name: p for p in base_pins}

    def after(src: str) -> None:
        if args.delete_shards:
            (shard_dir / src).unlink()

    hooks = slicing.Hooks(
        path=lambda src: str(shard_dir / src),
        before=lambda src: sources.fetch(by_name[src], shard_dir),
        after=after,
    )
    slicing.write(args.out, kvs, tensors, hooks, first.alignment)


def cmd_select(args) -> None:
    if args.trace:
        score, source = select_experts.scores_from_trace(args.trace, N_LAYER, args.n_in), "trace gate weights"
    else:
        score, source = select_experts.scores_from_profile(args.profile), "profile counts"
    if args.adaptive:
        keep = select_experts.keep_adaptive(score, args.n_keep, args.floor)
    else:
        keep = select_experts.keep_equal(score, args.n_keep)
    select_experts.write_keep(args.out, keep, score, source)
    worst = max(select_experts.dropped_mass(score, keep))
    print(f"dropped score mass: worst layer {worst:.1%}", file=sys.stderr)
    if worst > args.warn:
        print(f"warning: a layer drops more than {args.warn:.0%} of its score mass", file=sys.stderr)


PLE_TABLE = "per_layer_token_embd.weight"  # 54 GB in the base Q8_0, 29 GB in the Coder; never per expert


def slice_file(src: str, dst: str, keep: dict[int, list[int]], drop_ple: bool = False) -> None:
    """Cuts a model or a GGUF imatrix (legacy .dat imatrix files are not read) to keep. drop_ple leaves the PLE
    table out: the quantize input of a variant needs only the experts (assemble takes everything else from the
    Coder), and the table would cost 54 GB of disk and a pass through llama-quantize."""
    layout = sources.Reader(Path(src)).layout()
    kvs = layout.kvs
    if any(k.key == "general.architecture" for k in kvs) and scalar(kvs, arch_key(kvs, "expert_count")) is not None:
        kvs = slicing.model_kvs(layout, slicing.equal_count(keep))
    tensors = [t for t in slicing.plan_slice(layout, "in", keep) if not (drop_ple and t.name == PLE_TABLE)]
    slicing.write(dst, kvs, tensors, slicing.Hooks(path=lambda _: src), layout.alignment)


def cmd_slice(args) -> None:
    slice_file(args.input, args.out, select_experts.read_keep(args.keep))


def cmd_types(args) -> None:
    v = variants.VARIANTS[args.variant]
    Path(args.out).write_text(variants.tensor_types(v, args.ple_type))
    print(" ".join(variants.quantize_cmd(args.quantize, "<imatrix>", args.out, "<in>", "<out>")))


def assemble(coder_trunk: str, quant: str, keep: dict[int, list[int]], out: str) -> None:
    coder = sources.Reader(Path(coder_trunk)).layout()
    q = sources.Reader(Path(quant)).layout()
    tensors = slicing.plan_assemble(coder, q, keep)
    kvs = slicing.model_kvs(coder, slicing.equal_count(keep))
    files = {"coder": coder_trunk, "quant": quant}
    slicing.write(out, kvs, tensors, slicing.Hooks(path=files.__getitem__), coder.alignment)


SPLIT_KEYS = ("split.no", "split.count", "split.tensors.count")


def check_split(cl, fl) -> str:
    """The split metadata of the final file equals the Coder part's: same part number, part count and tensor total.
    A multi-part file only loads with its other parts next to it, so the result line names them."""
    for key in SPLIT_KEYS:
        if scalar(cl.kvs, key) != scalar(fl.kvs, key):
            raise SystemExit(f"check: {key} is {scalar(fl.kvs, key)}, the Coder part has {scalar(cl.kvs, key)}")
    count = scalar(fl.kvs, "split.count")
    if not count or count < 2:
        return "single file"
    total = scalar(fl.kvs, "split.tensors.count")
    return f"part {scalar(fl.kvs, 'split.no') + 1} of {count}, {len(fl.tensors)} of {total} tensors"


def check(coder_trunk: str, final: str, keep: dict[int, list[int]]) -> None:
    """Router rows byte-equal to the kept Coder rows, every non-expert tensor byte-equal, shapes and expert_count."""
    cr, fr = sources.Reader(Path(coder_trunk)), sources.Reader(Path(final))
    cl, fl = cr.layout(), fr.layout()
    n = slicing.equal_count(keep)
    check_split(cl, fl)
    if expert_count(fl) != n or [t.name for t in cl.tensors] != [t.name for t in fl.tensors]:
        raise SystemExit("check: expert_count or tensor list differs")
    for t in cl.tensors:
        layer = slicing.per_expert_layer(t.name)
        if layer is None:
            if cr.tensor(cl, t.name) != fr.tensor(fl, t.name):
                raise SystemExit(f"check: {t.name} is not a byte copy")
            continue
        if fl.by_name[t.name].dims != [*t.dims[:-1], n]:
            raise SystemExit(f"check: {t.name} has shape {fl.by_name[t.name].dims}")
        if slicing.ROUTER.match(t.name):
            per = t.nbytes // t.dims[-1]
            src, got = cr.tensor(cl, t.name), fr.tensor(fl, t.name)
            want = b"".join(src[e * per : (e + 1) * per] for e in keep[layer])
            if got != want:
                raise SystemExit(f"check: {t.name} rows are not the kept Coder rows")
    print(f"check: {final} ok ({n} experts per layer; {check_split(cl, fl)})", file=sys.stderr)


def cmd_slice_mtp(args) -> None:
    """The MTP head cut to a keep list (a byte copy, expert_count set), then checked against the source."""
    layout = sources.Reader(Path(args.input)).layout()
    keep = mtp.keep_for(layout, mtp.read_keep_list(args.keep_list))
    slice_file(args.input, args.out, keep)
    check(args.input, args.out, keep)
    mtp.check_expert_bytes(args.input, args.out, keep)


def cmd_assemble(args) -> None:
    assemble(args.coder_trunk, args.quant, select_experts.read_keep(args.keep), args.out)


def cmd_check(args) -> None:
    check(args.coder_trunk, args.final, select_experts.read_keep(args.keep))


def cmd_variant(args) -> None:
    v = variants.VARIANTS[args.variant]
    work = Path(args.work)
    keep = select_experts.read_keep(args.keep)
    if slicing.equal_count(keep) != v.n_expert:
        raise SystemExit(f"keep.json has {slicing.equal_count(keep)} experts per layer, variant {v.name} {v.n_expert}")
    q8, imat, types = work / f"{v.name}-q8_0.gguf", work / f"{v.name}-imatrix.gguf", work / f"{v.name}.types"
    quant, final = work / f"{v.name}-quant.gguf", work / f"coder-{v.name}-{v.n_expert}e.gguf"
    slice_file(args.intermediate, str(q8), keep, drop_ple=True)
    slice_file(args.imatrix, str(imat), keep)
    types.write_text(variants.tensor_types(v, args.ple_type))
    subprocess.run(variants.quantize_cmd(args.quantize, str(imat), str(types), str(q8), str(quant)), check=True)
    q8.unlink()
    assemble(args.coder_trunk, str(quant), keep, str(final))
    check(args.coder_trunk, str(final), keep)


def cmd_repack(args) -> None:
    if Path(args.out).exists():
        raise SystemExit(f"{args.out} exists")
    repack.copy(args.input, args.out)
    if args.tile:
        names = repack_tile.tile_file(args.out, args.pattern or repack_tile.LAYER_MATRICES)
        print(f"{args.out}: {len(names)} tensors tiled to TONA_Q6_K_T")
        return
    names = repack.repack_file(args.out, args.pattern or repack.EXPERTS)
    print(f"{args.out}: {len(names)} tensors repacked to TONA_IQ3_XXS_R")


def cmd_dense(args) -> None:
    counts = dense_repack.write_file(args.input, args.out)
    print(f"{args.out}: converted tensors by source type {counts.converted}")
    print("blocks compared bit for bit:", dense_repack.verify(args.input, args.out))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", default=str(Path(__file__).with_name("sources.toml")))
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("match")
    s.add_argument("--coder-dir"), s.add_argument("--base-dir"), s.add_argument("--out", required=True)
    s = sub.add_parser("slice-base")
    s.add_argument("--indices", required=True), s.add_argument("--shards-dir", required=True)
    s.add_argument("--out", required=True), s.add_argument("--delete-shards", action="store_true")
    s = sub.add_parser("select")
    s.add_argument("--trace", nargs="+"), s.add_argument("--profile"), s.add_argument("--n-in", type=int, default=256)
    s.add_argument("--n-keep", type=int, required=True), s.add_argument("--out", required=True)
    s.add_argument("--adaptive", action="store_true"), s.add_argument("--floor", type=int, default=64)
    s.add_argument("--warn", type=float, default=0.15)
    s = sub.add_parser("slice")
    s.add_argument("--input", required=True), s.add_argument("--keep", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("slice-mtp")
    s.add_argument("--input", required=True), s.add_argument("--keep-list", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("types")
    s.add_argument("--variant", choices=variants.VARIANTS, required=True), s.add_argument("--out", required=True)
    s.add_argument("--quantize", default="llama-quantize")
    s.add_argument("--ple-type", help="type of the PLE table in the source (Coder file: iq4_nl); default none")
    for name in ("assemble", "check"):
        s = sub.add_parser(name)
        s.add_argument("--coder-trunk", required=True), s.add_argument("--keep", required=True)
        s.add_argument("--quant" if name == "assemble" else "--final", required=True)
        if name == "assemble":
            s.add_argument("--out", required=True)
    s = sub.add_parser("variant")
    s.add_argument("--variant", choices=variants.VARIANTS, required=True), s.add_argument("--keep", required=True)
    s.add_argument("--intermediate", required=True), s.add_argument("--imatrix", required=True)
    s.add_argument("--coder-trunk", required=True), s.add_argument("--work", required=True)
    s.add_argument("--quantize", default="llama-quantize"), s.add_argument("--ple-type")
    s = sub.add_parser("dense")
    s.add_argument("--input", required=True), s.add_argument("--out", required=True)
    s = sub.add_parser("repack")
    s.add_argument("--input", required=True), s.add_argument("--out", required=True)
    s.add_argument("--pattern", help="regex of tensor names (default: gate/up experts; --tile: layer matrices)")
    s.add_argument("--tile", action="store_true", help="Q6_K layer matrices to TONA_Q6_K_T instead")
    return p


COMMANDS = {
    "match": cmd_match, "slice-mtp": cmd_slice_mtp,
    "slice-base": cmd_slice_base, "select": cmd_select, "slice": cmd_slice, "types": cmd_types,
    "assemble": cmd_assemble, "check": cmd_check, "variant": cmd_variant, "repack": cmd_repack,
    "dense": cmd_dense,
}  # fmt: skip


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    if args.cmd == "select" and not (args.trace or args.profile):
        raise SystemExit("select needs --trace or --profile")
    COMMANDS[args.cmd](args)


if __name__ == "__main__":
    main()
