#!/usr/bin/env python3
"""
run_surface_builder.py
======================
Command-line driver for the AS-ALD amorphous surface builder (Challenge 4),
intended for a local GPU node rather than a notebook.

Keeps surface_builder.py as the library; this script parses arguments, runs the
ensemble, writes structures, and prints a validation report.

Examples
--------
# quick pipeline check (ultra-short MD, gate off), both materials:
    python run_surface_builder.py --mode test --n-bulk 1 --no-gate

# full production, 3 bulks, require >=4 accepted surfaces each:
    python run_surface_builder.py --mode full --n-bulk 3 --target-accepted 4

# only SiO2, full protocol, float64 (highest accuracy, slower):
    python run_surface_builder.py --material SiO2 --mode full --dtype float64

# cooling-rate sweep + Problem-A / Problem-B reports:
    python run_surface_builder.py --mode full --quench-sweep 8,15,30 \
        --report-exposure --cluster-report

Outputs
-------
  <outdir>/<material>_surface_<i>.xyz    one file per accepted surface
  <outdir>/<material>_summary.json       density summary + QC metadata
  <outdir>/build_report.txt              human-readable validation report
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from ase.io import write

import surface_builder as sb


def parse_args():
    p = argparse.ArgumentParser(description="AS-ALD amorphous surface builder")
    p.add_argument("--material", choices=["SiO2", "SiNx", "both"], default="both")
    p.add_argument("--mode", choices=["test", "fast", "full"], default="full",
                   help="MD length: test=ultra-short, fast=short, full=Kim et al.")
    p.add_argument("--n-bulk", type=int, default=3,
                   help="number of independent bulk replicas (2 surfaces each)")
    p.add_argument("--target-accepted", type=int, default=None,
                   help="keep generating extra bulks until this many surfaces pass")
    p.add_argument("--max-extra-bulk", type=int, default=3)
    p.add_argument("--no-gate", action="store_true",
                   help="disable the quality gate (keep all slabs)")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--device", default=None,
                   help="torch device for MACE (auto-detect if omitted)")
    p.add_argument("--supercell", type=int, nargs=3, default=None,
                   metavar=("A", "B", "C"),
                   help="crystal supercell (default: 3 3 2 SiO2 / 2 2 3 SiNx)")
    p.add_argument("--no-cache", action="store_true",
                   help="disable bulk caching")
    p.add_argument("--cache-dir", default=None,
                   help="directory for cached bulk structures (default: bulk_cache/)")
    p.add_argument("--list-cache", action="store_true",
                   help="list cached bulks and exit")
    p.add_argument("--use-published", action="store_true",
                   help="load published amorphous bulks from the literature dir")
    p.add_argument("--literature-dir", default=None,
                   help="directory of published bulk structures (default: literature/)")
    p.add_argument("--prefix", default="surface",
                   help="output filename prefix: <material>_<prefix>_<i>.xyz")
    p.add_argument("--heartbeat-sec", type=float, default=15.0,
                   help="wall-clock seconds between MD progress heartbeats "
                        "(0 to silence)")
    p.add_argument("--outdir", default="surfaces_out")
    p.add_argument("--report-exposure", action="store_true",
                   help="print total-vs-exposed site densities (Problem-A check)")
    p.add_argument("--cluster-report", action="store_true",
                   help="print representative-site clustering (Problem-B check)")
    p.add_argument("--quench-sweep", type=str, default=None,
                   help="comma-separated quench durations (ps) cycled across "
                        "bulks for cooling-rate variation, e.g. '8,15,30'")
    return p.parse_args()


def make_calculator(dtype, device):
    try:
        from mace.calculators import mace_mp
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        calc = mace_mp(model="medium-omat-0", device=device, default_dtype=dtype)
        print(f"[calc] MACE-omat-0 loaded on {device} ({dtype})")
        return calc
    except Exception as e:
        print(f"[calc] ERROR loading MACE ({e}).")
        print("[calc] Refusing to fall back to LJ for a production run. "
              "Check your env / GPU visibility.")
        sys.exit(1)


def report_exposure(slab, material):
    counts = sb.classify_sites(slab, material, exposure_filter=True)
    area = sb.surface_area_nm2(slab)
    lines = []
    for st, v in counts.items():
        lines.append(f"    {st}: total={v['total']} ({v['total']/area:.1f}/nm^2), "
                     f"exposed={v['exposed']} ({v['exposed']/area:.1f}/nm^2)")
    return "\n".join(lines)


def report_clustering(slab, material, n_rep=3):
    counts = sb.classify_sites(slab, material, exposure_filter=True)
    prim = ["OH", "O_bridge"] if material == "SiO2" else ["NH2", "NH_bridge"]
    lines = []
    for st in prim:
        idx = counts.get(st, {}).get("indices", [])
        reps = sb.cluster_representative_sites(slab, idx, n_representatives=n_rep)
        lines.append(f"    {st}: {len(idx)} sites -> {len(reps)} representatives "
                     f"(weights {[round(w,2) for _, w in reps]})")
    return "\n".join(lines)


def run_material(material, calc, args, report_fh):
    header = f"\n=== Building {material} (mode={args.mode}, n_bulk={args.n_bulk}) ==="
    print(header); report_fh.write(header + "\n")

    t0 = time.time()
    quench_sweep = None
    if args.quench_sweep:
        quench_sweep = [float(x) for x in args.quench_sweep.split(",")]
        print(f"[{material}] cooling-rate sweep (quench ps): {quench_sweep}")

    kw = {}
    if args.supercell:
        kw["supercell"] = tuple(args.supercell)
    surfaces, summary = sb.build_surface_ensemble(
        material, calc,
        n_bulk=args.n_bulk,
        target_accepted=args.target_accepted,
        max_extra_bulk=args.max_extra_bulk,
        apply_gate=(not args.no_gate),
        quench_sweep=quench_sweep,
        use_cache=(not args.no_cache),
        cache_dir=args.cache_dir,
        use_published=args.use_published,
        literature_dir=args.literature_dir,
        **kw,
    )
    elapsed = time.time() - t0

    for i, s in enumerate(surfaces):
        # keep only scalar provenance ASE's extxyz writer can serialise
        s_clean = s.copy()
        keep = {}
        for k in ("cooling_rate_K_per_ps", "quench_ps", "clumping_R"):
            if k in s.info and s.info[k] is not None:
                keep[k] = float(s.info[k])
        n_strained = len(s.info.get("strained_sites", []))
        s_clean.info.clear()
        s_clean.info.update(keep)
        s_clean.info["n_strained_sites"] = n_strained
        write(os.path.join(args.outdir, f"{material}_{args.prefix}_{i}.xyz"), s_clean)

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o
    with open(os.path.join(args.outdir, f"{material}_summary.json"), "w") as jf:
        json.dump(_clean(summary), jf, indent=2)

    rpt = [f"  surfaces written: {len(surfaces)}",
           f"  accepted={summary.get('_n_accepted')} rejected={summary.get('_n_rejected')}",
           f"  wall time: {elapsed/60:.1f} min"]
    for st, info in summary.items():
        if not str(st).startswith("_"):
            rpt.append(f"    {st}: {info['mean_nm2']} +/- {info['std_nm2']} nm^-2 "
                       f"(literature {info['literature']})")
    if surfaces and args.report_exposure:
        rpt.append("  Problem-A (total vs exposed density), surface 0:")
        rpt.append(report_exposure(surfaces[0], material))
    if surfaces and args.cluster_report:
        rpt.append("  Problem-B (representative-site clustering), surface 0:")
        rpt.append(report_clustering(surfaces[0], material))

    block = "\n".join(rpt)
    print(block); report_fh.write(block + "\n")
    return surfaces, summary


def main():
    args = parse_args()

    # list cached bulks and exit (no calculator needed)
    if args.list_cache:
        sb.list_cached_bulks(args.cache_dir)
        return

    os.makedirs(args.outdir, exist_ok=True)
    sb.MD_HEARTBEAT_SEC = args.heartbeat_sec

    {"test": sb.use_test_protocol,
     "fast": sb.use_fast_protocol,
     "full": sb.use_full_protocol}[args.mode]()

    calc = make_calculator(args.dtype, args.device)

    materials = ["SiO2", "SiNx"] if args.material == "both" else [args.material]

    t_all = time.time()
    with open(os.path.join(args.outdir, "build_report.txt"), "w") as report_fh:
        report_fh.write(f"AS-ALD surface build report\n"
                        f"mode={args.mode} n_bulk={args.n_bulk} "
                        f"dtype={args.dtype} gate={not args.no_gate}\n")
        for material in materials:
            run_material(material, calc, args, report_fh)

    total = time.time() - t_all
    print(f"\n[done] all materials in {total:.0f}s ({total/60:.1f} min); "
          f"outputs in {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()
