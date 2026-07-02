#!/usr/bin/env python3
"""
run_surface_builder.py
======================
Command-line driver for the AS-ALD amorphous surface builder (Challenge 4),
intended for a local GPU cluster (e.g. V100 nodes) rather than a notebook.

Keeps surface_builder.py as the library; this script just parses arguments,
runs the ensemble, writes structures, and prints a validation report.

Examples
--------
# quick pipeline check (ultra-short MD, gate off), both materials:
    python run_surface_builder.py --mode test --n-bulk 1 --no-gate

# full overnight production, 3 bulks, require >=4 accepted surfaces each:
    python run_surface_builder.py --mode full --n-bulk 3 --target-accepted 4

# only SiO2, full protocol, float64 (highest accuracy, slower):
    python run_surface_builder.py --material SiO2 --mode full --dtype float64

# calibrate the Problem-A exposure filter and print total-vs-exposed OH:
    python run_surface_builder.py --material SiO2 --mode full --report-exposure

Outputs
-------
  <outdir>/<material>_surface_<i>.xyz        one file per accepted surface
  <outdir>/<material>_summary.json           density summary + QC metadata
  <outdir>/build_report.txt                  human-readable validation report
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from ase.io import write

import surface_builder as sb


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="AS-ALD amorphous surface builder")
    p.add_argument("--material", choices=["SiO2", "SiNx", "both"], default="both",
                   help="which substrate(s) to build")
    p.add_argument("--mode", choices=["test", "fast", "full"], default="full",
                   help="MD length: test=ultra-short, fast=short, full=Kim et al.")
    p.add_argument("--n-bulk", type=int, default=3,
                   help="number of independent bulk replicas (2 surfaces each)")
    p.add_argument("--target-accepted", type=int, default=None,
                   help="keep generating extra bulks until this many surfaces "
                        "pass the quality gate")
    p.add_argument("--max-extra-bulk", type=int, default=3,
                   help="cap on extra bulks generated to reach --target-accepted")
    p.add_argument("--no-gate", action="store_true",
                   help="disable the quality gate (keep all slabs; for testing)")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                   help="MACE precision (float32 faster; float64 more accurate)")
    p.add_argument("--device", default="cuda",
                   help="torch device for MACE (cuda / cuda:0 / cpu)")
    p.add_argument("--outdir", default="surfaces_out",
                   help="output directory")
    p.add_argument("--report-exposure", action="store_true",
                   help="print total-vs-exposed site densities (Problem-A check)")
    p.add_argument("--cluster-report", action="store_true",
                   help="print representative-site clustering (Problem-B check)")
    p.add_argument("--quench-sweep", type=str, default=None,
                   help="comma-separated quench durations (ps) to cycle across "
                        "bulks for cooling-rate variation, e.g. '8,15,30'. "
                        "Faster cooling (shorter) = more reactive sites; slower "
                        "(longer) = more relaxed. Omit for single default rate.")
    p.add_argument("--seed-offset", type=int, default=0,
                   help="added to all RNG seeds (for independent reruns)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Calculator with explicit device
# ---------------------------------------------------------------------------

def make_calculator(dtype, device):
    try:
        from mace.calculators import mace_mp
        calc = mace_mp(model="medium-omat-0", device=device, default_dtype=dtype)
        print(f"[calc] MACE-omat-0 loaded on {device} ({dtype})")
        return calc
    except Exception as e:
        print(f"[calc] ERROR loading MACE ({e}).")
        print("[calc] Refusing to fall back to LJ for a production run. "
              "Check your conda env / GPU visibility.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_material(material, calc, args, report_fh):
    header = f"\n=== Building {material} (mode={args.mode}, n_bulk={args.n_bulk}) ==="
    print(header); report_fh.write(header + "\n")

    t0 = time.time()
    quench_sweep = None
    if args.quench_sweep:
        quench_sweep = [float(x) for x in args.quench_sweep.split(",")]
        print(f"[{material}] cooling-rate sweep (quench ps): {quench_sweep}")
    surfaces, summary = sb.build_surface_ensemble(
        material, calc,
        n_bulk=args.n_bulk,
        target_accepted=args.target_accepted,
        max_extra_bulk=args.max_extra_bulk,
        apply_gate=(not args.no_gate),
        quench_sweep=quench_sweep,
    )
    elapsed = time.time() - t0

    # write structures. Move our custom metadata out of .info before writing,
    # because ASE's extxyz writer tries to serialise .info entries and chokes
    # on variable-length per-surface lists (strained_sites etc.) when surfaces
    # have different atom counts. We keep the metadata in the JSON summary
    # instead. Save as .xyz for the geometry; provenance lives in the report.
    for i, s in enumerate(surfaces):
        s_clean = s.copy()
        # preserve scalar provenance as simple strings ASE can serialise
        keep = {}
        for k in ("cooling_rate_K_per_ps", "quench_ps", "clumping_R"):
            if k in s.info and s.info[k] is not None:
                keep[k] = float(s.info[k])
        n_strained = len(s.info.get("strained_sites", []))
        s_clean.info.clear()
        s_clean.info.update(keep)
        s_clean.info["n_strained_sites"] = n_strained
        path = os.path.join(args.outdir, f"{material}_surface_{i}.xyz")
        write(path, s_clean)

    # write summary json (numpy-safe)
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

    # report
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
    os.makedirs(args.outdir, exist_ok=True)

    # select protocol
    {"test": sb.use_test_protocol,
     "fast": sb.use_fast_protocol,
     "full": sb.use_full_protocol}[args.mode]()

    calc = make_calculator(args.dtype, args.device)

    materials = ["SiO2", "SiNx"] if args.material == "both" else [args.material]

    with open(os.path.join(args.outdir, "build_report.txt"), "w") as report_fh:
        report_fh.write(f"AS-ALD surface build report\n"
                        f"mode={args.mode} n_bulk={args.n_bulk} "
                        f"dtype={args.dtype} gate={not args.no_gate}\n")
        for material in materials:
            run_material(material, calc, args, report_fh)

    print(f"\n[done] outputs in {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()
