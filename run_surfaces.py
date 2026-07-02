#!/usr/bin/env python3
"""
run_surfaces.py
===============
CLI driver for the AS-ALD amorphous surface builder (Challenge 4).

Replaces the Colab notebook cells with a standalone script that can run
on any system with a GPU (or CPU, with LJ fallback).

Usage:
    # Quick validation (~3-5 min per material on RTX 4060)
    python run_surfaces.py --mode test

    # Fast exploration (~10-15 min per material)
    python run_surfaces.py --mode fast

    # Full production run (~2-4 hours per material)
    python run_surfaces.py --mode full

    # Single material only
    python run_surfaces.py --mode test --materials SiO2

    # Custom number of bulk replicas
    python run_surfaces.py --mode full --n-bulk 5 --target-accepted 6

    # Skip quality gate (keep all surfaces)
    python run_surfaces.py --mode test --no-gate

    # Use float64 for higher accuracy (slower)
    python run_surfaces.py --mode full --dtype float64
"""

import argparse
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description="AS-ALD Amorphous Surface Builder (Challenge 4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["test", "fast", "full"], default="test",
        help="MD protocol: test (~3-5 min), fast (~10-15 min), full (~2-4 hrs) per material"
    )
    parser.add_argument(
        "--materials", nargs="+", default=["SiO2", "SiNx"],
        choices=["SiO2", "SiNx"],
        help="Which materials to build surfaces for (default: both)"
    )
    parser.add_argument(
        "--n-bulk", type=int, default=None,
        help="Number of bulk replicas (default: 1 for test/fast, 3 for full)"
    )
    parser.add_argument(
        "--target-accepted", type=int, default=None,
        help="Minimum number of accepted surfaces (triggers over-generation)"
    )
    parser.add_argument(
        "--no-gate", action="store_true",
        help="Disable quality gate (keep all surfaces)"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable bulk caching (always rerun melt-quench)"
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="Directory for cached bulk structures (default: bulk_cache/)"
    )
    parser.add_argument(
        "--list-cache", action="store_true",
        help="List cached bulks and exit"
    )
    parser.add_argument(
        "--use-published", action="store_true",
        help="Try to load published amorphous bulks from literature/ directory"
    )
    parser.add_argument(
        "--literature-dir", default=None,
        help="Directory containing published bulk structures (default: literature/)"
    )
    parser.add_argument(
        "--supercell", type=int, nargs=3, default=[2, 2, 2],
        metavar=("A", "B", "C"),
        help="Crystal supercell size (default: 2 2 2)"
    )
    parser.add_argument(
        "--dtype", default="float32", choices=["float32", "float64"],
        help="MACE precision (float32 is faster, float64 is more accurate)"
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Directory for output .xyz files (default: current directory)"
    )
    parser.add_argument(
        "--prefix", default=None,
        help="Output filename prefix (default: based on mode)"
    )

    args = parser.parse_args()

    # Set defaults based on mode
    if args.n_bulk is None:
        args.n_bulk = 1 if args.mode in ("test", "fast") else 3
    if args.prefix is None:
        args.prefix = {"test": "quick", "fast": "fast", "full": "prod"}[args.mode]

    # Import surface builder
    import surface_builder as sb

    # Handle --list-cache
    if args.list_cache:
        sb.list_cached_bulks(args.cache_dir)
        return

    supercell = tuple(args.supercell)

    # Select protocol
    {"test": sb.use_test_protocol,
     "fast": sb.use_fast_protocol,
     "full": sb.use_full_protocol}[args.mode]()

    # Load calculator
    print("\n--- Loading calculator ---")
    calc = sb.get_calculator(dtype=args.dtype)
    print(f"Calculator type: {type(calc).__name__}\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build surfaces for each material
    all_results = {}
    total_start = time.time()

    for material in args.materials:
        print(f"\n{'='*60}")
        print(f"  Building {material} surfaces ({args.mode.upper()} mode)")
        print(f"  n_bulk={args.n_bulk}, gate={'OFF' if args.no_gate else 'ON'}, "
              f"cache={'OFF' if args.no_cache else 'ON'}, supercell={supercell}")
        print(f"{'='*60}")

        mat_start = time.time()

        surfaces, summary = sb.build_surface_ensemble(
            material, calc,
            n_bulk=args.n_bulk,
            target_accepted=args.target_accepted,
            apply_gate=not args.no_gate,
            supercell=supercell,
            use_cache=not args.no_cache,
            cache_dir=args.cache_dir,
            use_published=args.use_published,
            literature_dir=args.literature_dir,
        )

        mat_elapsed = time.time() - mat_start

        # Save surfaces
        from ase.io import write
        for i, s in enumerate(surfaces):
            outpath = os.path.join(args.output_dir, f"{material}_{args.prefix}_{i}.xyz")
            write(outpath, s)
            print(f"  Saved: {outpath}")

        # Print summary
        print(f"\n--- {material} Summary ---")
        print(f"  Accepted: {summary['_n_accepted']}, "
              f"Rejected: {summary['_n_rejected']}")
        print(f"  Time: {mat_elapsed:.0f}s ({mat_elapsed/60:.1f} min)")
        print(f"  Exposed-site densities:")
        for st, info in summary.items():
            if not st.startswith("_"):
                lit = info['literature']
                lit_str = f"{lit[0]} ± {lit[1]}" if lit else "N/A"
                print(f"    {st:12s}: {info['mean_nm2']:5.2f} ± {info['std_nm2']:.2f} nm⁻² "
                      f"(literature: {lit_str} nm⁻²)")

        all_results[material] = (surfaces, summary)

        # Problem A check for SiO2: total vs exposed -OH density
        if material == "SiO2" and surfaces:
            print(f"\n  --- Problem A Check (total vs exposed -OH) ---")
            slab = surfaces[0]
            counts = sb.classify_sites(slab, "SiO2", exposure_filter=True)
            area = sb.surface_area_nm2(slab)
            for st, v in counts.items():
                print(f"    {st}: total={v['total']} ({v['total']/area:.1f}/nm²), "
                      f"exposed={v['exposed']} ({v['exposed']/area:.1f}/nm²)")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  All done in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  Output files in: {os.path.abspath(args.output_dir)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
