#!/usr/bin/env python3
"""
run_pipeline.py
===============
End-to-end driver for the AS-ALD in-silico co-scientist (Challenge 4).

Chains all phases into a single reproducible run:

    Phase 1  surface_builder      build (or load) amorphous SiO2 + SiNx surfaces
    Phase 0  structure_validation measure area + validate RDF/CN/density
    Phase 2  energetics           screen inhibitors/precursors per reactive site
    Phase 3  selection_agent      rank candidates, recommend inhibitor+precursor
    Phase 4  selectivity_model    predict selectivity vs 90%-at-10nm target
    Phase 5  (this script)        write JSON + markdown report

Typical use
-----------
    # fast smoke test of the whole chain (minutes; numbers are rough)
    python run_pipeline.py --mode test

    # reuse surfaces already built by run_surface_builder.py, screen the full library
    python run_pipeline.py --use-existing "SiO2_prod_*.xyz" "SiNx_prod_*.xyz"

    # production: full-accuracy surfaces + full library
    python run_pipeline.py --mode full --n-bulk 3 --target-accepted 4

Outputs: report.json, report.md, and one selectivity CSV per candidate.
"""

import argparse
import glob
import json
import os
import time


def build_or_load_surfaces(args, sb, calc):
    """Return {material: [slab, ...]} either from disk or a fresh build."""
    from ase.io import read
    surfaces = {}

    if args.use_existing:
        patterns = args.use_existing
        for material in args.materials:
            files = []
            for pat in patterns:
                files += [f for f in glob.glob(pat) if material in os.path.basename(f)]
            files = sorted(set(files))
            if not files:
                print(f"  [load] no existing files matched for {material}")
            surfaces[material] = [read(f) for f in files]
            print(f"  [load] {material}: {len(surfaces[material])} surface(s) "
                  f"from disk")
        return surfaces

    {"test": sb.use_test_protocol,
     "fast": sb.use_fast_protocol,
     "full": sb.use_full_protocol}[args.mode]()

    for material in args.materials:
        print(f"\n[Phase 1] Building {material} surfaces ({args.mode} mode)...")
        slabs, summary = sb.build_surface_ensemble(
            material, calc,
            n_bulk=args.n_bulk,
            target_accepted=args.target_accepted,
            apply_gate=not args.no_gate,
        )
        surfaces[material] = slabs
        from ase.io import write
        for i, s in enumerate(slabs):
            path = os.path.join(args.output_dir, f"{material}_{args.mode}_{i}.xyz")
            write(path, s)
        print(f"  saved {len(slabs)} {material} surface(s)")
    return surfaces


def validate_surfaces(surfaces):
    """Phase 0: structural validation + measured area for every surface."""
    import structure_validation as sv
    report = {}
    for material, slabs in surfaces.items():
        report[material] = []
        for i, slab in enumerate(slabs):
            area = sv.true_surface_area_nm2(slab)
            val = sv.validate_structure(slab, material, verbose=False)
            report[material].append({
                "index": i,
                "area_nm2": round(area, 3),
                "structure_ok": val["_ok"],
                "metrics": {k: v for k, v in val.items() if not k.startswith("_")},
            })
            print(f"  [Phase 0] {material}[{i}]: area={area:.2f} nm^2, "
                  f"structure {'OK' if val['_ok'] else 'off-target'} "
                  f"({val['_passed']}/{val['_total']})")
    return report


def main():
    ap = argparse.ArgumentParser(
        description="AS-ALD in-silico co-scientist — full pipeline (Challenge 4)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--mode", choices=["test", "fast", "full"], default="test")
    ap.add_argument("--materials", nargs="+", default=["SiO2", "SiNx"],
                    choices=["SiO2", "SiNx"])
    ap.add_argument("--n-bulk", type=int, default=1)
    ap.add_argument("--target-accepted", type=int, default=None)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--use-existing", nargs="+", default=None,
                    help="glob pattern(s) of existing .xyz surfaces to reuse")
    ap.add_argument("--reagents", nargs="+", default=None,
                    help="restrict screen to these reagent names")
    ap.add_argument("--max-sites", type=int, default=3,
                    help="representative sites per site-type per surface")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--output-dir", default=".")
    ap.add_argument("--report", default="report")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    import surface_builder as sb
    import inhibitor_library as lib
    import energetics as en
    import selection_agent as agent
    import selectivity_model as sm

    print("=" * 66)
    print("AS-ALD in-silico co-scientist — end-to-end pipeline")
    print("=" * 66)

    print("\n[calc] loading MLIP...")
    calc = sb.get_calculator(dtype=args.dtype)
    is_placeholder = type(calc).__name__ == "LennardJones"
    if is_placeholder:
        print("  [WARN] running on Lennard-Jones placeholder: energies are NOT "
              "physical. Install mace-torch for meaningful numbers.")

    # -- Phase 1: surfaces --
    surfaces = build_or_load_surfaces(args, sb, calc)
    if not any(surfaces.values()):
        print("No surfaces available; aborting.")
        return

    # -- Phase 0: validation --
    print("\n[Phase 0] Structural validation")
    validation = validate_surfaces(surfaces)

    # -- Phase 2: energetics screen --
    print("\n[Phase 2] Energetics screen")
    reagents = lib.get_reagents(names=args.reagents) if args.reagents \
        else lib.get_reagents()
    energetics = en.screen_reagents(surfaces, reagents, calc,
                                    max_sites=args.max_sites)

    # -- Phase 3: selection agent --
    print("\n[Phase 3] Selection agent")
    cfg = agent.load_config()
    # gather strain info: reagents whose favourable sites are strain-flagged
    selection = agent.select(energetics, cfg)
    print(agent.explain(selection))

    # -- Phase 4: selectivity vs target --
    print("\n[Phase 4] Selectivity model")
    p = sm.GrowthParams()
    selectivity = {}
    for cand in selection["inhibitors"]:
        dE_GS, dE_NGS = cand["dE_GS"], cand["dE_NGS"]
        if dE_NGS is None:
            continue
        res = sm.selectivity_curve(dE_GS if dE_GS is not None else 0.0, dE_NGS, p)
        print(sm.summarize(cand["name"], res, p))
        csv_path = os.path.join(args.output_dir,
                                f"selectivity_{cand['name']}.csv")
        sm.write_curve_csv(csv_path, res)
        selectivity[cand["name"]] = {
            k: (float(v) if hasattr(v, "__float__") and not hasattr(v, "__len__")
                else None)
            for k, v in res.items() if not hasattr(v, "__len__")
        }

    # -- Phase 5: report --
    report = {
        "mode": args.mode,
        "materials": args.materials,
        "placeholder_calc": is_placeholder,
        "validation": validation,
        "energetics": energetics,
        "selection": selection,
        "selectivity": selectivity,
        "growth_params": vars(p),
        "runtime_s": round(time.time() - t0, 1),
    }
    json_path = os.path.join(args.output_dir, f"{args.report}.json")
    # guard: --report may include a sub-path; make sure its dir exists so a full
    # (expensive) run is never lost to a missing-directory error at the last step
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    md_path = os.path.join(args.output_dir, f"{args.report}.md")
    with open(md_path, "w") as fh:
        fh.write(_render_markdown(report, selection, selectivity, p))

    print(f"\n[Phase 5] Wrote {json_path} and {md_path}")
    print(f"Total runtime: {report['runtime_s']:.0f} s")


def _render_markdown(report, selection, selectivity, p):
    rec = selection["recommendation"]
    L = ["# AS-ALD Co-Scientist Report", ""]
    if report["placeholder_calc"]:
        L += ["> **WARNING:** run on Lennard-Jones placeholder — energies are "
              "not physical. Re-run with MACE installed.", ""]
    L += [f"- Mode: `{report['mode']}`  |  materials: {report['materials']}",
          f"- Runtime: {report['runtime_s']} s", "",
          "## Recommendation", ""]
    if rec["inhibitor"]:
        L += [f"- **Inhibitor:** {rec['inhibitor']} "
              f"(dE_NGS={rec['inhibitor_dE_NGS']:+.2f} eV, "
              f"dE_GS={rec['inhibitor_dE_GS']:+.2f} eV, "
              f"contrast={rec['inhibitor_contrast']:+.2f} eV)"]
    else:
        L += ["- **Inhibitor:** none met criteria"]
    L += [f"- **Precursor:** {rec['precursor']}", "",
          "## Inhibitor ranking", "",
          "| reagent | dE_GS (eV) | dE_NGS (eV) | contrast | score | status |",
          "|---|---|---|---|---|---|"]
    for c in selection["inhibitors"]:
        gs = "n/a" if c["dE_GS"] is None else f"{c['dE_GS']:+.2f}"
        ng = "n/a" if c["dE_NGS"] is None else f"{c['dE_NGS']:+.2f}"
        ct = "n/a" if c["contrast"] is None else f"{c['contrast']:+.2f}"
        status = "accept" if c["accepted"] else "reject"
        L += [f"| {c['name']} | {gs} | {ng} | {ct} | {c['score']:+.2f} | {status} |"]
    L += ["", "## Selectivity vs target "
          f"({p.target_selectivity*100:.0f}% @ {p.target_thickness_nm:.0f} nm)", "",
          "| inhibitor | S at 10 nm | max thk @ target S (nm) | meets? |",
          "|---|---|---|---|"]
    for name, s in selectivity.items():
        sat = s.get("selectivity_at_target")
        mx = s.get("max_thickness_at_target_selectivity_nm")
        meets = "yes" if s.get("meets_target") else "no"
        sat_s = "n/a" if sat is None else f"{sat*100:.1f}%"
        mx_s = "n/a" if mx is None else f"{mx:.2f}"
        L += [f"| {name} | {sat_s} | {mx_s} | {meets} |"]
    L += ["", "_Generated by run_pipeline.py (Challenge 4)._"]
    return "\n".join(L)


if __name__ == "__main__":
    main()
