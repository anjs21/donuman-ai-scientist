# AS-ALD Co-Scientist — Full Pipeline (Phases 0–5)

This document covers the modules added on top of the surface builder
(`surface_builder.py` / `run_surfaces.py`, documented in `README.md`) to turn it
into a complete in-silico AI co-scientist for Challenge 4. It takes the
amorphous surfaces the builder produces and reasons all the way to the target
metric: **passivate SiN so SiOx grows on SiO with 90% selectivity at 10 nm.**

## Modules

| Phase | File | Role |
|---|---|---|
| 0 | `structure_validation.py` | Measured surface area (replaces the 1.5× roughness constant) + RDF / coordination / density validation vs literature. |
| 2a | `inhibitor_library.py` | Curated inhibitor/precursor/co-reactant library: gas-phase geometry builders + reaction templates (dissociative or physisorption). |
| 2b | `energetics.py` | MACE reaction/adsorption energy `dE` (eV) at each representative reactive site; local relaxation keeps each site cheap. |
| 3 | `selection_agent.py` + `selection_criteria.md` | Glass-box agent ranking candidates by GS/NGS selectivity contrast against editable thresholds; recommends inhibitor + precursor. |
| 4 | `selectivity_model.py` | Maps `dE` → site blocking → growth-per-cycle → thickness → selectivity vs cycle; reports whether the 90%-at-10nm target is met. |
| 5 | `run_pipeline.py` | Orchestrates all phases and writes `report.json` / `report.md` + selectivity CSVs. |

## Roles in the target process

- **Growth surface (GS) = SiO2** — grow SiOx here; inhibitor should *spare* it.
- **Non-growth surface (NGS) = SiNx** — block here; inhibitor should *bind* it.
- The agent ranks by `contrast = dE_GS − dE_NGS` (large positive = selective blocker).

## Running it

```bash
# fast smoke test of the whole chain (uses TEST-mode surfaces)
python3 run_pipeline.py --mode test

# reuse surfaces already built by run_surfaces.py, screen the full library
python3 run_pipeline.py --use-existing "SiO2_*_*.xyz" "SiNx_*_*.xyz"

# restrict the screen / control cost
python3 run_pipeline.py --use-existing "SiO2_*.xyz" "SiNx_*.xyz" \
        --reagents DMATMS TMCS pyridine BDMAS --max-sites 3
```

Each module self-tests standalone (`python3 <module>.py`) — the agent and
selectivity model run on synthetic energies with no GPU needed, so you can
exercise the decision logic instantly.

## Extending

- Add a chemistry: append a `Reagent` to `LIBRARY` in `inhibitor_library.py`.
- Change selection policy: edit thresholds in `selection_criteria.md` (read at run time).
- Calibrate kinetics: fit `GrowthParams` in `selectivity_model.py` to experimental data.

## Honest caveats

- MACE-MP-0 adsorption energies for bond-breaking chemistry are approximate;
  validate top candidates against DFT before trusting absolute `dE`.
- The selectivity model is a transparent phenomenological link, not a
  first-principles growth simulation — its parameters are meant to be calibrated.
