# AS-ALD In-Silico AI Co-Scientist — Challenge 4

An end-to-end, reproducible pipeline for **area-selective atomic layer
deposition (AS-ALD)** design. It builds realistic amorphous surfaces, screens
inhibitor/precursor chemistries against their reactive sites with a machine-
learning interatomic potential (MACE-MP-0), and reasons about which chemistry
delivers the target process:

> **Passivate SiN (nitride) so SiOx deposits on SiO (oxide) with 90%
> selectivity at 10 nm of oxide thickness.**

## What this implements (mapped to the challenge brief)

| Brief requirement | Module |
|---|---|
| **1. Amorphous surface builder** that reflects experiment | `surface_builder.py` (melt-quench + passivation + exposure filter) |
| — over-counting / structural realism controls | `structure_validation.py` (measured area, RDF, coordination, density) |
| **2. Agentic selection logic** for inhibitor/precursor candidates | `energetics.py` + `selection_agent.py` + `selection_criteria.md` |
| The 90%-at-10nm selectivity target | `selectivity_model.py` |
| Ease of use with Python | `run_surfaces.py`, `run_pipeline.py` (CLIs) |

## Pipeline (phases)

```
Phase 1  surface_builder.py       amorphous SiO2 + SiNx slabs (MACE melt-quench)
Phase 0  structure_validation.py  measured area, RDF/CN/density vs literature
Phase 2  energetics.py            per-site reaction/adsorption energy (dE, eV)
Phase 3  selection_agent.py       glass-box ranking by GS/NGS selectivity contrast
Phase 4  selectivity_model.py     dE -> growth-per-cycle -> thickness -> selectivity
Phase 5  run_pipeline.py          chains all of the above -> report.json / report.md
```

**Roles in the target process:** growth surface (GS) = SiO2 (grow film here),
non-growth surface (NGS) = SiNx (block here). A good inhibitor binds the NGS
strongly and the GS weakly — the agent ranks by `contrast = dE_GS − dE_NGS`.

## Install

```bash
pip install -r requirements.txt
# install a CUDA torch build matching your GPU driver, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

A CUDA GPU is strongly recommended (surfaces were validated on an RTX 4060).
Without MACE the code still runs on a Lennard-Jones placeholder for plumbing
tests, but the energies are not physical (the report is flagged accordingly).

## Quick start

```bash
# 1. Build a few quick surfaces (minutes; rough numbers)
python run_surfaces.py --mode test

# 2. Run the whole co-scientist end-to-end on those surfaces
python run_pipeline.py --mode test

# Reuse existing production surfaces and screen the full reagent library:
python run_pipeline.py --use-existing "SiO2_*_*.xyz" "SiNx_*_*.xyz"

# Production surfaces (overnight) then full pipeline:
python run_surfaces.py --mode full --n-bulk 3 --target-accepted 4
python run_pipeline.py --use-existing "SiO2_full_*.xyz" "SiNx_full_*.xyz"
```

Each module also runs standalone as a self-test:

```bash
python inhibitor_library.py     # builds every molecule, checks geometry
python selection_agent.py       # agent logic on synthetic energetics
python selectivity_model.py     # selectivity curves on synthetic dE
python structure_validation.py *.xyz   # validate built surfaces
```

## Outputs

- `report.json` / `report.md` — full run: validation, energetics, ranked
  candidates, recommended inhibitor + precursor, and selectivity vs target.
- `selectivity_<reagent>.csv` — cycle-resolved thickness + selectivity curves.
- `<material>_<mode>_<i>.xyz` — the amorphous surfaces.

## Extending

- **New chemistry:** add a `Reagent` to `LIBRARY` in `inhibitor_library.py`
  (geometry builder + reaction template). The screen, agent, and selectivity
  model pick it up automatically.
- **New film (ZrO2/TiO2/HfO2, SiNx):** add its precursor(s) and, if needed, a
  crystal builder + composition in `surface_builder.py`.
- **Selection policy:** edit thresholds in `selection_criteria.md` — the agent
  reads them at run time.
- **Kinetics calibration:** fit `GrowthParams` in `selectivity_model.py` to
  experimental break-through curves.

## Modeling caveats (be honest with the judges)

- MACE-MP-0 accuracy for bond-breaking adsorption is approximate; validate the
  top candidates' dE against DFT before trusting absolute values.
- Melt-quench uses small cells and fast quench rates; use the ensemble +
  quality gate (`surface_builder`) and structural validation (`Phase 0`) to
  bound the variance, and report spreads, not single numbers.
- `selectivity_model.py` is a transparent phenomenological link, not a
  first-principles growth simulation — its parameters are meant to be calibrated.
