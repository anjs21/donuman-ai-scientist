# WARDEN — results and status (Merck Innovation Cup, Challenge 4)

**Task.** Passivate SiNx so that SiO2 grows only on SiO2: ≥ 90 % selectivity at
10 nm, found in silico. Deliverables: (1) an amorphous surface builder and
(2) agentic selection of inhibitors and precursors.

**Bottom line.** The full pipeline runs end to end: validated amorphous surfaces →
literature-grounded proposals (LLM planner and an independent LLM critic,
verbatim-quote checks, live Valency search) → MACE adsorption on the slabs →
GFN2-xTB / MACE reaction barriers → uncertainty-aware multi-cycle process design →
a control gate that refuses to rank when known chemistry is not reproduced.
**No new molecule has yet passed the control-gated 90 %/10 nm criterion**, and
the pipeline says so instead of reporting a hit. Even the acylation control
loses its apparent selectivity once the agent explores every pathway. The main
reasons are method limitations that the pipeline found itself (section 3).

## 1. What is built

| Part | Where | What it does |
|---|---|---|
| Amorphous surface builder | `surface_builder.py`, `structure_validation.py`, `warden/surfaces.py` | Melt-quench SiO2 / SiNx surfaces, terminations, admission checks (coordination, bond lengths, density, defects, Si–Si bonds), site classification (OH, NH2, NH bridge) with population weights |
| Reaction templates | `warden/chemistry.py` | physisorption, silyl substitution, acylation, carbonyl addition, imine condensation (Si–NH2 + R2C=O → Si–N=CR2 + H2O). Element balance and connectivity are checked; nothing a model writes is executed |
| Adsorption screen | `warden/engine.py` | MACE-MATPES-r2SCAN-0 relaxations at representative sites of both slabs, with every failure recorded |
| Barriers | `warden/kinetics.py` | Climbing-image NEB at gas-phase site models or clusters carved from the slabs. Unphysical bands are rejected, never reported |
| Process model | `warden/process.py`, `warden/assess.py` | Eyring rates, Langmuir pre-complex occupancy, dose windows, redosing, cap loss and nucleation over many cycles; Monte Carlo over uncertain inputs gives P(S ≥ 90 % at 10 nm); "measure first" ranks the inputs that limit confidence |
| Control gate | `warden/assess.py`, `examples/controls-v3.json` | Known molecules (DMATMS, BDIPADS → oxide; acetyl chloride → nitride) must come out right before any candidate is ranked |
| Evidence-acquisition agent | `warden/decide.py` | Chooses the next calculation (molecule, site, pathway, orientation, method) by value of information, with learned method bias and failure rates; stops when the pick is confident |
| LLM campaign | `warden/campaign.py`, `warden/planner.py` | Planner and critic (OpenAI or local Ollama), literature search, analog design, budgets, resumable state; every step is logged |

## 2. Results

### Surfaces (Kim et al. reference slabs, Zenodo 10.5281/zenodo.18081511)

| Slab | Formula | Checks | Defects | Sites |
|---|---|---|---|---|
| SiO2_kim_1 | H30O118Si54 | 4/4 pass (density 2.15 g/cm³, Si CN 3.91) | none | 24 OH |
| SiNx_kim_1 | H59N99O8Si72 | 3/4 pass; density 2.14 g/cm³ fails and is exempted with citation | 36 % of Si mis-coordinated, 23 Si–Si bonds | 19 NH2, 14 NH bridge |

The admitted SiNx reference is far from ideal Si3N4; this matters for the barriers below.

### Control: acetyl chloride (acylation; amine ≫ alcohol is textbook chemistry)

| Site model | Ea SiO2 | Ea SiNx | Verdict |
|---|---:|---:|---|
| Gas-phase models (H3Si–OH, H3Si–NH2, (H3Si)2NH), GFN2-xTB, addition pathway only | 0.92 eV | 0.21 eV | prefers SiNx; nominal S = 0.989 at 10 nm, P(S ≥ 0.9) = 0.52 (373 K, 0.05 s dose, redose every cycle) |
| Same gas models after the agent tried the four-centre pathway at Si–OH | **0.00 eV** (barrierless) | 0.21 eV | prefers SiO2; nominal S = 0, so the control fails |
| Clusters carved from the amorphous slabs, GFN2-xTB | 0.04 eV | 0.06 eV | both essentially barrierless; no selectivity, so the control fails |

The apparent selectivity came from exploring one pathway. The site barrier is
the fastest route, and the evidence-acquisition agent chose to test the
unexplored one: it is barrierless at the silanol. The selectivity also does not
survive the amorphous surfaces. The gate therefore labels every run
`uncalibrated_fail`. DMATMS and BDIPADS did not yield usable barriers (8
attempts: SCF non-convergence or incomplete proton transfer), so the oxide side
of the gate is still untested. xTB may underestimate these barriers; without a
reference method this cannot be settled.

### Candidates

| Molecule | Evidence | Result |
|---|---|---|
| benzoyl chloride | xTB, gas sites | dEa −0.05 eV; P = 0.19, equal to the random-pick baseline |
| methyl chloroformate | xTB, gas sites | dEa −0.05 eV; P = 0.19 |
| pivaloyl chloride, acetic anhydride, N-acetylimidazole | xTB, gas sites | incomplete (failed sites are recorded, not guessed) |
| pivaldehyde, carbonyl addition | MACE, slabs | dE SiO2 −0.33, SiNx +0.36 eV: prefers SiO2, 0 nm at 90 % |
| pivaldehyde, imine condensation | MACE, slabs | imine +1.1 to +1.8 eV at both NH2 sites; adduct kept; same verdict |
| pivaldehyde, addition barriers | xTB, gas sites, after agent repair | Ea SiO2 0.09 eV, SiNx (NH2) 0.12 eV; both fast, no kinetic selectivity; NH-bridge site failed (SCF) |
| glutaraldehyde (added by the agent from the literature library) | xTB, gas sites | barrierless at Si–OH (four-centre) |

Experimentally, aldehydes attach to SiNx and not to SiO2 (Langmuir 2021,
10.1021/acs.langmuir.1c00212). The disagreement led to the method check in section 3.

### Agent benchmark (40 synthetic worlds per budget; regret reduction in decades, ± s.e.; positive = agent better)

| Budget | vs exhaustive cheap | vs screen-then-verify | vs uncertainty sampling | vs random |
|---|---|---|---|---|
| 20 | **+2.45 ± 0.74** | **+0.83 ± 0.40** | +0.62 ± 0.41 | +1.25 ± 0.69 |
| 40 | **+1.50 ± 0.49** | **+1.21 ± 0.46** | +0.24 ± 0.26 | **+1.49 ± 0.58** |
| 80 | −0.10 ± 0.09 | **+1.11 ± 0.42** | −0.05 ± 0.11 | **+3.29 ± 0.88** |

The agent beats the workflow baselines at tight budgets. Its edge over
uncertainty sampling is not significant at any budget (details in `AGENT_DECISIONS.md`).

## 3. What the pipeline found about its own methods

**MACE vs GFN2-xTB disagree by 0.5–1 eV on carbonyl chemistry.** Gas-phase model
reactions, best of 4 conformers each:

| Reaction (eV) | MACE-MATPES-r2SCAN-0 | GFN2-xTB |
|---|---:|---:|
| tBuCHO + CH3NH2 → imine + H2O | +0.66 | −0.34 |
| tBuCHO + H3SiNH2 → silyl imine + H2O | +1.06 | 0.00 |
| tBuCHO + CH3NH2 → hemiaminal | −0.75 | −1.26 |
| tBuCHO + H3SiOH → hemiacetal | −0.58 | −1.22 |

At +0.66 eV ordinary imine formation would have K ≈ 10⁻⁷ at 473 K, which is
implausible. The materials-trained MACE checkpoint is therefore biased against
the imine route, and the aldehyde verdict above is not trustworthy. xTB is known
to overbind such adducts, so neither method is a reference.

**MACE NEB failed at every carbonyl site.** Image-to-image jumps of 7–12 eV were
rejected by the band checks, so no barrier was reported. The submission
campaign therefore uses GFN2-xTB for barriers.

**Entropy is not in the screening energies.** At 473 K and ~1 Torr, adsorbing
the aldehyde costs about +0.47 eV of translational entropy, while condensation
(water released) gains about −0.56 eV. Recomputing the pivaldehyde sites with
this correction still gives a SiO2 preference.

## 4. Live campaign (OpenAI gpt-4.1 planner and critic, Valency search, MACE adsorption, xTB barriers)

Session 1 (`runs/campaign_submit`, 12 steps):

1. The planner proposed pivaldehyde from the Langmuir 2021 quote; the critic approved it.
   The adsorption result was reused from an identical earlier calculation
   (content-addressed cache), and the xTB barriers were computed fresh.
2. xTB found a barrier on SiO2 only. The planner asked the analog designer for
   bulkier and other aldehydes and proposed a tert-butyl analog.
3. **The critic rejected the analog** because the computed result for the parent
   contradicts its predicted selectivity and the proposal did not explain why.
4. The planner chose `repair`. The evidence-acquisition agent spent its 3-unit
   budget on the failed sites, retried on the other pathway and filled the SiNx
   NH2 barrier: 0.09 vs 0.12 eV, so no selectivity.
5. Several steps were lost to malformed replies (`candidate_ids: []`); empty
   optional fields are now dropped before validation (test added).

Session 2 (`runs/campaign_submit2`, warm-started from session 1's records, 12 steps):

- `repair` spent its budget on the remaining SiNx NH-bridge site. Three more
  orientations and pathways all failed (unresolved band, unconverged NEB, SCF
  failure); the failures are recorded, not guessed.
- One Valency search (`search_by_abstract`, PubMed) returned 10 new papers.
- The planner proposed four aldehyde analogs (ethyl, isopropyl, tert-butyl).
  **The critic rejected all four**, each time citing the computed parent result
  (SiO2 preference, no barrier contrast). Repeated analog requests were refused
  without spending budget.
- No new molecule was computed. The critic and the guards worked, but the
  planner stayed on the one literature lead that the computations had already
  refuted. It needs pressure to leave a refuted reaction class (next steps).

Evidence-acquisition agent (`runs/decide_submit`, xTB, gas sites, budget 8,
literature library attached, no LLM):

| # | Agent's choice and reason | Result |
|---:|---|---|
| 0 | Add glutaraldehyde: class-mean barriers gave a chance to beat the current best | barrierless at Si–OH |
| 1 | Acetyl chloride at Si–OH, untried four-centre pathway ("site barrier may be lower than the explored path") | barrierless: the control's selectivity disappears |
| 2–5 | DMATMS / BDIPADS untried pathways and orientations | all failed (recorded) |

It stopped at budget exhaustion with every candidate's dose-window margin
negative (best −4.7 decades): **no molecule tested so far has a feasible
selective dose window**.

## 5. Next steps (in order)

1. A DFT reference (for example r2SCAN-D3 or ωB97X-D on the four reactions above) to
   correct the method bias; `decide` already learns per-(method, surface,
   template) bias from paired references.
2. A robust transition-state search for the slab clusters (relaxed scan → saddle
   optimiser → frequency and IRC check) in place of plain NEB.
3. Free-energy corrections in the screen; then rerun the campaign with a larger budget.
4. Planner diversity: count a refuted hypothesis against its whole reaction
   class, so the planner moves on to other chemistry instead of proposing more
   analogs the critic will reject.

## 6. Reproduce

```bash
env -u PYTHONPATH python -m pytest -q        # 223 passed, 1 skipped
python -m warden fetch-model --name matpes-r2scan-0 --accept-license --verify
python -m warden llm-check --config config/v3-submit-openai.json
python -m warden campaign --db runs/campaign_submit/campaign.sqlite \
    --manifest examples/surfaces-kim-reference.json --config config/v3-submit-openai.json \
    --controls examples/controls-v3.json --evidence examples/evidence-literature.json \
    --bond config/bond.json
python -m warden assess --db <db> --config config/v3.json --controls examples/controls-v3.json
python -m warden benchmark --worlds 40 --budget 40
```

`OPENAI_API_KEY` is read from the environment only. `config/bond.json` (git-ignored)
is copied from `config/valency.example.json`.
