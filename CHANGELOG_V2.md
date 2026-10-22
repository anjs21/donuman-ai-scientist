# Changes from the linked repository

- Added the `warden` package, CLI, configurations, evidence/candidate/manifest
  schemas, offline demo, run guide, and regression tests.
- Replaced the fixed-candidate path in the supported workflow with SMILES-based
  proposals and explicitly supported reaction templates, including correct
  BDMAS and SiCl4 fragments in the new candidate definitions.
- Added persistent campaign state, structured planner actions, a critique pass,
  source quote checks, candidate deduplication, calculation budgets, and resume.
- Added an MCP client with tool discovery, allowlisting, and schema validation.
- Added converged-only serial calculations, geometry checks, representative site
  sampling, per-surface coverage gates, and content/version-aware cache keys.
- Corrected target-thickness evaluation: no clipped cycle count, no perfect
  selectivity at zero growth, and missing energies remain unknown.
- Corrected selected legacy ranking and surface geometry issues; seeded serial
  surface MD; stopped fallback calculators and failed surface relaxation.
- Added a local checkpoint argument to the surface builder and a manifest helper.
- Disabled the old UI/pipeline launch routes in this bundle to avoid bypassing
  v2 gates. Retained their source and other legacy modules for comparison.

Scientific scope and remaining limitations are documented in RUN_ORDER.md.

# v3 additions

- `warden/kinetics.py`: climbing-image NEB barriers on gas-phase site models and
  on H-capped clusters carved from admitted slabs; pre-complex binding, Eyring
  rates with Langmuir occupancy; guards for unconverged/discontinuous bands,
  changed connectivity, re-captured protons and open-shell clusters.
- `acylation` reaction template in candidate validation, MACE placement and the planner.
- `warden/process.py`: multi-cycle model with redosing, cap degradation,
  irreversible nucleation and steric shielding; Monte Carlo robust schedule
  design, dose windows, barrier targets, input sensitivity, calibration.
- `warden/assess.py`: control gate, method consensus, baselines and lift.
- CLI `kinetics` and `assess`; campaign runs them when `campaign.kinetics` is set.
- Surface validation reports coordination defects and Si-Si bonds.
- `warden/decide.py`: evidence-acquisition agent (hierarchical belief, method bias
  learning, empirical-Bayes failure model, knowledge-gradient + relevance values,
  confident stopping), live xTB/MACE executor, decision reports; CLI `decide`.
- `warden/benchmark.py`: equal-budget synthetic benchmark vs four baselines; CLI `benchmark`.
- Selectable reaction pathways in barrier endpoints; `process.stress_test`.

# Fix: barrierless paths and the confident-stop rule

- `kinetics.classify`: a converged, clean band that never rises by
  `min_barrier_eV` and ends downhill is now `barrierless` (a fast reaction),
  not a failure. Uphill bands without a maximum stay `no_barrier_resolved`.
- Site barriers are the fastest usable path over orientations, including
  barrierless ones (`select_best`). Previously a barrierless oxide path was
  discarded and a slower orientation reported, which can make a non-selective
  molecule look selective.
- A barrierless barrier is censored (Ea <= `min_barrier_eV`) and resolved
  against selectivity: 0 eV on SiO2, the bound on SiNx (`effective_barrier`).
- `decide`: barrierless attempts are successful observations, not method
  failures, both when seeded from stored records and in live runs.
- `reclassify` re-scores stored records under these rules, so `assess` and
  `decide` correct existing databases without recomputing.
- `decide.run` no longer stops as "confident" with a single candidate, or
  among only infeasible candidates while library molecules remain.
- Regression fixture: `tests/fixtures/kinetics/acetyl_chloride_xtb_gas.json`.
  Re-scored, acetyl chloride's oxide paths are barrierless in these xTB gas
  models, so the control computes `prefers_SiO2` (dEa -0.21 eV) and the
  assessment is `uncalibrated_fail`. The acetyl chloride row in V3_KINETICS.md
  predates this fix and should be re-run.

# Literature loop (Valency BOND)

- `warden lit-brief`: reads kinetics screens and the decision agent's
  observations; separates chemical outcomes from numerical failures; writes a
  brief, literature questions and starting `bond-call` arguments. No LLM.
- `warden lit-import`: validates literature-backed proposals (verbatim quotes,
  `literature_status` tested/analog, hypothesis, falsification, novelty) and
  writes a library for `decide --library`, literature-tested molecules first.
- Quote checks now match the readable text of captured records. Previously the
  escaped JSON written by `bond-call` made quotes containing subscripts, Å,
  quotation marks or line breaks impossible to match.
- `decide --library` re-checks evidence references against the database, so
  literature-backed entries are accepted.
- `bond-call` refuses tools whose names send or change data (e.g.
  `submit_feedback`) even if allowlisted, before connecting.
- `add-evidence` prints the evidence IDs it stores.
- Claude Code: install commands in BOND_SEARCH.md; project skill
  `.claude/skills/warden-literature`; tracked `evidence/` folder.
