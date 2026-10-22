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

# Agentic campaign additions (branch neji_dev_2)

- Planner actions `analogs` and `repair` beside `search`, `propose`, `refine`,
  `finish`; every action may carry a `lesson`. Budgets: `campaign.max_repairs`,
  `campaign.repair_budget`, `campaign.max_analog_calls`.
- Literature mining (`warden/literature.py`, `campaign.py`): results are stored as
  plain decoded text so verbatim quotes match; DOIs are tracked as `paper`
  records; identical searches are refused; tool descriptions are compacted.
  `config/valency.example.json` connects the Valency Bond MCP server
  (ChemRxiv, PubMed, arXiv...) through `mcp-remote` with read-only tools allowlisted.
- `warden/analogs.py`: deterministic leaving-group, steric and aromatic
  substituent analogs with descriptors, stored as quotable `analog_design`
  evidence; CLI `warden analogs`.
- Self-repair: `repair` runs `warden decide --ids ...` on failed/uncertain
  barriers; `assess` now merges valid decision-agent observations into failed
  kinetics sites (`kinetics.apply_observations`), and `decide` stores a compact
  `decision` record for the planner.
- `warden/notebook.py`: campaign notebook (planner lessons, outcomes, rejections)
  fed back each step; code-scored hypothesis status
  (supported/refuted/inconclusive/untested) from computed contrasts and barriers.
  Both appear in report.md/json with retrieved papers.
- CLI `warden search` stores one literature call as evidence.
- `examples/candidates-literature.json`: literature candidates from Valency with
  DOIs and verbatim abstract quotes (verified against the retrieved records).
  Propanoyl chloride and BDMADMS are computable now; the aldehydes (the
  literature's SiN-selective class) and the cyclic azasilane need new
  carbonyl-addition / ring-opening templates.
