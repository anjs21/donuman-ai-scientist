# Merge: literature + neji_dev_2 (branch merge-neji-dev-2)

Both branches started from master 623e2ac. `literature` reworked the
evidence-acquisition agent (`decide`) and the barrier physics; `neji_dev_2`
built the agentic campaign (live Valency search, analog design, aldehyde
template, notebook). Integration decisions:

- Kept from literature: the `decide` agent (class sharing, all library
  molecules valued, fastest orientation, untried orientations only, method
  pairing per orientation, budget ledger, add-with-first-calculation),
  barrierless paths as fast reactions, dose-window criterion, lit-brief /
  lit-import, benchmark. neji_dev_2 had not changed `decide`.
- Kept from neji_dev_2: `rotation()` no longer normalises its input in place
  (literature still displaced the leaving group in starting geometries),
  `carbonyl_addition`, `analogs`, campaign actions analogs/repair, notebook and
  hypothesis tracking, BOND result text and DOI tracking, reference-structure
  exemptions, gas/slab consensus, MACE engine settings for `decide`, UTF-8 I/O,
  literature candidates and evidence (`examples/*-literature.json`).
- Reconciled:
  - `Store.recent` existed twice with opposite order (git merged it silently);
    kept newest-first, which neji_dev_2's callers need.
  - Quote checks: normalised matching (literature) plus the 5-word minimum.
  - `reclassify`: barrierless-aware best attempt (literature) with binding
    re-based on each attempt's own reactant complex (neji_dev_2).
  - `apply_observations` (repair -> assess): a faster repaired path now also
    lowers a valid site; previously valid sites were never changed, which
    ignores e.g. a barrierless oxide route found by repair.
  - Hypothesis tracking: kinetics re-scored under current rules; a
    SiNx-selective prediction is `supported` only with a positive dose-window
    margin, otherwise `direction_only`.
  - One planner record (`decision`) exposed to the planner and critic as
    `agent_ranking`, with the meaning of the margin.
  - `decide --library` skips entries marked not computable, wrong-direction
    references (`predicted_direction: prefers_SiO2`, controls) and invalid
    entries, and reports them instead of aborting.
  - The `carbonyl_addition` four-centre tilt has the same period as the spin.
  - Benchmark worlds draw numbers only for templates present, so new
    templates do not change published benchmark worlds.

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

# OpenAI backend for planner and critic

- `warden/planner.py`: `OpenAI` backend (Chat Completions, non-strict JSON schema,
  local validation, retries on 429/5xx, refusal/truncation errors, token usage),
  key read only from an environment variable; `make_planner` with per-role
  backends; shared `Backend.plan/critique`. Legacy `llm` configs still mean Ollama.
- Campaign logs `llm_backends` and `llm_usage` events. CLI `llm-check`.
- `config/v3-openai.json`, `config/v3-hybrid.json`; `tests/test_llm.py` (mock server).

# MACE foundation checkpoints

- `warden/models.py` + CLI `fetch-model`: downloads only from a fixed registry of
  official ACEsuit URLs (MATPES-r2SCAN-0, MATPES-PBE-0, OMAT-0, MPA-0, MP-0a),
  streams to a temporary file, hashes, refuses tiny/incomplete files and
  overwrites without --force, requires --accept-license for ASL models, writes
  `<file>.provenance.json`; `--verify` loads it and evaluates NH3.
  Default destination `models/mace.model` (what `config/*.json` resolve to).
