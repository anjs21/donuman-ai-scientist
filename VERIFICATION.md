# Verification record

Base repository: anjs21/donuman-ai-scientist, master commit
54cb64c5a25a4c624e183253b6514af24d859020.

Executed in Python 3.12.14 with the versions in requirements-v2.txt.

- `python -m pytest -q`: **38 passed**.
- Compiled the new package, helper scripts, and changed legacy entrypoints.
- `pip check`: no broken requirements in the core verification environment.
- Ran the offline demo to completion and verified idempotent resume.
- Embedded all four seed inhibitors and both precursor definitions.
- Verified element balance for gas fragments and full surface-exchange fixtures.
- Reference oxide passed the current admission gates; reference nitride failed
  the density gate (about 2.135 versus 3.0 ± 0.2 g/cm³). The screen stops before
  loading MACE for this invalid pair.
- Exercised MCP inventory, allowlisted calls, and argument-schema rejection
  against a local mock MCP server using the actual pinned Python MCP SDK.
- Exercised Ollama's HTTP adapter against a local mock server.
- Tested convergence rejection, missing-energy handling, target-cycle regression,
  geometry rotations, cache invalidation by checkpoint contents/sampling,
  proposal critique rejection, bounded stopping, timeout/resume, and exclusion
  of synthetic or superseded failed-refinement results from real ranking.

## Re-verification on branch neji_dev_2 (from master 623e2ac, v3 kinetics + decision agent)

Executed in Python 3.13 with mcp 1.26.0 (pinned) and newer numpy/scipy/ase/rdkit
than requirements-v2.txt; tblite (xTB) not installed.

- `python -m pytest -q`: **77 passed, 1 skipped** (the xTB barrier test).
  Before the fixes below, master gave 71 passed, 1 skipped on the same tests.
- mcp 2.x removed `streamablehttp_client`; keep the 1.26.0 pin.
- Fixes covered by new regression tests: `decide --methods mace` now receives
  the engine block (model_path); `kinetics.reclassify` re-bases the pre-complex
  binding on the newly selected attempt's reactant energy; `assess.consensus`
  keeps gas-model and slab-cluster records of one method separately; the planner
  sees the newest evidence (not hash order), with per-item text capped and a
  refusal when the prompt would exceed `llm.num_ctx`; evidence quotes need at
  least 5 words.

## Not verified

No real MACE checkpoint was loaded; no new surface MD, adsorption energies,
DFT comparison, or experimental performance is reported. Live Ollama model
quality, live Valency BOND authentication/tool calls, CUDA behavior, and the
optional mace-torch installation were not exercised. The synthetic demo and
mock integration tests are software checks, not evidence of chemical discovery.

The full inherited legacy project was not tested or repaired. The supported
path is the v2 CLI described in RUN_ORDER.md. The archived source and tests
are supplied so these limitations can be evaluated and improved directly.
