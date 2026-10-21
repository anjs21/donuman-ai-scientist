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
