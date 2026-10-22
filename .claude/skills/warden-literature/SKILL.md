---
name: warden-literature
description: Use when the user asks to find new AS-ALD inhibitor candidates from the literature, to search Valency BOND for WARDEN, to act on a WARDEN literature brief, or to turn papers into candidates for `warden decide`. Runs the brief -> search -> capture -> propose -> import loop with auditable evidence.
---

# WARDEN literature loop (Valency BOND)

Goal of the project: an inhibitor that blocks SiNx while preserving SiO2 growth
(SiO2 deposits on SiO2, not on SiNx). WARDEN computes barriers; this skill finds
literature-supported molecules to compute next, guided by what the calculations
have already shown.

## Requirements

- Valency plugin for Claude Code, installed and signed in (see BOND_SEARCH.md).
  If its tools are unavailable, say so and stop. Do not invent results.
- WARDEN environment active (`python -m warden --help` works).
- `config/bond.json` (copied from `config/bond.example.json`, git-ignored) with
  `allowed_tools` set to the read-only tools reported by `bond-tools`.

## Rules

- Never write evidence text yourself. Evidence must be the provider's own record,
  captured by `python -m warden bond-call`. Your searches through the Valency
  plugin are for finding papers; `bond-call` is how a cited passage enters WARDEN.
- Quotes in proposals must be copied verbatim from the captured record
  (at least 12 characters). Paraphrases are rejected by `lit-import`.
- Never call `submit_feedback` or any tool that sends or changes data.
- Numerical failures in the brief are method problems, not evidence against a
  molecule. Do not search the literature to explain them.
- Label each molecule `tested` only if the cited source studies that molecule
  itself (on a relevant surface or reaction); otherwise `analog`.
- BOND covers arXiv, bioRxiv, medRxiv and PubMed, plus ChemRxiv where the live
  schema offers it. Much AS-ALD work is in journals outside this corpus: say when
  a question returns little, rather than stretching weak matches.
- Mark preprints as preprints. Do not re-propose SMILES listed under
  "Already computed" in the brief.

## Steps

1. **Brief.** On the machine holding the campaign database:
   `python -m warden lit-brief --db <db> --config config/v3.json --out evidence/brief`
   Read `evidence/brief/brief.md`. Work only on its "Literature questions".

2. **Search.** For each question, search with the Valency tools
   (`semantic_search_papers` or `search_by_abstract` for concepts;
   `search_by_title` for a known paper; `find_similar_papers` to expand from a good
   hit). Try ChemRxiv as a source if the live schema supports one. Note for each
   useful paper its stable Valency paper ID and the passage that matters.

3. **Capture.** For every paper you will cite, capture its record with WARDEN:
   write `{"paper_id": "<id>"}` (or the arguments the live schema requires) to a
   file, then
   `python -m warden bond-call --config config/bond.json --tool get_paper_by_id --arguments <file> --out evidence/searches/<short-name>.json`
   Check the live tool name and schema with `bond-tools` first if unsure.
   Then load it: `python -m warden add-evidence --db <db> --file evidence/searches/<short-name>.json`
   and note the printed `ev_...` ID.

4. **Propose.** Write `evidence/proposals.json`, a list of:

   ```json
   {"name": "acetic anhydride",
    "smiles": "C[C:1](=O)[O:2]C(C)=O",
    "template": "acylation",
    "role": "inhibitor",
    "literature_status": "tested",
    "hypothesis": "One sentence: why it should cap Si-NH2 faster than Si-OH.",
    "falsification": "One sentence: the computed result that would refute it.",
    "evidence": [{"id": "ev_...", "quote": "verbatim passage from the captured record"}]}
   ```

   Optional, only when a captured source reports or directly implies barriers for
   this molecule: `"prior_Ea_eV": {"SiO2": 1.2, "SiNx": 0.6}` and `"prior_sd_eV": 0.25`.
   `decide` then uses them instead of what the molecule's reaction class has shown
   so far. Leave them out otherwise; never invent numbers.

   Templates: `acylation` (carbonyl carbon `[C:1]`, leaving atom `:2` = Cl, O or N),
   `silyl_substitution` (`[Si:1]`, leaving Cl or N `:2`), `physisorption` (no maps).
   Neutral, closed-shell, one molecule. Prefer molecules that address the brief's
   chemical findings (e.g. "oxide_too_reactive" -> milder or more N-H-selective).

5. **Import.**
   `python -m warden lit-import --db <db> --candidates evidence/proposals.json --out evidence/library.json`
   Fix or drop every rejected entry; do not weaken quotes to make them pass.

6. **Hand over.** Show the user the accepted library (tested first), what each
   entry is for, and which brief questions found nothing. The agent then runs:
   `python -m warden decide --db <db> --sites gas --budget <n> --library evidence/library.json --dry-run`

## Moving between a laptop and a cluster

BOND sign-in needs a browser, so steps 2-4 usually run on a laptop while the
campaign database lives on the cluster. Files under `evidence/` are tracked by git.

- Cluster: step 1, then commit and push `evidence/brief/`.
- Laptop: pull; do steps 2-4 using a scratch database `evidence/scratch.sqlite`
  (git-ignored) as `<db>`, and run step 5 against it to catch quote errors early.
  Commit and push `evidence/searches/` and `evidence/proposals.json`.
- Cluster: pull; `add-evidence` each file in `evidence/searches/` into the campaign
  database, then run step 5 and `decide` there. The cluster import is the one that
  counts: it also rejects molecules already computed in the campaign.

Evidence IDs are content hashes, so the same capture file gives the same IDs in
both databases and the proposals' references stay valid.
