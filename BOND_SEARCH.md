# Literature-guided candidate selection

The supported workflow is `python -m warden`. Literature retrieval can run
independently of Ollama, MACE, or admitted slabs, using `bond-tools` and
`bond-call`. The full `campaign` additionally requires those live components.
Passing software tests or publishing a cloud environment does not establish
that live literature retrieval or chemical discovery has succeeded.

## Connect and discover

Create and activate the Python 3.12 environment described in RUN_ORDER.md;
install requirements-v3.txt for the core workflow plus xTB. Do not install the
legacy requirements.txt for this workflow.

The official [Valency connector](https://github.com/valency-oss/valency-connectors)
documents this installation for supported Codex CLI clients:

```bash
codex plugin marketplace add valency-oss/valency-connectors
codex plugin add valency@valency
codex mcp login valency
```

Complete browser authorization and start a new Codex session to load the tools.
Client-managed OAuth credentials are separate from WARDEN's Python adapter;
do not extract or copy client credentials. In hosted clients, use the supported
plugin installation and sign-in flow for that client.

For WARDEN, copy config/bond.example.json to the ignored config/bond.json
to use the OAuth-capable Node `mcp-remote` bridge, or use the HTTP example
only if an authorized BOND_TOKEN binding is already provided. Credentials
belong in secure settings, never in repository files or chat.
OAuth callbacks require an environment supported by the provider and client.

```bash
python -m warden bond-tools --config config/bond.json --out runs/bond-tools.json
```

Inspect the live inventory and argument schemas. Add only the literature
search/read tools you intend to use to allowed_tools in the local config.
Both examples start with an empty allowlist. Do not infer tool names or
argument schemas from example prompts. Network access needs mcp.valency.io
and oauth.valency.io (the authorization server advertised by BOND's protected
resource metadata), plus any further destinations reported by sign-in.

## Retrieve auditable evidence before screening

Put one discovered tool's arguments in runs/search-arguments.json, then run:

```bash
python -m warden bond-call --config config/bond.json \
  --tool ACTUAL_DISCOVERED_TOOL --arguments runs/search-arguments.json \
  --out runs/search-evidence-01.json
```

This command enforces the existing tool allowlist and live input schema,
preserves the raw response with query provenance and retrieval time, and
refuses to overwrite an existing output. It does not make an LLM call or
require screening inputs. Read the response and follow up with paper-reading
tools when available; search snippets alone may not support a chemical claim.
Use a new output path for each retrieval. Import selected evidence with:

```bash
python -m warden add-evidence --db runs/live/campaign.sqlite \
  --file runs/search-evidence-01.json
```

The same file is accepted by campaign --evidence. For autonomous additional
retrieval, pass --bond config/bond.json to campaign. Default campaign limits
allow only three search calls; this is a bounded search, not exhaustive coverage.

The [ChemRxiv guide](https://valency-oss.github.io/oss-hack/) describes
keyword and semantic search with source=chemrxiv. Category tools do not work
for that source. Confirm the live schema and available sources; search both
ChemRxiv and other relevant corpora. Label preprints and deduplicate published
versions. BOND coverage is not the entire published literature.

## Use previous results to choose questions

V3_KINETICS.md and AGENT_DECISIONS.md contain historical summaries. The original
live calculation databases are not bundled here. Synthetic demo/benchmark
outputs are not chemical evidence.

- Keep acetyl chloride as a comparison control: reported gas-model barriers
  favor SiNx, but reported process P(success)=0.52 and the control gate is
  incomplete. Its gas-phase preference is unconfirmed on amorphous surfaces.
- Benzoyl chloride shows little reported gas-model contrast. Prioritize related
  candidates only with a source-supported reason to recover surface selectivity.
- Pivaloyl chloride, methyl chloroformate, acetic anhydride, N-acetylimidazole,
  DMATMS and BDIPADS have incomplete or failed calculations. Numerical failure
  is not evidence of poor chemistry.
- Search for amine versus silanol acylation/silylation selectivity, vapor-phase
  delivery, coverage and steric shielding, thermal stability, cap persistence,
  and preservation of oxide growth under the actual ALD cycle.
- Compare lower-temperature acetyl-chloride conditions with milder electrophiles
  for practical high-temperature dose windows. Do not assume weakening an
  electrophile preserves surface contrast.

For each shortlisted molecule retain identity/SMILES, DOI or source URL, exact
supporting passage and location, surface termination, temperature/exposure,
experimental versus computed evidence, volatility/stability information,
contradictory results, and a falsifying next calculation. Clearly distinguish
literature-tested molecules from proposed analogs. Missing information stays
unknown. Quote matching checks text presence, not scientific relevance.

Only submit supported reaction templates to the existing chemistry validator.
Preserve surface admission gates; do not turn the historical unadmitted cluster
diagnostics into ranking evidence. `warden decide` selects calculations from
existing candidates and a supplied library; it does not itself search literature.
