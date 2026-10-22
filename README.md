# WARDEN v2 implementation bundle

Start with **[RUN_ORDER.md](RUN_ORDER.md)** for installation and the exact execution order.
Results and current status: **[RESULTS.md](RESULTS.md)**.

```bash
python -m pip install -r requirements-v2.txt
python -m pytest -q
python -m warden demo --out runs/demo
```

This update adds structure-based candidate proposals, a bounded feedback loop,
MCP literature retrieval, strict calculation/admission checks, persistent records,
and deterministic reports. The offline demo is synthetic; live services and
MACE scientific performance have not been verified. See [VERIFICATION.md](VERIFICATION.md).

**v3 kinetic layer:** site-resolved reaction barriers (GFN2-xTB or MACE NEB on
clusters carved from the amorphous slabs), an acylation chemistry template,
dose-window analysis, and an uncertainty-aware multi-cycle process designer
with control-gated ranking. See **[V3_KINETICS.md](V3_KINETICS.md)**.

**Evidence-acquisition agent:** decides which calculation to run next
(molecule, site, pathway, orientation, or method), with an equal-budget
benchmark against baselines. See **[AGENT_DECISIONS.md](AGENT_DECISIONS.md)**.

**LLM backends:** the campaign planner/critic can run on local Ollama or the
OpenAI API, per role. See **[LLM_PROVIDERS.md](LLM_PROVIDERS.md)**.

The original README is preserved as [README_LEGACY.md](README_LEGACY.md).
Use `python -m warden`, not the old UI/pipeline entrypoints.

For live literature retrieval, authentication, and candidate selection informed
by previous results, see [BOND_SEARCH.md](BOND_SEARCH.md). `warden bond-call`
saves source evidence independently of the full screening campaign.
