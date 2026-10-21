# WARDEN v2 implementation bundle

Start with **[RUN_ORDER.md](RUN_ORDER.md)** for installation and the exact execution order.

```bash
python -m pip install -r requirements-v2.txt
python -m pytest -q
python -m warden demo --out runs/demo
```

This update adds structure-based candidate proposals, a bounded feedback loop,
MCP literature retrieval, strict calculation/admission checks, persistent records,
and deterministic reports. The offline demo is synthetic; live services and
MACE scientific performance have not been verified. See [VERIFICATION.md](VERIFICATION.md).

The original README is preserved as [README_LEGACY.md](README_LEGACY.md).
Use `python -m warden`, not the old UI/pipeline entrypoints.
