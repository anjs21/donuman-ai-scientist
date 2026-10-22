"""Fetch MACE foundation checkpoints from a fixed registry of official URLs.

Only the URLs below can be downloaded (no arbitrary links, and the campaign
planner cannot call this). The file is streamed to a temporary path, hashed,
moved into place, and described in `<file>.provenance.json` (source URL,
SHA-256, size, license, time). An existing checkpoint is never overwritten
without --force. Source: https://github.com/ACEsuit/mace-foundations
"""
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import urllib.request
from .common import read_json, write_json

BASE = "https://github.com/ACEsuit"
REGISTRY = {
    "matpes-r2scan-0": {"url": f"{BASE}/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-r2scan-omat-ft.model",
                        "license": "ASL", "theory": "r2SCAN (MATPES)", "min_mace": "0.3.10"},
    "matpes-pbe-0": {"url": f"{BASE}/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model",
                     "license": "ASL", "theory": "PBE, no +U (MATPES)", "min_mace": "0.3.10"},
    "omat-0-medium": {"url": f"{BASE}/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model",
                      "license": "ASL", "theory": "PBE+U (OMAT)", "min_mace": "0.3.10"},
    "mpa-0-medium": {"url": f"{BASE}/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model",
                     "license": "MIT", "theory": "PBE+U (MPTrj + sAlex)", "min_mace": "0.3.10"},
    "mp-0a-medium": {"url": f"{BASE}/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model",
                     "license": "MIT", "theory": "PBE+U (MPTrj)", "min_mace": "0.3.6"},
}
RESTRICTED = {"ASL"}


def fetch(name, out, force=False, accept_license=False, timeout=120, chunk=1 << 20, log=print):
    if name not in REGISTRY:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(REGISTRY)}")
    entry = REGISTRY[name]
    if entry["license"] in RESTRICTED and not accept_license:
        raise ValueError(f"{name} is distributed under the {entry['license']} (academic, non-commercial); "
                         "re-run with --accept-license after checking it applies to your use")
    out = Path(out).expanduser().resolve()
    if out.exists() and not force:
        raise ValueError(f"{out} exists; use --force to replace it (its provenance file is replaced too)")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".part")
    digest, size = hashlib.sha256(), 0
    req = urllib.request.Request(entry["url"], headers={"User-Agent": "warden-fetch-model"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response, open(temporary, "wb") as f:
            total = int(response.headers.get("Content-Length") or 0)
            for part in iter(lambda: response.read(chunk), b""):
                f.write(part); digest.update(part); size += len(part)
            if total and size != total:
                raise IOError(f"Incomplete download: {size} of {total} bytes")
        if size < 1 << 20:
            raise IOError(f"Downloaded only {size} bytes; not a model checkpoint")
        os.replace(temporary, out)
    finally:
        if temporary.exists():
            temporary.unlink()
    record = {"name": name, **entry, "path": str(out), "sha256": digest.hexdigest(), "bytes": size,
              "downloaded_at": datetime.now(timezone.utc).isoformat(),
              "license_accepted": bool(accept_license) if entry["license"] in RESTRICTED else None,
              "source": "https://github.com/ACEsuit/mace-foundations"}
    write_json(provenance_path(out), record)
    log(f"{name}: {size / 1e6:.1f} MB -> {out}\nsha256 {record['sha256']}")
    return record


def provenance_path(model_path):
    p = Path(model_path)
    return p.with_name(p.name + ".provenance.json")


def provenance(model_path):
    """Provenance record for a checkpoint, if it was fetched here and still matches."""
    from .common import file_hash
    p = provenance_path(model_path)
    if not p.exists():
        return None
    record = read_json(p)
    record["matches_file"] = Path(model_path).exists() and file_hash(model_path) == record.get("sha256")
    return record


def verify(model_path, device="cpu"):
    """Load the checkpoint and evaluate one small molecule (finite energy and forces)."""
    import numpy as np
    from ase import Atoms
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths=str(model_path), device=device, default_dtype="float64")
    atoms = Atoms("NH3", positions=[[0, 0, 0], [0.94, 0, -0.33], [-0.47, 0.81, -0.33], [-0.47, -0.81, -0.33]])
    atoms.calc = calc
    energy, forces = atoms.get_potential_energy(), atoms.get_forces()
    if not (np.isfinite(energy) and np.isfinite(forces).all()):
        raise ValueError("Checkpoint produced non-finite energy or forces")
    return {"energy_eV": float(energy), "max_force_eV_A": float(np.abs(forces).max())}
