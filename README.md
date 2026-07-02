# AS-ALD In-Silico Co-Scientist

**Challenge 4 — Merck Innovation Cup**

An in-silico AI co-scientist for designing selective atomic layer growth schemes. This tool generates realistic amorphous SiO₂ and SiNₓ surfaces for area-selective ALD (AS-ALD) research, implementing the melt-quench-anneal protocol from [Kim et al., *Appl. Surf. Sci.* 2026, 730, 166294](https://arxiv.org/abs/2510.17356).

## 🎯 Goal

Achieve **90% selectivity at 10 nm oxide thickness** for area-selective deposition of SiOx on SiOx, passivating SiN (nitride) as the non-growth surface.

## 📁 Project Structure

```
AI-Scientist/
├── surface_builder.py       # Core module — amorphous surface construction pipeline
├── run_surfaces.py          # CLI driver script (replaces Colab notebook)
├── AS_ALD_surface_builder.ipynb  # Original Colab notebook (reference)
├── AS_ALD_surface_builder.py     # Converted notebook (reference)
├── AI_Scientist.pdf         # Challenge brief
├── bulk_cache/              # Auto-created: cached melt-quenched bulk structures
├── literature/              # User-provided: published amorphous bulk structures
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install torch ase mace-torch scipy
```

**Requirements:**
| Package | Purpose |
|---------|---------|
| `torch` | GPU-accelerated tensor operations |
| `ase` | Atomic Simulation Environment — structure manipulation, MD, optimization |
| `mace-torch` | MACE machine-learning interatomic potential |
| `scipy` | Scientific computing utilities |

### 2. Verify GPU (Optional but Recommended)

```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

### 3. Run the Surface Builder

```bash
cd AI-Scientist/

# Quick validation (~3-5 min on GPU, ~10-15 min on CPU)
python3 run_surfaces.py --mode test

# Fast exploration (~10-15 min on GPU)
python3 run_surfaces.py --mode fast

# Full production run (~2-4 hours on GPU)
python3 run_surfaces.py --mode full
```

## 📖 Usage Guide

### CLI Options

```
python3 run_surfaces.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | `test` | MD protocol: `test` (~3-5 min), `fast` (~10-15 min), `full` (~2-4 hrs) per material |
| `--materials` | `SiO2 SiNx` | Materials to build: `SiO2`, `SiNx`, or both |
| `--n-bulk` | 1 (test/fast), 3 (full) | Number of independent bulk replicas |
| `--target-accepted` | — | Minimum accepted surfaces (triggers over-generation) |
| `--no-gate` | off | Disable quality gate (keep all surfaces) |
| `--supercell A B C` | `2 2 2` | Crystal supercell dimensions |
| `--no-cache` | off | Disable bulk caching (always rerun melt-quench) |
| `--cache-dir` | `bulk_cache/` | Directory for cached bulk structures |
| `--list-cache` | — | List cached bulks and exit |
| `--use-published` | off | Load published amorphous bulks from literature directory |
| `--literature-dir` | `literature/` | Directory containing published bulk structures |
| `--dtype` | `float32` | MACE precision (`float32` faster, `float64` more accurate) |
| `--output-dir` | `.` | Directory for output `.xyz` files |
| `--prefix` | auto | Output filename prefix |

### Examples

```bash
# Single material, test mode
python3 run_surfaces.py --mode test --materials SiO2

# Full production with quality gate, targeting 4 accepted surfaces
python3 run_surfaces.py --mode full --n-bulk 3 --target-accepted 4

# Use a larger supercell for better statistics
python3 run_surfaces.py --mode full --supercell 3 3 2

# Use published structures (skips melt-quench entirely)
python3 run_surfaces.py --mode test --use-published

# Check what's in the cache
python3 run_surfaces.py --list-cache

# Output to a specific directory
python3 run_surfaces.py --mode test --output-dir results/
```

## ⚡ Performance Optimizations

### Bulk Caching

The most expensive step is **melt-quench** (premelt → melt → quench → relax), which converts a crystal into an amorphous bulk. This is a **one-time cost** per (material, seed, protocol) combination.

- **First run**: Melt-quench runs normally, then saves the result to `bulk_cache/`
- **Subsequent runs**: Instantly loads the cached bulk, **skipping ~80% of total runtime**
- Cache keys include material, seed, supercell size, and protocol parameters — switching modes won't cause stale cache hits

```bash
# First run: ~5 min (melt-quench runs)
python3 run_surfaces.py --mode test --materials SiO2

# Second run: ~1 min (melt-quench skipped, only cleave+passivate+anneal)
python3 run_surfaces.py --mode test --materials SiO2
```

To clear the cache:
```bash
rm -rf bulk_cache/
```

### Published Bulk Structures

You can skip melt-quench entirely by providing published amorphous structures from literature (e.g., Kim et al.'s supplementary data):

1. Create a `literature/` directory:
   ```bash
   mkdir -p literature/
   ```

2. Place structure files (`.xyz`, `.cif`, `.extxyz`, `.vasp`, `.poscar`) with names starting with the material:
   ```
   literature/
   ├── SiO2_amorphous_Kim2026.xyz
   └── SiNx_amorphous_Kim2026.xyz
   ```

3. Run with `--use-published`:
   ```bash
   python3 run_surfaces.py --mode test --use-published
   ```

### Supercell Size

The default supercell is `(2,2,2)`, optimized for speed:

| Material | (2,2,2) | (3,3,2) — original |
|----------|---------|---------------------|
| SiO₂ | 72 atoms | 162 atoms |
| SiNₓ | 112 atoms | 168 atoms |

Smaller systems run **2–4× faster** (MD scales ~O(N²)). Use larger supercells for production quality:
```bash
python3 run_surfaces.py --mode full --supercell 3 3 2
```

## 🔬 Pipeline Overview

The surface builder implements this multi-stage protocol:

```
┌─────────────────────┐
│  1. Build Crystal    │  α-quartz (SiO₂) or β-Si₃N₄ (SiNₓ)
│     Supercell        │  Default: (2,2,2)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  2. Melt-Quench      │  6000K premelt → 3000K melt → staged quench → 0K
│     (CACHED)         │  Cached to bulk_cache/ after first run
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  3. Cleave + Vacuum  │  Cut along z-axis (2 terminations per bulk)
│                      │  15 Å vacuum gap inserted
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  4. Passivate        │  Saturate dangling bonds per Table 1 (Kim et al.)
│                      │  SiO₂: -OH, SiNₓ: -NH₂, -NH-
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  5. Anneal + Relax   │  1000K / 5ps anneal (regularizes Si=NH → Si-NH-Si)
│                      │  BFGS final relaxation
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  6. Classify Sites   │  Exposure filter (solves Problem A: over-counting)
│     + Quality Gate   │  Strain flagging, clumping ratio, density gate
└─────────────────────┘
```

### Key Features

| Feature | Problem Solved |
|---------|----------------|
| **Exposure filter** | SiOx models over-count reactive sites vs experiment (6.1 vs ~4.5 OH/nm²) |
| **Anneal regularization** | SiNx –NH₂/–NH sites have irregular spacing → excess calculations |
| **Quality gate** | Statistical outlier rejection using literature mean ± n·σ bounds |
| **Strain flagging** | Identifies strained 3-membered-ring sites (non-transferable ΔE) |
| **Clumping ratio** | PBC-aware Clark-Evans metric detects unrealistic site clustering |

## 📊 Expected Output

Each run produces `.xyz` files (viewable in OVITO, ASE GUI, VESTA) and a console summary:

```
=== SiO2 (TEST mode) ===
    melt-quench bulk (72 atoms)...
      [premelt] 6000K 0.4ps (200 steps) done in 12s
      [melt] 3000K 1.0ps (500 steps) done in 28s
      quenching...
      final BFGS relax...
    [cache] saved bulk → bulk_cache/SiO2_bulk_a3b2c1d4.xyz
    anneal + relax (94 atoms)...

[SiO2] bulk 0 term 0: {'OH': 5.83, 'O_bridge': 2.41} | strained=1 R=1.12 | ACCEPT

--- SiO2 Summary ---
  Accepted: 2, Rejected: 0
  Exposed-site densities:
    OH          :  5.83 ± 0.24 nm⁻² (literature: 6.1 ± 0.4 nm⁻²)
    O_bridge    :  2.41 ± 0.33 nm⁻² (literature: 2.8 ± 0.7 nm⁻²)
```

## ⏱️ Runtime Estimates

| Mode | RTX 4060 (8GB) | A100 (64GB) | CPU Only |
|------|---------------|-------------|----------|
| **TEST** (1 bulk, both materials) | ~8-12 min | ~3-5 min | ~30-45 min |
| **FAST** (1 bulk) | ~25-30 min | ~10-12 min | ~1-2 hrs |
| **FULL** (3 bulks, both materials) | ~5-8 hrs | ~2-3 hrs | ~12-20 hrs |

> **Note:** Second runs with caching are **~5× faster** since melt-quench is skipped.

## 🛠️ Using as a Python Module

```python
import surface_builder as sb

# Load calculator
calc = sb.get_calculator()

# Set protocol
sb.use_test_protocol()  # or use_fast_protocol() / use_full_protocol()

# Build surface ensemble
surfaces, summary = sb.build_surface_ensemble(
    material="SiO2",
    calc=calc,
    n_bulk=1,
    apply_gate=False,   # keep all surfaces
    use_cache=True,     # cache melt-quenched bulks
)

# Analyze a surface
slab = surfaces[0]
counts = sb.classify_sites(slab, "SiO2", exposure_filter=True)
area = sb.surface_area_nm2(slab)
for site_type, v in counts.items():
    print(f"{site_type}: {v['exposed']/area:.1f}/nm²")

# List cached bulks
sb.list_cached_bulks()
```

## 📚 References

1. Kim et al., "Machine-learning interatomic potential-driven amorphous surface model for area-selective atomic layer deposition", *Appl. Surf. Sci.* 2026, 730, 166294. [arXiv:2510.17356](https://arxiv.org/abs/2510.17356)
2. Bobb-Semple et al., "Area-Selective Atomic Layer Deposition Assisted by Self-Assembled Monolayers", *Chem. Mater.* 2020, 32, 4920–4953.
3. MACE: Batatia et al., "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields", NeurIPS 2022.

## 📝 License

This project was developed for the Merck Innovation Cup 2026, Challenge 4.
