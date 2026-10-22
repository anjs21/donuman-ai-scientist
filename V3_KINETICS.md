# WARDEN v3 — kinetic selectivity layer

v3 adds what thermodynamic screening cannot see: **reaction barriers**, the
**dose window** they imply, and a **multi-cycle process model with cap
degradation and redosing**. It does this in a way that differs from a single
gas-phase barrier plus one deterministic selectivity number:

| | Typical approach | WARDEN v3 |
|---|---|---|
| Barrier model | One gas-phase model per surface | Gas-phase models **and** H-capped clusters carved around representative sites of the amorphous slabs, population-weighted (`warden/kinetics.py`) |
| Inhibitor chemistry | Fixed hypothesis | `acylation` template (acyl chloride / anhydride / acyl-imidazole / chloroformate capping N-H or O-H) alongside silylation and physisorption; the planner can propose any of them |
| Rate | Barrier difference | Eyring rate with Langmuir pre-complex occupancy (Sackur-Tetrode + Campbell-Sellers adsorbate entropy), so T and partial pressure enter |
| Process | One schedule, one S | Grid over T x dose x redose interval; **P(S >= 0.9 at 10 nm)** by Monte Carlo over method error and uncomputed inputs; most robust practical schedule |
| Design target | Maximise contrast | Analytic **dose window** and the **barrier band** (`target_Ea_NGS_max_eV`, `target_Ea_GS_min_eV`) that puts it inside practical dose times; the agent steers reactivity into the band |
| What next | — | Spearman ranking of uncertain inputs against S: **measure_first** |
| Experiment | — | `calibrate()` fits p_nuc, shielding, k_deg, theta_max to thickness-vs-cycle data |
| Trust | One reproduced control | **Control gate**: ranking is labelled `control_calibrated` only if every control reproduces; otherwise `uncalibrated_fail/incomplete` |
| Methods | Two methods agree on direction | Per-candidate method disagreement becomes the relative barrier uncertainty in the Monte Carlo |
| Baselines | One | Electrophilicity-only, thermodynamics-only, barrier-contrast-only, random; lift in P(success) |

## Process model

Per surface, each site class *i* (weight w_i, barrier Ea_i) has capped
fraction theta_i and irreversibly nucleated fraction nu_i. Per ALD cycle:

```
redose (every N):  theta_i += max(0, theta_max (1 - nu_i) - theta_i) * P_cap(Ea_i, T, P, t_dose)
precursor:         nu_i    += (1 - theta_i - nu_i) * p_nuc * (1 - theta)^m
growth:            thk     += GPC * sum_i w_i nu_i
co-reactant:       theta_i *= 1 - k_deg            (k_deg_GS on the oxide)
```

Nuclei cannot be re-passivated, so redosing has diminishing returns. Missing
barriers make the candidate *insufficient evidence*; they are never treated as
"slow". `degradation_per_cycle(Ea, T, t)` converts a computed cap-degradation
barrier into k_deg.

## Run order

```bash
python -m pip install -r requirements-v3.txt        # adds tblite (GFN2-xTB)
env -u PYTHONPATH python -m pytest -q               # 62 tests incl. one real xTB barrier

# 1. Barriers on gas-phase site models (minutes per molecule on CPU)
python -m warden kinetics --db runs/v3_gas/campaign.sqlite --config config/v3.json \
       --candidates examples/seeds-v3-acylation.json --sites gas
python -m warden kinetics --db runs/v3_gas/campaign.sqlite --config config/v3.json \
       --candidates examples/controls-v3.json --sites gas

# 2. Barriers on clusters carved from admitted slabs (fail-closed on admission)
python -m warden kinetics --db runs/v3_slab/campaign.sqlite --config config/v3.json \
       --candidates examples/seeds-v3-acylation.json --sites manifest --manifest <manifest.json>

# 3. Control-gated, uncertainty-aware ranking and process design
python -m warden assess --db runs/v3_gas/campaign.sqlite --config config/v3.json \
       --controls examples/controls-v3.json --out runs/v3_gas/assess

# 4. Closed loop: campaign runs screen -> kinetics -> assess after each proposal
python -m warden campaign --db runs/live/campaign.sqlite --manifest <manifest.json> \
       --config config/v3.json --controls examples/controls-v3.json --evidence <evidence.json>
```

`--method mace` reuses the v2 MACE checkpoint for barriers; running both
methods into the same database lets `assess` report method disagreement.
`--allow-unadmitted` is diagnostic only: such results are excluded from ranking.

## Results from this environment (2026-10-21, GFN2-xTB, CPU, gas-phase site models)

Reproduce with the commands above; report in `runs/v3_gas/assess/assessment.md`.

| Molecule | Ea SiO2 -OH | Ea SiNx (NH2 / NH bridge) | dEa (GS - NGS) | Outcome |
|---|---:|---:|---:|---|
| acetyl chloride (control) | 0.92 | 0.36 / 0.06 | **+0.71** | prefers SiNx, as expected; practical dose window at 373-423 K |
| benzoyl chloride | 0.20 | 0.21 (mean) | -0.01 | no kinetic selectivity |
| pivaloyl chloride | 0.14 | 0.15 / unresolved | — | incomplete |
| methyl chloroformate | unresolved | 0.27 (mean) | — | incomplete |
| acetic anhydride | 0.07 | unresolved | — | incomplete |
| N-acetylimidazole | 0.44 | unresolved | — | incomplete |
| DMATMS, BDIPADS (controls) | unresolved | unresolved | — | silylation NEB did not converge |

Acetyl chloride, nominal inputs: **S = 0.989 at 10 nm** (373 K, 50 ms dose,
redose every cycle). Over the stated priors: **P(success) = 0.52**, so the
schedule is not robust; Spearman ranking says measure steric shielding
(rho 0.72) and cap saturation theta_max (0.43) first. At 473 K and above the
window drops below practical dose times: the barriers would need to be about
1.0 eV on both surfaces, i.e. a milder electrophile with the same contrast.

Control gate: `uncalibrated_incomplete` (one of three controls computed).

Slab-cluster diagnostic (acetyl chloride, `--allow-unadmitted`, excluded from
ranking): valid barriers at 2/4 SiO2 sites (0.06, 0.11 eV; 42% of population)
and 2/8 SiNx sites (0.11, 1.84 eV; 15%); all four NH-bridge clusters were
refused as open-shell (slab dangling bonds). The apparent reversal
(dEa = -1.07 eV) rests on too few sites to interpret, and the process model
refuses it. The gas-phase preference is therefore **not yet confirmed on
amorphous environments**; a defect-free nitride and a robust TS search come first.
Barriers within 0.05 eV of zero are labelled `no_barrier_resolved` and
excluded; earlier near-zero values were artefacts of a shallow starting complex.

## Limitations

- NEB on finite models with a fixed number of images; one conformer per molecule;
  the climbing image is the barrier estimate. Bands that jump > 1.5 eV between
  images or squeeze atoms are labelled `band_unresolved`, not used.
- Silylation four-centre paths often fail to converge with plain NEB; these are
  recorded as failures. A dedicated TS search (dimer/Sella) is the next step.
- GFN2-xTB barriers need DFT spot checks; the default method sigma (0.15-0.20 eV)
  is a stated prior, replaced by measured disagreement when two methods exist.
- Process priors (p_nuc, shielding, theta_max, k_deg) are assumptions until
  `calibrate()` is run on measured growth curves.
- Clusters with an odd electron count (dangling bonds in the slab) are refused.
- The bundled SiNx reference fails density admission and has 36% mis-coordinated
  Si and 23 Si-Si bonds (`validate-surfaces` now reports `coordination_defects`).
