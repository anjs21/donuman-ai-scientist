# Inhibitor / Precursor Selection Criteria

Supplemental knowledge file for the AS-ALD selection agent (Challenge 4,
`selection_agent.py`). The agent reads the machine-readable block at the bottom;
the prose explains the reasoning so a human co-scientist can audit and edit it.

## Problem framing

**Target process:** passivate SiN (nitride) so that SiOx deposits selectively
on SiO (oxide), reaching **90% selectivity at 10 nm** of oxide thickness.

- **Growth surface (GS) = SiO2** — we WANT film growth here. An inhibitor must
  bind this surface *weakly* so precursor chemisorption is unimpeded.
- **Non-growth surface (NGS) = SiNx** — we want to BLOCK growth here. An
  inhibitor must bind this surface *strongly* to passivate the -NH2 / -NH sites.

So the agent looks for an inhibitor with a large **selectivity contrast**:

```
contrast = dE_ads(GS)  -  dE_ads(NGS)      [eV]
```

Both dE are adsorption/reaction energies (negative = favourable). A large
positive contrast means "strongly bound on the NGS, weakly bound on the GS" —
exactly the blocking selectivity we need.

## Ranking criteria (in priority order)

1. **Blocks the NGS.** Require `dE_ads(NGS) <= bind_threshold_eV`. If the
   inhibitor does not actually stick to SiNx, it cannot passivate it.
2. **Spares the GS.** Prefer `dE_ads(GS) >= spare_threshold_eV` (weak/unfavourable
   on oxide). Penalise inhibitors that also cap the SiO2 -OH we need for growth.
3. **Maximise contrast.** Among candidates passing 1–2, rank by `contrast`.
4. **Volatility / processability.** Prefer `high` volatility (clean vapour
   dosing, no residue). Used as a tie-breaker and a soft penalty.
5. **Strained-site robustness.** Down-weight inhibitors whose favourable binding
   comes only from a few strained hot-spots (flagged by
   `surface_builder.flag_strained_sites`) rather than typical sites — those
   dE values are less transferable.

## Precursor selection (for the SiOx film on the GS)

A precursor is scored on the *opposite* logic: it must chemisorb **favourably on
the GS (SiO2 -OH)** to grow film, i.e. `dE_ads(GS) <= precursor_threshold_eV`.
Selectivity is delivered by the inhibitor, not the precursor, so precursor
ranking is simply "most favourable on the growth surface, high volatility".

## Editable thresholds

The agent parses the fenced `yaml`-style block below. Units: eV.

```selection-config
bind_threshold_eV: -0.30       # inhibitor must be at least this favourable on NGS
spare_threshold_eV: -0.20      # inhibitor should be no more favourable than this on GS
precursor_threshold_eV: -0.30  # precursor must be at least this favourable on GS
contrast_weight: 1.0           # weight on (dE_GS - dE_NGS)
volatility_bonus: 0.15         # score bonus for high-volatility reagents
volatility_penalty: 0.15       # score penalty for low-volatility reagents
strain_penalty: 0.10           # penalty if favourable binding is strain-dominated
min_contrast_eV: 0.20          # below this, treat GS/NGS as non-selective
```
