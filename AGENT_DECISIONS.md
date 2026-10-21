# WARDEN evidence-acquisition agent

The agent decides **which evidence to obtain next** — another molecule, site,
reaction pathway, orientation, or a more accurate method — and we measure the
benefit of those decisions against strong baselines under equal budgets.
Code: `warden/decide.py` (agent), `warden/benchmark.py` (evaluation),
`warden/process.py::stress_test` (recipe stress), `warden/kinetics.py`
(pathways `addition`, `four_centre`, `backside`).

## How it decides

| Component | Implementation |
|---|---|
| Surface-specific chemistry | Hierarchical Gaussian belief over every barrier: candidate × amorphous site × pathway. A site's effective barrier is the minimum over pathways, so unexplored pathways keep real uncertainty. `site_robustness` bootstraps the site set to test whether the pick survives different sites. |
| Method reliability | Per-method noise **and** systematic bias per (surface, template), learned separately for barriers and reaction energies from results paired with the reference method. Cheap results are bias-corrected; residual bias does not average away, so escalation is valued only when cheap results cannot separate the leaders. |
| Agent decisions | Each step values every action by knowledge gradient on the dose-window feasibility margin + decision-relevant uncertainty (P(top two) × P(leaf sets the window) × uncertainty this method can remove), × learned success probability (empirical-Bayes per method/template/pathway/site type, decaying on retries), ÷ cost. Stops at P(correct selection) ≥ 0.95 or when nothing is worth its cost. Every decision is logged with its reason and alternatives. |
| Process prediction | `posterior_design` runs the uncertainty-aware process design on posterior barriers, then `stress_test` sweeps cap loss, incomplete re-passivation (theta_max), slower re-passivation, unwanted oxide inhibition and oxide-cap persistence. |
| Evaluation | `warden benchmark`: synthetic worlds with hidden barriers, template-dependent method biases and failure rates. All policies share the same inference and pick rule; only acquisition differs. |

## Commands

```bash
env -u PYTHONPATH python -m pytest -q                       # 72 tests
python -m warden benchmark --worlds 40 --budget 40           # equal-budget comparison
python -m warden decide --db <db> --sites gas --budget 10 --dry-run   # what to compute next, and why
python -m warden decide --db <db> --sites gas --budget 6              # run it (xTB, live)
python -m warden decide --db <db> --sites manifest --manifest <m.json> --library examples/seeds-v3-acylation.json
```

Reports: `<out>/decisions.md` and `decisions.json` (decision trace, method
bias table, failure model, site robustness, process recipe, stress test).

## Benchmark results (40 synthetic worlds per budget; paired regret reduction, decades, ± s.e.)

| Budget | vs exhaustive cheap | vs screen-then-verify | vs uncertainty sampling | Agent compute used |
|---:|---:|---:|---:|---:|
| 20 | **+3.01 ± 0.78** | **+2.06 ± 0.63** | -0.13 ± 0.45 | 92% |
| 40 | **+1.70 ± 0.54** | **+1.70 ± 0.60** | -0.23 ± 0.36 | 84% |
| 80 | +0.06 ± 0.17 | **+1.49 ± 0.41** | +0.16 ± 0.12 | 76% |

After switching the failure prior to empirical Bayes (needed for realistic live
behaviour), budget 40 gives +1.46 ± 0.59, +1.46 ± 0.67 and -0.47 ± 0.40.

Reading: the agent clearly beats workflow-style baselines (screen everything
cheaply; screen then verify the top two with the reference method) at tight
budgets, and matches exhaustive screening at large budgets with ~25% less
compute. It does **not** beat uncertainty sampling on decision quality; it
matches it while stopping earlier and explaining each choice. These are tests
of acquisition strategy on synthetic worlds, not chemical evidence.

## Live run (GFN2-xTB, gas-phase sites, budget 6)

Seeded with the earlier kinetics screen, the agent chose to test untried
pathways on the silanol site for the three leading candidates (an alternative
oxide pathway with a lower barrier would erase selectivity). All six
calculations failed (unresolved band, unconverged band, SCF failures, product
not formed); the agent lowered its success estimates after each failure and
switched orientation rather than repeating. With the empirical-Bayes prior its
next-action success estimates are now <= 22%, which makes the limiting factor
explicit: the NEB backend, not the decision logic. A more robust
transition-state method is the highest-value addition.

## Limitations

- Gaussian belief with stated priors; leaf data enter the hyper-mean slightly
  twice (standard empirical-Bayes approximation).
- Bias is shared per (method, surface, template); cross-candidate bias
  correlation is ignored, which makes P(correct selection) conservative.
- No DFT backend is wired for live runs; `dft` exists in the benchmark and as
  a method an external result import can supply.
