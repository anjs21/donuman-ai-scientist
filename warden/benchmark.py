"""Equal-budget evaluation of evidence-acquisition policies on synthetic worlds.

Each world hides true barriers (candidate x site x pathway), per-method
systematic biases that differ by surface, noise, and failure rates. Every
policy uses the same Belief inference and the same final pick rule; only the
choice of which calculation to buy differs. This measures decision quality
(correct pick, regret) and efficiency (cost spent) -- it is a test of the
agent's strategy, not of any chemistry.
"""
import numpy as np
from .common import clean
from .decide import DEFAULTS, Action, Agent, Belief, Observation, window_width
from .kinetics import PATHWAYS

WORLD = {"n_candidates": 6, "sites": {"SiO2": 3, "SiNx": 4},
         "bias_sd_eV": {"xtb": 0.20, "mace": 0.30, "dft": 0.03},
         "fail": {"xtb": (0.15, 0.35), "mace": (0.10, 0.30), "dft": (0.10, 0.20)},
         "unrunnable_fraction": 0.0}


def make_world(seed, cfg=None, world=None):
    cfg = {**DEFAULTS, **(cfg or {})}; world = {**WORLD, **(world or {})}
    rng = np.random.default_rng(seed)
    sites = []
    for m, n in world["sites"].items():
        types = ["OH"] if m == "SiO2" else ["NH2", "NH_bridge"]
        for i in range(n):
            sites.append({"material": m, "label": f"{m}:{i}", "site_type": types[i % len(types)],
                          "population": int(rng.integers(1, 10))})
    candidates, truth = {}, {}
    for c in range(world["n_candidates"]):
        cid = f"c{c}"
        template = "acylation" if c % 3 else "silyl_substitution"
        candidates[cid] = {"id": cid, "name": cid, "template": template}
        mu = {m: rng.normal(cfg["prior_mean_eV"][m], cfg["prior_sd_eV"]) for m in ("SiO2", "SiNx")}
        for s in sites:
            base = mu[s["material"]] + rng.normal(0, cfg["site_sd_eV"])
            for p in PATHWAYS[template]:
                truth[(cid, s["label"], p)] = base + rng.normal(0, cfg["pathway_sd_eV"])
    # Cheap-method errors depend on the chemistry: bias per method, surface and template.
    bias = {(meth, m, t): rng.normal(0, sd) for meth, sd in world["bias_sd_eV"].items()
            for m in ("SiO2", "SiNx") for t in PATHWAYS}
    return {"sites": sites, "candidates": candidates, "truth": truth, "bias": bias,
            "fail": world["fail"], "seed": seed}


def true_width(world, cid, cfg):
    cfg = {**DEFAULTS, **cfg}
    template = world["candidates"][cid]["template"]
    eff = {m: np.array([min(world["truth"][(cid, s["label"], p)] for p in PATHWAYS[template])
                        for s in world["sites"] if s["material"] == m]) for m in ("SiO2", "SiNx")}
    return float(window_width(eff["SiO2"][None], eff["SiNx"][None], cfg, clip=False)[0])


class SyntheticExecutor:
    def __init__(self, world, cfg):
        self.world, self.cfg = world, {**DEFAULTS, **cfg}
        self.rng = np.random.default_rng([world["seed"], 7])

    def execute(self, a, belief):
        w = self.world
        material = next(s["material"] for s in w["sites"] if s["label"] == a.site)
        index = PATHWAYS[w["candidates"][a.candidate]["template"]].index(a.pathway)
        cost = self.cfg["methods"][a.method]["cost"]
        if self.rng.random() < w["fail"][a.method][index]:
            return Observation(a.candidate, a.site, a.pathway, a.method, "failed", reason="synthetic_failure", cost=cost)
        truth = w["truth"][(a.candidate, a.site, a.pathway)]
        bias = w["bias"][(a.method, material, w["candidates"][a.candidate]["template"])]
        sd = self.cfg["methods"][a.method]["sigma_barrier_eV"]
        y = truth + bias + self.rng.normal(0, 0.5 * sd)
        r = truth - 0.6 + 0.5 * bias + self.rng.normal(0, 0.5 * sd)
        return Observation(a.candidate, a.site, a.pathway, a.method, "valid", float(y), float(r), cost=cost)


# --- baseline policies: same inference, different evidence choice ---------------

def _keys(belief, method_order):
    for cid in belief.candidates:
        for m in ("SiO2", "SiNx"):
            for label, p in belief.leaves(cid, m):
                yield cid, label, p


def policy_exhaustive_cheap(belief, budget_left, state, cfg):
    """Round-robin over every candidate x site x pathway with the cheapest method."""
    cheap = min(state["methods"], key=lambda m: cfg["methods"][m]["cost"])
    keys = list(_keys(belief, None))
    key = keys[state.setdefault("i", 0) % len(keys)]
    state["i"] += 1
    return Action("observe", *key, cheap, cost=cfg["methods"][cheap]["cost"])


def policy_screen_then_verify(belief, budget_left, state, cfg):
    """Cheap screen of every candidate at the most populated site of each surface,
    then the reference method on the top two, then round-robin cheap."""
    cheap = min(state["methods"], key=lambda m: cfg["methods"][m]["cost"])
    ref = cfg["reference_method"] if cfg["reference_method"] in state["methods"] else cheap
    if "plan" not in state:
        rep = {m: max((s for s in belief.sites if s["material"] == m), key=lambda s: s["population"])["label"]
               for m in ("SiO2", "SiNx")}
        state["rep"] = rep
        state["plan"] = [(c, rep[m], belief.leaves(c, m)[0][1], cheap) for c in belief.candidates for m in rep]
    if not state["plan"] and not state.get("verified"):
        scores = belief.scores()
        top = sorted(scores, key=scores.get, reverse=True)[:2]
        state["plan"] = [(c, state["rep"][m], belief.leaves(c, m)[0][1], ref) for c in top for m in state["rep"]]
        state["verified"] = True
    while state["plan"]:
        c, s, p, meth = state["plan"].pop(0)
        if cfg["methods"][meth]["cost"] <= budget_left:
            return Action("observe", c, s, p, meth, cost=cfg["methods"][meth]["cost"])
    return policy_exhaustive_cheap(belief, budget_left, state, cfg)


def policy_uncertainty(belief, budget_left, state, cfg):
    """Cheapest method on the leaf with the largest posterior variance."""
    cheap = min(state["methods"], key=lambda m: cfg["methods"][m]["cost"])
    best, key = -1, None
    for cid in belief.candidates:
        for m in ("SiO2", "SiNx"):
            for i, (label, p) in enumerate(belief.leaves(cid, m)):
                _, var = belief.leaf_marginal(cid, m, i)
                if var > best:
                    best, key = var, (cid, label, p)
    return Action("observe", *key, cheap, cost=cfg["methods"][cheap]["cost"])


def policy_random(belief, budget_left, state, cfg):
    rng = state.setdefault("rng", np.random.default_rng(0))
    keys = list(_keys(belief, None))
    methods = [m for m in state["methods"] if cfg["methods"][m]["cost"] <= budget_left]
    if not methods:
        return None
    key = keys[rng.integers(len(keys))]
    meth = methods[rng.integers(len(methods))]
    return Action("observe", *key, meth, cost=cfg["methods"][meth]["cost"])


BASELINES = {"exhaustive_cheap": policy_exhaustive_cheap, "screen_then_verify": policy_screen_then_verify,
             "uncertainty_sampling": policy_uncertainty, "random": policy_random}


def run_policy(name, world, budget, cfg, methods, tolerance=0.1):
    cfg = {**DEFAULTS, **cfg}
    belief = Belief(world["sites"], world["candidates"], cfg)
    executor = SyntheticExecutor(world, cfg)
    spent, n_fail, n_actions = 0., 0, 0
    if name == "warden_voi":
        agent = Agent(belief, methods, (), cfg)
        while True:
            if belief.obs and belief.pcs() >= cfg["target_pcs"]:
                break
            a, _ = agent.choose(budget - spent)
            if a is None:
                break
            o = executor.execute(a, belief); belief.add(o)
            spent += o.cost; n_fail += o.status != "valid"; n_actions += 1
    else:
        state = {"methods": list(methods)}
        while True:
            a = BASELINES[name](belief, budget - spent, state, cfg)
            if a is None or a.cost > budget - spent:
                break
            o = executor.execute(a, belief); belief.add(o)
            spent += o.cost; n_fail += o.status != "valid"; n_actions += 1
    scores = belief.scores()
    pick = max(scores, key=scores.get)
    truth = {c: true_width(world, c, cfg) for c in world["candidates"]}
    best = max(truth.values())
    # Margins saturate at the full practical range, so near-ties count as correct.
    return {"policy": name, "pick": pick, "correct": truth[pick] >= best - tolerance,
            "regret_decades": best - truth[pick], "spent": spent, "actions": n_actions,
            "failures": n_fail, "pcs_reported": belief.pcs()}


def benchmark(n_worlds=20, budget=60., cfg=None, methods=("xtb", "dft"), policies=None, world=None, seed=0,
              tolerance=0.1):
    cfg = {**DEFAULTS, **(cfg or {})}
    policies = policies or ["warden_voi", *BASELINES]
    rows = []
    for w in range(n_worlds):
        world_w = make_world(seed + w, cfg, world)
        for name in policies:
            rows.append({"world": w, **run_policy(name, world_w, budget, cfg, methods, tolerance)})
    summary = []
    for name in policies:
        r = [x for x in rows if x["policy"] == name]
        if not r:
            continue
        correct = np.array([x["correct"] for x in r], float)
        regret = np.array([x["regret_decades"] for x in r])
        summary.append({"policy": name, "worlds": len(r),
                        "p_correct": float(correct.mean()), "p_correct_se": float(correct.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else None,
                        "mean_regret_decades": float(regret.mean()),
                        "mean_spent": float(np.mean([x["spent"] for x in r])),
                        "mean_failures": float(np.mean([x["failures"] for x in r]))})
    return clean({"budget": budget, "methods": list(methods), "summary": summary, "runs": rows,
                  "note": "Synthetic worlds test acquisition strategy only; they are not chemical evidence."})
