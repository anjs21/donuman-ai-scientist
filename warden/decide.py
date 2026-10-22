"""Evidence-acquisition agent: decides which calculation to run next.

The agent holds a hierarchical Gaussian belief over every barrier it could
compute -- candidate x surface site x reaction pathway -- and scores each
candidate by the width (decades) of its practical dose window at the best
temperature, the quantity that decides kinetic AS-ALD feasibility (see
process.dose_window). Each step it values every available action:

    another molecule | another site | another pathway | another orientation
    (retry of a failed key) | a more accurate method on an existing key

by its knowledge gradient (expected gain in the best attainable score), times
the action's learned success probability, per unit cost. Paired results from
two methods update that method's error separately for barriers and reaction
energies, so escalation to an expensive method happens only when the cheap one
cannot separate the leading candidates. It stops when the choice is confident
(probability of correct selection) or no action is worth its cost.

Every decision is logged with the numbers behind it.
"""
from dataclasses import dataclass, field
import math
import numpy as np
from scipy.stats import norm
from .common import clean
from .kinetics import H_EVS, KB_EV, PATHWAYS

DEFAULTS = {
    "prior_mean_eV": {"SiO2": 0.9, "SiNx": 0.6}, "prior_sd_eV": 0.5,
    "site_sd_eV": 0.2, "pathway_sd_eV": 0.2,
    "methods": {"xtb": {"cost": 1.0, "sigma_barrier_eV": 0.15, "sigma_reaction_eV": 0.10, "bias_sd_eV": 0.20},
                "mace": {"cost": 0.3, "sigma_barrier_eV": 0.25, "sigma_reaction_eV": 0.15, "bias_sd_eV": 0.30},
                "dft": {"cost": 25.0, "sigma_barrier_eV": 0.05, "sigma_reaction_eV": 0.04, "bias_sd_eV": 0.05}},
    "reference_method": "dft",
    "temperatures": [373., 423., 473., 523.], "practical_dose_s": [0.05, 30.], "occupancy": 1.0,
    "capture_NGS": 0.99, "capture_GS": 0.01,
    "draws": 128, "kg_points": 5, "target_pcs": 0.95, "min_value_per_cost": 1e-4,
    "new_molecule_cost": 2.0, "relevance_weight": 0.5, "success_prior": [3., 1.], "retry_decay": 0.7, "seed": 0}


def window_width(ea_gs, ea_ngs, cfg, clip=True):
    """Practical dose-window width (decades) at the best temperature.

    ea_*: (..., sites). The slowest nitride site must be capped (capture_NGS)
    before the fastest oxide site reaches capture_GS. Computed in log space.
    With clip=False the result is a signed feasibility margin: negative values
    say by how many decades the window misses, which keeps the agent's
    objective informative where every candidate is still infeasible.
    """
    t_lo, t_hi = np.log10(cfg["practical_dose_s"])
    best = np.full(np.shape(ea_gs)[:-1], -np.inf)
    for T in cfg["temperatures"]:
        kt = KB_EV * T
        log_pre = np.log10(cfg["occupancy"] * kt / H_EVS)
        lo = np.log10(-np.log1p(-cfg["capture_NGS"])) - log_pre + ea_ngs.max(-1) / (kt * np.log(10))
        hi = np.log10(-np.log1p(-cfg["capture_GS"])) - log_pre + ea_gs.min(-1) / (kt * np.log(10))
        best = np.maximum(best, np.minimum(hi, t_hi) - np.maximum(lo, t_lo))
    return np.maximum(best, 0.) if clip else best


class MethodModel:
    """Per-method noise plus a systematic bias per (surface, template).

    Results paired with the reference method estimate each bias (Gaussian,
    prior bias_sd_eV), separately for barriers and reaction energies. Cheap
    results are bias-corrected; the remaining bias variance does not average
    away with repeated cheap calculations, which is what makes escalation
    worth paying for.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.pairs = {}  # (method, quantity, material, template) -> differences vs reference

    def noise(self, method, quantity="barrier"):
        return self.cfg["methods"][method][f"sigma_{quantity}_eV"]

    def bias(self, method, material, template, quantity="barrier"):
        """(mean, variance) of the method's systematic offset from the reference."""
        b0 = self.cfg["methods"][method].get("bias_sd_eV", 0.)
        if method == self.cfg["reference_method"]:
            return 0., b0 ** 2
        d = self.pairs.get((method, quantity, material, template), [])
        s2 = self.noise(method, quantity) ** 2 + self.noise(self.cfg["reference_method"], quantity) ** 2
        prec = 1 / max(b0, 1e-6) ** 2 + len(d) / s2
        return float(sum(d) / s2 / prec), float(1 / prec)

    def sigma(self, method, quantity="barrier", material=None, template=None):
        """Total error sd for reporting: noise and unexplained bias."""
        if material is None:
            b0 = self.cfg["methods"][method].get("bias_sd_eV", 0.)
            return math.sqrt(self.noise(method, quantity) ** 2 + b0 ** 2)
        mean, var = self.bias(method, material, template, quantity)
        return math.sqrt(self.noise(method, quantity) ** 2 + var)

    def add(self, method, quantity, material, template, difference):
        self.pairs.setdefault((method, quantity, material, template), []).append(float(difference))

    def report(self):
        out = []
        for (m, q, mat, t), d in sorted(self.pairs.items()):
            mean, var = self.bias(m, mat, t, q)
            out.append({"method": m, "quantity": q, "material": mat, "template": t, "n_pairs": len(d),
                        "mean_difference_eV": float(np.mean(d)),
                        "rms_difference_eV": float(np.sqrt(np.mean(np.square(d)))),
                        "bias_estimate_eV": mean, "bias_sd_eV": math.sqrt(var)})
        return out


class FailureModel:
    """Beta success rate per (method, template, pathway, site_type); retries decay per key."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.counts = {}
        self.key_failures = {}

    def p(self, method, template, pathway, site_type, key):
        # Empirical Bayes: an untried class starts from this method's pooled
        # success rate, so a backend that fails often is not assumed reliable.
        a, b = self.cfg["success_prior"]
        s_all = sum(v[0] for k, v in self.counts.items() if k[0] == method)
        f_all = sum(v[1] for k, v in self.counts.items() if k[0] == method)
        pooled = (s_all + a) / (s_all + f_all + a + b)
        s, f = self.counts.get((method, template, pathway, site_type), (0, 0))
        n0 = a + b
        return (s + n0 * pooled) / (s + f + n0) * self.cfg["retry_decay"] ** self.key_failures.get(key, 0)

    def add(self, method, template, pathway, site_type, key, ok):
        s, f = self.counts.get((method, template, pathway, site_type), (0, 0))
        self.counts[(method, template, pathway, site_type)] = (s + ok, f + (not ok))
        if not ok:
            self.key_failures[key] = self.key_failures.get(key, 0) + 1

    def report(self):
        return [{"method": k[0], "template": k[1], "pathway": k[2], "site_type": k[3],
                 "successes": s, "failures": f} for k, (s, f) in sorted(self.counts.items())]


@dataclass
class Observation:
    candidate: str
    site: str
    pathway: str
    method: str
    status: str
    Ea_eV: float = None
    dE_rxn_eV: float = None
    reason: str = None
    cost: float = 0.
    source: str = "agent"


class Belief:
    def __init__(self, sites, candidates, cfg):
        self.cfg = cfg
        self.sites = [s for s in sites if not s.get("error")]
        self.unrunnable = [s for s in sites if s.get("error")]
        self.candidates = dict(candidates)  # id -> spec (needs template, name)
        self.obs = []
        self.methods = MethodModel(cfg)
        self.failures = FailureModel(cfg)
        self._z = {}

    # --- structure -------------------------------------------------------------
    def leaves(self, cid, material):
        template = self.candidates[cid]["template"]
        return [(s["label"], p) for s in self.sites if s["material"] == material for p in PATHWAYS[template]]

    def site(self, label):
        return next(s for s in self.sites if s["label"] == label)

    def add(self, o):
        self.obs.append(o)
        spec = self.candidates[o.candidate]
        site = self.site(o.site)
        key = (o.candidate, o.site, o.pathway, o.method)
        self.failures.add(o.method, spec["template"], o.pathway, site["site_type"], key, o.status == "valid")
        if o.status != "valid":
            return
        ref = self.cfg["reference_method"]
        for other in self.obs[:-1]:
            if (other.status == "valid" and other.candidate == o.candidate and other.site == o.site
                    and other.pathway == o.pathway and {other.method, o.method} >= {ref} and other.method != o.method):
                cheap, exact = (o, other) if other.method == ref else (other, o)
                t = spec["template"]
                self.methods.add(cheap.method, "barrier", site["material"], t, cheap.Ea_eV - exact.Ea_eV)
                if cheap.dE_rxn_eV is not None and exact.dE_rxn_eV is not None:
                    self.methods.add(cheap.method, "reaction", site["material"], t, cheap.dE_rxn_eV - exact.dE_rxn_eV)

    def _data(self, cid, leaves, material):
        """Bias-corrected leaf means, noise precisions, and the residual bias
        variance of the data (precision-weighted over the methods used)."""
        template = self.candidates[cid]["template"]
        ybar = np.zeros(len(leaves)); prec = np.zeros(len(leaves)); bvar = 0.; total = 0.
        index = {l: i for i, l in enumerate(leaves)}
        for o in self.obs:
            if o.candidate == cid and o.status == "valid" and (o.site, o.pathway) in index:
                i = index[(o.site, o.pathway)]
                mean, var = self.methods.bias(o.method, material, template)
                w = 1 / self.methods.noise(o.method) ** 2
                ybar[i] = (ybar[i] * prec[i] + (o.Ea_eV - mean) * w) / (prec[i] + w)
                prec[i] += w
                bvar += w * var; total += w
        return ybar, prec, (bvar / total if total else 0.), total

    def _noise(self, cid, n_leaves):
        if cid not in self._z:
            rng = np.random.default_rng([self.cfg["seed"], len(self._z)])
            self._z[cid] = {m: (rng.standard_normal(self.cfg["draws"]),
                                rng.standard_normal((self.cfg["draws"], n_leaves[m])),
                                rng.standard_normal(self.cfg["draws"])) for m in n_leaves}
        return self._z[cid]

    # --- posterior samples --------------------------------------------------------
    def samples(self, cid, extra=None):
        """Effective per-site barrier samples per material, shape (K, D, sites).

        A shared residual-bias draw shifts all of a candidate's data on one
        surface together. extra = (material, leaf_index, y_values[K], noise_var,
        bias_var): hypothetical observations for the knowledge gradient, using
        common random numbers so values are smooth in y.
        """
        mats = ("SiO2", "SiNx")
        leaves = {m: self.leaves(cid, m) for m in mats}
        z = self._noise(cid, {m: len(leaves[m]) for m in mats})
        t2 = self.cfg["site_sd_eV"] ** 2 + self.cfg["pathway_sd_eV"] ** 2
        out = {}
        for m in mats:
            ybar, prec, bvar, total = self._data(cid, leaves[m], m)
            ybar, prec = ybar[None], prec[None]
            if extra and extra[0] == m:
                _, i, ys, var, b_extra = extra
                ys = np.asarray(ys, float)
                ybar = np.repeat(ybar, len(ys), 0); prec = np.repeat(prec, len(ys), 0)
                ybar[:, i] = (ybar[:, i] * prec[:, i] + ys / var) / (prec[:, i] + 1 / var)
                prec[:, i] += 1 / var
                bvar = (bvar * total + b_extra / var) / (total + 1 / var)
            bias = z[m][2][None, :] * math.sqrt(bvar)                      # (1, D)
            data = ybar[:, None, :] - bias[:, :, None]                      # (K, D, L)
            w = prec / (1 + t2 * prec)  # 1 / (t2 + 1/prec), zero when unobserved
            s0 = self.cfg["prior_sd_eV"]
            prec_mu = 1 / s0 ** 2 + w.sum(-1)                               # (K,)
            mean_mu = (self.cfg["prior_mean_eV"][m] / s0 ** 2 + (w[:, None, :] * data).sum(-1)) / prec_mu[:, None]
            mu = mean_mu + z[m][0][None] / np.sqrt(prec_mu)[:, None]        # (K, D)
            prec_leaf = 1 / t2 + prec
            leaf = ((mu[:, :, None] / t2 + prec[:, None, :] * data) / prec_leaf[:, None, :]
                    + z[m][1][None] / np.sqrt(prec_leaf)[:, None, :])
            labels = [l for l, _ in leaves[m]]
            site_order = list(dict.fromkeys(labels))
            out[m] = np.stack([leaf[:, :, [j for j, l in enumerate(labels) if l == s]].min(-1)
                               for s in site_order], -1)
            out["leaf_" + m] = leaf
        return out

    def binding(self, cid):
        """P(each leaf is the one that sets the window): the fastest oxide leaf
        and, on the nitride, the lowest pathway of the slowest site."""
        s = self.samples(cid)
        out = {}
        gs = s["leaf_SiO2"][0]
        out["SiO2"] = np.bincount(gs.argmin(-1), minlength=gs.shape[-1]) / gs.shape[0]
        leaves = self.leaves(cid, "SiNx")
        labels = [l for l, _ in leaves]
        order = list(dict.fromkeys(labels))
        slow = s["SiNx"][0].argmax(-1)
        ngs = s["leaf_SiNx"][0]
        hits = np.zeros(len(leaves))
        for d, site_j in enumerate(slow):
            idx = [j for j, l in enumerate(labels) if l == order[site_j]]
            hits[idx[int(np.argmin(ngs[d, idx]))]] += 1
        out["SiNx"] = hits / ngs.shape[0]
        return out

    def contention(self):
        """P(candidate is among the top two) under the joint posterior."""
        ids = list(self.candidates)
        w = np.stack([self.widths(c)[0] for c in ids])
        rank = (-w).argsort(0).argsort(0)
        return {c: float(np.mean(rank[i] <= 1)) for i, c in enumerate(ids)}

    def widths(self, cid, extra=None):
        s = self.samples(cid, extra)
        return window_width(s["SiO2"], s["SiNx"], self.cfg, clip=False)  # (K, D) signed margin

    def scores(self):
        return {cid: float(self.widths(cid)[0].mean()) for cid in self.candidates}

    def pcs(self):
        """Posterior probability that the current pick is truly best."""
        if not self.candidates:
            return 0.
        w = {cid: self.widths(cid)[0] for cid in self.candidates}
        means = {c: v.mean() for c, v in w.items()}
        pick = max(means, key=means.get)
        top = np.max(np.stack(list(w.values())), axis=0)
        return float(np.mean(w[pick] >= top - 1e-12))

    def leaf_marginal(self, cid, material, i):
        leaf = self.samples(cid)["leaf_" + material][0, :, i]
        return float(leaf.mean()), float(leaf.var())

    def posterior_means(self, cid):
        """Posterior-mean effective barrier per site (for process design)."""
        s = self.samples(cid)
        out = []
        for m in ("SiO2", "SiNx"):
            labels = list(dict.fromkeys(l for l, _ in self.leaves(cid, m)))
            for j, label in enumerate(labels):
                observed = any(o.candidate == cid and o.site == label and o.status == "valid" for o in self.obs)
                out.append({"material": m, "label": label, "site_type": self.site(label)["site_type"],
                            "population": self.site(label)["population"], "status": "valid",
                            "Ea_eV": float(s[m][0, :, j].mean()), "Ea_sd_eV": float(s[m][0, :, j].std()),
                            "observed": observed, "precomplex_binding_eV": None})
        return out


@dataclass
class Action:
    kind: str
    candidate: str = None
    site: str = None
    pathway: str = None
    method: str = None
    orientation: int = 0
    value: float = 0.
    p_success: float = 1.
    cost: float = 1.
    why: str = ""
    extra: dict = field(default_factory=dict)

    def score(self):
        return self.value * self.p_success / self.cost


class Agent:
    """Knowledge-gradient evidence acquisition over molecules, sites, pathways and methods."""

    def __init__(self, belief, methods_available, library=(), cfg=None):
        self.belief = belief
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.methods_available = list(methods_available)
        self.library = list(library)

    def actions(self):
        b, cfg = self.belief, self.cfg
        scores = b.scores()
        best = max(scores.values()) if scores else 0.
        q = norm.ppf((np.arange(cfg["kg_points"]) + .5) / cfg["kg_points"])
        acts = []
        contention = b.contention() if len(b.candidates) > 1 else {c: 1. for c in b.candidates}
        sens = 1 / (KB_EV * np.mean(cfg["temperatures"]) * np.log(10))  # decades per eV
        for cid in b.candidates:
            others = max([v for c, v in scores.items() if c != cid], default=-np.inf)
            spec = b.candidates[cid]
            binding = b.binding(cid)
            for m in ("SiO2", "SiNx"):
                for i, (label, pathway) in enumerate(b.leaves(cid, m)):
                    mean, var = b.leaf_marginal(cid, m, i)
                    site = b.site(label)
                    for method in self.methods_available:
                        bmean, bvar = b.methods.bias(method, m, spec["template"])
                        noise2 = b.methods.noise(method) ** 2
                        sig2 = noise2 + bvar
                        ys = mean + np.sqrt(var + sig2) * q
                        w = b.widths(cid, (m, i, ys, noise2, bvar)).mean(-1)
                        kg = float(np.mean(np.maximum(w, others)) - max(scores[cid], others))
                        # Decision-relevant uncertainty: candidate in contention x leaf sets the
                        # window x fraction of the leaf's uncertainty this method removes.
                        # Only uncertainty above this method's own bias floor is removable by it.
                        reducible = max(var - bvar, 0.)
                        relevance = (contention[cid] * binding[m][i] * math.sqrt(reducible)
                                     * reducible / (reducible + noise2) * sens * cfg["relevance_weight"])
                        value = max(kg, 0.) + relevance
                        key = (cid, label, pathway, method)
                        attempts = sum(1 for o in b.obs if (o.candidate, o.site, o.pathway, o.method) == key)
                        acts.append(Action("observe", cid, label, pathway, method, attempts,
                                           max(value, 0.), b.failures.p(method, spec["template"], pathway,
                                                                         site["site_type"], key),
                                           cfg["methods"][method]["cost"],
                                           extra={"leaf_sd_eV": math.sqrt(var), "method_sd_eV": math.sqrt(sig2),
                                                  "knowledge_gradient": max(kg, 0.), "relevance": relevance,
                                                  "p_contention": contention[cid], "p_binding_leaf": float(binding[m][i])}))
        if self.library:
            spec = self.library[0]
            probe = Belief(b.sites, {"_new": spec}, b.cfg)
            w = probe.widths("_new")[0]
            acts.append(Action("add_molecule", value=float(np.mean(np.maximum(w, best)) - best),
                               cost=cfg["new_molecule_cost"], extra={"name": spec["name"]}))
        return acts

    def explain(self, a, previous):
        if a.kind == "add_molecule":
            return f"add {a.extra['name']}: prior chance to beat the current best justifies its cost"
        reason = []
        if previous and previous.get("failed_here"):
            reason.append(f"retry with orientation {a.orientation} after {previous['failed_here']} failure(s)")
        if previous and previous.get("observed_cheaper"):
            reason.append(f"escalate to {a.method}: cheaper-method sd {previous['cheap_sd']:.2f} eV cannot separate the leaders")
        if previous and previous.get("new_pathway"):
            reason.append(f"try pathway {a.pathway}: site barrier may be lower than the explored path")
        if previous and previous.get("new_site"):
            reason.append(f"new site {a.site}: unobserved site could reorder candidates")
        return "; ".join(reason) or f"observe {a.site} ({a.pathway}) with {a.method}"

    def choose(self, budget_left):
        acts = [a for a in self.actions() if a.cost <= budget_left]
        if not acts:
            return None, []
        acts.sort(key=lambda a: -a.score())
        best = acts[0]
        if best.score() < self.cfg["min_value_per_cost"]:
            return None, acts[:5]
        if best.kind == "observe":
            b = self.belief
            same = [o for o in b.obs if o.candidate == best.candidate and o.site == best.site]
            ctx = {"failed_here": sum(1 for o in same if o.pathway == best.pathway and o.method == best.method
                                      and o.status != "valid"),
                   "observed_cheaper": any(o.status == "valid" and o.pathway == best.pathway and o.method != best.method
                                           and b.cfg["methods"][o.method]["cost"] < b.cfg["methods"][best.method]["cost"]
                                           for o in same),
                   "cheap_sd": best.extra.get("method_sd_eV", 0.),
                   "new_pathway": any(o.pathway != best.pathway for o in same) and not any(o.pathway == best.pathway for o in same),
                   "new_site": not same}
            if ctx["observed_cheaper"]:
                cheap = [o.method for o in same if o.status == "valid" and o.method != best.method]
                ctx["cheap_sd"] = b.methods.sigma(cheap[0], material=b.site(best.site)["material"],
                                                  template=b.candidates[best.candidate]["template"])
            best.why = self.explain(best, ctx)
        else:
            best.why = self.explain(best, None)
        return best, acts[:5]


def run(belief, executor, budget, library=(), methods=("xtb",), cfg=None, log=None, max_steps=200):
    """Closed loop: choose, execute, update, until confident or out of budget."""
    agent = Agent(belief, methods, library, cfg)
    spent, trace = 0., []
    for step in range(max_steps):
        pcs = belief.pcs()
        if pcs >= agent.cfg["target_pcs"] and belief.obs:
            trace.append({"step": step, "stop": "confident", "pcs": pcs}); break
        action, shortlist = agent.choose(budget - spent)
        if action is None:
            trace.append({"step": step, "stop": "no_action_worth_its_cost" if shortlist else "budget_exhausted",
                          "pcs": pcs}); break
        record = {"step": step, "pcs_before": pcs, "action": clean(vars(action)),
                  "alternatives": [clean({k: getattr(a, k) for k in ("kind", "candidate", "site", "pathway", "method")}
                                         | {"value_per_cost": a.score()}) for a in shortlist[1:4]]}
        if action.kind == "add_molecule":
            spec = agent.library.pop(0)
            belief.candidates[spec["id"]] = spec
            spent += action.cost
            record["added"] = spec["name"]
        else:
            o = executor.execute(action, belief)
            spent += o.cost
            belief.add(o)
            record["result"] = clean(vars(o))
        record["spent"] = spent
        trace.append(record)
        if log:
            log(record)
    return {"trace": trace, "spent": spent, "scores": belief.scores(), "pcs": belief.pcs(),
            "pick": max(belief.scores(), key=belief.scores().get) if belief.candidates else None,
            "methods": belief.methods.report(), "failures": belief.failures.report()}


def site_robustness(belief, n_boot=200, seed=0):
    """Does the ordering survive different amorphous sites? Bootstrap the site
    set (per material) on posterior-mean barriers; report how often the pick stays top."""
    rng = np.random.default_rng(seed)
    means = {cid: {m: belief.samples(cid)[m][0].mean(0) for m in ("SiO2", "SiNx")} for cid in belief.candidates}
    scores = belief.scores()
    pick = max(scores, key=scores.get)
    stays = 0
    for _ in range(n_boot):
        idx = {m: rng.integers(0, len(means[pick][m]), len(means[pick][m])) for m in ("SiO2", "SiNx")}
        w = {cid: float(window_width(v["SiO2"][idx["SiO2"]][None], v["SiNx"][idx["SiNx"]][None], belief.cfg, clip=False)[0])
             for cid, v in means.items()}
        stays += w[pick] >= max(w.values()) - 1e-12
    return {"pick": pick, "fraction_pick_stays_top": stays / n_boot, "n_bootstrap": n_boot}


# --- live execution and reporting -----------------------------------------------

class LiveExecutor:
    """Runs the chosen NEB barrier (one orientation, one pathway, one method)."""

    def __init__(self, sites, kinetics_config, artifact_dir, cost=None):
        from . import kinetics as kin
        self.kin = kin
        self.sites = {s["label"]: s for s in sites if not s.get("error")}
        self.config = {**kin.DEFAULTS, **kinetics_config}
        self.artifact_dir = artifact_dir
        self.cost = cost or DEFAULTS["methods"]
        self._factories = {}

    def factory(self, method):
        if method not in self._factories:
            self._factories[method] = self.kin.calculator_factory({**self.config, "method": method})
        return self._factories[method]

    def execute(self, a, belief):
        import time
        spec = belief.candidates[a.candidate]
        s = self.sites[a.site]
        start = time.time()
        try:
            r = self.kin.barrier(s["substrate"], s["site"], s["normal"], s["frozen"], spec, self.factory(a.method),
                                 {**self.config, "method": a.method},
                                 f"{self.artifact_dir}/{spec['id']}_{a.site.replace(':', '_')}_{a.pathway}_{a.method}_{a.orientation}.extxyz",
                                 a.orientation, a.pathway)
            ok = r["status"] == "valid"
            return Observation(a.candidate, a.site, a.pathway, a.method, r["status"],
                               r["Ea_eV"] if ok else None, r["dE_rxn_eV"] if ok else None,
                               None if ok else r["status"], self.cost[a.method]["cost"],
                               source=f"live:{time.time() - start:.0f}s")
        except Exception as exc:
            return Observation(a.candidate, a.site, a.pathway, a.method, "failed", reason=str(exc),
                               cost=self.cost[a.method]["cost"], source=f"live:{time.time() - start:.0f}s")


def observations_from_kinetics(records):
    """Seed the belief with earlier kinetics screens (default pathway per template)."""
    out = []
    for k in records:
        pathway = PATHWAYS[k["template"]][0]
        for site in k["sites"]:
            for a in site.get("attempts", []):
                ok = a.get("status") == "valid"
                out.append(Observation(k["candidate_id"], site["label"], pathway, k["method"],
                                       a.get("status", "failed"), a.get("Ea_eV") if ok else None,
                                       a.get("dE_rxn_eV") if ok else None, a.get("reason"),
                                       source="kinetics_record"))
    return out


def posterior_design(belief, cid, mass_amu=100.):
    """Process design and stress test from posterior-mean barriers (flagged)."""
    from dataclasses import replace
    from .process import ProcessParams, design, stress_test
    record = {"candidate_id": cid, "name": belief.candidates[cid].get("name"), "mass_amu": mass_amu,
              "sites": belief.posterior_means(cid)}
    params = ProcessParams(use_precomplex=False)
    d = design(record, params, n_draws=150)
    rec = d["recommended"]
    stress = stress_test(record, replace(params, T_K=rec["T_K"], dose_s=rec["dose_s"]), rec["redose_every"])
    return {"basis": "posterior-mean barriers; unobserved sites filled from the hierarchy",
            "sites": record["sites"], "recommended": rec, "measure_first": d["measure_first"][:4],
            "dose_windows": d["dose_windows"], "stress_test": stress}


def markdown(result):
    names = result.get("names", {})
    n = lambda c: names.get(c, c)
    lines = ["# WARDEN evidence-acquisition report", "",
             "**Screening only. Scores are dose-window feasibility margins (decades, best T); "
             "negative means infeasible by that much.**", "",
             f"Pick: **{n(result['pick'])}** | P(correct selection) {result['pcs']:.2f} | "
             f"spent {result['spent']:.1f} / {result['budget']:.1f} cost units | stop: {result['stop']}", "",
             "## Candidate scores (posterior mean margin)", "", "| Candidate | Margin (decades) | P(top two) |", "|---|---:|---:|"]
    for c, v in sorted(result["scores"].items(), key=lambda x: -x[1]):
        lines.append(f"| {n(c)} | {v:.2f} | {result['contention'].get(c, 0):.2f} |")
    lines += ["", "## Decisions", "", "| # | Action | Value/cost | P(success) | Result | Why |", "|---:|---|---:|---:|---|---|"]
    for t in result["trace"]:
        if "action" not in t:
            continue
        a = t["action"]
        what = (f"add {t.get('added')}" if a["kind"] == "add_molecule"
                else f"{n(a['candidate'])} @ {a['site']} / {a['pathway']} / {a['method']} (orient {a['orientation']})")
        res = t.get("result") or {}
        outcome = (f"Ea {res['Ea_eV']:.2f} eV" if res.get("status") == "valid"
                   else res.get("reason") or res.get("status") or "added")
        score = a["value"] * a["p_success"] / a["cost"]
        lines.append(f"| {t['step']} | {what} | {score:.3g} | {a['p_success']:.2f} | {outcome} | {a['why']} |")
    lines += ["", "## Method reliability (paired with reference)", ""]
    if result["methods"]:
        lines += ["| Method | Quantity | Surface | Template | Pairs | Bias estimate (eV) | Bias sd (eV) |", "|---|---|---|---|---:|---:|---:|"]
        for m in result["methods"]:
            lines.append(f"| {m['method']} | {m['quantity']} | {m['material']} | {m['template']} | {m['n_pairs']} | "
                         f"{m['bias_estimate_eV']:+.2f} | {m['bias_sd_eV']:.2f} |")
    else:
        lines.append("No paired reference results yet; cheap-method bias is carried as prior uncertainty.")
    lines += ["", "## Failure model", "", "| Method | Template | Pathway | Site type | OK | Failed |", "|---|---|---|---|---:|---:|"]
    for f in result["failures"]:
        lines.append(f"| {f['method']} | {f['template']} | {f['pathway']} | {f['site_type']} | {f['successes']} | {f['failures']} |")
    rob = result.get("robustness")
    if rob:
        lines += ["", "## Site robustness", "",
                  f"Bootstrap over amorphous sites: the pick stays top in {rob['fraction_pick_stays_top']:.0%} of {rob['n_bootstrap']} resamples."]
    pd = result.get("process")
    if pd:
        rec = pd["recommended"]
        lines += ["", f"## Process prediction for {n(result['pick'])}", "", f"Basis: {pd['basis']}.", "",
                  f"Recipe: {rec['T_K']:.0f} K, {rec['dose_s']} s dose, redose every {rec['redose_every']} -> "
                  f"P(S >= target) {rec['P_success']:.2f}, median S {rec['S_median'] if rec['S_median'] is None else round(rec['S_median'], 3)}", "",
                  "| Stress | Values that still meet target | Fails somewhere |", "|---|---|---|"]
        for name, sres in pd["stress_test"].items():
            lines.append(f"| {name} | {sres['passing_values']} | {sres['fails_somewhere']} |")
    if result.get("next"):
        lines += ["", "## Next recommended actions", ""]
        for a in result["next"]:
            lines.append(f"- {a}")
    lines += ["", result.get("note", "")]
    return "\n".join(lines) + "\n"
