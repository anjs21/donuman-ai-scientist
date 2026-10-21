"""Mechanistic multi-cycle AS-ALD model with uncertainty-aware schedule design.

Per surface, each reactive-site class i (population weight w_i, barrier Ea_i)
carries a capped fraction theta_i and an irreversibly nucleated fraction nu_i.
One ALD cycle:

    redose (every N cycles): theta_i += avail_i * P_cap(Ea_i, T, P, t_dose)
                             avail_i = max(0, theta_max*(1 - nu_i) - theta_i)
    precursor:  nu_i += (1 - theta_i - nu_i) * p_nuc * (1 - theta)^m
    growth:     thickness += GPC * sum_i w_i nu_i
    co-reactant: theta_i *= 1 - k_deg

Nuclei cannot be re-passivated, so redosing has diminishing returns, and caps
lost to the co-reactant expose sites between doses. The exponent m is steric
shielding of an open site by neighbouring caps.

Instead of one deterministic selectivity, inputs that are not computed (p_nuc,
m, theta_max, k_deg) are sampled from stated priors, and barriers are perturbed
by method error. The output is P(S >= target at target thickness) for each
schedule, the most robust schedule, and a ranking of which uncertain input
controls success -- i.e. what to measure first.
"""
from dataclasses import dataclass, asdict, replace
import numpy as np
from scipy.stats import spearmanr
from scipy.optimize import minimize
from .kinetics import H_EVS, KB_EV, precomplex_fraction, rate_constant
from .common import clean


@dataclass
class ProcessParams:
    T_K: float = 473.0            # substrate temperature during inhibitor dose
    pressure_Pa: float = 133.0    # inhibitor partial pressure (~1 Torr)
    dose_s: float = 5.0
    gpc_A: float = 1.2            # SiO2 growth per cycle on open oxide
    target_thickness_nm: float = 10.0
    target_selectivity: float = 0.90
    max_cycles: int = 300
    theta_max: float = 0.99       # steric saturation of the cap layer
    p_nuc: float = 1.0            # nucleation probability at an accessible open site
    shielding: float = 1.0        # m in (1 - theta)^m
    k_deg: float = 0.03           # NGS cap fraction lost per cycle
    k_deg_GS: float = 0.03        # GS cap loss (ester-type caps can self-clean)
    cycle_time_s: float = 20.0
    use_precomplex: bool = True   # Langmuir pre-complex occupancy from binding energy
    min_practical_dose_s: float = 0.05   # shortest dose a production tool can meter
    max_practical_dose_s: float = 30.0


PRIORS = {"p_nuc": ("uniform", 0.2, 1.0), "shielding": ("uniform", 0.0, 3.0),
          "theta_max": ("uniform", 0.95, 0.999), "k_deg": ("loguniform", 0.005, 0.1),
          "k_deg_GS": ("loguniform", 0.005, 0.5)}
SIGMA = {"common_eV": 0.20, "relative_eV": 0.15, "site_eV": 0.10}


def site_classes(kinetics, material):
    """(weights, Ea, binding, mass) for one surface from a kinetics record."""
    rows = [s for s in kinetics["sites"] if s["material"] == material]
    if not rows or any(s["status"] != "valid" for s in rows):
        # Missing barriers are unknown, not slow: refuse rather than guess.
        raise ValueError(f"{kinetics.get('name')}: incomplete barriers on {material}")
    w = np.array([s["population"] for s in rows], float)
    return {"w": w / w.sum(), "Ea": np.array([s["Ea_eV"] for s in rows], float),
            "binding": np.array([s["precomplex_binding_eV"] if s.get("precomplex_binding_eV") is not None
                                 else 0. for s in rows], float),
            "labels": [s["label"] for s in rows], "mass": kinetics.get("mass_amu", 100.)}


def rates(classes, p, ea=None):
    """Capping rate (1/s) per site class; `ea` may carry a leading draws axis."""
    ea = classes["Ea"] if ea is None else ea
    occ = (np.array([precomplex_fraction(b, classes["mass"], p.T_K, p.pressure_Pa)
                     for b in classes["binding"]]) if p.use_precomplex else 1.)
    kt = KB_EV * p.T_K
    return occ * kt / H_EVS * np.exp(-ea / kt)


def capture(classes, p, ea=None):
    """Per-dose capture probability."""
    return -np.expm1(-rates(classes, p, ea) * p.dose_s)


def simulate(gs, ngs, p, redose_every, draws=None):
    """Vectorized over Monte Carlo draws. `draws` maps parameter -> array."""
    draws = draws or {}
    n = len(next(iter(draws.values()))) if draws else 1
    get = lambda k: np.broadcast_to(np.asarray(draws.get(k, getattr(p, k)), float), (n,))
    theta_max, p_nuc, m = (get(k)[:, None] for k in ("theta_max", "p_nuc", "shielding"))
    out = {}
    for name, cls in (("GS", gs), ("NGS", ngs)):
        k_deg = get("k_deg_GS" if name == "GS" else "k_deg")[:, None]
        shift = draws.get("shift_" + name, np.zeros(n))
        jitter = draws.get("jitter_" + name, np.zeros((n, len(cls["w"]))))
        pc = capture(cls, p, cls["Ea"][None, :] + shift[:, None] + jitter)
        w = cls["w"][None, :]
        theta = np.zeros((n, len(cls["w"]))); nu = np.zeros_like(theta)
        thk = np.zeros((n, p.max_cycles + 1))
        for c in range(p.max_cycles):
            if c % redose_every == 0:
                theta += np.maximum(0., theta_max * (1 - nu) - theta) * pc
            covered = (w * theta).sum(axis=1, keepdims=True)
            access = np.clip(1 - covered, 0, 1) ** m
            nu += np.clip(1 - theta - nu, 0, 1) * p_nuc * access
            thk[:, c + 1] = thk[:, c] + p.gpc_A * (w * nu).sum(axis=1)
            theta *= 1 - k_deg
        out[name] = thk
    return out


def at_target(thk_gs, thk_ngs, p):
    target = p.target_thickness_nm * 10
    n = thk_gs.shape[0]
    S = np.full(n, np.nan); cycles = np.full(n, np.nan); ngs = np.full(n, np.nan)
    for d in range(n):
        hit = np.flatnonzero(thk_gs[d] >= target)
        if not len(hit):
            continue
        c = hit[0]
        f = (target - thk_gs[d, c - 1]) / (thk_gs[d, c] - thk_gs[d, c - 1]) if c else 1.
        cycles[d] = c - 1 + f
        ngs[d] = thk_ngs[d, c - 1] + f * (thk_ngs[d, c] - thk_ngs[d, c - 1]) if c else thk_ngs[d, c]
        S[d] = (target - ngs[d]) / (target + ngs[d])
    return S, cycles, ngs


def sample(n, priors=None, sigma=None, n_sites=None, seed=0):
    rng = np.random.default_rng(seed)
    priors = {**PRIORS, **(priors or {})}; sigma = {**SIGMA, **(sigma or {})}
    draws = {}
    for k, (kind, lo, hi) in priors.items():
        draws[k] = (rng.uniform(lo, hi, n) if kind == "uniform"
                    else np.exp(rng.uniform(np.log(lo), np.log(hi), n)))
    common = rng.normal(0, sigma["common_eV"], n)
    for name in ("GS", "NGS"):
        draws["shift_" + name] = common + rng.normal(0, sigma["relative_eV"], n)
        draws["jitter_" + name] = rng.normal(0, sigma["site_eV"], (n, (n_sites or {}).get(name, 1)))
    return draws


def evaluate_schedule(gs, ngs, p, redose_every, draws):
    res = simulate(gs, ngs, p, redose_every, draws)
    S, cycles, ngs_thk = at_target(res["GS"], res["NGS"], p)
    ok = np.nan_to_num(S, nan=-1) >= p.target_selectivity
    redoses = np.ceil(np.nan_to_num(cycles, nan=p.max_cycles) / redose_every)
    time = np.nan_to_num(cycles, nan=np.inf) * p.cycle_time_s + redoses * p.dose_s
    q = lambda a, x: float(np.nanpercentile(a, x)) if np.isfinite(a).any() else None
    return {"redose_every": redose_every, "dose_s": p.dose_s,
            "P_success": float(ok.mean()), "S_median": q(S, 50), "S_p10": q(S, 10), "S_p90": q(S, 90),
            "cycles_median": q(cycles, 50), "time_median_s": q(np.where(np.isfinite(time), time, np.nan), 50),
            "reached_fraction": float(np.isfinite(S).mean()), "_S": S}


def nominal(gs, ngs, p, redose_every):
    res = simulate(gs, ngs, p, redose_every)
    S, cycles, ngs_thk = at_target(res["GS"], res["NGS"], p)
    return {"redose_every": redose_every, "S_at_target": float(S[0]) if np.isfinite(S[0]) else None,
            "cycles_to_target": float(cycles[0]) if np.isfinite(cycles[0]) else None,
            "NGS_thickness_nm": float(ngs_thk[0] / 10) if np.isfinite(ngs_thk[0]) else None}


def dose_window(gs, ngs, p, capture_NGS=0.99, capture_GS=0.01):
    """Dose times that cap >= capture_NGS of every NGS class and <= capture_GS of
    every GS class, at nominal barriers. Empty (None) if no such window exists."""
    lo = float(-np.log1p(-capture_NGS) / rates(ngs, p).min())
    hi = float(-np.log1p(-capture_GS) / rates(gs, p).max())
    kt = KB_EV * p.T_K
    prefactor = lambda cls: (rates(cls, p) * np.exp(cls["Ea"] / kt))
    t_lo, t_hi = p.min_practical_dose_s, p.max_practical_dose_s
    return {"T_K": p.T_K, "pressure_Pa": p.pressure_Pa, "min_dose_s": lo, "max_dose_s": hi,
            "exists": hi > lo, "width_decades": float(np.log10(hi / lo)),
            "practical": bool(hi > lo and hi >= t_lo and lo <= t_hi),
            # Barriers that would place the window inside the practical dose range.
            "target_Ea_NGS_max_eV": float(kt * np.log(prefactor(ngs).min() * t_hi / -np.log1p(-capture_NGS))),
            "target_Ea_GS_min_eV": float(kt * np.log(prefactor(gs).max() * t_lo / -np.log1p(-capture_GS)))}


def design(kinetics, params=None, temperatures=(373., 423., 473., 523.), doses=(1e-4, 1e-3, .01, .05, .2, 1., 5., 20.),
           intervals=range(1, 11), n_draws=300, p_required=0.8, priors=None, sigma=None, seed=0):
    """Robust process window for one inhibitor; common random numbers across schedules."""
    p = params or ProcessParams()
    gs, ngs = site_classes(kinetics, "SiO2"), site_classes(kinetics, "SiNx")
    sigma = {**SIGMA, **(sigma or {})}
    if kinetics.get("method_disagreement_eV") is not None:
        sigma["relative_eV"] = max(sigma["relative_eV"], kinetics["method_disagreement_eV"])
    draws = sample(n_draws, priors, sigma, {"GS": len(gs["w"]), "NGS": len(ngs["w"])}, seed)
    table, windows = [], []
    for T in temperatures:
        windows.append(dose_window(gs, ngs, replace(p, T_K=float(T))))
        for dose in doses:
            q = replace(p, T_K=float(T), dose_s=float(dose))
            for every in intervals:
                row = evaluate_schedule(gs, ngs, q, every, draws)
                row["T_K"] = float(T)
                row["nominal"] = nominal(gs, ngs, q, every)
                table.append(row)
    for r in table:
        r["practical_dose"] = p.min_practical_dose_s <= r["dose_s"] <= p.max_practical_dose_s
    usable = [r for r in table if r["practical_dose"]] or table
    robust = [r for r in usable if r["P_success"] >= p_required]
    best = (min(robust, key=lambda r: r["time_median_s"] or np.inf) if robust
            else max(usable, key=lambda r: (r["P_success"], r["S_median"] or -1)))
    sens = []
    S = best["_S"]
    for k in [*{**PRIORS, **(priors or {})}, "shift_GS", "shift_NGS"]:
        x = draws[k]
        mask = np.isfinite(S)
        rho = spearmanr(x[mask], S[mask]).statistic if mask.sum() > 3 and np.ptp(x[mask]) > 0 else np.nan
        sens.append({"input": k, "spearman_rho": None if np.isnan(rho) else float(rho)})
    sens.sort(key=lambda r: -abs(r["spearman_rho"] or 0))
    for r in table:
        r.pop("_S")
    return clean({"candidate_id": kinetics.get("candidate_id"), "name": kinetics.get("name"),
                  "params": asdict(p), "priors": {**PRIORS, **(priors or {})}, "sigma": sigma,
                  "p_required": p_required, "dose_windows": windows, "schedules": table,
                  "recommended": best, "robust": bool(robust), "measure_first": sens,
                  "limitation": "Process surrogate: priors are stated assumptions, not measurements. "
                                "Calibrate with calibrate() before quoting absolute selectivity."})


def degradation_per_cycle(Ea_eV, T_K, exposure_s, occupancy=1.0):
    """Cap loss per co-reactant step from a computed degradation barrier."""
    return 1 - float(np.exp(-rate_constant(Ea_eV, T_K, occupancy) * exposure_s))


def calibrate(kinetics, observed, params=None, redose_every=1, fit=("p_nuc", "shielding", "k_deg", "theta_max")):
    """Fit uncomputed process inputs to measured thickness-vs-cycle data.

    observed: [{"cycle": int, "thk_GS_A": float|None, "thk_NGS_A": float|None}, ...]
    """
    p = params or ProcessParams()
    gs, ngs = site_classes(kinetics, "SiO2"), site_classes(kinetics, "SiNx")
    bounds = {"p_nuc": (1e-3, 1.), "shielding": (0., 6.), "k_deg": (1e-4, .5),
              "k_deg_GS": (1e-4, .9), "theta_max": (.5, .9999)}
    max_cycle = max(o["cycle"] for o in observed)
    p = replace(p, max_cycles=max(max_cycle, 1))

    def unpack(x):
        vals = {}
        for k, v in zip(fit, x):
            lo, hi = bounds[k]
            vals[k] = lo + (hi - lo) / (1 + np.exp(-v))
        return replace(p, **vals)

    def loss(x):
        res = simulate(gs, ngs, unpack(x), redose_every)
        err = 0.
        for o in observed:
            for key, name in (("thk_GS_A", "GS"), ("thk_NGS_A", "NGS")):
                if o.get(key) is not None:
                    err += (res[name][0, o["cycle"]] - o[key]) ** 2
        return err

    x0 = np.zeros(len(fit))
    fitres = minimize(loss, x0, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-6})
    fitted = unpack(fitres.x)
    return {"params": asdict(fitted), "rmse_A": float(np.sqrt(fitres.fun / max(1, sum(
        (o.get("thk_GS_A") is not None) + (o.get("thk_NGS_A") is not None) for o in observed)))),
        "converged": bool(fitres.success), "fitted": list(fit)}


STRESS = {
    "cap_loss_per_cycle": ("k_deg", [0.005, 0.01, 0.03, 0.1, 0.2]),
    "incomplete_repassivation_theta_max": ("theta_max", [0.9, 0.95, 0.99, 0.999]),
    "slower_repassivation_NGS_shift_eV": ("shift_NGS", [0.0, 0.1, 0.2, 0.3, 0.5]),
    "oxide_inhibition_GS_shift_eV": ("shift_GS", [0.0, -0.1, -0.2, -0.3, -0.4]),
    "oxide_cap_persistence_k_deg_GS": ("k_deg_GS", [0.5, 0.1, 0.03, 0.01]),
}


def stress_test(kinetics, params, redose_every):
    """One-at-a-time sweeps around a recipe: how S at target responds to cap
    loss, incomplete or slower re-passivation, and unwanted oxide inhibition."""
    gs, ngs = site_classes(kinetics, "SiO2"), site_classes(kinetics, "SiNx")
    out = {}
    for name, (key, values) in STRESS.items():
        rows = []
        for v in values:
            draws = {key: np.array([v], float)}
            res = simulate(gs, ngs, params, redose_every, draws)
            S, cycles, ngs_thk = at_target(res["GS"], res["NGS"], params)
            rows.append({"value": v, "S_at_target": float(S[0]) if np.isfinite(S[0]) else None,
                         "cycles_to_target": float(cycles[0]) if np.isfinite(cycles[0]) else None,
                         "meets_target": bool(np.isfinite(S[0]) and S[0] >= params.target_selectivity)})
        passing = [r["value"] for r in rows if r["meets_target"]]
        out[name] = {"parameter": key, "sweep": rows,
                     "passing_values": passing, "fails_somewhere": len(passing) < len(rows)}
    return clean(out)
