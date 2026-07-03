"""AS-ALD Co-Scientist — on-device LLM assistant (Streamlit prototype).

Two capabilities on top of the existing pipeline, driven by a local Ollama
model (default qwen2.5:7b-instruct), so unpublished energetics never leave the
machine:

  Job 2  Explain     read a pipeline report and stream a scientific discussion.
  Job 3  Tune & re-run
                      diagnose why a run under-performed, propose selection-
                      threshold changes as structured JSON, apply them, and
                      re-run selection instantly from the cached energetics.
                      A separate (gated) button launches a full pipeline re-run
                      as a background subprocess for the expensive MD/MLIP path.

Run with:  streamlit run app.py
The model only ever runs *between* pipeline runs, so it doesn't contend with
MACE for the 8 GB of VRAM.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import streamlit as st

from agent import agent_loop
from agent import llm
from agent import report_tools as rt

st.set_page_config(page_title="AS-ALD Co-Scientist", page_icon="🧪",
                   layout="wide")

REPO = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# Full-pipeline subprocess helpers (non-blocking background run)
# --------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    """True if the process is still executing (Linux).

    A finished subprocess child we spawned becomes a *zombie* until reaped, and
    ``os.kill(pid, 0)`` still succeeds on a zombie -- so it would look "running"
    forever. Read /proc state instead and treat 'Z' (defunct) as finished.
    """
    try:
        with open(f"/proc/{pid}/stat") as fh:
            state = fh.read().rsplit(")", 1)[1].split()[0]
        return state != "Z"
    except (FileNotFoundError, IndexError):
        return False
    except OSError:
        return False


def _read_tail(logpath: str, n: int = 4000) -> str:
    try:
        with open(logpath) as fh:
            return fh.read()[-n:]
    except OSError:
        return ""


def _launch_pipeline(mode: str, surface_glob: str | None = None,
                     reagents: list[str] | None = None,
                     materials: list[str] | None = None,
                     auto: bool = False, explain_goal: str | None = None,
                     model: str | None = None,
                     n_workers: int = 1) -> None:
    """Launch run_pipeline.py in the background.

    When `surface_glob` is given it is passed as --use-existing, so Phase 1
    (melt-quench MD) is skipped and only the adsorption energetics (Phase 2)
    are recomputed on the existing slabs. Without it, surfaces are rebuilt
    from scratch. `materials` restricts the substrates; `reagents` restricts
    the molecules screened.

    If `auto` is set, the status panel waits for completion and then streams a
    summary (`explain_goal`, using `model`) of the newly written report instead
    of showing a manual refresh button.
    """
    logdir = os.path.join(REPO, "runs")
    os.makedirs(logdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    logpath = os.path.join(logdir, f"pipeline_{mode}_{stamp}.log")
    outdir = os.path.join(logdir, f"run_{mode}_{stamp}")
    os.makedirs(outdir, exist_ok=True)
    cmd = [sys.executable, "run_pipeline.py", "--mode", mode,
           "--output-dir", outdir, "--report", os.path.join(outdir, "report")]
    if materials:
        cmd += ["--materials", *materials]
    if surface_glob:
        cmd += ["--use-existing", surface_glob]
    if reagents:
        cmd += ["--reagents", *reagents]
    if n_workers > 1:
        cmd += ["--n-workers", str(n_workers)]
    logf = open(logpath, "w")
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=logf,
                            stderr=subprocess.STDOUT)
    st.session_state["pipeline"] = {
        "pid": proc.pid, "log": logpath, "mode": mode, "started": stamp,
        "cmd": " ".join(cmd), "outdir": outdir,
        "report_json": os.path.join(outdir, "report.json"),
        "t0": time.time(), "auto": auto, "explain_goal": explain_goal,
        "model": model, "explanation": None,
    }
    st.success(f"Launched (pid {proc.pid}):\n\n`{' '.join(cmd[2:])}`")


def _show_pipeline_status() -> None:
    info = st.session_state.get("pipeline")
    if not info:
        return
    st.markdown("---")
    st.markdown(f"**Background pipeline** — mode `{info['mode']}`, "
                f"pid {info['pid']}, started {info['started']}")
    alive = _pid_alive(info["pid"])
    tail = _read_tail(info["log"])
    if info.get("auto"):
        _auto_watch(info, alive, tail)
    else:
        st.write("Status:", "🟢 running" if alive else "⚪ finished")
        if st.button("Refresh status / tail log", key="refresh_pipe"):
            st.rerun()
        st.code(tail or "(no output yet)", language="text")


def _auto_watch(info: dict, alive: bool, tail: str) -> None:
    """Wait-and-summarize path: poll until the job ends, then load the new
    report and stream a summary once. No manual refresh needed."""
    elapsed = int(time.time() - info.get("t0", time.time()))
    if alive:
        st.info(f"⏳ Waiting for the job to finish… {elapsed}s elapsed. "
                "Leave this tab open — results appear here automatically.")
        with st.expander("Live log tail"):
            st.code(tail or "(no output yet)", language="text")
        time.sleep(4)
        st.rerun()
        return

    rp = info.get("report_json")
    if not (rp and os.path.exists(rp)):
        st.error(f"Job ended after ~{elapsed}s but wrote no report — see the log:")
        st.code(tail or "(no output)", language="text")
        return
    st.success(f"✅ Job finished in ~{elapsed}s.")
    try:
        new_report = rt.load_report(rp)
    except (OSError, json.JSONDecodeError) as e:
        st.error(f"Could not read {rp}: {e}")
        return

    nrec = new_report.get("selection", {}).get("recommendation", {})
    m1, m2 = st.columns(2)
    m1.metric("Recommended inhibitor",
              nrec.get("inhibitor") or "— none met criteria")
    m2.metric("Recommended precursor", nrec.get("precursor") or "—")
    st.caption(f"New report: `{os.path.relpath(rp, REPO)}` "
               "(also selectable in the sidebar).")

    goal, mdl = info.get("explain_goal"), info.get("model")
    if not (goal and mdl):
        return
    st.subheader("Automatic summary")
    if info.get("explanation"):
        st.markdown(info["explanation"])           # already generated; re-render
    elif not llm.is_up():
        st.warning("Model offline — start Ollama to get the auto-summary.")
    else:
        try:
            info["explanation"] = st.write_stream(
                llm.chat_stream(mdl, _explain_messages(new_report, goal)))
        except llm.OllamaError as e:
            st.error(str(e))


# --------------------------------------------------------------------------
# Explanation prompt (shared by the Explain tab and the auto-summary)
# --------------------------------------------------------------------------

EXPLAIN_SYSTEM = (
    "You are an expert computational surface chemist assisting with "
    "area-selective ALD (AS-ALD). The goal process passivates SiNx "
    "(non-growth surface, -NH2/-NH sites) so SiOx grows selectively on SiO2 "
    "(growth surface, -OH sites); target is 90% selectivity at 10 nm oxide. "
    "Adsorption/reaction energies dE are in eV, negative = favourable; "
    "contrast = dE_GS - dE_NGS, large positive = selective blocker. Ground "
    "every claim in the numbers provided. If the run used a placeholder "
    "calculator or a coarse (test) mode, say so and temper conclusions "
    "accordingly. Do not invent data."
)

EXPLAIN_GOALS = {
    "Discussion section (paper-style)":
        "Write a rigorous Discussion section for a methods paper: interpret the "
        "selectivity contrast per candidate, connect the adsorption energetics "
        "to the ASD mechanism, and state the principal limitations of this run.",
    "Plain-language summary":
        "Explain in plain language what this run found and what it means for "
        "designing a selective ALD scheme.",
    "Next experiments to run":
        "Propose a concrete, prioritized set of next computational experiments "
        "(new candidates, parameters, or validation) that would most improve "
        "confidence in the recommendation.",
}


def _explain_messages(report: dict, goal_key: str) -> list[dict]:
    digest = rt.report_digest(report)
    return [
        {"role": "system", "content": EXPLAIN_SYSTEM},
        {"role": "user",
         "content": f"{EXPLAIN_GOALS[goal_key]}\n\nRUN REPORT\n----------\n{digest}"},
    ]


# --------------------------------------------------------------------------
# Sidebar: server status, model, report selection
# --------------------------------------------------------------------------

st.sidebar.title("🧪 AS-ALD Co-Scientist")
st.sidebar.caption("On-device assistant")

up = llm.is_up()
if up:
    st.sidebar.success(f"Ollama online · {llm.OLLAMA_HOST}")
    models = llm.list_models()
else:
    st.sidebar.error(f"Ollama not reachable at {llm.OLLAMA_HOST}")
    st.sidebar.code("ollama serve", language="bash")
    models = []

default_model = llm.DEFAULT_MODEL
if models:
    idx = models.index(default_model) if default_model in models else 0
    model = st.sidebar.selectbox("Model", models, index=idx)
else:
    model = st.sidebar.text_input("Model", value=default_model)

reports = rt.find_reports(REPO)
if not reports:
    st.sidebar.warning("No report*.json found under the repo.")
    st.stop()
rel = [os.path.relpath(p, REPO) for p in reports]
report_path = reports[rel.index(st.sidebar.selectbox("Report", rel))]

try:
    report = rt.load_report(report_path)
except (json.JSONDecodeError, OSError) as e:
    st.error(f"Could not read {report_path}: {e}")
    st.stop()

if report.get("placeholder_calc"):
    st.sidebar.warning("This run used the Lennard-Jones placeholder — energies "
                       "are not physical.")


# --------------------------------------------------------------------------
# Header + at-a-glance recommendation
# --------------------------------------------------------------------------

rec = report.get("selection", {}).get("recommendation", {})
c1, c2, c3 = st.columns(3)
c1.metric("Recommended inhibitor", rec.get("inhibitor") or "— none met criteria")
c2.metric("Recommended precursor", rec.get("precursor") or "—")
n_meets = sum(1 for s in report.get("selectivity", {}).values()
              if s.get("meets_target"))
c3.metric("Candidates meeting target", n_meets)

tab_submit, tab_explain, tab_agent = st.tabs(
    ["🚀 Submit job", "📖 Explain", "🤖 Agent"])


# --------------------------------------------------------------------------
# Submit job — choose substrate(s), structure, and inhibitors to screen
# --------------------------------------------------------------------------

with tab_submit:
    st.subheader("Submit a screening job")
    st.caption("Pick the substrate(s), the structure to screen on, and which "
               "inhibitors to test. Runs run_pipeline.py in the background; "
               "MACE uses the GPU, so avoid running the model meanwhile.")

    j_materials = st.multiselect(
        "Substrate(s)", ["SiO2", "SiNx"], default=["SiO2", "SiNx"],
        help="SiO2 = growth surface (grow SiOx here); SiNx = non-growth "
             "surface (passivate). Both are needed to score selectivity "
             "contrast.")

    surf_sets = rt.find_surface_sets()
    rel_to_dir = {os.path.relpath(d, REPO): d for d in surf_sets}
    source = st.radio(
        "Structure", ["Use existing slabs (fast — skips melt-quench MD)",
                      "Build fresh from bulk (runs melt-quench MD — slow)"],
        key="submit_source")
    use_existing = source.startswith("Use existing")

    surface_glob, mode, set_covers = None, "test", []
    if use_existing:
        if not surf_sets:
            st.warning("No reusable surface slabs on disk — build fresh instead.")
        else:
            chosen = st.selectbox(
                "Structure set", list(rel_to_dir),
                format_func=lambda r: f"{r}/  "
                f"({', '.join(rt.set_materials(surf_sets[rel_to_dir[r]])) or 'no slabs'})")
            files = surf_sets[rel_to_dir[chosen]]
            set_covers = rt.set_materials(files)
            surface_glob = os.path.join(rel_to_dir[chosen], "*.*")
    else:
        mode = st.selectbox(
            "MD protocol", ["test", "fast", "full"], key="submit_mode",
            help="test ≈ minutes (code check), fast ≈ 10–15 min, "
                 "full ≈ hours per material.")

    cats = rt.reagents_by_category()
    j_inhibitors = st.multiselect(
        "Inhibitors to test", cats.get("inhibitor", []),
        default=cats.get("inhibitor", []),
        help="The candidates you want screened for selective blocking.")
    j_precursors = st.multiselect(
        "Precursors (for the SiOx film / recommendation)",
        cats.get("precursor", []), default=cats.get("precursor", []),
        help="Needed for a precursor recommendation and the selectivity curve.")

    j_reagents = j_inhibitors + j_precursors

    j_n_workers = st.slider(
        "⚡ Parallel site workers (Phase 2 energetics)", 1, 4, 2,
        key="submit_n_workers",
        help="Number of concurrent local relaxations during the energetics screen. "
             "Higher = faster Phase 2, but uses more GPU memory. 2 is a safe default.")

    st.markdown("**When it finishes**")
    j_auto = st.checkbox(
        "Wait and summarize automatically (no manual refresh)", value=True,
        key="submit_auto",
        help="The app watches the job and, when it ends, loads the new report "
             "and writes a summary with the local model. Leave the tab open.")
    j_goal = None
    if j_auto:
        j_goal = st.selectbox("Summary style", list(EXPLAIN_GOALS),
                              key="submit_goal",
                              disabled=not up,
                              help=None if up else "Model offline — the job will "
                              "still run; start Ollama to get the summary.")

    problems = []
    if not j_materials:
        problems.append("choose at least one substrate")
    if not j_inhibitors:
        problems.append("choose at least one inhibitor")
    if use_existing:
        if not surf_sets:
            problems.append("no existing slabs — build fresh")
        else:
            missing = [m for m in j_materials if m not in set_covers]
            if missing:
                problems.append("chosen structure set has no slab for "
                                + ", ".join(missing))

    preview = ["run_pipeline.py", "--mode", mode, "--materials", *j_materials]
    if surface_glob:
        preview += ["--use-existing", os.path.relpath(surface_glob, REPO)]
    if j_reagents:
        preview += ["--reagents", *j_reagents]
    st.caption("Command that will run:")
    st.code(" ".join(preview), language="bash")

    if problems:
        st.info("Before submitting: " + "; ".join(problems) + ".")
    if st.button("Submit job", type="primary", disabled=bool(problems),
                 key="submit_job"):
        _launch_pipeline(mode, surface_glob=surface_glob,
                         reagents=j_reagents or None,
                         materials=j_materials or None,
                         auto=j_auto, explain_goal=j_goal, model=model,
                         n_workers=j_n_workers)


# --------------------------------------------------------------------------
# Job 2 — Explain
# --------------------------------------------------------------------------

with tab_explain:
    st.subheader("Scientific discussion of this run")
    st.caption("The model reads the report digest below and writes a discussion "
               "section. Streams live; nothing leaves the machine.")

    with st.expander("Report digest sent to the model"):
        st.code(rt.report_digest(report), language="text")

    audience = st.radio("Write for", list(EXPLAIN_GOALS), horizontal=True)

    if st.button("Generate", type="primary", disabled=not up, key="gen_explain"):
        try:
            st.write_stream(
                llm.chat_stream(model, _explain_messages(report, audience)))
        except llm.OllamaError as e:
            st.error(str(e))


# --------------------------------------------------------------------------
# Job 3 — Tune & re-run
# --------------------------------------------------------------------------

# with tab_tune:
#     st.subheader("Diagnose, tune thresholds, re-run selection")
#     st.caption("Threshold changes only touch the selection + selectivity "
#                "phases, so re-running is instant — the MD/MLIP energetics are "
#                "reused from this report.")

#     cfg = rt.current_config()

#     # -- 1. Ask the model for a structured suggestion --
#     st.markdown("**1 · Ask the model to diagnose this run**")
#     if st.button("Diagnose & suggest changes", disabled=not up, key="diagnose"):
#         digest = rt.report_digest(report)
#         system = (
#             "You tune the selection thresholds of an AS-ALD screening pipeline. "
#             "The pipeline already computed adsorption energies; you only adjust "
#             "the decision thresholds, you do NOT invent energies. Goal: find an "
#             "inhibitor that binds SiNx (NGS) strongly and SiO2 (GS) weakly "
#             "(large positive contrast = dE_GS - dE_NGS). If no inhibitor was "
#             "accepted, decide whether a threshold is too strict given the "
#             "observed numbers, and propose the smallest change that would admit "
#             "a genuinely selective candidate WITHOUT accepting a non-selective "
#             "one. Only propose changes you can justify from the numbers. "
#             "Thresholds (eV unless noted): "
#             "bind_threshold_eV (inhibitor must be at least this favourable on "
#             "NGS), spare_threshold_eV (should be no more favourable than this on "
#             "GS), min_contrast_eV (minimum GS-NGS contrast), "
#             "precursor_threshold_eV, contrast_weight, volatility_bonus, "
#             "volatility_penalty, strain_penalty."
#         )
#         user = (f"Current thresholds: {json.dumps(cfg)}\n\n"
#                 f"RUN REPORT\n----------\n{digest}\n\n"
#                 "Return a diagnosis, a list of threshold changes (key, to, "
#                 "reason), and the expected effect.")
#         try:
#             with st.spinner("Model is analysing the run…"):
#                 suggestion = llm.chat_json(
#                     model,
#                     [{"role": "system", "content": system},
#                      {"role": "user", "content": user}],
#                     rt.SUGGESTION_SCHEMA)
#             st.session_state["suggestion"] = suggestion
#         except llm.OllamaError as e:
#             st.error(str(e))

#     sugg = st.session_state.get("suggestion")
#     if sugg:
#         st.info(f"**Diagnosis.** {sugg.get('diagnosis','')}")
#         st.write(f"*Expected effect:* {sugg.get('expected_effect','')}")
#         for ch in sugg.get("changes", []):
#             k = ch.get("key")
#             if k in cfg:
#                 st.write(f"- `{k}`: {cfg[k]} → **{ch.get('to')}** — "
#                          f"{ch.get('reason','')}")

#     # -- 2. Editable thresholds (seeded by the suggestion) --
#     st.markdown("**2 · Review / edit thresholds**")
#     proposed = {c["key"]: c["to"] for c in (sugg or {}).get("changes", [])
#                 if c.get("key") in cfg}
#     overrides = {}
#     cols = st.columns(2)
#     for i, (k, v) in enumerate(cfg.items()):
#         lo, hi = rt.CONFIG_BOUNDS.get(k, (v - 1, v + 1))
#         start = float(proposed.get(k, v))
#         start = max(lo, min(hi, start))
#         step = 0.05 if hi - lo <= 3 else 0.1
#         overrides[k] = cols[i % 2].slider(
#             k, float(lo), float(hi), start, step=step,
#             help="⟵ model-proposed value" if k in proposed else None)

#     changed_keys = {k: ov for k, ov in overrides.items()
#                     if abs(ov - cfg[k]) > 1e-9}

#     # -- 3. Instant re-selection --
#     st.markdown("**3 · Re-run selection (instant, no GPU)**")
#     if st.button("Re-run selection with these thresholds", type="primary",
#                  key="reselect"):
#         try:
#             new = rt.reselect(report, overrides)
#             st.session_state["reselect_result"] = new
#         except (ValueError, KeyError) as e:
#             st.error(f"Re-selection failed: {e}")

#     new = st.session_state.get("reselect_result")
#     if new:
#         old_rec = rec.get("inhibitor")
#         new_rec = new["selection"]["recommendation"]["inhibitor"]
#         a, b = st.columns(2)
#         a.metric("Inhibitor (this report)", old_rec or "— none")
#         b.metric("Inhibitor (after re-selection)", new_rec or "— none",
#                  delta=("changed" if new_rec != old_rec else "same"),
#                  delta_color="normal" if new_rec != old_rec else "off")
#         rows = []
#         for c in new["selection"]["inhibitors"]:
#             rows.append({
#                 "reagent": c["name"],
#                 "dE_GS": c["dE_GS"], "dE_NGS": c["dE_NGS"],
#                 "contrast": c["contrast"], "score": c["score"],
#                 "status": "accept" if c["accepted"]
#                 else "reject: " + "; ".join(c["reasons"]),
#             })
#         st.dataframe(rows, width="stretch", hide_index=True)
#         if new["selectivity"]:
#             st.caption("Selectivity vs 90% @ 10 nm target")
#             st.dataframe(
#                 [{"inhibitor": n,
#                   "S@target": s.get("selectivity_at_target"),
#                   "meets_target": bool(s.get("meets_target"))}
#                  for n, s in new["selectivity"].items()],
#                 width="stretch", hide_index=True)

#         # -- 4. Persist to criteria file --
#         st.markdown("**4 · Persist thresholds & (optionally) full re-run**")
#         if changed_keys:
#             if st.button("Write these thresholds to selection_criteria.md",
#                          key="persist"):
#                 diffs = rt.apply_config_to_md(overrides)
#                 if diffs:
#                     st.success("Updated: " + ", ".join(
#                         f"{k} {o}→{n}" for k, o, n in diffs))
#                 else:
#                     st.info("No values differed from the file.")
#         else:
#             st.caption("Sliders match the current file — nothing to persist.")

#         st.caption("Need *new* adsorption energies (a new candidate molecule or "
#                    "a different structure)? Threshold tuning above doesn't — but "
#                    "for that, use the **🚀 Submit job** tab to launch a screen "
#                    "with your chosen substrate, structure, and inhibitors.")


# --------------------------------------------------------------------------
# Job 4 — Autonomous Phase 2 screening agent (finds the best inhibitor)
# --------------------------------------------------------------------------

def _render_screen_table(screened: dict) -> None:
    st.dataframe(
        [{"reagent": c.get("name"), "dE_GS": c.get("dE_GS"),
          "dE_NGS": c.get("dE_NGS"), "contrast": c.get("contrast"),
          "binds_NGS": c.get("binds_NGS"), "selective": c.get("selective"),
          **({"error": c["error"]} if "error" in c else {})}
         for c in screened.values()],
        width="stretch", hide_index=True)


def _render_agent_event(ev: dict) -> None:
    """Render one trace event from agent_loop.run_agent into the page."""
    step, kind = ev.get("step"), ev["type"]
    if kind == "setup":
        st.caption(f"Surfaces {ev['surfaces']} loaded from "
                   f"`{os.path.relpath(ev['dir'], REPO)}`. "
                   + ("⚠️ Lennard-Jones placeholder — energies not physical."
                      if ev["placeholder"] else "MLIP calculator ready."))
    elif kind == "assistant":
        st.markdown(f"**Step {step} · reasoning**")
        st.markdown(f"> {ev['text']}")
    elif kind == "tool_call":
        st.markdown(f"**Step {step} · action** — calling `{ev['name']}`")
        if ev["args"]:
            st.code(json.dumps(ev["args"], indent=2), language="json")
    elif kind == "tool_result":
        res, name = ev["result"], ev["name"]
        with st.expander(f"Step {step} · result of `{name}`", expanded=True):
            if name == "screen" and isinstance(res.get("screened"), dict):
                _render_screen_table(res["screened"])
            elif name == "rank_screened" and isinstance(res.get("ranking"), list):
                best = res.get("best_selective")
                st.caption(f"best selective so far: "
                           f"{best['name'] if best else '— none yet'}")
                _render_screen_table({r["name"]: r for r in res["ranking"]})
            else:
                st.code(json.dumps(res, indent=2, default=str), language="json")
    elif kind == "final":
        st.success("Agent finished")
        st.markdown(ev["text"])
    elif kind == "error":
        st.error(ev["message"])


with tab_agent:
    st.subheader("Autonomous screening agent — find the best inhibitor")
    st.caption("The on-device model drives the **Phase 2 energetics screen**: it "
               "calls `screen(...)` on batches of inhibitors, reads the "
               "dE_GS / dE_NGS / contrast that come back, and iterates toward the "
               "most selective candidate — screening incrementally instead of "
               "brute-forcing the whole library. Each screen runs the MLIP, so "
               "this uses the GPU; don't run a pipeline job at the same time.")

    surf_sets_a = rt.find_surface_sets()
    if not surf_sets_a:
        st.warning("No reusable surface slabs on disk. Build surfaces first "
                   "(Submit job tab, or run_surface_builder.py) before screening.")
    rel_dirs_a = {os.path.relpath(d, REPO): d for d in surf_sets_a}

    default_goal = (
        "Screen the inhibitor library and identify the single best selective "
        "inhibitor for passivating SiNx (NGS) while sparing SiO2 (GS): the "
        "largest positive contrast (dE_GS - dE_NGS) with favourable NGS binding. "
        "Screen strategically and report the winner with its numbers.")
    a_goal = st.text_area("Goal", value=default_goal, height=110, key="agent_goal")

    c1a, c2a, c3a = st.columns(3)
    chosen_dir = None
    if rel_dirs_a:
        chosen_dir = rel_dirs_a[c1a.selectbox(
            "Surface set", list(rel_dirs_a),
            format_func=lambda r: f"{r}/ "
            f"({', '.join(rt.set_materials(surf_sets_a[rel_dirs_a[r]])) or '—'})",
            key="agent_surf")]
    a_steps = c2a.slider("Max steps", 3, 20, agent_loop.MAX_STEPS_DEFAULT,
                         key="agent_steps")
    a_sites = c3a.slider("Sites per screen", 1, 4, 2, key="agent_sites",
                         help="Representative sites per site-type per surface. "
                         "Fewer = faster but noisier energies.")

    disabled_a = (not up) or (not surf_sets_a)
    if not up:
        st.info("Model offline — start Ollama (`ollama serve`) to run the agent.")
    if st.button("Run screening agent", type="primary", disabled=disabled_a,
                 key="run_agent"):
        with st.spinner("Agent screening… each screen runs the MLIP, so this "
                        "can take a while; the trace fills in as it goes."):
            try:
                for ev in agent_loop.run_agent(model, a_goal,
                                               surface_dir=chosen_dir,
                                               max_steps=a_steps,
                                               max_sites=a_sites):
                    _render_agent_event(ev)
            except llm.OllamaError as e:
                st.error(str(e))


# --------------------------------------------------------------------------
# Background pipeline status (rendered once, below the tabs)
# --------------------------------------------------------------------------

_show_pipeline_status()
