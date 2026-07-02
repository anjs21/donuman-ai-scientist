"""On-device LLM assistant for the AS-ALD co-scientist.

A thin layer over a local Ollama server (default qwen2.5:7b-instruct) that adds
two capabilities to the existing pipeline:

  Job 2  explain   -- read a report.json and write a scientific discussion.
  Job 3  tune      -- inspect a run, propose selection-threshold changes as
                      structured JSON, and re-run selection instantly from the
                      already-computed energetics (no GPU / MACE needed).

Everything runs against http://localhost:11434 using the Python standard
library only; no cloud calls and no extra runtime dependencies beyond Streamlit
for the UI.
"""
