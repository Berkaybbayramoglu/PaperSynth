# Research Synthesis System

A multi-agent pipeline that reads a folder of research papers (PDF / TXT / MD)
and produces a single, dense synthesis report in both plain text and Word
(`.docx`) formats. It runs entirely against a **local LLM** (LM Studio by
default) — no cloud API key required.

## How it works

```
papers/  ─►  Reader  ─►  Analyzer  ─►  Synthesizer  ─►  Writer  ─►  report.txt / .docx
                            │              │              ▲
                            ▼              │              │
                          Memory           │       Reviewer / Chief Reviewer
                          (cache)          │              │
                                           ▼              │
                                   Metrics Extractor      │
                                                          │
                                                     Validator
                                                  (anti-hallucination)
```

Each stage is an agent (a Semantic Kernel chat completion call with its own
prompt). The Writer / Reviewer / Validator loop iterates up to
`max_iterations` times until the report is approved by both reviewers and
no fabricated content is detected.

## The agents

| Agent | Role |
|---|---|
| `ReaderAgent` | Extracts text from PDFs and asks the LLM to pull out every detail. |
| `AnalyzerAgent` | Writes a 5-section detailed analysis for each paper (cached via `MemoryAgent`). |
| `MemoryAgent` | Disk-backed JSON cache of per-paper analyses, keyed by content hash. |
| `SynthesizerAgent` | Merges all analyses into a paper-by-paper + thematic summary. |
| `MetricsExtractorAgent` | Pulls out every numeric result (accuracy, F1, dataset size, etc.). |
| `WriterAgent` | Drafts the final report; chunks large input and writes with cumulative context. |
| `ReviewerAgent` | Strict first-pass reviewer; rejects anything vague or under-detailed. |
| `ChiefReviewerAgent` | Final approval; rejects reports that are too brief or repetitive. |
| `ValidationAgent` | Cross-checks the report against the synthesis to catch hallucinations. |
| `OrderVerificationAgent` | Makes sure papers stay in their original order through the pipeline. |
| `ProgressTracker` | Writes a JSON log of every stage and event for inspection. |

## Project layout

```
research-synthesis-system/
├── main.py                       # Entry point: wires agents, runs the pipeline
├── requirements.txt
├── sample_papers/                # Drop your PDFs here (samples included)
└── research_synthesis/
    ├── agents/                   # One file per agent
    │   ├── reader.py
    │   ├── analyzer.py
    │   ├── synthesizer.py
    │   ├── writer.py
    │   ├── reviewer.py
    │   ├── chief_reviewer.py
    │   ├── validator.py
    │   ├── metrics_extractor.py
    │   ├── order_verifier.py
    │   └── memory.py
    ├── core/
    │   ├── behavior_tree.py      # Top-level orchestrator
    │   └── progress_tracker.py   # JSON-backed event log
    ├── output/
    │   └── word_export.py        # Renders the .docx report
    └── utils/
        └── text_cleaning.py      # Strips markdown from LLM output
```

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Start a local LLM**

   Install [LM Studio](https://lmstudio.ai/), load a chat model (the default
   is `qwen3-vl-8b`), and start the local server on
   `http://localhost:1234/v1`. To use a different model or endpoint, edit
   `LLM_BASE_URL` / `LLM_MODEL_ID` at the top of `main.py`.

3. **Add your papers**

   Drop your `.pdf`, `.txt`, or `.md` files into `sample_papers/` (or any
   directory you point the script at).

## Usage

```bash
# Uses ./sample_papers by default
python main.py

# Or pick a different directory
python main.py path/to/my_papers/
```

After the pipeline finishes you'll get:

- `research_report_<timestamp>.txt` — plain-text report
- `research_report_<timestamp>.docx` — Word version with headings and styling
- `analysis_cache.json` — per-paper analysis cache (speeds up reruns)
- `synthesis_progress.json` — detailed per-stage event log

## Configuration

The most useful knobs live in `main.py` and on `ResearchSynthesisBehaviorTree.run`:

| Parameter | Default | What it does |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:1234/v1` | OpenAI-compatible endpoint of your local LLM |
| `LLM_MODEL_ID` | `qwen3-vl-8b` | Model name the endpoint should serve |
| `max_iterations` | `3` | Max Writer ↔ Reviewer rounds before giving up |
| `parallel_processing` | `False` | Set `True` only if your LLM server handles parallel requests |
| `read_batch_size` | `6` | Papers read in parallel per batch (when `parallel_processing=True`) |
| `analyze_batch_size` | `4` | Papers analyzed in parallel per batch (when `parallel_processing=True`) |

Sequential mode is recommended for LM Studio and most single-GPU local
LLMs — they queue requests internally anyway.

## Output structure

The generated report has four sections:

1. **OVERVIEW** — brief introduction to all papers and their domains
2. **DETAILED PAPER-BY-PAPER ANALYSIS** — 5–7 paragraphs per paper
3. **INTEGRATED SYNTHESIS AND COMPARISON** — thematic comparison with
   subsections A (Approaches), B (Contexts), C (Instruments),
   D (Techniques), E (Findings), F (Future Directions)
4. **CONCLUSIONS** — 2–3 paragraph wrap-up
