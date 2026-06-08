# LLM-Powered Content Enrichment Pipeline

A small, self-contained Python tool that takes a draft article and enriches it with
media and hyperlinks chosen by a Large Language Model. It selects a hero image, an
in-context image, and two relevant links from a simulated asset database, decides where
each belongs, writes fresh anchor text for the links, and outputs a single enriched
Markdown file.

## Executive Summary

The design philosophy is **"let the LLM decide *what*, let the code control *how*."** The
model makes the editorial judgements that need language understanding — which assets are
relevant, where they fit, what the links should say. Everything mechanical and error-prone
— parsing, validating, and stitching the final document — is handled by plain Python.

The most important consequence of that split: **the original article text is never sent
back through the model for rewriting.** Instead the article is broken into numbered
paragraphs, the LLM only ever returns *indices* ("put the hero after paragraph 0"), and the
code inserts content *between* paragraphs. The source prose therefore cannot be silently
altered, truncated, or hallucinated — a common failure mode when you ask an LLM to
"return the article with the images added."

The tool ships with a deterministic **mock backend as the default**, so it runs instantly
with no API key and no network. Two real backends — **Gemini** and a local **Ollama**
model — sit behind the same interface and can be switched on with a flag.

## Technical Design

### Architecture

```
article.md ──► split into numbered paragraphs
                     │
          ┌──────────┴───────────────────────────────┐
          │  LLM pipeline (3 separate calls)          │
          │  1. Selection   → hero, in-context, links │
          │  2. Placement   → paragraph index per item│
          │  3. Anchor text → link display text       │
          └──────────┬───────────────────────────────┘
                     │  (each stage validated + fallback)
                     ▼
            assemble Markdown (insert between paragraphs)
                     ▼
            article.enriched.md
```

### Why three separate LLM calls

The pipeline runs **selection → placement → anchor-text as three distinct calls** rather
than one mega-prompt. This maps directly onto the three editorial decisions, and it means
each stage has a small, well-defined input/output contract that can be logged and
validated independently. When something goes wrong, the logs tell you *which* decision
failed, and the fallback for that one stage kicks in without poisoning the others. The
trade-off is more latency and tokens than a single combined call — discussed below.

### Pluggable backends

All three backends expose one method, `generate(prompt, stage) -> str`, and are expected
to return JSON:

- **`mock`** (default) — returns canned, schema-valid JSON per stage. Zero dependencies,
  fully deterministic, runs offline. This is what satisfies the assignment's "mock the API
  response" requirement.
- **`gemini`** — Google `gemini-2.5-flash` via **LangChain** (`langchain-google-genai`),
  using JSON-mode (`response_mime_type: "application/json"`) so the output is already clean
  JSON. Verified end-to-end against the live API.
- **`ollama`** — any local open-source model served by [Ollama](https://ollama.com),
  defaulting to **`qwen2.5`**, using Ollama's `format: "json"` constraint.

Because they share an interface, the pipeline code doesn't know or care which one it's
talking to.

### Robustness

Every stage **parses, validates, and falls back** rather than trusting the model:

- JSON parsing tolerates code fences / stray prose by extracting the first `{...}` block.
- Selection checks that returned IDs actually exist in the database, that the hero and
  in-context items are distinct images, and that exactly two valid links came back —
  backfilling from the DB if not.
- Placement clamps every paragraph index into range and defaults to sensible positions if
  the model returns garbage. The hero image is always forced to the top.
- Anchor text falls back to the link's database title for any link the model skipped.

All decisions and justifications are emitted through Python's `logging` module, so a run is
fully auditable (use `-v` for debug detail).

## Prompt Engineering Strategy

Three techniques carry most of the weight:

1. **Strict JSON output contracts.** Every prompt ends with the exact JSON shape expected
   and the instruction to respond with JSON only. Combined with each backend's native JSON
   mode, this is what makes parsing reliable instead of a guessing game.

2. **Reasoning as a required field (lightweight chain-of-thought).** The selection prompt
   forces a `justification` field. Making the model articulate *why* it chose each asset
   nudges it toward better, more grounded picks — and conveniently doubles as the audit log
   the assignment asks for, at no extra cost.

3. **Few-shot-style schema priming.** Showing the literal target structure in the prompt
   acts as a one-shot example that locks the output format, which matters far more for
   small open-source models than for frontier ones.

A deliberate fourth choice is the **paragraph-index placement prompt**: rather than asking
the model to produce the final document, it sees the article as an indexed list and returns
only "after paragraph N." This keeps the model's job small and verifiable and is the key to
the no-text-corruption guarantee described above.

### Model rationale

- **Gemini `gemini-2.0-flash`** for the API path — it's fast, has a generous free tier, and
  its JSON mode makes structured extraction dependable, which is exactly what this
  selection/placement task needs.
- **`qwen2.5` (7B) via Ollama** for the open-source path — in this size class it's the most
  reliable at honouring strict "return only this JSON" instructions. Good alternatives if
  you prefer: `llama3.1` (8B, strong all-rounder) or `gemma2` (9B). For low-RAM machines,
  `phi3.5` or `llama3.2:3b`.

## Quick Start (Input → Run → Output)

There is no interactive prompt or text box. The **input is an article file**, you pass
its path to the script, and the **output is written to the `output/` folder**.

### Step 1 — Provide the input (the article)

Put your draft article in a plain-text or Markdown file. You can either use one of the
included samples (`sample_article.md`, `articles/houseplants.md`) or create your own:

```bash
# create your own input file (any .md or .txt file works)
nano my_article.md        # paste your article text and save
```

### Step 2 — Run the app

Pass the article's path as the first argument:

```bash
python run.py my_article.md
```

That single argument **is the input**. Common variations:

```bash
python run.py my_article.md                       # default: mock backend, .docx output
python run.py my_article.md -o output/my.md       # Markdown output instead of .docx
python run.py my_article.md --backend gemini       # use the real Gemini model
python run.py my_article.md -v                      # verbose: show every decision
```

### Step 3 — Get the output

The enriched file is written to:

```
output/<your-article-name>.docx
```

For example, `python run.py my_article.md` produces **`output/my_article.docx`**.
Open it in Microsoft Word, Google Docs, or LibreOffice. The `output/` folder is created
automatically if it doesn't exist.

> **Note on your own topics:** the built-in media/links catalogue is about electric
> vehicles. For an article on a different topic, use the Gemini (or Ollama) backend **and**
> supply a matching catalogue with `--assets your_assets.json` (see
> `assets/houseplants_assets.json` for the format). The default `mock` backend always
> returns the EV picks, so it only makes sense for `sample_article.md`.

---

## Usage

No installation is required for the default mock backend (standard library only).

```bash
# Default: deterministic mock backend, no API key, no network
python run.py sample_article.md

# Point at a custom media/links catalogue (JSON)
python run.py articles/houseplants.md --assets assets/houseplants_assets.json

# Get Markdown instead of .docx (format follows the file extension)
python run.py sample_article.md -o output/sample_article.md

# Verbose logging (see every stage's decision)
python run.py sample_article.md -v
```

**Output** defaults to `output/<article-name>.docx` (the folder is created
automatically). The output format follows the extension of `-o`, so
`-o output/name.md` produces Markdown instead. The `.docx` path needs
`python-docx`; if it isn't installed the tool logs a warning and writes Markdown
instead rather than failing.

### Using the live Gemini backend

```bash
pip install langchain-google-genai

# Provide the key either as an env var or in a .env file (auto-loaded).
# .env accepts GEMINI_API_KEY, GEMINI-KEY, GEMINI_KEY or GOOGLE_API_KEY.
echo 'GEMINI_API_KEY=your-key-here' > .env

python run.py articles/houseplants.md --backend gemini --assets assets/houseplants_assets.json
# optionally pin a model:
python run.py articles/houseplants.md --backend gemini --model gemini-2.5-flash
```

### Using a local open-source model (Ollama)

```bash
# one-time: install Ollama, then pull a model
ollama pull qwen2.5
python run.py sample_article.md --backend ollama
# or another model:
python run.py sample_article.md --backend ollama --model llama3.1
```

## Assumptions and Trade-offs

- **Mock is the default, by design.** The assignment requires the API response to be
  mocked, and the tool must always run for a reviewer. The mock's canned picks line up with
  the EV sample article (`sample_article.md`) so the demo output is coherent; the real
  backends are there to prove the integration is genuine, not hand-waved. Note: because the
  mock is deterministic, running it on a *different* article still returns the EV-shaped
  IDs — the validators then backfill safely from whatever catalogue is loaded. For arbitrary
  articles use the `gemini` or `ollama` backend.
- **Custom catalogues via `--assets`.** The built-in EV database lives in `run.py`; any other
  catalogue can be supplied as a JSON file (`{"media": [...], "links": [...]}`). The
  houseplants sample (`articles/houseplants.md` + `assets/houseplants_assets.json`) was
  enriched with the live Gemini backend.
- **Output format: `.docx` by default, Markdown on request.** The assignment describes a
  Markdown deliverable; both are supported and the renderers share one `build_inserts()`
  structure, so the document is identical either way. `.docx` embeds images when the URLs are
  reachable and otherwise falls back to a labelled caption + clickable link, so the asset
  placement is recorded regardless. Choose Markdown with `-o output/name.md`.
- **Three calls vs. one.** Separate calls cost more latency/tokens than a single combined
  prompt, but they give clean per-stage logging, validation, and fallback. For a low-volume
  editorial tool that trade is clearly worth it; at scale you'd likely merge selection and
  placement into one structured call.
- **Links are inserted as their own callout lines** (`> 📎 Related: ...`) rather than woven
  into existing sentences. Weaving reads more naturally but means editing the source prose,
  which reopens the text-corruption risk this design exists to avoid. The callout approach
  is the safe, deterministic choice.
- **In-memory databases.** The media/link "databases" are Python lists in `run.py`, as
  specified. A real deployment would query a CMS/asset store; the `*_by_id` and `describe`
  helpers are the seam where that swap would happen.
- **Counts follow the spec, with an optional featured video.** One hero image, one
  in-context image, and two links are required as specified. On top of that the selection
  stage may *optionally* pick one **featured video** if one genuinely fits (it returns
  `null` otherwise), rendered via a `video_block()` as an HTML5 `<video>` tag with a
  clickable fallback link. This exercises the "images **and** videos" part of the brief
  without forcing an irrelevant video in. In testing, the mock (EV) embeds the road-test
  video, while live Gemini correctly declined a video for the houseplants article.
- **No semantic re-ranking.** Selection relies entirely on the LLM reading the titles,
  descriptions, and tags. With a larger database you'd add an embedding-based pre-filter to
  hand the model a shortlist instead of the whole catalogue.

## Files

| File | Purpose |
|---|---|
| `run.py` | The entire pipeline: data, backends, prompts, stages, assembly, CLI |
| `sample_article.md` | EV sample article (matches the built-in mock picks) |
| `articles/houseplants.md` | Second sample article (different topic) |
| `assets/houseplants_assets.json` | Custom media + links catalogue for the houseplants article |
| `output/` | Generated enriched articles (`<article-name>.docx`) |
| `requirements.txt` | `python-docx` for .docx output; `langchain-google-genai` for the Gemini backend |
| `README.md` | This document |
