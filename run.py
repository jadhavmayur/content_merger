"""
Content enrichment pipeline.

Takes a draft article, asks an LLM to pick relevant media + links from a small
simulated database, figures out where they belong, writes fresh anchor text, and
spits out an enriched Markdown file.

Three backends are supported (pick with --backend):
  - mock    : deterministic, no network. This is the default so the script always runs.
  - gemini  : Google Gemini via LangChain (needs GEMINI_API_KEY + langchain-google-genai).
  - ollama  : a local open-source model via Ollama (default: qwen2.5).

The pipeline runs as three separate LLM calls (selection -> placement -> anchor text)
so each step can be logged and validated on its own.
"""

import argparse
import json
import logging
import os
import re
import sys
import urllib.request

log = logging.getLogger("enrich")


# ---------------------------------------------------------------------------
# Simulated databases. In a real system these would come from a CMS / asset store.
# Topic is electric vehicles so the "right" picks are obvious enough to eyeball,
# and there are a few distractors mixed in so selection isn't trivial.
# ---------------------------------------------------------------------------

MEDIA_DB = [
    {
        "id": "img_ev_charging",
        "type": "image",
        "title": "EV charging at a public station",
        "description": "A modern electric car plugged into a fast charger at dusk.",
        "tags": ["ev", "charging", "infrastructure", "hero"],
        "url": "https://media.example.com/img/ev_charging.jpg",
    },
    {
        "id": "img_battery_pack",
        "type": "image",
        "title": "Lithium-ion battery pack cutaway",
        "description": "Cutaway diagram of a modern EV battery module.",
        "tags": ["battery", "technology", "range"],
        "url": "https://media.example.com/img/battery_pack.jpg",
    },
    {
        "id": "img_city_traffic",
        "type": "image",
        "title": "Rush hour traffic",
        "description": "Generic congested city street, mostly petrol cars.",
        "tags": ["traffic", "city", "commute"],
        "url": "https://media.example.com/img/city_traffic.jpg",
    },
    {
        "id": "vid_ev_review",
        "type": "video",
        "title": "2025 EV road test",
        "description": "Ten minute video review of a popular electric sedan.",
        "tags": ["ev", "review", "video"],
        "url": "https://media.example.com/vid/ev_review.mp4",
    },
    {
        "id": "img_solar_roof",
        "type": "image",
        "title": "Rooftop solar panels",
        "description": "Residential solar installation on a sunny day.",
        "tags": ["solar", "energy", "home"],
        "url": "https://media.example.com/img/solar_roof.jpg",
    },
]

LINK_DB = [
    {
        "id": "link_battery_guide",
        "title": "How EV batteries actually work",
        "description": "Deep dive into lithium-ion chemistry, degradation and range.",
        "tags": ["battery", "technology", "range"],
        "url": "https://example.com/guides/ev-batteries",
    },
    {
        "id": "link_charging_map",
        "title": "Find a charging station near you",
        "description": "Interactive map of public charging points worldwide.",
        "tags": ["charging", "infrastructure", "tools"],
        "url": "https://example.com/charging-map",
    },
    {
        "id": "link_incentives",
        "title": "Government EV incentives by region",
        "description": "Up to date rebates, tax credits and grants for EV buyers.",
        "tags": ["policy", "incentives", "cost"],
        "url": "https://example.com/ev-incentives",
    },
    {
        "id": "link_recipe",
        "title": "30 minute weeknight pasta",
        "description": "An easy pasta recipe. Completely unrelated distractor.",
        "tags": ["food", "recipe"],
        "url": "https://example.com/pasta",
    },
]


# ---------------------------------------------------------------------------
# LLM backends. They all expose the same generate(prompt, stage) -> str method
# and are expected to return a JSON string. The `stage` argument is only used by
# the mock so it knows which canned answer to hand back.
# ---------------------------------------------------------------------------

class LLMError(Exception):
    pass


class MockLLM:
    """Deterministic stand-in so the pipeline runs with no network or API key.

    Returns canned, schema-valid JSON for each stage. The picks line up with the
    sample article (electric vehicles), which keeps the demo output coherent.
    """

    name = "mock"

    _CANNED = {
        "selection": {
            "hero_image_id": "img_ev_charging",
            "in_context_image_id": "img_battery_pack",
            "featured_video_id": "vid_ev_review",
            "link_ids": ["link_battery_guide", "link_incentives"],
            "justification": (
                "The charging photo is a strong, on-topic hero. The battery cutaway "
                "supports the section on range/technology. The road-test video gives "
                "readers a richer look at a real EV, and the battery guide and incentives "
                "page are the two links readers are most likely to follow."
            ),
        },
        "placement": {
            "hero_image_id": {"after_paragraph": 0},
            "in_context_image_id": {"after_paragraph": 3},
            "featured_video_id": {"after_paragraph": 2},
            "links": [
                {"id": "link_battery_guide", "after_paragraph": 3},
                {"id": "link_incentives", "after_paragraph": 4},
            ],
        },
        "anchor": {
            "anchors": [
                {"id": "link_battery_guide", "anchor_text": "how an EV battery really works"},
                {"id": "link_incentives", "anchor_text": "what incentives you may qualify for"},
            ]
        },
    }

    def generate(self, prompt, stage):
        log.debug("MockLLM serving canned response for stage=%s", stage)
        return json.dumps(self._CANNED[stage])


class GeminiLLM:
    """Google Gemini backend via LangChain (langchain-google-genai)."""

    name = "gemini"

    def __init__(self, model="gemini-2.5-flash"):
        # Accept a few common spellings so a .env with GEMINI-KEY / GOOGLE_API_KEY works too.
        api_key = next(
            (os.environ[k] for k in
             ("GEMINI_API_KEY", "GEMINI-KEY", "GEMINI_KEY", "GOOGLE_API_KEY")
             if os.environ.get(k)),
            None,
        )
        if not api_key:
            raise LLMError("no Gemini API key found (set GEMINI_API_KEY or GEMINI-KEY)")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise LLMError("langchain-google-genai not installed (pip install langchain-google-genai)") from e

        # Ask for raw JSON back; saves us from stripping prose/code fences.
        self._llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0,
            model_kwargs={"generation_config": {"response_mime_type": "application/json"}},
        )

    def generate(self, prompt, stage):
        return self._llm.invoke(prompt).content


class OllamaLLM:
    """Local open-source model via Ollama's HTTP API.

    qwen2.5-7b is the default because it's reliable at returning the strict JSON
    these prompts ask for. Swap with --model if you prefer llama3.1, gemma2, etc.
    """

    name = "ollama"

    def __init__(self, model="qwen2.5", host="http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def generate(self, prompt, stage):
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "format": "json",   # Ollama will constrain the output to valid JSON
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            self.host + "/api/generate", data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = json.loads(r.read())
        except Exception as e:
            raise LLMError(f"Ollama request failed: {e}") from e
        return body.get("response", "")


def get_backend(name, model=None):
    if name == "mock":
        return MockLLM()
    if name == "gemini":
        return GeminiLLM(model) if model else GeminiLLM()
    if name == "ollama":
        return OllamaLLM(model) if model else OllamaLLM()
    raise ValueError(f"unknown backend: {name}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dotenv(path=".env"):
    """Tiny .env reader (no dependency). Loads KEY=VALUE lines into os.environ,
    tolerating spaces around '=' and surrounding quotes. Existing env vars win."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def load_assets(path):
    """Load the media + links databases from a JSON file: {"media": [...], "links": [...]}.
    Replaces the built-in defaults so the tool can be pointed at any catalogue."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    media, links = data.get("media", []), data.get("links", [])
    if not media or not links:
        raise ValueError("assets file must contain non-empty 'media' and 'links' lists")
    return media, links


def parse_json(text):
    """Best-effort JSON parse. Real models sometimes wrap output in ```json fences
    or add a stray sentence, so we pull out the first {...} block before giving up.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise LLMError("could not parse JSON from model output")


def split_paragraphs(article):
    """Split into paragraphs on blank lines. We keep the original text untouched and
    only ever insert *between* these blocks, so the article can't be corrupted."""
    return [p.strip() for p in re.split(r"\n\s*\n", article.strip()) if p.strip()]


def media_by_id(item_id):
    return next((m for m in MEDIA_DB if m["id"] == item_id), None)


def link_by_id(item_id):
    return next((l for l in LINK_DB if l["id"] == item_id), None)


def describe(items):
    return "\n".join(
        f"- {i['id']}: {i['title']} — {i['description']} (tags: {', '.join(i['tags'])})"
        for i in items
    )


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def selection_prompt(article):
    return f"""You are a content editor enriching an article with media and links.

ARTICLE:
{article}

AVAILABLE MEDIA:
{describe(MEDIA_DB)}

AVAILABLE LINKS:
{describe(LINK_DB)}

Pick the items that genuinely fit this article:
- exactly ONE hero image (sets the tone, sits at the top)
- exactly ONE in-context image (illustrates a specific point in the body)
- exactly TWO links readers would actually want to follow
- optionally ONE featured video if a video genuinely fits; use null if none do

Explain your reasoning briefly. Respond ONLY with JSON in this shape:
{{
  "hero_image_id": "<id>",
  "in_context_image_id": "<id>",
  "featured_video_id": "<id or null>",
  "link_ids": ["<id>", "<id>"],
  "justification": "<why these picks>"
}}"""


def placement_prompt(paragraphs, selection):
    numbered = "\n".join(f"[{i}] {p}" for i, p in enumerate(paragraphs))
    # Only ask the model to place the video if one was actually selected.
    if selection.get("featured_video_id"):
        video_line = f"\n- featured video: {selection['featured_video_id']}"
        video_field = '\n  "featured_video_id": {"after_paragraph": <int>},'
    else:
        video_line = video_field = ""
    return f"""Decide where each selected item should go in the article.

The article paragraphs are numbered below. For each item, give the paragraph index
it should appear AFTER (0 = right after the first/title paragraph).

PARAGRAPHS:
{numbered}

ITEMS TO PLACE:
- hero image: {selection['hero_image_id']}
- in-context image: {selection['in_context_image_id']}{video_line}
- links: {', '.join(selection['link_ids'])}

Respond ONLY with JSON:
{{
  "hero_image_id": {{"after_paragraph": <int>}},
  "in_context_image_id": {{"after_paragraph": <int>}},{video_field}
  "links": [{{"id": "<id>", "after_paragraph": <int>}}, {{"id": "<id>", "after_paragraph": <int>}}]
}}"""


def anchor_prompt(article, link_ids):
    links = [link_by_id(i) for i in link_ids]
    return f"""Write fresh, compelling anchor text for these links so they read
naturally inside the article. Keep each under ~8 words. Don't just reuse the title.

ARTICLE (for tone/context):
{article}

LINKS:
{describe([l for l in links if l])}

Respond ONLY with JSON:
{{"anchors": [{{"id": "<id>", "anchor_text": "<text>"}}]}}"""


# ---------------------------------------------------------------------------
# Pipeline stages. Each one calls the LLM, then validates the result and falls
# back to something safe if the model returned junk — that's the robustness story.
# ---------------------------------------------------------------------------

def run_selection(llm, article):
    raw = llm.generate(selection_prompt(article), stage="selection")
    data = parse_json(raw)

    hero = media_by_id(data.get("hero_image_id"))
    incontext = media_by_id(data.get("in_context_image_id"))
    links = [l for l in data.get("link_ids", []) if link_by_id(l)]

    # Validate, and patch up anything missing rather than crashing.
    if hero is None or hero["type"] != "image":
        log.warning("invalid hero image, falling back to first image in DB")
        hero = next(m for m in MEDIA_DB if m["type"] == "image")
    if incontext is None or incontext["type"] != "image" or incontext["id"] == hero["id"]:
        log.warning("invalid in-context image, falling back")
        incontext = next(m for m in MEDIA_DB if m["type"] == "image" and m["id"] != hero["id"])
    if len(links) != 2:
        log.warning("expected 2 valid links, got %d — backfilling", len(links))
        pool = [l["id"] for l in LINK_DB if l["id"] not in links]
        links = (links + pool)[:2]

    # The featured video is optional: keep it only if it's a real video item.
    video = media_by_id(data.get("featured_video_id"))
    if video is not None and video["type"] != "video":
        log.warning("featured_video_id is not a video — ignoring it")
        video = None

    selection = {
        "hero_image_id": hero["id"],
        "in_context_image_id": incontext["id"],
        "featured_video_id": video["id"] if video else None,
        "link_ids": links,
        "justification": data.get("justification", ""),
    }
    log.info("Selection: hero=%s in_context=%s video=%s links=%s",
             hero["id"], incontext["id"], selection["featured_video_id"], links)
    log.info("Justification: %s", selection["justification"])
    return selection


def run_placement(llm, paragraphs, selection):
    last = len(paragraphs) - 1

    def clamp(idx, default):
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            return default
        return max(0, min(idx, last))

    try:
        data = parse_json(llm.generate(placement_prompt(paragraphs, selection), stage="placement"))
    except LLMError:
        log.warning("placement output unparseable — using sensible defaults")
        data = {}

    placement = {
        "hero": 0,  # hero always rides at the very top regardless of model whim
        "in_context": clamp(data.get("in_context_image_id", {}).get("after_paragraph"), min(2, last)),
        "links": {},
    }
    if selection.get("featured_video_id"):
        placement["video"] = clamp(data.get("featured_video_id", {}).get("after_paragraph"), min(3, last))
    given = {l.get("id"): l.get("after_paragraph") for l in data.get("links", [])}
    for n, link_id in enumerate(selection["link_ids"]):
        placement["links"][link_id] = clamp(given.get(link_id), min(n + 1, last))

    log.info("Placement: %s", placement)
    return placement


def run_anchor_text(llm, article, selection):
    try:
        data = parse_json(llm.generate(anchor_prompt(article, selection["link_ids"]), stage="anchor"))
        anchors = {a["id"]: a["anchor_text"] for a in data.get("anchors", []) if a.get("anchor_text")}
    except LLMError:
        log.warning("anchor text output unparseable — using link titles instead")
        anchors = {}

    # Any link the model skipped gets its DB title as a safe fallback.
    for link_id in selection["link_ids"]:
        if link_id not in anchors:
            anchors[link_id] = link_by_id(link_id)["title"]
    log.info("Anchor text: %s", anchors)
    return anchors


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def media_block(item):
    return f"![{item['title']}]({item['url']})\n*{item['description']}*"


def video_block(item):
    # Markdown has no native video embed, so we render an HTML5 <video> tag (which
    # most renderers honour) with a plain clickable link as the fallback.
    return (
        f'<video src="{item["url"]}" controls width="100%"></video>\n'
        f"🎬 [Watch: {item['title']}]({item['url']})\n*{item['description']}*"
    )


def link_block(item, anchor):
    return f"> 📎 Related: [{anchor}]({item['url']})"


def build_inserts(paragraphs, selection, placement, anchors):
    """Map each paragraph index to the list of items to insert after it.
    Each item is a (kind, data, anchor) tuple so both the Markdown and the DOCX
    renderers can consume the same structure."""
    inserts = {i: [] for i in range(len(paragraphs))}
    inserts[placement["hero"]].append(("image", media_by_id(selection["hero_image_id"]), None))
    inserts[placement["in_context"]].append(("image", media_by_id(selection["in_context_image_id"]), None))
    if selection.get("featured_video_id") and "video" in placement:
        inserts[placement["video"]].append(("video", media_by_id(selection["featured_video_id"]), None))
    for link_id, idx in placement["links"].items():
        inserts[idx].append(("link", link_by_id(link_id), anchors[link_id]))
    return inserts


def assemble(paragraphs, selection, placement, anchors):
    """Render the enriched article as Markdown text."""
    inserts = build_inserts(paragraphs, selection, placement, anchors)
    out = []
    for i, para in enumerate(paragraphs):
        out.append(para)
        for kind, item, anchor in inserts[i]:
            if kind == "image":
                out.append(media_block(item))
            elif kind == "video":
                out.append(video_block(item))
            elif kind == "link":
                out.append(link_block(item, anchor))
    return "\n\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# DOCX rendering (optional output format, needs python-docx)
# ---------------------------------------------------------------------------

def _docx_hyperlink(paragraph, url, text):
    """Add a real, clickable hyperlink to a python-docx paragraph (styled blue/underlined)."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    part = paragraph.part
    r_id = part.relate_to(
        url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    color = OxmlElement("w:color"); color.set(qn("w:val"), "0563C1"); rpr.append(color)
    underline = OxmlElement("w:u"); underline.set(qn("w:val"), "single"); rpr.append(underline)
    run.append(rpr)
    t = OxmlElement("w:t"); t.text = text; run.append(t)
    link.append(run)
    paragraph._p.append(link)


def write_docx(paragraphs, selection, placement, anchors, out_path):
    """Render the enriched article as a Word .docx file.

    Images are downloaded and embedded when reachable; if a URL can't be fetched
    (e.g. the simulated placeholder URLs), we fall back to a labelled caption + link
    so the document still records exactly which asset goes where."""
    import io
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor

    doc = Document()
    inserts = build_inserts(paragraphs, selection, placement, anchors)

    def caption(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.italic = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0x60, 0x60, 0x60)

    def add_image(item):
        try:
            with urllib.request.urlopen(item["url"], timeout=10) as r:
                doc.add_picture(io.BytesIO(r.read()), width=Inches(5.5))
        except Exception:
            p = doc.add_paragraph()
            p.add_run(f"[Image: {item['title']}] ").bold = True
            _docx_hyperlink(p, item["url"], item["url"])
        caption(item["description"])

    def add_video(item):
        p = doc.add_paragraph()
        p.add_run("🎬 Watch: ").bold = True
        _docx_hyperlink(p, item["url"], item["title"])
        caption(item["description"])

    def add_link(item, anchor):
        p = doc.add_paragraph()
        p.add_run("📎 Related: ").bold = True
        _docx_hyperlink(p, item["url"], anchor)

    for i, para in enumerate(paragraphs):
        # Headings: a paragraph that starts with one or more '#' becomes a Word heading.
        stripped = para.lstrip("#")
        level = len(para) - len(stripped)
        if 1 <= level <= 4 and stripped.startswith(" "):
            doc.add_heading(stripped.strip(), level=min(level, 4))
        else:
            doc.add_paragraph(para)

        for kind, item, anchor in inserts[i]:
            if kind == "image":
                add_image(item)
            elif kind == "video":
                add_video(item)
            elif kind == "link":
                add_link(item, anchor)

    doc.save(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Enrich a Markdown/text article with LLM-chosen media and links.")
    ap.add_argument("article", help="path to the input article (.md or .txt)")
    ap.add_argument("-o", "--output",
                    help="output path; extension picks the format (default: output/<article>.docx)")
    ap.add_argument("-b", "--backend", default="mock", choices=["mock", "gemini", "ollama"],
                    help="LLM backend to use (default: mock)")
    ap.add_argument("-m", "--model", help="override the model name for gemini/ollama")
    ap.add_argument("-a", "--assets", help="JSON file with custom media/links databases")
    ap.add_argument("-v", "--verbose", action="store_true", help="show debug logging")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    load_dotenv()  # pull GEMINI key etc. from .env if present

    if args.assets:
        global MEDIA_DB, LINK_DB
        try:
            MEDIA_DB, LINK_DB = load_assets(args.assets)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            log.error("could not load assets '%s': %s", args.assets, e)
            return 1
        log.info("Loaded assets from %s (%d media, %d links)", args.assets, len(MEDIA_DB), len(LINK_DB))

    if not os.path.isfile(args.article):
        log.error("article not found: %s", args.article)
        return 1
    with open(args.article, encoding="utf-8") as f:
        article = f.read()
    if not article.strip():
        log.error("article is empty")
        return 1

    try:
        llm = get_backend(args.backend, args.model)
    except LLMError as e:
        log.error("could not start '%s' backend: %s", args.backend, e)
        return 1
    log.info("Using backend: %s", llm.name)

    paragraphs = split_paragraphs(article)
    log.info("Article has %d paragraphs", len(paragraphs))

    try:
        selection = run_selection(llm, article)
        placement = run_placement(llm, paragraphs, selection)
        anchors = run_anchor_text(llm, article, selection)
    except LLMError as e:
        log.error("pipeline failed: %s", e)
        return 1

    # Default output: output/<article-name>.docx . Format follows the extension,
    # so `-o something.md` still gives Markdown.
    if args.output:
        out_path = args.output
    else:
        stem = os.path.splitext(os.path.basename(args.article))[0]
        out_path = os.path.join("output", stem + ".docx")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if out_path.lower().endswith(".docx"):
        try:
            write_docx(paragraphs, selection, placement, anchors, out_path)
        except ImportError:
            # Don't fail the run just because python-docx is missing — emit Markdown instead.
            out_path = os.path.splitext(out_path)[0] + ".md"
            log.warning("python-docx not installed; writing Markdown to %s instead "
                        "(pip install python-docx for .docx output)", out_path)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(assemble(paragraphs, selection, placement, anchors))
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(assemble(paragraphs, selection, placement, anchors))

    log.info("Wrote enriched article -> %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
