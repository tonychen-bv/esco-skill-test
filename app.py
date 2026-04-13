"""
app.py — ESCO Skill Gap Analyzer (Streamlit UI)

Prerequisites:
    1. Run embed_esco.py first to generate esco_embeddings/
    2. Copy .env.example → .env and fill in your Azure OpenAI keys

Run:
    streamlit run app.py
"""

import io
import json
import os
import re

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
EMBEDDINGS_DIR = os.path.join(os.path.dirname(__file__), "esco_embeddings")
ESCO_DATA_DIR = os.getenv(
    "ESCO_DATA_DIR",
    "/Users/tonychen/Downloads/ESCO dataset - v1.2.1 - classification - en - csv",
)
EMBEDDING_MODEL = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
LLM_MODEL = os.getenv("AZURE_LLM_DEPLOYMENT", "gpt-5.4-nano")
EMBEDDING_DIM = 256
SKILL_MATCH_THRESHOLD = 0.50
TOP_OCC_CANDIDATES = 3   # internal: fetch top-N, only best match is displayed

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
)


# ── File extraction ───────────────────────────────────────────────────────────
def extract_text_from_file(uploaded_file) -> str:
    """Extract plain text from an uploaded PDF or DOCX file."""
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    uploaded_file.seek(0)  # reset so Streamlit can re-read if needed

    if name.endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(data))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages).strip()
        except Exception as e:
            st.error(f"Failed to read PDF: {e}")
            return ""

    elif name.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(data))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(paragraphs).strip()
        except Exception as e:
            st.error(f"Failed to read DOCX: {e}")
            return ""

    return ""


# ── Data loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ESCO embeddings…")
def load_esco_data():
    occ_emb = np.load(os.path.join(EMBEDDINGS_DIR, "occupations_embeddings.npy"))
    with open(os.path.join(EMBEDDINGS_DIR, "occupations_meta.json"), encoding="utf-8") as f:
        occ_meta = json.load(f)

    skill_emb = np.load(os.path.join(EMBEDDINGS_DIR, "skills_embeddings.npy"))
    with open(os.path.join(EMBEDDINGS_DIR, "skills_meta.json"), encoding="utf-8") as f:
        skill_meta = json.load(f)

    with open(os.path.join(EMBEDDINGS_DIR, "occ_skill_relations.json"), encoding="utf-8") as f:
        relations = json.load(f)

    # Normalise embeddings for fast cosine similarity (dot product)
    occ_emb = occ_emb / (np.linalg.norm(occ_emb, axis=1, keepdims=True) + 1e-9)
    skill_emb = skill_emb / (np.linalg.norm(skill_emb, axis=1, keepdims=True) + 1e-9)

    # Full occupation detail lookup keyed by conceptUri (for rich metadata display)
    occ_detail: dict[str, dict] = {}
    detail_path = os.path.join(EMBEDDINGS_DIR, "occupations_detail.json")
    if os.path.exists(detail_path):
        with open(detail_path, encoding="utf-8") as f:
            occ_detail = json.load(f)

    return occ_emb, occ_meta, skill_emb, skill_meta, relations, occ_detail


# ── UI helpers ────────────────────────────────────────────────────────────────
def render_occ_match(occ: dict, score: float, detail: dict):
    """Display a single best-match occupation card with full metadata."""
    st.markdown(f"### {occ['preferredLabel']}")
    st.caption(f"Similarity score: `{score:.3f}`")

    if detail.get("altLabels"):
        st.markdown("**Also known as**")
        st.caption(detail["altLabels"].replace("\n", " · "))

    if detail.get("definition"):
        st.markdown("**Definition**")
        st.markdown(detail["definition"])

    if detail.get("description"):
        st.markdown("**Description**")
        st.markdown(detail["description"])

    if detail.get("scopeNote"):
        st.markdown("**Scope note**")
        st.caption(detail["scopeNote"])


# ── Azure helpers ─────────────────────────────────────────────────────────────
def get_embedding(text: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
        dimensions=EMBEDDING_DIM,
    )
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


def semantic_search_occupations(query: str, occ_emb, occ_meta, top_k: int = 5):
    vec = get_embedding(query)
    scores = occ_emb @ vec
    indices = np.argsort(scores)[::-1][:top_k]
    return [(occ_meta[i], float(scores[i])) for i in indices]


def semantic_search_skills(query: str, skill_emb, skill_meta, top_k: int = 3):
    vec = get_embedding(query)
    scores = skill_emb @ vec
    indices = np.argsort(scores)[::-1][:top_k]
    return [(skill_meta[i], float(scores[i])) for i in indices]


def llm_parse_current(current_text: str) -> dict:
    """Extract job title and skills from a current-state description."""
    prompt = f"""You are a career advisor assistant. Extract structured information from the user's current state description.

Current state description:
\"\"\"{current_text}\"\"\"

Return a JSON object with exactly these fields:
{{
  "current_title": "the most specific job title mentioned or implied (in English)",
  "current_skills": ["skill or tool 1", "skill or tool 2", ...]
}}

Rules:
- current_title must be in English
- current_skills should be concrete skills, tools, technologies, or competencies (not vague adjectives)
- Extract up to 50 skills maximum"""

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


def llm_parse_target(target_text: str) -> dict:
    """Extract target job title from a target-state description."""
    prompt = f"""You are a career advisor assistant. Extract the target job title from the user's description.

Target state description:
\"\"\"{target_text}\"\"\"

Return a JSON object with exactly this field:
{{
  "target_title": "the most specific target job title (in English)"
}}

Rules:
- target_title must be in English"""

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


def llm_infer_next_role(current_title: str, current_occ_label: str, current_occ_skills: list[dict]) -> str:
    """Given a current occupation, ask LLM to suggest the most natural next career step."""
    skill_sample = ", ".join(s["skillLabel"] for s in current_occ_skills[:15] if s["relationType"] == "essential")
    prompt = f"""You are a career development expert.

A professional currently works as: "{current_occ_label}" (matched from user input: "{current_title}")
Key skills for this role: {skill_sample}

What is the single most natural and common next career step (advancement) for this role?
Return only the job title in English — no explanation, no punctuation, just the title.
Example output: Software Architect"""

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_completion_tokens=20,
    )
    return resp.choices[0].message.content.strip()


def compute_gap_merged(
    user_matched_skills: list[dict],
    current_occ_skills: list[dict],
    target_occ_skills: list[dict],
    current_name: str = "Current role",
    target_name: str = "Target role",
) -> list[dict]:
    """
    Build the gap list from (current + target) skills minus user's matched URIs.

    Deduplication rules per unique skillUri:
    - Appears in one role only → source = current_name or target_name
    - Appears in both roles with same relationType → deduplicated, source = both label
    - Appears in both roles with different relationType → kept as two separate entries
    """
    matched_uris = {m["esco_uri"] for m in user_matched_skills}
    both_name = f"{current_name} + {target_name}"

    by_uri: dict[str, list] = {}
    for s in (current_occ_skills or []):
        if s["skillUri"] not in matched_uris:
            by_uri.setdefault(s["skillUri"], []).append((current_name, s))
    for s in (target_occ_skills or []):
        if s["skillUri"] not in matched_uris:
            by_uri.setdefault(s["skillUri"], []).append((target_name, s))

    gap: list[dict] = []
    for entries in by_uri.values():
        if len(entries) == 1:
            source, s = entries[0]
            gap.append({**s, "source": source})
        else:
            _, s_curr = entries[0]
            _, s_tgt = entries[1]
            if s_curr["relationType"] == s_tgt["relationType"]:
                gap.append({**s_curr, "source": both_name})
            else:
                gap.append({**s_curr, "source": current_name})
                gap.append({**s_tgt, "source": target_name})
    return gap


def llm_development_plan(
    target_occ: dict,
    essential_gaps: list[dict],
    user_matched_skills: list[dict],
    current_occ: dict | None = None,
    target_is_inferred: bool = False,
    user_profile_text: str = "",
) -> dict:
    """
    Generate a 70-20-10 development plan calibrated to the user's seniority level.

    Returns a dict:
    {
      "experience": [{"action": str, "addresses": [skill_label, ...]}, ...],
      "social":     [...],
      "formal":     [...],
    }
    """
    def fmt_have(matched: list[dict]) -> str:
        if not matched:
            return "  (none — starting from scratch)"
        return "\n".join(f"  - {m['esco_label']} (from: \"{m['user_skill']}\")" for m in matched)

    gap_labels = [s["skillLabel"] for s in essential_gaps[:30]]
    gap_list = "\n".join(f'  - "{label}"' for label in gap_labels)

    current_context = (
        f"Currently works as: {current_occ.get('preferredLabel', 'unknown')}"
        if current_occ
        else "Starting from scratch (no current role provided)"
    )
    inferred_note = " (inferred next step)" if target_is_inferred else ""
    profile_snippet = user_profile_text[:1200].strip() if user_profile_text else "(not provided)"

    prompt = f"""You are an expert career development advisor using the 70-20-10 development model.

## User profile (raw input — use this to infer seniority level)
{profile_snippet}

## Structured context
- {current_context}
- Target role{inferred_note}: {target_occ.get('preferredLabel', 'Unknown')}
- Skills the user already has:
{fmt_have(user_matched_skills)}

## Essential skill gaps
(Use EXACTLY these labels when referencing skills in the "addresses" field)
{gap_list}

---

## Your task

Step 1 — silently infer the user's seniority level (e.g. junior, mid-level, senior, staff, principal) from their profile and existing skills. Every action item must be calibrated to that level in terms of complexity, autonomy, and scope.

Step 2 — generate a development plan across three sections: experience, social, formal.
- Each item MUST reference one or more skills from the gap list above using their exact labels.
- Focus on the most impactful gaps — you do not need to cover every gap in every section.
- Within each section, order items by impact: the item that will give the user the greatest career leverage comes first.
- Section constraints:
  - experience: exactly 2 items — one for the first half (Month 1–6) and one for the second half (Month 7–12). Each experience item may span up to 6 months.
  - social: up to 3 items.
  - formal: up to 3 items.

Step 3 — assign a realistic schedule to every item:
- The entire plan must fit within 12 months (Month 1 through Month 12).
- experience items: first item within Month 1–6, second item within Month 7–12, each up to 6 months long.
- social and formal items: each takes a maximum of 3 months.
- All items must be fully independent — no item should require another item to be completed first. Each item must be actionable on its own from day one of its scheduled window.
- Overlaps between items are allowed, but the total concurrent load must remain achievable — do not schedule more than 2–3 items at the same time across all sections.
- Express the schedule as "Month X–Y" (e.g. "Month 1–3", "Month 7–12", "Month 4–6").

Return ONLY a JSON object with this exact structure (no markdown, no extra text):
{{
  "experience": [
    {{"title": "3–5 word summary", "action": "...", "addresses": ["exact skill label", ...], "schedule": "Month X–Y"}},
    ...
  ],
  "social": [
    {{"title": "3–5 word summary", "action": "...", "addresses": ["exact skill label", ...], "schedule": "Month X–Y"}},
    ...
  ],
  "formal": [
    {{"title": "3–5 word summary", "action": "...", "addresses": ["exact skill label", ...], "schedule": "Month X–Y"}},
    ...
  ]
}}

"title" must be a concise 3–5 word label that captures the essence of the action (e.g. "Lead cross-functional API project", "Shadow a senior architect", "Complete AWS Solutions Architect cert"). Used as a header and in the schedule timeline.

---

## Section-specific rules

**experience** — on-the-job projects, stretch assignments, responsibilities to seek:
- Describe what to do AND what a successful outcome looks like (be concrete: deliverables, decisions made, scope owned, measurable impact).
- Frame each item around the result the user should be able to demonstrate at the end, not just the activity.

**social** — peer learning, mentorship, community engagement:
- Focus on the specific competency the user should gain through the interaction, not the activity itself.
- Describe what kind of person to seek (what they should have done / be able to demonstrate), what to ask or observe, and how to convert the interaction into a lasting capability.
- Avoid generic advice like "join a community" — instead describe what the user should be able to do differently as a result.

**formal** — courses, certifications, books, structured learning:
- Name real, specific resources (actual course titles, cert programs, authors, platforms).
- For each resource, also describe how to apply the learning back into on-the-job experience: what project, task, or responsibility should the user tackle immediately after or alongside the formal learning to cement it.

**General rules for all "action" text:**
- Write 3–5 sentences of substantive, expert-level guidance — not a one-liner.
- Calibrate complexity and autonomy to the inferred seniority level.
- Do NOT mention skill names, do NOT include phrases like "to improve X", "to develop Y", "in order to build Z", or any reference to which skill is being addressed. The skill mapping is shown separately as tags."""

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_completion_tokens=2500,
    )
    raw = resp.choices[0].message.content or ""
    finish_reason = resp.choices[0].finish_reason
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, KeyError) as e:
        st.warning(f"⚠️ Development plan parse error: `{e}`")
        st.caption(f"finish_reason: `{finish_reason}`")
        with st.expander("Raw LLM response (debug)"):
            st.text(raw or "(empty)")
        return {"experience": [], "social": [], "formal": []}


# ── PDF Export ───────────────────────────────────────────────────────────────
def generate_pdf_report(
    name: str,
    current_occ: dict | None,
    target_occ: dict,
    user_matched_skills: list[dict],
    gap_essential: list[dict],
    plan: dict,
    target_is_inferred: bool = False,
) -> bytes:
    """Generate a styled Career Development Plan PDF using reportlab."""
    import io as _io
    from datetime import date
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, KeepTogether, HRFlowable,
    )

    # ── Palette ───────────────────────────────────────────────────────────────
    C_NAVY   = colors.HexColor("#1e3a5f")
    C_EXP    = colors.HexColor("#27ae60")
    C_SOC    = colors.HexColor("#e67e22")
    C_FORM   = colors.HexColor("#2980b9")
    C_GAP    = colors.HexColor("#c0392b")
    C_LIGHT  = colors.HexColor("#f4f6f9")
    C_BORDER = colors.HexColor("#d0d7e3")
    C_MUTED  = colors.HexColor("#7f8c8d")
    C_WHITE  = colors.white

    PAGE_W, PAGE_H = A4
    MARGIN = 2 * cm
    W = PAGE_W - 2 * MARGIN  # usable width

    buf = _io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
    )

    # ── Styles ────────────────────────────────────────────────────────────────
    base = getSampleStyleSheet()
    def sty(name, **kw):
        return ParagraphStyle(name, parent=base["Normal"], **kw)

    s_title    = sty("title",   fontSize=22, textColor=C_WHITE, leading=28, alignment=TA_LEFT)
    s_subtitle = sty("sub",     fontSize=10, textColor=C_WHITE, leading=14)
    s_h2       = sty("h2",      fontSize=13, textColor=C_NAVY,  spaceBefore=14, spaceAfter=6, leading=18)
    s_body     = sty("body",    fontSize=9,  leading=14, spaceAfter=4)
    s_small    = sty("small",   fontSize=8,  textColor=C_MUTED, leading=12)
    s_tag_exp  = sty("tag_exp", fontSize=7,  textColor=C_EXP)
    s_tag_soc  = sty("tag_soc", fontSize=7,  textColor=C_SOC)
    s_tag_form = sty("tag_form",fontSize=7,  textColor=C_FORM)
    s_chip_lbl = sty("chip",    fontSize=8,  textColor=C_WHITE, alignment=TA_CENTER)
    s_gap_type = sty("gtype",   fontSize=8,  textColor=C_GAP,   spaceBefore=6, spaceAfter=2)

    def colored_header_table(text_para, bg_color, padding=12):
        t = Table([[text_para]], colWidths=[W])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), bg_color),
            ("TOPPADDING",    (0,0), (-1,-1), padding),
            ("BOTTOMPADDING", (0,0), (-1,-1), padding),
            ("LEFTPADDING",   (0,0), (-1,-1), 14),
            ("RIGHTPADDING",  (0,0), (-1,-1), 14),
        ]))
        return t

    def schedule_chip(schedule_text, color):
        t = Table([[Paragraph(f"  {schedule_text}  ", s_chip_lbl)]], colWidths=[3.2*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), color),
            ("TOPPADDING",    (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
            ("LEFTPADDING",   (0,0), (-1,-1), 4),
            ("RIGHTPADDING",  (0,0), (-1,-1), 4),
        ]))
        return t

    story = []

    # ── Page 1: Cover ─────────────────────────────────────────────────────────
    today = date.today().strftime("%B %d, %Y")
    current_label = current_occ["preferredLabel"] if current_occ else "—"
    target_label  = target_occ["preferredLabel"]
    inferred_note = " (inferred)" if target_is_inferred else ""

    # Hero banner
    hero_inner = Table([
        [Paragraph("Career Development Plan", s_title)],
        [Paragraph(f"{name}  ·  Generated {today}", s_subtitle)],
    ], colWidths=[W - 28])
    story.append(colored_header_table(hero_inner, C_NAVY, padding=20))
    story.append(Spacer(1, 16))

    # Role transition card
    role_table = Table(
        [[
            Table([
                [Paragraph("Current Role", s_small)],
                [Paragraph(f"<b>{current_label}</b>", s_body)],
            ], colWidths=[W * 0.43]),
            Paragraph("<b>→</b>", sty("arrow", fontSize=18, alignment=TA_CENTER, textColor=C_MUTED)),
            Table([
                [Paragraph(f"Target Role{inferred_note}", s_small)],
                [Paragraph(f"<b>{target_label}</b>", s_body)],
            ], colWidths=[W * 0.43]),
        ]],
        colWidths=[W * 0.45, W * 0.10, W * 0.45],
    )
    role_table.setStyle(TableStyle([
        ("VALIGN",  (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN",   (1,0), (1,-1),  "CENTER"),
        ("BACKGROUND", (0,0), (0,-1), C_LIGHT),
        ("BACKGROUND", (2,0), (2,-1), colors.HexColor("#eaf4ff")),
        ("BOX",     (0,0), (0,-1), 0.5, C_BORDER),
        ("BOX",     (2,0), (2,-1), 0.5, C_BORDER),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
    ]))
    story.append(role_table)
    story.append(Spacer(1, 20))

    # ── Skill type color map (mirrors Streamlit UI) ───────────────────────────
    SKILL_TYPE_COLORS = {
        "knowledge":        colors.HexColor("#4e9af1"),
        "skill/competence": colors.HexColor("#27ae60"),
        "language":         colors.HexColor("#9b59b6"),
        "others":           colors.HexColor("#888888"),
    }

    def skill_table(skills_by_type: dict[str, list[str]]) -> Table:
        """Render skill groups as a two-column badge table: [type badge | pill row]."""
        TYPE_ORDER = ["knowledge", "skill/competence", "language", "others"]
        sorted_types = sorted(
            skills_by_type.items(),
            key=lambda kv: TYPE_ORDER.index(kv[0]) if kv[0] in TYPE_ORDER else 99,
        )
        rows = []
        for skill_type, labels in sorted_types:
            type_color = SKILL_TYPE_COLORS.get(skill_type, SKILL_TYPE_COLORS["others"])
            type_cell = Paragraph(
                f"<b>{skill_type.title()}</b>",
                sty(f"stc_{skill_type}", fontSize=7.5, textColor=C_WHITE, alignment=TA_CENTER),
            )
            # Skills as small pill table (3 per row)
            pill_rows = []
            row_buf = []
            for label in labels:
                row_buf.append(
                    Paragraph(label, sty("pill", fontSize=7.5, textColor=colors.black))
                )
                if len(row_buf) == 3:
                    pill_rows.append(row_buf)
                    row_buf = []
            if row_buf:  # pad last row
                while len(row_buf) < 3:
                    row_buf.append(Paragraph("", s_body))
                pill_rows.append(row_buf)

            pill_col_w = (W * 0.74) / 3
            pills_tbl = Table(pill_rows, colWidths=[pill_col_w] * 3)
            pill_style = [
                ("FONTSIZE",      (0,0), (-1,-1), 7.5),
                ("TOPPADDING",    (0,0), (-1,-1), 3),
                ("BOTTOMPADDING", (0,0), (-1,-1), 3),
                ("LEFTPADDING",   (0,0), (-1,-1), 5),
                ("RIGHTPADDING",  (0,0), (-1,-1), 5),
                ("BOX",           (0,0), (-1,-1), 0.4, C_BORDER),
                ("INNERGRID",     (0,0), (-1,-1), 0.2, C_BORDER),
                ("BACKGROUND",    (0,0), (-1,-1), C_LIGHT),
                ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ]
            pills_tbl.setStyle(TableStyle(pill_style))
            rows.append([type_cell, pills_tbl])

        tbl = Table(rows, colWidths=[W * 0.22, W * 0.78])
        tbl_style = [
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (0,-1),  6),
            ("RIGHTPADDING",  (0,0), (0,-1),  6),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [C_LIGHT, C_WHITE]),
        ]
        # Color each type badge cell individually
        for row_idx, (skill_type, _) in enumerate(sorted_types):
            c = SKILL_TYPE_COLORS.get(skill_type, SKILL_TYPE_COLORS["others"])
            tbl_style.append(("BACKGROUND", (0, row_idx), (0, row_idx), c))
        tbl.setStyle(TableStyle(tbl_style))
        return tbl

    # Skills you bring
    story.append(HRFlowable(width=W, color=C_BORDER))
    story.append(Paragraph("Skills You Bring", s_h2))
    if user_matched_skills:
        by_type: dict[str, list] = {}
        for m in user_matched_skills:
            t = (m.get("skillType") or "others").lower().strip()
            by_type.setdefault(t, []).append(m["esco_label"])
        story.append(skill_table(by_type))
    else:
        story.append(Paragraph("No matched skills found.", s_small))
    story.append(Spacer(1, 16))

    # Essential skill gaps
    story.append(HRFlowable(width=W, color=C_BORDER))
    story.append(Paragraph("Essential Skill Gaps to Close", s_h2))
    if gap_essential:
        by_type2: dict[str, list] = {}
        for s in gap_essential:
            t = (s.get("skillType") or "others").lower().strip()
            by_type2.setdefault(t, []).append(s["skillLabel"])
        story.append(skill_table(by_type2))
    else:
        story.append(Paragraph("No essential skill gaps — your profile already covers all requirements.", s_small))

    story.append(PageBreak())

    # ── Page 2: Gantt + Plan ──────────────────────────────────────────────────
    story.append(Paragraph("12-Month Development Schedule", s_h2))

    # Build item list for Gantt
    SECTION_META = [
        ("experience", "Experience (70%)", C_EXP),
        ("social",     "Social (20%)",     C_SOC),
        ("formal",     "Formal (10%)",     C_FORM),
    ]
    gantt_items = []
    for sec_key, sec_label, sec_color in SECTION_META:
        for item in plan.get(sec_key, []):
            schedule = item.get("schedule", "")
            m = re.search(r"(\d+)[^\d]+(\d+)", schedule)
            start = int(m.group(1)) if m else 1
            end   = int(m.group(2)) if m else 1
            end   = max(start, min(end, 12))
            title = item.get("title") or item.get("action", "")[:40]
            gantt_items.append({
                "label": title,
                "section": sec_label[:3],
                "start": start, "end": end,
                "color": sec_color,
            })

    if gantt_items:
        LABEL_W  = W * 0.38
        MONTH_W  = (W * 0.62) / 12
        ROW_H    = 18

        # Header row: label + M1…M12
        header_row = [Paragraph("<b>Activity</b>",
                                sty("gh", fontSize=7, textColor=C_WHITE))]
        for i in range(1, 13):
            header_row.append(Paragraph(f"<b>M{i}</b>",
                                        sty("gm", fontSize=7, textColor=C_WHITE, alignment=TA_CENTER)))

        gantt_data = [header_row]
        span_cmds  = [
            ("BACKGROUND", (0,0), (-1,0), C_NAVY),
            ("GRID",       (0,0), (-1,-1), 0.3, C_BORDER),
            ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
            ("LEFTPADDING",   (0,0), (-1,-1), 4),
            ("RIGHTPADDING",  (0,0), (-1,-1), 4),
        ]

        for r_idx, gi in enumerate(gantt_items, start=1):
            row = [Paragraph(gi["label"],
                             sty("gl", fontSize=6.5, textColor=colors.black))]
            for m in range(1, 13):
                if gi["start"] <= m <= gi["end"]:
                    row.append(Paragraph("", s_body))
                else:
                    row.append("")
            gantt_data.append(row)

            # Colored bar span
            col_start = gi["start"]      # 1-based col index (label=0, M1=1)
            col_end   = gi["end"]
            if col_start != col_end:
                span_cmds.append(("SPAN",       (col_start, r_idx), (col_end, r_idx)))
            span_cmds.append(("BACKGROUND", (col_start, r_idx), (col_end, r_idx), gi["color"]))
            # Alternate row background for non-bar cells
            bg = C_LIGHT if r_idx % 2 == 0 else C_WHITE
            for m_col in range(0, 13):
                if not (col_start <= m_col <= col_end):
                    span_cmds.append(("BACKGROUND", (m_col, r_idx), (m_col, r_idx), bg))

        col_widths = [LABEL_W] + [MONTH_W] * 12
        gantt_table = Table(gantt_data, colWidths=col_widths, rowHeights=ROW_H)
        gantt_table.setStyle(TableStyle(span_cmds))
        story.append(gantt_table)

    story.append(Spacer(1, 24))

    # ── Development Plan sections ─────────────────────────────────────────────
    tag_styles = {"experience": s_tag_exp, "social": s_tag_soc, "formal": s_tag_form}

    for sec_key, sec_label, sec_color in SECTION_META:
        items = plan.get(sec_key, [])
        if not items:
            continue

        story.append(KeepTogether([
            colored_header_table(
                Paragraph(f"<b>{sec_label}</b>",
                          sty(f"sh_{sec_key}", fontSize=12, textColor=C_WHITE)),
                sec_color, padding=10,
            ),
            Spacer(1, 8),
        ]))

        for item in items:
            schedule  = item.get("schedule", "")
            title     = item.get("title", "")
            action    = item.get("action", "")
            addresses = item.get("addresses", [])
            t_style   = tag_styles[sec_key]

            s_item_title = sty(
                f"it_{sec_key}", fontSize=10, textColor=C_NAVY,
                spaceBefore=4, spaceAfter=3, leading=14,
            )
            block = [
                schedule_chip(f"  {schedule}", sec_color),
                Spacer(1, 5),
            ]
            if title:
                block.append(Paragraph(f"<b>{title}</b>", s_item_title))
            block.append(Paragraph(action, s_body))
            if addresses:
                block.append(Paragraph(
                    "  ".join(f"[ {a} ]" for a in addresses),
                    t_style,
                ))
            block.append(Spacer(1, 10))
            story.append(KeepTogether(block))

        story.append(Spacer(1, 8))

    doc.build(story)
    return buf.getvalue()


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ESCO Skill Gap Analyzer", page_icon="🎯", layout="wide")
st.title("🎯 ESCO Skill Gap Analyzer")
st.caption("Powered by ESCO v1.2.1 + Azure OpenAI")

# Check embeddings exist
if not os.path.exists(os.path.join(EMBEDDINGS_DIR, "occupations_embeddings.npy")):
    st.error(
        "Embedding files not found. Please run `python embed_esco.py` first, "
        "then restart this app."
    )
    st.stop()

occ_emb, occ_meta, skill_emb, skill_meta, relations, occ_detail = load_esco_data()

# ── README / Guide ────────────────────────────────────────────────────────────
with st.expander("📖 How to use this tool", expanded=False):
    st.markdown("""
## Welcome to the ESCO Skill Gap Analyzer

This tool helps you understand what skills you need to reach your target role — and delivers a concrete, personalised development plan to get there. Powered by the **ESCO v1.2.1** European Skills/Competences taxonomy and **Azure OpenAI**.

---

### ✍️ How to fill in the inputs

You can use **either or both** input fields:

| Mode | What you fill in | What you get |
|------|-----------------|--------------|
| **Current only** | Your current role & skills | System infers your most natural next career step, then produces a full gap analysis and plan toward it |
| **Target only** | The role you want to reach | A complete skill roadmap starting from scratch — no current profile needed |
| **Both** | Current + target | Gap analysis between your existing skills and the target role, with a tailored development plan |

You can type a free-text description **or upload a PDF / DOCX file** (e.g. your résumé for Current, a job description for Target). The more detail you provide, the more accurate the results.

---

### 📦 What each output section means

**🔍 Extracted Information**
What the AI parsed from your input — your current job title and extracted skills list, and/or your target title. Review this to confirm the AI understood your input correctly before proceeding.

**🏷 Role Match**
Your input is matched against ~3,000 ESCO occupations using semantic similarity. The best-matching occupation is shown with its official definition, description, alternative labels, and scope note.

**📋 Skills for Target Role**
The official ESCO skill list for your matched target occupation, split into **Essential** (must-have) and **Optional** (nice-to-have), grouped by skill type (Knowledge / Skill & Competence / Language).

**✅ Skills You Already Have**
Your extracted skills matched against the full ESCO skill vocabulary (cosine similarity ≥ 0.50). Each matched skill shows its ESCO label, similarity score, and whether it is required by the target role (**Required ✅**).

> ⚠️ Similarity is computed on full ESCO skill embeddings (label + description + scope note), not just the skill name — so even an exact name match will score below 1.0. Multiple ESCO skills above the threshold may be matched per extracted skill; duplicates are deduplicated by URI, keeping the highest score.

**❌ Skill Gaps**
Target role skills **not** covered by your matched profile — the gap to close. Grouped by Essential / Optional → skill type.

**🗺 Development Plan**
A personalised **70-20-10** plan calibrated to your inferred seniority level, scheduled across a 12-month horizon:
- 🛠 **Experience (70%)** — 2 substantial on-the-job assignments (one per half-year), outcome-oriented with clear deliverables
- 🤝 **Social (20%)** — up to 3 competency-focused peer learning or mentorship engagements
- 📚 **Formal (10%)** — up to 3 specific courses, certifications, or books, each with guidance on applying the learning back on the job

Each item has a **title**, a suggested schedule (🗓 Month X–Y), a detailed action description, and skill gap tags. All items are fully independent.

**📄 Export Development Plan**
After the plan is generated, enter your name and download a structured PDF — includes your profile summary, skill gaps, a 12-month Gantt chart, and the full development plan with coloured sections.

---

### 💡 Tips
- The more detail you provide (years of experience, tools used, past projects), the more accurately the system infers your seniority level and tailors the plan.
- Uploading a full résumé or job description gives better results than a one-line summary.
- Skill gaps are computed by exact ESCO URI matching — not inferred by AI — so the gap list is precise and auditable.
""")

# ── Input section ─────────────────────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    st.subheader("Current State *(optional)*")
    current_file = st.file_uploader(
        "Upload resume / profile (PDF or DOCX)",
        type=["pdf", "docx"],
        key="current_file",
    )
    if current_file:
        current_file_text = extract_text_from_file(current_file)
        if current_file_text:
            st.caption(f"Extracted {len(current_file_text):,} chars from **{current_file.name}**")
            with st.expander("Preview extracted text"):
                st.text(current_file_text[:1000] + ("…" if len(current_file_text) > 1000 else ""))
        else:
            st.warning("Could not extract text from this file.")
        current_text = current_file_text
    else:
        current_text = st.text_area(
            "Or describe your current role and skills",
            placeholder=(
                "e.g. I'm a backend software developer with 3 years of experience. "
                "I work mainly with Python, FastAPI, PostgreSQL, and Docker. "
                "I've built several REST APIs and have some experience with AWS."
            ),
            height=180,
            key="current_input",
        )

with col2:
    st.subheader("Target State *(optional)*")
    target_file = st.file_uploader(
        "Upload job description (PDF or DOCX)",
        type=["pdf", "docx"],
        key="target_file",
    )
    if target_file:
        target_file_text = extract_text_from_file(target_file)
        if target_file_text:
            st.caption(f"Extracted {len(target_file_text):,} chars from **{target_file.name}**")
            with st.expander("Preview extracted text"):
                st.text(target_file_text[:1000] + ("…" if len(target_file_text) > 1000 else ""))
        else:
            st.warning("Could not extract text from this file.")
        target_text = target_file_text
    else:
        target_text = st.text_area(
            "Or describe the role you want to move towards",
            placeholder=(
                "e.g. I want to become a software architect or tech lead, "
                "responsible for system design and leading a small engineering team."
            ),
            height=180,
            key="target_input",
        )

analyze_btn = st.button("🔍 Analyze Skill Gap", type="primary", use_container_width=True)

# ── Analysis ──────────────────────────────────────────────────────────────────
if analyze_btn:
    has_current = bool(current_text.strip())
    has_target = bool(target_text.strip())

    if not has_current and not has_target:
        st.warning("Please fill in at least one field before analyzing.")
        st.stop()

    # Determine mode
    if has_current and has_target:
        mode = "both"
    elif has_current:
        mode = "current_only"
    else:
        mode = "target_only"

    st.divider()

    # ── Step 1: Parse inputs ──────────────────────────────────────────────────
    current_occ = None
    current_occ_skills = []
    extracted_skills = []
    target_is_inferred = False

    with st.spinner("Parsing your input…"):
        if has_current:
            parsed_current = llm_parse_current(current_text)
            extracted_title = parsed_current.get("current_title", "")
            extracted_skills = parsed_current.get("current_skills", [])
        if has_target:
            parsed_target = llm_parse_target(target_text)
            target_title = parsed_target.get("target_title", "")

    with st.expander("Extracted Information", expanded=False):
        ei_cols = st.columns(2) if (has_current and has_target) else [st.container()]
        col_idx = 0
        if has_current:
            with ei_cols[col_idx]:
                st.markdown("**Current Title**")
                st.markdown(f"`{extracted_title}`")
                st.markdown("**Extracted Skills**")
                # Render skills as wrapped badges using markdown spans
                badge_html = " ".join(
                    f'<span style="display:inline-block;border:1px solid rgba(128,128,128,0.45);'
                    f'border-radius:4px;padding:2px 8px;margin:2px;font-size:0.82em">{s}</span>'
                    for s in extracted_skills
                )
                st.markdown(badge_html or "_none_", unsafe_allow_html=True)
            col_idx += 1
        if has_target:
            with ei_cols[col_idx] if has_current else ei_cols[0]:
                st.markdown("**Target Title**")
                st.markdown(f"`{target_title}`")

    # ── Step 2: Match current occupation (if provided) ────────────────────────
    if has_current:
        with st.spinner("Matching current occupation in ESCO…"):
            current_occ_candidates = semantic_search_occupations(
                extracted_title, occ_emb, occ_meta, top_k=TOP_OCC_CANDIDATES
            )
        current_occ = current_occ_candidates[0][0]
        current_occ_skills = relations.get(current_occ["conceptUri"], [])

        st.subheader("Current Role Match")
        best_occ, best_score = current_occ_candidates[0]
        render_occ_match(best_occ, best_score, occ_detail.get(best_occ["conceptUri"], {}))

    # ── Step 3: Determine target occupation ───────────────────────────────────
    if has_target:
        with st.spinner("Matching target occupation in ESCO…"):
            target_occ_candidates = semantic_search_occupations(
                target_title, occ_emb, occ_meta, top_k=TOP_OCC_CANDIDATES
            )
        target_occ = target_occ_candidates[0][0]
    else:
        # current_only: infer next role from LLM, then ESCO-match it
        with st.spinner("Inferring your next career step…"):
            inferred_title = llm_infer_next_role(
                extracted_title, current_occ["preferredLabel"], current_occ_skills
            )
            target_occ_candidates = semantic_search_occupations(
                inferred_title, occ_emb, occ_meta, top_k=TOP_OCC_CANDIDATES
            )
        target_occ = target_occ_candidates[0][0]
        target_is_inferred = True
        st.info(f"No target provided — inferred next step: **{inferred_title}**")

    target_occ_skills = relations.get(target_occ["conceptUri"], [])

    st.subheader("Target Role Match" + (" *(inferred)*" if target_is_inferred else ""))
    best_tgt, best_tgt_score = target_occ_candidates[0]
    render_occ_match(best_tgt, best_tgt_score, occ_detail.get(best_tgt["conceptUri"], {}))

    # ── Shared display constants ───────────────────────────────────────────────
    N_COLS = 3
    SKILL_TYPE_ICON = {
        "knowledge": "📖",
        "skill/competence": "🔧",
        "language": "🌐",
        "others": "📎",
    }
    SKILL_TYPE_COLOR = {
        "knowledge": "#4e9af1",
        "skill/competence": "#2ecc71",
        "language": "#9b59b6",
        "others": "#888888",
    }
    TYPE_SORT_ORDER = ["knowledge", "skill/competence", "language", "others"]

    def normalize_type(t: str) -> str:
        """Return 'others' for empty/unknown skill types."""
        return t.strip() if t and t.strip() else "others"

    def sort_types(groups: dict) -> list[tuple]:
        """Return (type, items) pairs sorted by TYPE_SORT_ORDER, 'others' last."""
        def key(k):
            try:
                return TYPE_SORT_ORDER.index(k)
            except ValueError:
                return len(TYPE_SORT_ORDER)
        return sorted(groups.items(), key=lambda kv: key(kv[0]))

    def skill_type_header(skill_type: str, count: int):
        icon = SKILL_TYPE_ICON.get(skill_type, "📎")
        color = SKILL_TYPE_COLOR.get(skill_type, "#888888")
        st.markdown(
            f'<div style="border-left:4px solid {color};padding:2px 10px;margin:6px 0 4px 0">'
            f'<strong>{icon} {skill_type}</strong>'
            f'<span style="opacity:0.6;font-size:0.85em"> — {count} item(s)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    def render_typed_group(group: list[dict]):
        """Render a skillType-grouped 3-col grid (shared by skill list and gap section)."""
        by_type: dict[str, list] = {}
        for s in group:
            by_type.setdefault(normalize_type(s["skillType"]), []).append(s)
        for skill_type, items in sort_types(by_type):
            skill_type_header(skill_type, len(items))
            rows = [items[i:i + N_COLS] for i in range(0, len(items), N_COLS)]
            for row in rows:
                cols = st.columns(N_COLS)
                for col, s in zip(cols, row):
                    col.markdown(s["skillLabel"])

    def render_skill_list(skills: list[dict], _title: str):
        """Two separate expanders (Essential / Optional) — avoids Streamlit nested expander limit."""
        essential = [s for s in skills if s["relationType"] == "essential"]
        optional = [s for s in skills if s["relationType"] != "essential"]
        if essential:
            with st.expander(f"🔴 Essential ({len(essential)})", expanded=False):
                render_typed_group(essential)
        if optional:
            with st.expander(f"🔵 Optional ({len(optional)})", expanded=False):
                render_typed_group(optional)

    # ── Step 4: Display ESCO skill list (target role only) ───────────────────
    essential_target = [s for s in target_occ_skills if s["relationType"] == "essential"]
    st.subheader(f"Skills for: {target_occ['preferredLabel']}")
    st.caption(f"{len(essential_target)} essential / {len(target_occ_skills) - len(essential_target)} optional")
    render_skill_list(target_occ_skills, f"View all {len(target_occ_skills)} skills")

    # ── Step 5: Match user's skills to ESCO ───────────────────────────────────
    # For each extracted skill, keep ALL ESCO matches above threshold (not just top-1).
    # Deduplicate by ESCO URI — retain the entry with the highest similarity score.
    user_matched_skills = []
    if extracted_skills:
        best_by_uri: dict[str, dict] = {}
        with st.spinner("Matching your skills to ESCO vocabulary…"):
            for user_skill in extracted_skills:
                matches = semantic_search_skills(user_skill, skill_emb, skill_meta, top_k=10)
                for esco_skill, score in matches:
                    if score < SKILL_MATCH_THRESHOLD:
                        break  # results are sorted descending; no need to check further
                    uri = esco_skill["conceptUri"]
                    if uri not in best_by_uri or score > best_by_uri[uri]["score"]:
                        best_by_uri[uri] = {
                            "user_skill": user_skill,
                            "esco_label": esco_skill["preferredLabel"],
                            "esco_uri": uri,
                            "skillType": esco_skill["skillType"],
                            "score": score,
                        }
        user_matched_skills = list(best_by_uri.values())

    # Sort by similarity descending
    user_matched_skills.sort(key=lambda m: m["score"], reverse=True)

    # Annotate whether each matched skill is required by the target role
    target_skill_uris = {s["skillUri"] for s in target_occ_skills}
    for m in user_matched_skills:
        m["in_target"] = m["esco_uri"] in target_skill_uris

    # Derive display names for roles (used throughout gap analysis section)
    cur_name = current_occ["preferredLabel"] if current_occ else None
    tgt_name = target_occ["preferredLabel"]

    # ── Step 6: Programmatic gap computation (target role only) ──────────────
    gap_skills = compute_gap_merged(
        user_matched_skills, [], target_occ_skills,
        current_name=cur_name or "current role",
        target_name=tgt_name,
    )
    gap_essential = [s for s in gap_skills if s["relationType"] == "essential"]
    gap_optional = [s for s in gap_skills if s["relationType"] != "essential"]

    # ── Step 7: Structured display ────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Gap Analysis")

    # Skills You Already Have
    st.markdown("### ✅ Skills You Already Have")
    st.caption(
        f"Matched from your profile to ESCO skill vocabulary (similarity ≥ {SKILL_MATCH_THRESHOLD}).  \n"
        "⚠️ Note: similarity is computed on the full ESCO skill embedding "
        "(label + description + scope note), not just the skill name — "
        "so even an exact name match will score below 1.0."
    )
    if user_matched_skills:
        with st.expander(f"View {len(user_matched_skills)} matched skill(s)", expanded=False):
            h1, h2, h3, h4 = st.columns([3, 3, 1, 1])
            h1.markdown("**Your description**")
            h2.markdown("**ESCO skill**")
            h3.markdown("**Required**")
            h4.markdown("**Sim.**")
            for m in user_matched_skills:
                c1, c2, c3, c4 = st.columns([3, 3, 1, 1])
                c1.markdown(f"`{m['user_skill']}`")
                c2.markdown(m["esco_label"])
                c3.markdown("✅" if m["in_target"] else "—")
                c4.markdown(f"`{m['score']:.2f}`")
    else:
        st.info(f"No skills from your profile could be matched above the threshold ({SKILL_MATCH_THRESHOLD}).")

    # Skill Gaps
    st.markdown("### ❌ Skill Gaps")
    st.caption(
        f"{len(gap_essential)} essential gap(s)  ·  {len(gap_optional)} optional gap(s)  "
        f"— target skills not found in your matched profile"
    )
    def render_gap_section(skills: list[dict], relation_label: str, relation_icon: str, expanded: bool = True):
        if not skills:
            return
        with st.expander(f"{relation_icon} {relation_label} ({len(skills)})", expanded=expanded):
            render_typed_group(skills)

    if gap_skills:
        render_gap_section(gap_essential, "Essential (must-have)", "🔴", expanded=True)
        render_gap_section(gap_optional, "Optional (nice-to-have)", "🔵", expanded=False)
    else:
        st.success("No skill gaps detected — your profile already covers all target role skills!")

    # Development Plan
    st.markdown("### 🗺 Development Plan")
    st.caption("Based on essential skill gaps only · structured as Experience / Social / Formal (70-20-10)")
    if not gap_essential:
        st.success("No essential skill gaps — no development plan needed!")
    else:
        with st.spinner("Generating development plan…"):
            plan = llm_development_plan(
                target_occ=target_occ,
                essential_gaps=gap_essential,
                user_matched_skills=user_matched_skills,
                current_occ=current_occ,
                target_is_inferred=target_is_inferred,
                user_profile_text=current_text if has_current else "",
            )

        # Debug: show plan keys if all sections are empty
        if not any(plan.get(k) for k in ("experience", "social", "formal")):
            st.warning("⚠️ Development plan returned empty — showing raw response for debugging.")
            st.json(plan)

        # Build a set of valid gap labels for badge validation
        gap_label_set = {s["skillLabel"] for s in gap_essential}

        def render_plan_section(items: list[dict], header: str):
            if not items:
                return
            st.markdown(f"#### {header}")
            for item in items:
                title     = item.get("title", "")
                action    = item.get("action", "")
                addresses = [a for a in item.get("addresses", []) if a in gap_label_set]
                schedule  = item.get("schedule", "")
                schedule_chip = (
                    f'<span style="display:inline-block;border:1px solid rgba(128,128,128,0.4);'
                    f'border-radius:4px;padding:1px 8px;margin-right:8px;font-size:0.8em;'
                    f'opacity:0.75">🗓 {schedule}</span>'
                    if schedule else ""
                )
                title_html = f'<strong style="font-size:1em">{title}</strong>' if title else ""
                st.markdown(
                    f'<div style="margin:12px 0 3px 0">{schedule_chip}{title_html}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(action)
                if addresses:
                    badge_html = " ".join(
                        f'<span style="display:inline-block;border:1px solid rgba(78,154,241,0.6);'
                        f'color:#4e9af1;border-radius:4px;padding:1px 7px;margin:1px;font-size:0.78em">'
                        f'{label}</span>'
                        for label in addresses
                    )
                    st.markdown(
                        f'<div style="margin:2px 0 10px 0">{badge_html}</div>',
                        unsafe_allow_html=True,
                    )

        render_plan_section(plan.get("experience", []), "🛠 Experience (70%)")
        render_plan_section(plan.get("social", []),     "🤝 Social (20%)")
        render_plan_section(plan.get("formal", []),     "📚 Formal (10%)")

        # Store export data in session_state so it survives reruns
        st.session_state["pdf_export"] = {
            "current_occ": current_occ,
            "target_occ": target_occ,
            "user_matched_skills": user_matched_skills,
            "gap_essential": gap_essential,
            "plan": plan,
            "target_is_inferred": target_is_inferred,
        }

    # Legend
    st.divider()
    st.caption("🔴 Essential skill  |  🔵 Optional skill  |  Similarity scores are cosine similarity (0–1)")

# ── PDF Export (outside analyze block so it survives reruns) ──────────────────
if "pdf_export" in st.session_state:
    st.divider()
    st.markdown("#### 📄 Export Development Plan")
    export_name = st.text_input(
        "Your name (for the PDF)",
        placeholder="e.g. Jane Smith",
        key="export_name",
    )
    if export_name.strip():
        pdf_bytes = generate_pdf_report(
            name=export_name.strip(),
            **st.session_state["pdf_export"],
        )
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name=f"development_plan_{export_name.strip().replace(' ', '_')}.pdf",
            mime="application/pdf",
        )
    else:
        st.caption("Enter your name above to enable the download button.")
