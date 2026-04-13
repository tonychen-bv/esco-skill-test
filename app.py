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
    occ_csv_path = os.path.join(ESCO_DATA_DIR, "occupations_en.csv")
    if os.path.exists(occ_csv_path):
        occ_df = pd.read_csv(occ_csv_path, usecols=[
            "conceptUri", "altLabels", "definition", "scopeNote", "description",
        ])
        for _, row in occ_df.iterrows():
            occ_detail[str(row["conceptUri"])] = {
                "altLabels": str(row["altLabels"]).strip() if pd.notna(row["altLabels"]) else "",
                "definition": str(row["definition"]).strip() if pd.notna(row["definition"]) else "",
                "scopeNote": str(row["scopeNote"]).strip() if pd.notna(row["scopeNote"]) else "",
                "description": str(row["description"]).strip() if pd.notna(row["description"]) else "",
            }

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
) -> dict:
    """
    Generate a 70-20-10 development plan focused on essential skill gaps.

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

    prompt = f"""You are an expert career development advisor using the 70-20-10 development model.

Context:
- {current_context}
- Target role{inferred_note}: {target_occ.get('preferredLabel', 'Unknown')}

Skills the user already has:
{fmt_have(user_matched_skills)}

Essential skill gaps (use EXACTLY these labels when referencing skills in your output):
{gap_list}

Create a development plan with three sections: experience, social, formal.
Each section has 3–5 action items. Each item MUST reference one or more skills from the gap list above using their exact labels.
Focus on the most impactful gaps — you do not need to cover every gap in every section.

Return ONLY a JSON object with this exact structure (no markdown, no extra text):
{{
  "experience": [
    {{"action": "...", "addresses": ["exact skill label", ...]}},
    ...
  ],
  "social": [
    {{"action": "...", "addresses": ["exact skill label", ...]}},
    ...
  ],
  "formal": [
    {{"action": "...", "addresses": ["exact skill label", ...]}},
    ...
  ]
}}

experience: on-the-job projects, stretch assignments, or responsibilities to seek.
social: mentors to find, communities to join, shadowing or peer-learning approaches.
formal: specific courses, certifications, books, or platforms (name actual resources).

Rules for "action" text:
- Write 3–5 sentences of substantive, expert-level guidance — not a one-liner.
- Be specific and professional: explain what exactly to do, how to approach it, what good execution looks like, and what outcome or competency gain to expect.
- Tailor the depth to someone who is serious about career growth and can handle nuanced advice.
- Do NOT mention skill names, do NOT include phrases like "to improve X", "to develop Y", "in order to build Z", or any reference to which skill is being addressed. The skill mapping is shown separately as tags."""

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_completion_tokens=2500,
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, KeyError):
        # Fallback: return empty structure so UI doesn't crash
        return {"experience": [], "social": [], "formal": []}


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

This tool helps you understand where you stand today and what it takes to reach your next career goal — powered by the **ESCO v1.2.1** European Skills/Competences taxonomy and **Azure OpenAI**.

---

### ✍️ How to fill in the inputs

You can use **either or both** input fields. Each mode produces different output:

| Mode | What you fill in | What you get |
|------|-----------------|--------------|
| **Current only** | Your current role & skills | System infers your most natural next career step, then runs a full gap analysis toward it |
| **Target only** | The role you want to reach | A complete skill roadmap starting from scratch — no current profile needed |
| **Both** | Current + target | A precise gap analysis comparing where you are to where you want to be |

You can type a free-text description **or upload a PDF / DOCX file** (e.g. your résumé for Current, a job description for Target).

---

### 📦 What each output section means

**🔍 Extracted Information**
What the AI parsed from your input — your current job title, extracted skills list, and/or target title. Review this to confirm the AI understood your input correctly.

**🏷 Current / Target Role Match**
Your input is matched against all ~3,000 ESCO occupations using semantic similarity. The best-matching ESCO occupation is shown with its official definition, description, alternative labels, and scope note.

**📋 Skills for [Role]**
The official ESCO skill list for each matched occupation, split into **Essential** (must-have) and **Optional** (nice-to-have), further grouped by skill type (Knowledge / Skill & Competence / Language).

**✅ Skills You Already Have**
Your extracted skills matched against the full ESCO skill vocabulary (threshold ≥ 0.50 cosine similarity). Shows which ESCO skill each of your skills maps to, and whether it appears in your current role, target role, or both.

> ⚠️ Scores are computed on full ESCO skill embeddings (label + description + scope note), so even an exact name match will score below 1.0.

**❌ Skill Gaps**
Skills required by your current and/or target role that are **not** found in your matched profile — the actual gap you need to close. Grouped by Essential / Optional → role source → skill type.

**🗺 Development Plan**
A personalised **70-20-10** learning plan to close your essential skill gaps:
- 🛠 **Experience (70%)** — on-the-job projects and stretch assignments
- 🤝 **Social (20%)** — mentors, communities, peer learning
- 📚 **Formal (10%)** — courses, certifications, books, platforms

Each action item is tagged with the specific skill gaps it addresses.

---

### 💡 Tips
- The more detail you provide in your description, the more accurate the skill extraction.
- Uploading a full résumé or job description gives better results than a one-line summary.
- The skill gap is computed programmatically (URI matching), not inferred by the AI — so it's precise.
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

    # ── Step 4: Display ESCO skill lists ──────────────────────────────────────
    skill_cols = st.columns(2) if has_current else [st.container()]

    if has_current:
        essential_current = [s for s in current_occ_skills if s["relationType"] == "essential"]
        with skill_cols[0]:
            st.subheader(f"Skills for: {current_occ['preferredLabel']}")
            st.caption(f"{len(essential_current)} essential / {len(current_occ_skills) - len(essential_current)} optional")
            render_skill_list(current_occ_skills, f"View all {len(current_occ_skills)} skills")

    essential_target = [s for s in target_occ_skills if s["relationType"] == "essential"]
    target_col = skill_cols[1] if has_current else skill_cols[0]
    with target_col:
        st.subheader(f"Skills for: {target_occ['preferredLabel']}")
        st.caption(f"{len(essential_target)} essential / {len(target_occ_skills) - len(essential_target)} optional")
        render_skill_list(target_occ_skills, f"View all {len(target_occ_skills)} skills")

    # ── Step 5: Match user's skills to ESCO ───────────────────────────────────
    user_matched_skills = []
    if extracted_skills:
        with st.spinner("Matching your skills to ESCO vocabulary…"):
            for user_skill in extracted_skills:
                matches = semantic_search_skills(user_skill, skill_emb, skill_meta, top_k=1)
                if matches:
                    esco_skill, score = matches[0]
                    if score >= SKILL_MATCH_THRESHOLD:
                        user_matched_skills.append(
                            {
                                "user_skill": user_skill,
                                "esco_label": esco_skill["preferredLabel"],
                                "esco_uri": esco_skill["conceptUri"],
                                "skillType": esco_skill["skillType"],
                                "score": score,
                            }
                        )

    # Sort by similarity descending
    user_matched_skills.sort(key=lambda m: m["score"], reverse=True)

    # Derive display names for roles (used throughout gap analysis section)
    cur_name = current_occ["preferredLabel"] if current_occ else None
    tgt_name = target_occ["preferredLabel"]
    both_name = f"{cur_name} + {tgt_name}" if cur_name else tgt_name

    # Annotate each matched skill with which role(s) it appears in
    current_skill_uris = {s["skillUri"] for s in current_occ_skills}
    target_skill_uris = {s["skillUri"] for s in target_occ_skills}
    for m in user_matched_skills:
        in_cur = m["esco_uri"] in current_skill_uris
        in_tgt = m["esco_uri"] in target_skill_uris
        if in_cur and in_tgt:
            m["in_roles"] = both_name
        elif in_cur:
            m["in_roles"] = cur_name
        elif in_tgt:
            m["in_roles"] = tgt_name
        else:
            m["in_roles"] = "—"

    # ── Step 6: Programmatic gap computation ──────────────────────────────────
    gap_skills = compute_gap_merged(
        user_matched_skills, current_occ_skills, target_occ_skills,
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
            h1, h2, h3, h4 = st.columns([3, 3, 2, 1])
            h1.markdown("**Your description**")
            h2.markdown("**ESCO skill**")
            h3.markdown("**In role**")
            h4.markdown("**Sim.**")
            for m in user_matched_skills:
                c1, c2, c3, c4 = st.columns([3, 3, 2, 1])
                c1.markdown(f"`{m['user_skill']}`")
                c2.markdown(m["esco_label"])
                c3.markdown(m["in_roles"])
                c4.markdown(f"`{m['score']:.2f}`")
    else:
        st.info(f"No skills from your profile could be matched above the threshold ({SKILL_MATCH_THRESHOLD}).")

    # Skill Gaps
    st.markdown("### ❌ Skill Gaps")
    st.caption(
        f"{len(gap_essential)} essential gap(s)  ·  {len(gap_optional)} optional gap(s)  "
        f"— target skills not found in your matched profile"
    )
    # Source ordering and icons use actual role names resolved above
    source_order = (
        [cur_name, tgt_name, both_name] if cur_name else [tgt_name]
    )
    source_icon = {
        cur_name: "👤",
        tgt_name: "🎯",
        both_name: "👥",
    }

    def render_gap_section(skills: list[dict], relation_label: str, relation_icon: str, expanded: bool = True):
        if not skills:
            return
        by_source: dict[str, list[dict]] = {}
        for s in skills:
            by_source.setdefault(s["source"], []).append(s)
        with st.expander(f"{relation_icon} {relation_label} ({len(skills)})", expanded=expanded):
            for source in source_order:
                items_in_source = by_source.get(source, [])
                if not items_in_source:
                    continue
                icon = source_icon.get(source, "•")
                st.markdown(f"**{icon} {source}** ({len(items_in_source)})")
                render_typed_group(items_in_source)
                st.markdown("")  # spacer between sources

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
            )

        # Build a set of valid gap labels for badge validation
        gap_label_set = {s["skillLabel"] for s in gap_essential}

        def render_plan_section(items: list[dict], header: str):
            if not items:
                return
            st.markdown(f"#### {header}")
            for item in items:
                action = item.get("action", "")
                addresses = [a for a in item.get("addresses", []) if a in gap_label_set]
                st.markdown(f"- {action}")
                if addresses:
                    badge_html = " ".join(
                        f'<span style="display:inline-block;border:1px solid rgba(78,154,241,0.6);'
                        f'color:#4e9af1;border-radius:4px;padding:1px 7px;margin:1px;font-size:0.78em">'
                        f'{label}</span>'
                        for label in addresses
                    )
                    st.markdown(
                        f'<div style="margin:-6px 0 6px 20px">{badge_html}</div>',
                        unsafe_allow_html=True,
                    )

        render_plan_section(plan.get("experience", []), "🛠 Experience (70%)")
        render_plan_section(plan.get("social", []),     "🤝 Social (20%)")
        render_plan_section(plan.get("formal", []),     "📚 Formal (10%)")

    # Legend
    st.divider()
    st.caption("🔴 Essential skill  |  🔵 Optional skill  |  Similarity scores are cosine similarity (0–1)")
