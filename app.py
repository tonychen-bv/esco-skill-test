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
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
EMBEDDINGS_DIR = os.path.join(os.path.dirname(__file__), "esco_embeddings")
EMBEDDING_MODEL = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
LLM_MODEL = os.getenv("AZURE_LLM_DEPLOYMENT", "gpt-4.1-mini")
EMBEDDING_DIM = 256
SKILL_MATCH_THRESHOLD = 0.50
TOP_OCC_CANDIDATES = 3   # show top-N occupation matches for user to confirm

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

    return occ_emb, occ_meta, skill_emb, skill_meta, relations


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
        max_tokens=20,
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


def llm_learning_recommendations(
    target_occ: dict,
    gap_skills: list[dict],
    user_matched_skills: list[dict],
    current_occ: dict | None = None,
    target_is_inferred: bool = False,
) -> str:
    """Generate prioritized learning recommendations based on the pre-computed gap."""

    def fmt_gap(skills: list[dict], max_n: int = 40) -> str:
        essential = [s for s in skills if s.get("relationType") == "essential"][:max_n]
        optional = [s for s in skills if s.get("relationType") != "essential"][:max_n]
        lines = []
        for s in essential:
            lines.append(f"  - {s['skillLabel']} [ESSENTIAL]")
        for s in optional:
            lines.append(f"  - {s['skillLabel']} [optional]")
        return "\n".join(lines) if lines else "  (no gaps — user already covers all target skills)"

    def fmt_have(matched: list[dict]) -> str:
        if not matched:
            return "  (none — starting from scratch)"
        return "\n".join(f"  - {m['esco_label']} (from: \"{m['user_skill']}\")" for m in matched)

    current_context = (
        f"Currently works as: {current_occ.get('preferredLabel', 'unknown')}"
        if current_occ
        else "Starting from scratch (no current role provided)"
    )
    inferred_note = " (inferred next step)" if target_is_inferred else ""

    prompt = f"""You are an expert career development advisor.

Context:
- {current_context}
- Target role{inferred_note}: {target_occ.get('preferredLabel', 'Unknown')}

Skills the user already has (matched to ESCO):
{fmt_have(user_matched_skills)}

Skill gaps (target role requires these, not in user's profile):
{fmt_gap(gap_skills)}

Provide 3-5 concrete, prioritized **Learning Recommendations** to bridge the gap.
Focus on [ESSENTIAL] gaps first. Be specific and actionable. Use bullet points."""

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800,
    )
    return resp.choices[0].message.content


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

occ_emb, occ_meta, skill_emb, skill_meta, relations = load_esco_data()

# ── Input section ─────────────────────────────────────────────────────────────
st.info(
    "Fill in **at least one** field.  \n"
    "- **Current only** → the system infers your natural next career step  \n"
    "- **Target only** → shows a full skill roadmap starting from scratch  \n"
    "- **Both** → gap analysis between your current profile and target role"
)

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
        for i, (occ, score) in enumerate(current_occ_candidates):
            label = "✅ Best match" if i == 0 else f"Alternative {i}"
            st.markdown(f"**{label}** — {occ['preferredLabel']} `sim: {score:.3f}`")
            st.caption(f"ISCO Group: {occ.get('iscoGroup', 'N/A')}  |  Code: {occ.get('code', 'N/A')}")
            if i == 0:
                st.divider()

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
    for i, (occ, score) in enumerate(target_occ_candidates):
        label = "✅ Best match" if i == 0 else f"Alternative {i}"
        st.markdown(f"**{label}** — {occ['preferredLabel']} `sim: {score:.3f}`")
        st.caption(f"ISCO Group: {occ.get('iscoGroup', 'N/A')}  |  Code: {occ.get('code', 'N/A')}")
        if i == 0:
            st.divider()

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

    def render_skill_list(skills: list[dict], title: str):
        """Grouped skill list inside an expander: essential/optional → skillType → 3-col grid."""
        essential = [s for s in skills if s["relationType"] == "essential"]
        optional = [s for s in skills if s["relationType"] != "essential"]
        with st.expander(title, expanded=False):
            for rel_label, rel_icon, group in [
                ("Essential", "🔴", essential),
                ("Optional", "🔵", optional),
            ]:
                if not group:
                    continue
                st.markdown(f"**{rel_icon} {rel_label} ({len(group)})**")
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
                st.markdown("")  # spacer between essential/optional

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

    def render_gap_section(skills: list[dict], relation_label: str, relation_icon: str):
        if not skills:
            return
        by_source: dict[str, list[dict]] = {}
        for s in skills:
            by_source.setdefault(s["source"], []).append(s)
        with st.expander(f"{relation_icon} {relation_label} ({len(skills)})", expanded=True):
            for source in source_order:
                items_in_source = by_source.get(source, [])
                if not items_in_source:
                    continue
                icon = source_icon.get(source, "•")
                st.markdown(f"**{icon} {source}** ({len(items_in_source)})")
                by_type: dict[str, list[dict]] = {}
                for s in items_in_source:
                    by_type.setdefault(normalize_type(s["skillType"]), []).append(s)
                for skill_type, type_items in sort_types(by_type):
                    skill_type_header(skill_type, len(type_items))
                    rows = [type_items[i:i + N_COLS] for i in range(0, len(type_items), N_COLS)]
                    for row in rows:
                        cols = st.columns(N_COLS)
                        for col, s in zip(cols, row):
                            col.markdown(s["skillLabel"])
                st.markdown("")  # spacer between sources

    if gap_skills:
        render_gap_section(gap_essential, "Essential (must-have)", "🔴")
        render_gap_section(gap_optional, "Optional (nice-to-have)", "🔵")
    else:
        st.success("No skill gaps detected — your profile already covers all target role skills!")

    # Learning Recommendations
    st.markdown("### 📚 Learning Recommendations")
    with st.spinner("Generating recommendations…"):
        recommendations = llm_learning_recommendations(
            target_occ=target_occ,
            gap_skills=gap_skills,
            user_matched_skills=user_matched_skills,
            current_occ=current_occ,
            target_is_inferred=target_is_inferred,
        )
    st.markdown(recommendations)

    # Legend
    st.divider()
    st.caption("🔴 Essential skill  |  🔵 Optional skill  |  Similarity scores are cosine similarity (0–1)")
