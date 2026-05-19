"""
app.py — Individual Development Planner (Streamlit UI)

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
SKILL_MATCH_THRESHOLD = 0.60
OCC_MATCH_THRESHOLD  = 0.3   # min cosine similarity to accept a current-role ESCO match
TOP_OCC_CANDIDATES = 3   # internal: fetch top-N, only best match is displayed

LANG_OPTIONS = {"English": "en", "中文": "zh", "日本語": "ja"}
LANG_LABELS = {
    "en": {
        "report_title": "Individual Development Plan",
        "generated": "Generated",
        "current_role": "Current Role",
        "target_role": "Target Role",
        "inferred": "inferred",
        "skills_you_bring": "Skills You Bring",
        "skill_gaps_section": "Essential Skill Gaps Overview",
        "top3_section": "Critical Competency Gap Analysis",
        "competency_n": "Competency",
        "skills_included": "Skills included",
        "gap_analysis_lbl": "Gap analysis",
        "dev_schedule": "12-Month Development Schedule",
        "experience_label": "Experience (70%)",
        "social_label": "Social (20%)",
        "formal_label": "Formal (10%)",
        "kpi_label": "KPI",
        "no_gaps": "No essential skill gaps — your profile already covers all requirements.",
        "no_skills": "No matched skills found.",
        "gantt_activity": "Activity",
        "stype_knowledge": "Knowledge",
        "stype_skill_competence": "Skill/Competence",
        "stype_language": "Language",
        "stype_others": "Others",
        "competency_count_label": "Critical Competency Gaps",
        "score_current": "Now",
        "score_target": "Target",
    },
    "zh": {
        "report_title": "個人發展計劃",
        "generated": "生成日期",
        "current_role": "現任職位",
        "target_role": "目標職位",
        "inferred": "系統推測",
        "skills_you_bring": "現有技能",
        "skill_gaps_section": "核心技能差距總覽",
        "top3_section": "關鍵能力差距分析",
        "competency_n": "能力項目",
        "skills_included": "涵蓋技能",
        "gap_analysis_lbl": "差距分析",
        "dev_schedule": "12個月發展時程",
        "experience_label": "實踐學習 (70%)",
        "social_label": "社群學習 (20%)",
        "formal_label": "正式學習 (10%)",
        "kpi_label": "關鍵指標 (KPI)",
        "no_gaps": "無核心技能差距——您的技能已涵蓋目標職位所有要求。",
        "no_skills": "未找到符合的技能。",
        "gantt_activity": "活動項目",
        "stype_knowledge": "知識",
        "stype_skill_competence": "技能／職能",
        "stype_language": "語言",
        "stype_others": "其他",
        "competency_count_label": "關鍵能力差距",
        "score_current": "現在",
        "score_target": "目標",
    },
    "ja": {
        "report_title": "個人開発計画",
        "generated": "作成日",
        "current_role": "現在のポジション",
        "target_role": "目標ポジション",
        "inferred": "推定",
        "skills_you_bring": "現有スキル",
        "skill_gaps_section": "必須スキルギャップ概要",
        "top3_section": "重要なコンピテンシーギャップ分析",
        "competency_n": "コンピテンシー",
        "skills_included": "含まれるスキル",
        "gap_analysis_lbl": "ギャップ分析",
        "dev_schedule": "12ヶ月開発スケジュール",
        "experience_label": "経験学習 (70%)",
        "social_label": "社会的学習 (20%)",
        "formal_label": "公式学習 (10%)",
        "kpi_label": "KPI指標",
        "no_gaps": "必須スキルギャップなし——あなたのスキルはすでに目標ポジションの要件をすべてカバーしています。",
        "no_skills": "マッチするスキルが見つかりませんでした。",
        "gantt_activity": "活動項目",
        "stype_knowledge": "知識",
        "stype_skill_competence": "スキル／コンピテンシー",
        "stype_language": "言語",
        "stype_others": "その他",
        "competency_count_label": "コンピテンシーギャップ",
        "score_current": "現在",
        "score_target": "目標",
    },
}

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
)


# ── Translation helper ────────────────────────────────────────────────────────
def translate_batch(texts: list[str], language: str, token_log: list | None = None, step_name: str = "Translate") -> list[str]:
    """Batch-translate a list of English ESCO terms to the target language.
    Returns the original list unchanged when language is 'en' or input is empty.
    """
    if language == "en" or not texts:
        return texts
    lang_name = {"zh": "Traditional Chinese (繁體中文)", "ja": "Japanese (日本語)"}.get(language, "English")
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    prompt = (
        f"Translate the following professional job titles and skill terms from English to {lang_name}. "
        "These are from the ESCO European Skills/Competences taxonomy.\n\n"
        "Return ONLY a JSON object: {\"translations\": [\"term 1 translation\", ...]}\n"
        "Preserve the same order. Do not add explanations.\n\n"
        f"{numbered}"
    )
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        _log_usage(token_log, step_name, resp.usage)
        result = json.loads(resp.choices[0].message.content)
        translated = result.get("translations", [])
        if len(translated) == len(texts):
            return translated
    except Exception:
        pass
    return texts


def _log_usage(log: list | None, step: str, usage) -> None:
    """Append token usage for one LLM call to the log list."""
    if log is None or usage is None:
        return
    log.append({
        "step": step,
        "input": getattr(usage, "prompt_tokens", 0),
        "output": getattr(usage, "completion_tokens", 0),
    })


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


def llm_parse_current(current_text: str, token_log: list | None = None) -> dict:
    """Extract job title and skills from a current-state description."""
    prompt = f"""You are a career advisor assistant. Extract structured information from the user's current state description.

Current state description:
\"\"\"{current_text}\"\"\"

Return a JSON object with exactly these fields:
{{
  "current_title": "the explicit job title stated by the user (in English), or empty string if none",
  "current_skills": ["skill or tool 1", "skill or tool 2", ...]
}}

Rules:
- current_title must be the job title the user explicitly states (e.g. "I am a Software Engineer", "currently a PM"). Do NOT infer or guess a title from skills or partial descriptions — if the user does not clearly name their role, return "".
- current_skills should be concrete skills, tools, technologies, or competencies (not vague adjectives)
- Extract up to 50 skills maximum"""

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    _log_usage(token_log, "Parse Current Profile", resp.usage)
    return json.loads(resp.choices[0].message.content)


def llm_parse_target(target_text: str, token_log: list | None = None) -> dict:
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
    _log_usage(token_log, "Parse Target Role", resp.usage)
    return json.loads(resp.choices[0].message.content)


def llm_select_target_occupation(
    user_description: str,
    extracted_skills: list[str],
    candidates: list[tuple],
    occ_detail: dict,
    token_log: list | None = None,
) -> int:
    """
    From top-N ESCO occupation candidates (by embedding similarity), use LLM to select
    the best match for the user's described target role. Returns 0-based index.
    """
    skills_str = ", ".join(extracted_skills[:20]) if extracted_skills else "(not provided)"

    cand_blocks = []
    for i, (occ, score) in enumerate(candidates):
        detail = occ_detail.get(occ.get("conceptUri", ""), {})
        alts = [a.strip() for a in (detail.get("altLabels") or "").split("\n") if a.strip()]
        alt_str = " · ".join(alts[:3])
        desc = (detail.get("description") or detail.get("scopeNote") or "")[:250].strip()
        block = f"[{i}] {occ['preferredLabel']}"
        if alt_str:
            block += f"  (a.k.a. {alt_str})"
        if desc:
            block += f"\n    {desc}"
        cand_blocks.append(block)

    prompt = f"""You are a career advisor helping match a user's target role to the best ESCO occupation.

## User's target role description:
\"\"\"{user_description}\"\"\"

## User's existing skills (for context):
{skills_str}

## ESCO occupation candidates (pre-ranked by semantic similarity, best first):
{chr(10).join(cand_blocks)}

## Task:
Select the single candidate that best represents what the user intends as their target role.
Take into account: the specific wording of their description, seniority level implied, industry context, and their existing skills.
If the top candidate (index 0) is clearly correct, select it. Only deviate if another candidate is a substantially better fit.

Return ONLY a JSON object with no explanation:
{{"selected_index": <integer 0–{len(candidates) - 1}>}}"""

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    _log_usage(token_log, "Select Target Occupation", resp.usage)
    try:
        idx = int(json.loads(resp.choices[0].message.content).get("selected_index", 0))
        return max(0, min(len(candidates) - 1, idx))
    except Exception:
        return 0


def llm_infer_next_role(current_title: str, current_occ_label: str, current_occ_skills: list[dict], token_log: list | None = None) -> str:
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
    _log_usage(token_log, "Infer Next Role", resp.usage)
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


def llm_identify_top_competencies(
    gap_essential: list[dict],
    user_matched_skills: list[dict],
    current_occ: dict | None,
    target_occ: dict,
    language: str = "en",
    token_log: list | None = None,
) -> dict:
    """
    From essential skill gaps, identify the top 3 most critical competency clusters
    with gap analysis for each.

    Returns: {"competencies": [{"name": str, "skills": [str], "gap_analysis": str}, ...]}
    """
    gap_labels = [s["skillLabel"] for s in gap_essential[:40]]
    gap_list = "\n".join(f'  - "{label}"' for label in gap_labels)
    user_skills = ", ".join(m["esco_label"] for m in user_matched_skills[:20]) or "(none identified)"
    current_ctx = current_occ["preferredLabel"] if current_occ else "not specified"

    lang_instruction = {
        "en": "Respond entirely in English.",
        "zh": "請以繁體中文回應所有內容（能力名稱、差距分析文字皆須為繁體中文）。",
        "ja": "すべての内容を日本語で回答してください（コンピテンシー名・ギャップ分析テキストを含む）。",
    }[language]

    prompt = f"""You are an expert career development advisor.

{lang_instruction}

## Context
- Current role: {current_ctx}
- Target role: {target_occ.get('preferredLabel', 'Unknown')}
- User's existing skills: {user_skills}

## Essential skill gaps to address:
{gap_list}

## Your task
From the skill gaps above, identify the most critical competency areas the user must develop for a successful transition to the target role.

Each competency area:
1. Groups 1–several related skills from the gap list above
2. Represents a distinct, meaningful capability dimension
3. Is prioritized by career impact and urgency (most critical first)

Always identify exactly 3 competency areas — the 3 most critical ones, ranked by career impact and urgency.

For each competency, provide:
- A concise gap analysis: what the user currently lacks and what level they must reach.
- A current_score (1–5): estimate the user's current proficiency based on their existing skills. 1 = no exposure, 5 = expert.
- A target_score (1–5): the proficiency level the target role actually requires for this competency. This is NOT necessarily high — a role may only require level 2 or 3 for a given competency even if it is critical. Score objectively based on what the role demands, not on assumed difficulty. Must always be > current_score (since this is a gap by definition).

Return ONLY a JSON object:
{{
  "competencies": [
    {{
      "name": "Competency area name (3–6 words)",
      "skills": ["exact skill label from gap list", ...],
      "gap_analysis": "2–3 sentences on the current gap and the target level.",
      "current_score": 2,
      "target_score": 4
    }},
    ...
  ]
}}

Rules:
- Exactly 3 competency areas
- Each skill label appears in at most one competency
- All text in the specified language (except current_score and target_score which are always integers)
- gap_analysis must be specific, not generic
- gap_analysis must use positive, growth-oriented language — describe what the user has the opportunity to develop and grow into, NOT what they lack or are missing. Avoid phrases like "lacks", "insufficient", "missing", "unable to"; use phrases like "has room to grow", "can strengthen", "opportunity to develop", "building towards"
- current_score and target_score are integers 1–5; target_score > current_score
- target_score reflects what the role objectively requires — do not inflate it; it can be 2, 3, 4, or 5"""

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    _log_usage(token_log, "Identify Key Competencies", resp.usage)
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {"competencies": []}


def _enforce_no_overlap(plan: dict) -> dict:
    """Post-process: shift items within each section so no two items share a month."""
    for sec_key in ("experience", "social", "formal"):
        items = plan.get(sec_key, [])
        prev_end = 0
        for item in items:
            mm = re.search(r"(\d+)[^\d]+(\d+)", item.get("schedule", ""))
            if not mm:
                continue
            s = int(mm.group(1))
            e = int(mm.group(2))
            dur = max(1, e - s)
            if s <= prev_end:           # overlap — push forward
                s = prev_end + 1
                e = min(12, s + dur)
            prev_end = e
            item["schedule"] = f"Month {s}–{e}"
    return plan


def llm_development_plan(
    target_occ: dict,
    top_competencies: list[dict],
    user_matched_skills: list[dict],
    current_occ: dict | None = None,
    target_is_inferred: bool = False,
    user_profile_text: str = "",
    language: str = "en",
    token_log: list | None = None,
) -> dict:
    """
    Generate a 70-20-10 development plan anchored to the identified competency gaps,
    with measurable KPIs for every item.

    Returns a dict:
    {
      "experience": [{"title", "action", "addresses", "schedule", "kpi"}, ...],
      "social":     [...],
      "formal":     [...],
    }
    """
    def fmt_have(matched: list[dict]) -> str:
        if not matched:
            return "  (none — starting from scratch)"
        return "\n".join(f"  - {m['esco_label']}" for m in matched)

    comp_text = ""
    for i, comp in enumerate(top_competencies, 1):
        comp_text += f"\n{i}. **{comp['name']}**\n"
        comp_text += f"   Gap analysis: {comp['gap_analysis']}\n"
        comp_text += f"   Skills: {', '.join(comp.get('skills', []))}\n"

    current_context = (
        f"Currently works as: {current_occ.get('preferredLabel', 'unknown')}"
        if current_occ
        else "Starting from scratch (no current role provided)"
    )
    inferred_note = " (inferred next step)" if target_is_inferred else ""
    profile_snippet = user_profile_text[:1200].strip() if user_profile_text else "(not provided)"

    lang_instruction = {
        "en": "Write all content in English.",
        "zh": "請以繁體中文撰寫所有內容（標題、行動說明、KPI 指標皆須為繁體中文）。",
        "ja": "すべての内容を日本語で記述してください（タイトル、アクション説明、KPI指標を含む）。",
    }[language]

    prompt = f"""You are an expert career development advisor using the 70-20-10 development model.

{lang_instruction}

## User profile (raw input — use this to infer seniority level)
{profile_snippet}

## Structured context
- {current_context}
- Target role{inferred_note}: {target_occ.get('preferredLabel', 'Unknown')}
- Skills the user already has:
{fmt_have(user_matched_skills)}

## Critical Competency Gaps (anchor every plan item to these)
{comp_text}

---

## Your task

Step 1 — silently infer the user's seniority level (junior / mid-level / senior / staff / principal) from their profile and existing skills. Calibrate all items to that level.

Step 2 — generate a development plan across three sections: experience, social, formal.
- Every item MUST address at least one of the competency areas listed above. Use the EXACT competency name in "addresses".
- Within each section, order items by impact: highest career leverage first.
- Section constraints:
  - experience: exactly 2 items — first for Month 1–6, second for Month 7–12.
  - social: up to 3 items.
  - formal: up to 3 items.

Step 3 — assign a schedule that is both logically sequenced and non-overlapping within each section:

**NON-OVERLAP RULE — this is a hard constraint, not a guideline:**
- Within each section (experience / social / formal), NO two items may share any month. Item B in the same section must start strictly AFTER Item A ends.
- Items from DIFFERENT sections may run concurrently. This means at most 3 items can be active in any given month (one per section).

VALID example (social section):
  - Social item 1: Month 1–2
  - Social item 2: Month 4–5   ← starts after month 2 ends ✓
  - Social item 3: Month 7–8   ← starts after month 5 ends ✓

INVALID example (social section — REJECT this):
  - Social item 1: Month 1–3
  - Social item 2: Month 2–4   ← month 2–3 overlaps with item 1 ✗

**Sequencing logic:**
- Formal items provide foundational knowledge; schedule them early so they precede or overlap the experience/social items that depend on them.
- Social items (mentorship, peer learning) work best when they accompany or slightly precede the experience items they support.
- Experience items are the deepest application — place the first experience item after or alongside early formal/social items, and the second experience item in the second half after further preparation.
- Within Month 1–6: favour formal learning and early social engagement. Within Month 7–12: favour the second experience item and any remaining social/formal items.

**Scheduling format:**
- experience: first item within Month 1–6, second item within Month 7–12, each spanning up to 6 months. The two experience items must NOT overlap.
- social and formal: each item spans 1–3 months. Items within the same section must NOT overlap.
- Express all schedules as "Month X–Y" (e.g. "Month 2–4", "Month 8–12").

**Before outputting JSON, verify each section:**
- For each section, list the [start, end] ranges in order and confirm no two ranges share a month.
- If any overlap exists, fix the schedules before outputting.

Step 4 — define a measurable KPI for every item:
- The KPI must be a concrete, verifiable success metric — NOT "complete the task" or "attend the course".
- Examples: "Deliver a working prototype reviewed by ≥2 senior engineers", "Achieve AWS SAA-C03 certification with score ≥ 750", "Lead 3 design reviews with documented decisions".
- One KPI per item.

Return ONLY a JSON object (no markdown, no extra text):
{{
  "experience": [
    {{
      "title": "3–5 word summary",
      "action": "...",
      "addresses": ["exact competency name from top 3"],
      "schedule": "Month X–Y",
      "kpi": "Specific measurable success indicator"
    }},
    ...
  ],
  "social": [...],
  "formal": [...]
}}

---

## Section-specific rules

**experience** — on-the-job projects, stretch assignments, responsibilities to seek:
- Describe what to do AND what a successful outcome looks like (deliverables, decisions owned, measurable impact).
- Frame each item around the result the user should demonstrate at the end.

**social** — peer learning, mentorship, community engagement:
- Describe what kind of person to seek and what capability the user should gain.
- Avoid generic advice; describe what the user can do differently as a result.

**formal** — courses, certifications, books, structured learning:
- Name real, specific resources (actual titles, cert programs, authors, platforms).
- Describe how to apply the learning back on the job immediately.

**General rules for all "action" text:**
- Write 3–5 sentences of substantive, expert-level guidance.
- Calibrate to the inferred seniority level.
- Do NOT mention competency/skill names in the action text, and do NOT use phrases like "to improve X" or "in order to develop Y"."""

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    _log_usage(token_log, "Generate Development Plan", resp.usage)
    raw = resp.choices[0].message.content or ""
    finish_reason = resp.choices[0].finish_reason
    try:
        plan = json.loads(raw)
        return _enforce_no_overlap(plan)
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
    top_competencies: list[dict],
    plan: dict,
    target_is_inferred: bool = False,
    language: str = "en",
    token_log: list | None = None,
) -> bytes:
    """Generate a styled Individual Development Plan PDF using reportlab."""
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
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # ── Register CJK-compatible font ─────────────────────────────────────────
    _FONT_NAME = "NotoSansSC"
    _font_candidates = [
        # Bundled font (primary — works on all platforms)
        os.path.join(os.path.dirname(__file__), "fonts", "NotoSansSC-Regular.ttf"),
        # Linux system font (Streamlit Community Cloud fallback)
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        # macOS system fallback
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    _font_registered = False
    for _fp in _font_candidates:
        if os.path.exists(_fp):
            try:
                pdfmetrics.registerFont(TTFont(_FONT_NAME, _fp))
                _font_registered = True
                break
            except Exception:
                continue
    if not _font_registered:
        _FONT_NAME = "Helvetica"  # ASCII-only fallback

    # ── Color system ─────────────────────────────────────────────────────────
    C_NAVY     = colors.HexColor("#0D2137")   # deep navy — primary
    C_INK      = colors.HexColor("#1E293B")   # body text
    C_SLATE    = colors.HexColor("#64748B")   # secondary text
    C_SILVER   = colors.HexColor("#CBD5E1")   # borders
    C_CLOUD    = colors.HexColor("#F1F5F9")   # section backgrounds
    C_WHITE    = colors.white

    C_EXP      = colors.HexColor("#0D9488")   # experience — teal
    C_SOC      = colors.HexColor("#B45309")   # social — amber
    C_FORM     = colors.HexColor("#1D4ED8")   # formal — blue

    C_COMP     = [
        colors.HexColor("#5B21B6"),   # competency 1 — purple
        colors.HexColor("#065F46"),   # competency 2 — dark teal
        colors.HexColor("#1E3A8A"),   # competency 3 — dark blue
        colors.HexColor("#92400E"),   # competency 4 — amber
        colors.HexColor("#831843"),   # competency 5 — rose
    ]
    C_COMP_HEX = ["#5B21B6", "#065F46", "#1E3A8A", "#92400E", "#831843"]
    # Light tints for score-bar "gap" cells (current < i <= target)
    C_COMP_LIGHT = [
        colors.HexColor("#DDD6FE"),   # light purple
        colors.HexColor("#A7F3D0"),   # light teal
        colors.HexColor("#BFDBFE"),   # light blue
        colors.HexColor("#FDE68A"),   # light amber
        colors.HexColor("#FBCFE8"),   # light rose
    ]

    STYPE_HEX = {
        "knowledge":        "#1D4ED8",
        "skill/competence": "#0D9488",
        "language":         "#7C3AED",
        "others":           "#64748B",
    }
    # ── Page geometry ─────────────────────────────────────────────────────────
    PAGE_W, PAGE_H = A4
    MARGIN  = 1.8 * cm
    W       = PAGE_W - 2 * MARGIN
    ACCENT  = 5          # left-accent bar width in points

    buf = _io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=MARGIN,
    )

    # ── Type scale ────────────────────────────────────────────────────────────
    def sty(n, **kw):
        return ParagraphStyle(n, fontName=_FONT_NAME, **kw)

    S_BODY    = sty("body",   fontSize=9,    textColor=C_INK,   leading=14)
    S_SMALL   = sty("small",  fontSize=7.5,  textColor=C_SLATE, leading=11)

    # ── Helpers ───────────────────────────────────────────────────────────────
    lbl = LANG_LABELS[language]

    def rule(w=W, t=0.4, c=C_SILVER, before=6, after=6):
        return HRFlowable(width=w, thickness=t, color=c, spaceBefore=before, spaceAfter=after)

    def strip(text: str, bg=C_NAVY, fg=C_WHITE, fs=9, pad=8) -> Table:
        """Full-width colored strip used as section header."""
        p = Paragraph(f"<b>{text}</b>", sty("st", fontSize=fs, textColor=fg, leading=fs + 3))
        t = Table([[p]], colWidths=[W])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), bg),
            ("TOPPADDING",    (0,0),(-1,-1), pad),
            ("BOTTOMPADDING", (0,0),(-1,-1), pad),
            ("LEFTPADDING",   (0,0),(-1,-1), 14),
            ("RIGHTPADDING",  (0,0),(-1,-1), 14),
        ]))
        return t

    def accent_card(content_rows: list, accent_color, border_color=C_SILVER) -> Table:
        """Card with a colored left accent bar (ACCENT pt) and content column."""
        bar = Table([[""]], colWidths=[ACCENT])
        bar.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), accent_color),
            ("TOPPADDING",    (0,0),(-1,-1), 0),
            ("BOTTOMPADDING", (0,0),(-1,-1), 0),
        ]))
        content = Table(content_rows, colWidths=[W - ACCENT])
        content.setStyle(TableStyle([
            ("TOPPADDING",    (0,0),(-1,-1), 6),
            ("BOTTOMPADDING", (0,0),(-1,-1), 6),
            ("LEFTPADDING",   (0,0),(-1,-1), 12),
            ("RIGHTPADDING",  (0,0),(-1,-1), 10),
            ("VALIGN",        (0,0),(-1,-1), "TOP"),
        ]))
        outer = Table([[bar, content]], colWidths=[ACCENT, W - ACCENT])
        outer.setStyle(TableStyle([
            ("BOX",           (0,0),(-1,-1), 0.4, border_color),
            ("VALIGN",        (0,0),(-1,-1), "TOP"),
            ("TOPPADDING",    (0,0),(-1,-1), 0),
            ("BOTTOMPADDING", (0,0),(-1,-1), 0),
            ("LEFTPADDING",   (0,0),(-1,-1), 0),
            ("RIGHTPADDING",  (0,0),(-1,-1), 0),
        ]))
        return outer

    # Width of the score badge placed in the card header (right-aligned)
    SCORE_BADGE_W = 88   # pt — wide enough for bar + label

    def score_bar(current: int, target: int, comp_idx: int) -> Table:
        """Compact score badge (squares + numeric label) for the card header top-right."""
        SZ, SP = 12, 3          # square size, inter-square spacer
        BAR_W = SZ * 5 + SP * 4  # 72 pt

        C_NONE = colors.HexColor("#E2E8F0")
        solid  = C_COMP[comp_idx % len(C_COMP)]
        light  = C_COMP_LIGHT[comp_idx % len(C_COMP_LIGHT)]

        cur = max(0, min(5, current))
        tgt = max(cur, min(5, target))

        # — 5-square bar ——————————————————————————————————————
        sq_row = [""] * 9   # 5 squares + 4 spacer cols
        bar = Table([sq_row],
                    colWidths=[SZ, SP, SZ, SP, SZ, SP, SZ, SP, SZ],
                    rowHeights=[SZ])
        cmds = [
            ("TOPPADDING",    (0,0),(-1,-1), 0),
            ("BOTTOMPADDING", (0,0),(-1,-1), 0),
            ("LEFTPADDING",   (0,0),(-1,-1), 0),
            ("RIGHTPADDING",  (0,0),(-1,-1), 0),
        ]
        for i in range(5):
            col = i * 2
            bg = solid if i < cur else (light if i < tgt else C_NONE)
            cmds.append(("BACKGROUND", (col, 0), (col, 0), bg))
            if i < 4:
                cmds.append(("BACKGROUND", (col+1, 0), (col+1, 0), colors.white))
        bar.setStyle(TableStyle(cmds))

        # — label row: "現在 2/5  →  目標 4/5" ————————————————
        lang_lbl = LANG_LABELS[language]
        label_p = Paragraph(
            f'<font color="{C_COMP_HEX[comp_idx]}" size="6.5"><b>'
            f'{lang_lbl["score_current"]} {cur}</b></font>'
            f'<font color="#94A3B8" size="6.5">/5 → </font>'
            f'<font color="#1E293B" size="6.5"><b>{lang_lbl["score_target"]} {tgt}</b>/5</font>',
            sty("sblbl", fontSize=6.5, leading=10, alignment=TA_CENTER),
        )

        # — assemble: bar centered, label below ———————————————
        # Outer column is SCORE_BADGE_W; bar is BAR_W, so we center it with padding
        pad = (SCORE_BADGE_W - BAR_W) / 2
        widget = Table(
            [[bar], [label_p]],
            colWidths=[BAR_W],
        )
        widget.setStyle(TableStyle([
            ("ALIGN",         (0,0),(-1,-1), "CENTER"),
            ("TOPPADDING",    (0,0),(-1,-1), 0),
            ("BOTTOMPADDING", (0,0),(-1,-1), 0),
            ("LEFTPADDING",   (0,0),(0,0),   0),
            ("RIGHTPADDING",  (0,0),(0,0),   0),
            ("TOPPADDING",    (0,1),(0,1),   3),
        ]))

        # Wrap in SCORE_BADGE_W container so outer table column aligns cleanly
        wrapper = Table([[widget]], colWidths=[SCORE_BADGE_W])
        wrapper.setStyle(TableStyle([
            ("ALIGN",         (0,0),(-1,-1), "CENTER"),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0),(-1,-1), 0),
            ("BOTTOMPADDING", (0,0),(-1,-1), 0),
            ("LEFTPADDING",   (0,0),(-1,-1), int(pad)),
            ("RIGHTPADDING",  (0,0),(-1,-1), int(pad)),
        ]))
        return wrapper

    def sched_chip(text: str, bg) -> Table:
        p = Paragraph(f"<b>{text}</b>",
                     sty("sc", fontSize=7, textColor=C_WHITE, leading=9, alignment=TA_CENTER))
        t = Table([[p]], colWidths=[3.2 * cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), bg),
            ("TOPPADDING",    (0,0),(-1,-1), 3),
            ("BOTTOMPADDING", (0,0),(-1,-1), 3),
            ("LEFTPADDING",   (0,0),(-1,-1), 6),
            ("RIGHTPADDING",  (0,0),(-1,-1), 6),
        ]))
        return t

    # Width available inside accent_card's content column (left pad 12 + right pad 10 = 22)
    CARD_INNER_W = W - ACCENT - 22

    def kpi_box(label_text: str, value_text: str) -> Table:
        label_p = Paragraph(
            f"<b>{label_text}</b>",
            sty("kl", fontSize=7, textColor=C_SLATE, leading=10),
        )
        value_p = Paragraph(
            value_text,
            sty("kv", fontSize=8.5, textColor=colors.HexColor("#064E3B"), leading=13),
        )
        t = Table([[label_p, value_p]],
                  colWidths=[CARD_INNER_W * 0.14, CARD_INNER_W * 0.86])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), colors.HexColor("#ECFDF5")),
            ("BOX",           (0,0),(-1,-1), 0.4, colors.HexColor("#6EE7B7")),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("LEFTPADDING",   (0,0),(-1,-1), 8),
            ("RIGHTPADDING",  (0,0),(-1,-1), 8),
        ]))
        return t

    # ── Translation ──────────────────────────────────────────────────────────
    _occ_labels_raw = [
        current_occ["preferredLabel"] if current_occ else None,
        target_occ["preferredLabel"],
    ]
    _skill_labels_raw = (
        [m["esco_label"] for m in user_matched_skills]
        + [s["skillLabel"] for s in gap_essential]
    )
    _unique_skill_labels = list(dict.fromkeys(_skill_labels_raw))

    if language != "en":
        _occ_inputs    = [l for l in _occ_labels_raw if l]
        _occ_map       = dict(zip(_occ_inputs,
                                  translate_batch(_occ_inputs, language,
                                                  token_log=token_log,
                                                  step_name="Translate PDF (occupations)")))
        _skill_map     = dict(zip(_unique_skill_labels,
                                  translate_batch(_unique_skill_labels, language,
                                                  token_log=token_log,
                                                  step_name="Translate PDF (skills)")))
    else:
        _occ_map, _skill_map = {}, {}

    def _t_occ(label):   return _occ_map.get(label, label)
    def _t_skill(label): return _skill_map.get(label, label)
    def _t_stype(raw):
        key_map = {"knowledge": "stype_knowledge",
                   "skill/competence": "stype_skill_competence",
                   "language": "stype_language"}
        return lbl.get(key_map.get(raw.lower(), "stype_others"), raw.title())

    # ── Derived data ──────────────────────────────────────────────────────────
    story        = []
    today        = date.today().strftime("%B %d, %Y")
    current_label = _t_occ(current_occ["preferredLabel"]) if current_occ else "—"
    target_label  = _t_occ(target_occ["preferredLabel"])
    inferred_sfx  = f"  ({lbl['inferred']})" if target_is_inferred else ""

    TYPE_ORDER = ["knowledge", "skill/competence", "language", "others"]

    def group_by_type(items, label_key):
        d = {}
        for x in items:
            t = (x.get("skillType") or "others").lower().strip()
            d.setdefault(t, []).append(x[label_key])
        return d

    have_by_type = group_by_type(user_matched_skills, "esco_label")
    gap_by_type  = group_by_type(gap_essential,       "skillLabel")

    SECTION_META = [
        ("experience", lbl["experience_label"], C_EXP),
        ("social",     lbl["social_label"],     C_SOC),
        ("formal",     lbl["formal_label"],     C_FORM),
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — COVER + SKILLS PROFILE
    # ═══════════════════════════════════════════════════════════════════════════

    # ── Hero banner ───────────────────────────────────────────────────────────
    hero_rows = Table([
        [Paragraph(lbl["report_title"].upper(),
                   sty("hl", fontSize=8, textColor=colors.HexColor("#94A3B8"), leading=11))],
        [Paragraph(f"<b>{name}</b>",
                   sty("hn", fontSize=26, textColor=C_WHITE, leading=32))],
        [Spacer(1, 4)],
        [Paragraph(today,
                   sty("hd", fontSize=8, textColor=colors.HexColor("#64748B"), leading=11))],
    ], colWidths=[W - 28])
    hero_rows.setStyle(TableStyle([
        ("TOPPADDING",    (0,0),(-1,-1), 1),
        ("BOTTOMPADDING", (0,0),(-1,-1), 1),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
        ("RIGHTPADDING",  (0,0),(-1,-1), 0),
    ]))
    hero_outer = Table([[hero_rows]], colWidths=[W])
    hero_outer.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_NAVY),
        ("TOPPADDING",    (0,0),(-1,-1), 24),
        ("BOTTOMPADDING", (0,0),(-1,-1), 24),
        ("LEFTPADDING",   (0,0),(-1,-1), 20),
        ("RIGHTPADDING",  (0,0),(-1,-1), 20),
    ]))
    story.append(hero_outer)
    story.append(Spacer(1, 14))

    # ── Role transition ───────────────────────────────────────────────────────
    def role_box(role_lbl, role_name, bg, accent_c) -> Table:
        acc = Table([[""]], colWidths=[4])
        acc.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), accent_c),
            ("TOPPADDING",    (0,0),(-1,-1), 0),
            ("BOTTOMPADDING", (0,0),(-1,-1), 0),
        ]))
        txt = Table([
            [Paragraph(role_lbl.upper(), sty("rbl", fontSize=6.5, textColor=C_SLATE, leading=9))],
            [Paragraph(f"<b>{role_name}</b>", sty("rbn", fontSize=10, textColor=C_INK, leading=14))],
        ], colWidths=[W * 0.44 - 4])
        txt.setStyle(TableStyle([
            ("TOPPADDING",    (0,0),(-1,-1), 10),
            ("BOTTOMPADDING", (0,0),(-1,-1), 10),
            ("LEFTPADDING",   (0,0),(-1,-1), 10),
            ("RIGHTPADDING",  (0,0),(-1,-1), 8),
        ]))
        box = Table([[acc, txt]], colWidths=[4, W * 0.44 - 4])
        box.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), bg),
            ("BOX",           (0,0),(-1,-1), 0.4, C_SILVER),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0),(-1,-1), 0),
            ("BOTTOMPADDING", (0,0),(-1,-1), 0),
            ("LEFTPADDING",   (0,0),(-1,-1), 0),
            ("RIGHTPADDING",  (0,0),(-1,-1), 0),
        ]))
        return box

    role_row = Table(
        [[
            role_box(lbl["current_role"], current_label,
                     colors.HexColor("#F8FAFC"), C_SLATE),
            Paragraph("<b>→</b>", sty("arr", fontSize=16, textColor=C_SILVER, alignment=TA_CENTER)),
            role_box(f"{lbl['target_role']}{inferred_sfx}", target_label,
                     colors.HexColor("#EFF6FF"), C_EXP),
        ]],
        colWidths=[W * 0.46, W * 0.08, W * 0.46],
    )
    role_row.setStyle(TableStyle([
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("ALIGN",         (1,0),(1,-1),  "CENTER"),
        ("TOPPADDING",    (0,0),(-1,-1), 0),
        ("BOTTOMPADDING", (0,0),(-1,-1), 0),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
        ("RIGHTPADDING",  (0,0),(-1,-1), 0),
    ]))
    story.append(role_row)
    story.append(Spacer(1, 14))

    # ── Metrics row ───────────────────────────────────────────────────────────
    def metric_cell(num, label_text, accent) -> Table:
        t = Table([
            [Paragraph(f"<b>{num}</b>",
                       sty("mv", fontSize=22, textColor=accent, leading=26, alignment=TA_CENTER))],
            [Paragraph(label_text,
                       sty("mk", fontSize=6.5, textColor=C_SLATE, leading=9, alignment=TA_CENTER))],
        ], colWidths=[W / 3 - 2])
        t.setStyle(TableStyle([
            ("TOPPADDING",    (0,0),(-1,-1), 10),
            ("BOTTOMPADDING", (0,0),(-1,-1), 10),
            ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ]))
        return t

    metrics = Table(
        [[
            metric_cell(str(len(user_matched_skills)), lbl["skills_you_bring"], C_FORM),
            metric_cell(str(len(gap_essential)),       lbl["skill_gaps_section"], colors.HexColor("#DC2626")),
            metric_cell(str(len(top_competencies)), lbl["competency_count_label"], C_COMP[0]),
        ]],
        colWidths=[W / 3] * 3,
    )
    metrics.setStyle(TableStyle([
        ("BOX",        (0,0),(0,-1), 0.4, C_SILVER),
        ("BOX",        (1,0),(1,-1), 0.4, C_SILVER),
        ("BOX",        (2,0),(2,-1), 0.4, C_SILVER),
        ("BACKGROUND", (0,0),(-1,-1), C_CLOUD),
        ("LINEAFTER",  (0,0),(1,-1), 0.4, C_SILVER),
        ("TOPPADDING",    (0,0),(-1,-1), 0),
        ("BOTTOMPADDING", (0,0),(-1,-1), 0),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
        ("RIGHTPADDING",  (0,0),(-1,-1), 0),
    ]))
    story.append(metrics)
    story.append(Spacer(1, 16))

    # ── Skills profile — side-by-side comparison ──────────────────────────────
    COL_W = (W - 0.4 * cm) / 2

    def skill_col(title_text, by_type: dict, title_bg) -> Table:
        rows = [[Table(
            [[Paragraph(f"<b>{title_text}</b>",
                        sty("sct", fontSize=8, textColor=C_WHITE, leading=11))]],
            colWidths=[COL_W],
        )]]
        rows[0][0].setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), title_bg),
            ("TOPPADDING",    (0,0),(-1,-1), 6),
            ("BOTTOMPADDING", (0,0),(-1,-1), 6),
            ("LEFTPADDING",   (0,0),(-1,-1), 10),
            ("RIGHTPADDING",  (0,0),(-1,-1), 10),
        ]))
        if not by_type:
            rows.append([Paragraph("—", S_SMALL)])
        else:
            sorted_types = sorted(by_type.keys(),
                                  key=lambda k: TYPE_ORDER.index(k) if k in TYPE_ORDER else 99)
            for sk_t in sorted_types:
                labels = by_type[sk_t]
                hex_c  = STYPE_HEX.get(sk_t, "#64748B")
                type_p = Paragraph(
                    f'<font color="{hex_c}">■</font> <b>{_t_stype(sk_t)}</b>',
                    sty("td", fontSize=7.5, textColor=C_INK, leading=11),
                )
                skills_p = Paragraph(
                    ",  ".join(_t_skill(lb) for lb in labels),
                    sty("sk", fontSize=7, textColor=C_SLATE, leading=11, leftIndent=12),
                )
                rows.append([type_p])
                rows.append([skills_p])

        tbl = Table(rows, colWidths=[COL_W])
        style = [
            ("TOPPADDING",    (0,0),(0,0), 0),
            ("BOTTOMPADDING", (0,0),(0,0), 0),
            ("LEFTPADDING",   (0,0),(0,0), 0),
            ("RIGHTPADDING",  (0,0),(0,0), 0),
            ("TOPPADDING",    (0,1),(-1,-1), 4),
            ("BOTTOMPADDING", (0,1),(-1,-1), 2),
            ("LEFTPADDING",   (0,1),(-1,-1), 10),
            ("RIGHTPADDING",  (0,1),(-1,-1), 8),
            ("BOX",           (0,0),(-1,-1), 0.4, C_SILVER),
        ]
        tbl.setStyle(TableStyle(style))
        return tbl

    skills_cmp = Table(
        [[skill_col(lbl["skills_you_bring"],   have_by_type, colors.HexColor("#1E3A8A")),
          Spacer(0.4 * cm, 1),
          skill_col(lbl["skill_gaps_section"], gap_by_type,  colors.HexColor("#991B1B"))]],
        colWidths=[COL_W, 0.4 * cm, COL_W],
    )
    skills_cmp.setStyle(TableStyle([
        ("VALIGN",        (0,0),(-1,-1), "TOP"),
        ("TOPPADDING",    (0,0),(-1,-1), 0),
        ("BOTTOMPADDING", (0,0),(-1,-1), 0),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
        ("RIGHTPADDING",  (0,0),(-1,-1), 0),
    ]))
    story.append(skills_cmp)

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 2 — COMPETENCY GAP ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    story.append(PageBreak())

    story.append(strip(lbl["top3_section"], bg=C_NAVY, fs=11, pad=11))
    story.append(Spacer(1, 14))

    for ci, comp in enumerate(top_competencies):
        c_bg  = C_COMP[ci % len(C_COMP)]
        c_hex = C_COMP_HEX[ci % len(C_COMP_HEX)]
        skills_text = "  ·  ".join(_t_skill(s) for s in comp.get("skills", []))

        num_p = Paragraph(
            f"<b>{ci + 1}</b>",
            sty("cn", fontSize=20, textColor=C_WHITE, leading=24, alignment=TA_CENTER),
        )
        num_tbl = Table([[num_p]], colWidths=[1.4 * cm])
        num_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), c_bg),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0),(-1,-1), 12),
            ("BOTTOMPADDING", (0,0),(-1,-1), 12),
            ("LEFTPADDING",   (0,0),(-1,-1), 0),
            ("RIGHTPADDING",  (0,0),(-1,-1), 0),
        ]))

        # Gap analysis highlight box
        ga_inner_w = W - 1.4 * cm - ACCENT - 24  # subtract card body L/R padding
        ga_lbl_p = Paragraph(
            f"<b>{lbl['gap_analysis_lbl']}</b>",
            sty("gal", fontSize=7, textColor=c_bg, leading=10),
        )
        ga_val_p = Paragraph(
            comp.get("gap_analysis", ""),
            sty("gav", fontSize=8.5, textColor=C_INK, leading=13),
        )
        ga_box = Table(
            [[ga_lbl_p], [ga_val_p]],
            colWidths=[ga_inner_w],
        )
        ga_box.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), colors.HexColor("#F5F3FF")),
            ("LINEAFTER",     (0,0),(0,-1),  2, c_bg),
            ("LINEBEFORE",    (0,0),(0,-1),  3, c_bg),
            ("BOX",           (0,0),(-1,-1), 0.4, C_SILVER),
            ("TOPPADDING",    (0,0),(-1,-1), 6),
            ("BOTTOMPADDING", (0,0),(-1,-1), 6),
            ("LEFTPADDING",   (0,0),(-1,-1), 8),
            ("RIGHTPADDING",  (0,0),(-1,-1), 8),
        ]))

        cur_score = int(comp.get("current_score") or 1)
        tgt_score = int(comp.get("target_score") or 4)
        tgt_score = max(cur_score + 1, min(5, tgt_score))  # enforce tgt > cur

        # Header row: competency name (left) + score badge (right-aligned)
        comp_body_w   = W - 1.4 * cm - ACCENT        # body column total width
        comp_inner_w  = comp_body_w - 22              # subtract body L(12) + R(10) padding
        name_col_w    = comp_inner_w - SCORE_BADGE_W - 6  # 6 pt gap

        name_p = Paragraph(
            f"<b>{comp.get('name', '')}</b>",
            sty("cname", fontSize=11, textColor=C_INK, leading=15),
        )
        header_tbl = Table(
            [[name_p, score_bar(cur_score, tgt_score, ci)]],
            colWidths=[name_col_w, SCORE_BADGE_W],
        )
        header_tbl.setStyle(TableStyle([
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0),(-1,-1), 0),
            ("BOTTOMPADDING", (0,0),(-1,-1), 0),
            ("LEFTPADDING",   (0,0),(-1,-1), 0),
            ("RIGHTPADDING",  (0,0),(-1,-1), 0),
            ("ALIGN",         (1,0),(1,0),   "RIGHT"),
        ]))

        body_rows = [
            [header_tbl],
            [Spacer(1, 5)],
            [Paragraph(
                f'<font color="{c_hex}">■</font>  '
                f'<font size="7" color="#64748B">{lbl["skills_included"]}:</font>  '
                f'<font size="7.5" color="#1E293B">{skills_text}</font>',
                sty("cs", fontSize=7.5, leading=12),
            )],
            [Spacer(1, 8)],
            [ga_box],
        ]
        body_tbl = Table(body_rows, colWidths=[comp_body_w])
        body_tbl.setStyle(TableStyle([
            ("TOPPADDING",    (0,0),(-1,-1), 6),
            ("BOTTOMPADDING", (0,0),(-1,-1), 6),
            ("LEFTPADDING",   (0,0),(-1,-1), 12),
            ("RIGHTPADDING",  (0,0),(-1,-1), 10),
            ("BACKGROUND",    (0,0),(-1,-1), C_CLOUD),
        ]))

        card_inner = Table([[num_tbl, body_tbl]],
                           colWidths=[1.4 * cm, W - 1.4 * cm])
        card_inner.setStyle(TableStyle([
            ("VALIGN",        (0,0),(-1,-1), "TOP"),
            ("TOPPADDING",    (0,0),(-1,-1), 0),
            ("BOTTOMPADDING", (0,0),(-1,-1), 0),
            ("LEFTPADDING",   (0,0),(-1,-1), 0),
            ("RIGHTPADDING",  (0,0),(-1,-1), 0),
            ("BOX",           (0,0),(-1,-1), 0.5, C_SILVER),
        ]))
        story.append(KeepTogether([card_inner, Spacer(1, 10)]))

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 3+ — DEVELOPMENT PLAN
    # ═══════════════════════════════════════════════════════════════════════════
    story.append(PageBreak())

    story.append(strip(lbl["dev_schedule"], bg=C_NAVY, fs=11, pad=11))
    story.append(Spacer(1, 12))

    # ── Gantt chart ───────────────────────────────────────────────────────────
    gantt_items = []
    for sec_key, sec_label, sec_color in SECTION_META:
        for item in plan.get(sec_key, []):
            sch = item.get("schedule", "")
            mm  = re.search(r"(\d+)[^\d]+(\d+)", sch)
            s   = int(mm.group(1)) if mm else 1
            e   = max(s, min(int(mm.group(2)) if mm else 1, 12))
            gantt_items.append({
                "label": item.get("title") or item.get("action", "")[:40],
                "start": s, "end": e, "color": sec_color,
            })

    if gantt_items:
        LBL_W   = W * 0.36
        MON_W   = (W * 0.64) / 12
        ROW_H   = 17
        C_GRID  = colors.HexColor("#E2E8F0")
        C_EVROW = colors.HexColor("#F8FAFC")

        hdr = [Paragraph(f"<b>{lbl['gantt_activity']}</b>",
                         sty("gh", fontSize=7, textColor=C_WHITE, leading=9))]
        for i in range(1, 13):
            hdr.append(Paragraph(f"<b>{i}</b>",
                                 sty("gm", fontSize=6.5, textColor=C_WHITE,
                                     leading=9, alignment=TA_CENTER)))
        g_data  = [hdr]
        g_cmds  = [
            ("BACKGROUND",    (0,0),(-1,0),  C_NAVY),
            ("GRID",          (0,0),(-1,-1), 0.25, C_GRID),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0),(-1,-1), 3),
            ("BOTTOMPADDING", (0,0),(-1,-1), 3),
            ("LEFTPADDING",   (0,0),(-1,-1), 4),
            ("RIGHTPADDING",  (0,0),(-1,-1), 4),
        ]
        for ri, gi in enumerate(gantt_items, start=1):
            row = [Paragraph(gi["label"], sty("gl", fontSize=6.5, textColor=C_INK, leading=9))]
            for m in range(1, 13):
                row.append(Paragraph("", S_BODY) if gi["start"] <= m <= gi["end"] else "")
            g_data.append(row)
            cs, ce = gi["start"], gi["end"]
            if cs != ce:
                g_cmds.append(("SPAN", (cs, ri), (ce, ri)))
            # Bar: slightly lighter version via alpha-like solid color
            bar_c = gi["color"]
            g_cmds.append(("BACKGROUND", (cs, ri), (ce, ri), bar_c))
            bg_row = C_EVROW if ri % 2 == 0 else C_WHITE
            for mc in range(0, 13):
                if not (cs <= mc <= ce):
                    g_cmds.append(("BACKGROUND", (mc, ri), (mc, ri), bg_row))

        gantt = Table(g_data, colWidths=[LBL_W] + [MON_W] * 12, rowHeights=ROW_H)
        gantt.setStyle(TableStyle(g_cmds))
        story.append(gantt)

    story.append(Spacer(1, 20))

    # ── Legend strip ─────────────────────────────────────────────────────────
    def legend_chip(label_text, color) -> Table:
        p = Paragraph(
            f'<font color="white"><b>{label_text}</b></font>',
            sty("lc", fontSize=7.5, textColor=C_WHITE, leading=10, alignment=TA_CENTER),
        )
        t = Table([[p]], colWidths=[W / 3 - 4])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), color),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ]))
        return t

    legend = Table(
        [[legend_chip(lbl["experience_label"], C_EXP),
          Spacer(4, 1),
          legend_chip(lbl["social_label"],     C_SOC),
          Spacer(4, 1),
          legend_chip(lbl["formal_label"],     C_FORM)]],
        colWidths=[W / 3 - 4, 4, W / 3 - 4, 4, W / 3 - 4],
    )
    legend.setStyle(TableStyle([
        ("TOPPADDING",    (0,0),(-1,-1), 0),
        ("BOTTOMPADDING", (0,0),(-1,-1), 0),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
        ("RIGHTPADDING",  (0,0),(-1,-1), 0),
    ]))
    story.append(legend)
    story.append(Spacer(1, 20))

    # ── Plan item cards ───────────────────────────────────────────────────────
    for sec_key, sec_label, sec_color in SECTION_META:
        items = plan.get(sec_key, [])
        if not items:
            continue

        story.append(KeepTogether([
            strip(sec_label, bg=sec_color, fs=10, pad=9),
            Spacer(1, 10),
        ]))

        for item in items:
            title    = item.get("title", "")
            action   = item.get("action", "")
            schedule = item.get("schedule", "")
            kpi      = item.get("kpi", "")
            addrs    = item.get("addresses", [])

            # Title row: title left, schedule chip right
            title_p = Paragraph(
                f"<b>{title}</b>" if title else "",
                sty("it", fontSize=10, textColor=C_NAVY, leading=14),
            )
            chip = sched_chip(schedule, sec_color) if schedule else Spacer(1, 1)
            title_row = Table([[title_p, chip]],
                              colWidths=[CARD_INNER_W - 3.8 * cm, 3.8 * cm])
            title_row.setStyle(TableStyle([
                ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
                ("TOPPADDING",    (0,0),(-1,-1), 2),
                ("BOTTOMPADDING", (0,0),(-1,-1), 4),
                ("LEFTPADDING",   (0,0),(-1,-1), 0),
                ("RIGHTPADDING",  (0,0),(-1,-1), 0),
                ("ALIGN",         (1,0),(1,-1), "RIGHT"),
            ]))

            card_rows: list = [[title_row]]
            card_rows.append([Paragraph(action, S_BODY)])
            if kpi:
                card_rows.append([Spacer(1, 6)])
                card_rows.append([kpi_box(lbl["kpi_label"], kpi)])
            if addrs:
                tags_text = "  ".join(
                    f'<font color="{C_COMP_HEX[i % len(C_COMP_HEX)]}">[{a}]</font>'
                    for i, a in enumerate(addrs)
                )
                card_rows.append([Spacer(1, 4)])
                card_rows.append([Paragraph(tags_text,
                                            sty("tg", fontSize=7, textColor=C_SLATE, leading=11))])

            story.append(KeepTogether([
                accent_card(card_rows, sec_color),
                Spacer(1, 8),
            ]))

        story.append(Spacer(1, 6))

    doc.build(story)
    return buf.getvalue()


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Individual Development Planner", page_icon="🎯", layout="wide")
st.title("🎯 Individual Development Planner")
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
## Welcome to the Individual Development Planner

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
Your extracted skills matched against the full ESCO skill vocabulary (cosine similarity ≥ 0.60). Each matched skill shows its ESCO label, similarity score, and whether it is required by the target role (**Required ✅**).

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

st.markdown("---")
report_lang = LANG_OPTIONS[
    st.radio(
        "Report language / 報告語言 / レポート言語",
        options=list(LANG_OPTIONS.keys()),
        horizontal=True,
        key="report_lang",
    )
]

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

    # ── Token usage log (reset each run) ─────────────────────────────────────
    token_log: list[dict] = []

    # ── Step 1: Parse inputs ──────────────────────────────────────────────────
    current_occ = None
    current_occ_skills = []
    extracted_skills = []
    target_is_inferred = False

    with st.spinner("Parsing your input…"):
        if has_current:
            parsed_current = llm_parse_current(current_text, token_log=token_log)
            extracted_title = parsed_current.get("current_title", "")
            extracted_skills = parsed_current.get("current_skills", [])
        if has_target:
            parsed_target = llm_parse_target(target_text, token_log=token_log)
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
        if not extracted_title:
            st.info("No explicit job title found in your current profile — current role will be treated as unspecified.")
        else:
            with st.spinner("Matching current occupation in ESCO…"):
                current_occ_candidates = semantic_search_occupations(
                    extracted_title, occ_emb, occ_meta, top_k=TOP_OCC_CANDIDATES
                )
            best_occ, best_score = current_occ_candidates[0]
            if best_score >= OCC_MATCH_THRESHOLD:
                current_occ = best_occ
                current_occ_skills = relations.get(current_occ["conceptUri"], [])
                st.subheader("Current Role Match")
                render_occ_match(best_occ, best_score, occ_detail.get(best_occ["conceptUri"], {}))
            else:
                st.info(f'Current role "{extracted_title}" could not be confidently matched to an ESCO occupation (best score: {best_score:.2f} < {OCC_MATCH_THRESHOLD}) — current role will be treated as unspecified.')

    # ── Step 3: Determine target occupation ───────────────────────────────────
    TOP_TARGET_CANDIDATES = 5

    if has_target:
        with st.spinner("Retrieving top target occupation candidates from ESCO…"):
            target_occ_candidates = semantic_search_occupations(
                target_title, occ_emb, occ_meta, top_k=TOP_TARGET_CANDIDATES
            )
        with st.spinner("Selecting best-matching target occupation…"):
            selected_idx = llm_select_target_occupation(
                user_description=target_text,
                extracted_skills=extracted_skills,
                candidates=target_occ_candidates,
                occ_detail=occ_detail,
                token_log=token_log,
            )
        target_occ = target_occ_candidates[selected_idx][0]
    else:
        # current_only: infer next role from LLM, then ESCO-match it
        with st.spinner("Inferring your next career step…"):
            inferred_title = llm_infer_next_role(
                extracted_title, current_occ["preferredLabel"], current_occ_skills,
                token_log=token_log,
            )
            target_occ_candidates = semantic_search_occupations(
                inferred_title, occ_emb, occ_meta, top_k=TOP_TARGET_CANDIDATES
            )
        with st.spinner("Selecting best-matching target occupation…"):
            selected_idx = llm_select_target_occupation(
                user_description=inferred_title,
                extracted_skills=extracted_skills,
                candidates=target_occ_candidates,
                occ_detail=occ_detail,
                token_log=token_log,
            )
        target_occ = target_occ_candidates[selected_idx][0]
        target_is_inferred = True
        st.info(f"No target provided — inferred next step: **{inferred_title}**")

    target_occ_skills = relations.get(target_occ["conceptUri"], [])

    st.subheader("Target Role Match" + (" *(inferred)*" if target_is_inferred else ""))
    selected_tgt, selected_tgt_score = target_occ_candidates[selected_idx]
    render_occ_match(selected_tgt, selected_tgt_score, occ_detail.get(selected_tgt["conceptUri"], {}))

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

    # Top 3 Competencies + Development Plan
    st.markdown("### 🗺 Development Plan")
    st.caption("Based on essential skill gaps · Key competencies identified · 70-20-10 framework")
    if not gap_essential:
        st.success("No essential skill gaps — no development plan needed!")
    else:
        with st.spinner("Identifying top 3 critical competency gaps…"):
            top_comp_result = llm_identify_top_competencies(
                gap_essential=gap_essential,
                user_matched_skills=user_matched_skills,
                current_occ=current_occ,
                target_occ=target_occ,
                language=report_lang,
                token_log=token_log,
            )
        top_competencies = top_comp_result.get("competencies", [])

        # Translate ESCO skill labels inside competencies when language != English
        if report_lang != "en" and top_competencies:
            _all_comp_skills = list({s for comp in top_competencies for s in comp.get("skills", [])})
            _translated_cs = translate_batch(_all_comp_skills, report_lang,
                                             token_log=token_log, step_name="Translate (competency skills)")
            _comp_skill_map = dict(zip(_all_comp_skills, _translated_cs))
            for comp in top_competencies:
                comp["skills"] = [_comp_skill_map.get(s, s) for s in comp.get("skills", [])]

        # Display top 3 competencies
        if top_competencies:
            st.markdown("#### 🎯 Critical Competency Gap Analysis")
            st.caption("Selected from essential skill gaps — highest career impact first")
            COMP_COLORS = ["#5B21B6", "#065F46", "#1E3A8A", "#92400E", "#831843"]
            for ci, comp in enumerate(top_competencies):
                c_color = COMP_COLORS[ci % len(COMP_COLORS)]
                skills_pills = " ".join(
                    f'<span style="display:inline-block;background:{c_color}18;'
                    f'border:1px solid {c_color}44;border-radius:4px;'
                    f'padding:1px 7px;margin:1px;font-size:0.78em;color:{c_color}">'
                    f'{s}</span>'
                    for s in comp.get("skills", [])
                )
                st.markdown(
                    f'<div style="border-left:4px solid {c_color};padding:10px 14px;'
                    f'margin:8px 0;background:{c_color}08;border-radius:0 6px 6px 0">'
                    f'<strong style="font-size:1.05em;color:{c_color}">'
                    f'{ci+1}. {comp.get("name","")}</strong><br>'
                    f'<span style="font-size:0.85em;opacity:0.8">{comp.get("gap_analysis","")}</span><br>'
                    f'<div style="margin-top:6px">{skills_pills}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with st.spinner("Generating 70-20-10 development plan…"):
            plan = llm_development_plan(
                target_occ=target_occ,
                top_competencies=top_competencies,
                user_matched_skills=user_matched_skills,
                current_occ=current_occ,
                target_is_inferred=target_is_inferred,
                user_profile_text=current_text if has_current else "",
                language=report_lang,
                token_log=token_log,
            )

        if not any(plan.get(k) for k in ("experience", "social", "formal")):
            st.warning("⚠️ Development plan returned empty — showing raw response for debugging.")
            st.json(plan)

        comp_name_set = {c["name"] for c in top_competencies}

        def render_plan_section(items: list[dict], header: str):
            if not items:
                return
            st.markdown(f"#### {header}")
            kpi_label = LANG_LABELS[report_lang]["kpi_label"]
            for item in items:
                title     = item.get("title", "")
                action    = item.get("action", "")
                addresses = [a for a in item.get("addresses", []) if a in comp_name_set]
                schedule  = item.get("schedule", "")
                kpi       = item.get("kpi", "")
                chip_html = (
                    f'<span style="display:inline-block;border:1px solid rgba(128,128,128,0.4);'
                    f'border-radius:4px;padding:1px 8px;margin-right:8px;font-size:0.8em;'
                    f'opacity:0.75">🗓 {schedule}</span>'
                    if schedule else ""
                )
                title_html = f'<strong style="font-size:1em">{title}</strong>' if title else ""
                st.markdown(
                    f'<div style="margin:12px 0 3px 0">{chip_html}{title_html}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(action)
                if addresses:
                    badge_html = " ".join(
                        f'<span style="display:inline-block;border:1px solid rgba(78,154,241,0.6);'
                        f'color:#4e9af1;border-radius:4px;padding:1px 7px;margin:1px;font-size:0.78em">'
                        f'{a}</span>'
                        for a in addresses
                    )
                    st.markdown(
                        f'<div style="margin:2px 0 6px 0">{badge_html}</div>',
                        unsafe_allow_html=True,
                    )
                if kpi:
                    st.markdown(
                        f'<div style="background:#eaf4ff;border:1px solid #d0d7e3;border-radius:5px;'
                        f'padding:6px 12px;margin:4px 0 10px 0;font-size:0.85em">'
                        f'<strong>{kpi_label}:</strong> {kpi}</div>',
                        unsafe_allow_html=True,
                    )

        render_plan_section(plan.get("experience", []), "🛠 Experience (70%)")
        render_plan_section(plan.get("social", []),     "🤝 Social (20%)")
        render_plan_section(plan.get("formal", []),     "📚 Formal (10%)")

        st.session_state["pdf_export"] = {
            "current_occ": current_occ,
            "target_occ": target_occ,
            "user_matched_skills": user_matched_skills,
            "gap_essential": gap_essential,
            "top_competencies": top_competencies,
            "plan": plan,
            "target_is_inferred": target_is_inferred,
            "language": report_lang,
        }

    # Always save token_log at the end of the analyze block regardless of which branch ran
    st.session_state["token_log"] = token_log

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
        _pdf_token_log = list(st.session_state.get("token_log", []))
        pdf_bytes = generate_pdf_report(
            name=export_name.strip(),
            token_log=_pdf_token_log,
            **st.session_state["pdf_export"],
        )
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name=f"development_plan_{export_name.strip().replace(' ', '_')}.pdf",
            mime="application/pdf",
        )

        # ── Token usage summary ───────────────────────────────────────────────
        if _pdf_token_log:
            st.markdown("##### 📊 Token Usage Summary")
            total_in  = sum(r["input"]  for r in _pdf_token_log)
            total_out = sum(r["output"] for r in _pdf_token_log)
            rows = [{"Step": r["step"], "Input tokens": r["input"],
                     "Output tokens": r["output"],
                     "Total": r["input"] + r["output"]} for r in _pdf_token_log]
            rows.append({"Step": "**Total**", "Input tokens": total_in,
                         "Output tokens": total_out, "Total": total_in + total_out})
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.caption("Enter your name above to enable the download button.")
