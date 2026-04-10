"""
embed_esco.py — One-time preprocessing script

Run this ONCE before starting the Streamlit app.
Generates embeddings for all ESCO occupations and skills,
and builds the occupation→skill relations lookup.

Usage:
    python embed_esco.py

Output (in ./esco_embeddings/):
    occupations_embeddings.npy   — shape (N_occ, 256)
    occupations_meta.json        — [{conceptUri, preferredLabel, code, iscoGroup}]
    skills_embeddings.npy        — shape (N_skill, 256)
    skills_meta.json             — [{conceptUri, preferredLabel, skillType, reuseLevel}]
    occ_skill_relations.json     — {occupationUri: [{skillUri, skillLabel, relationType, skillType}]}
"""

import os
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
ESCO_DATA_DIR = os.getenv(
    "ESCO_DATA_DIR",
    "/Users/tonychen/Downloads/ESCO dataset - v1.2.1 - classification - en - csv",
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "esco_embeddings")
EMBEDDING_MODEL = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
EMBEDDING_DIM = 256
BATCH_SIZE = 100

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
)


# ── Helpers ──────────────────────────────────────────────────────────────────
def safe_str(val, max_len: int = 0) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    s = str(val).strip()
    return s[:max_len] if max_len else s


def build_occ_text(row) -> str:
    parts = [safe_str(row["preferredLabel"])]
    alt = safe_str(row["altLabels"], 200)
    if alt:
        parts.append(alt)
    desc = safe_str(row["description"], 300)
    if desc:
        parts.append(desc)
    return " | ".join(parts)


def build_skill_text(row) -> str:
    parts = [safe_str(row["preferredLabel"])]
    alt = safe_str(row["altLabels"], 200)
    if alt:
        parts.append(alt)
    desc = safe_str(row["description"], 300)
    if desc:
        parts.append(desc)
    scope = safe_str(row["scopeNote"], 200)
    if scope:
        parts.append(scope)
    return " | ".join(parts)


def get_embeddings_batch(texts: list[str], retries: int = 3) -> list[list[float]]:
    for attempt in range(retries):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
                dimensions=EMBEDDING_DIM,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  Retry {attempt+1}/{retries} after error: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


def embed_texts(texts: list[str], desc: str = "") -> np.ndarray:
    all_embeddings: list[list[float]] = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=desc):
        batch = texts[i : i + BATCH_SIZE]
        embeddings = get_embeddings_batch(batch)
        all_embeddings.extend(embeddings)
        time.sleep(0.1)  # gentle rate-limit buffer
    return np.array(all_embeddings, dtype=np.float32)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Occupations ────────────────────────────────────────────────────────
    occ_emb_path = os.path.join(OUTPUT_DIR, "occupations_embeddings.npy")
    occ_meta_path = os.path.join(OUTPUT_DIR, "occupations_meta.json")

    if os.path.exists(occ_emb_path) and os.path.exists(occ_meta_path):
        print("Occupation embeddings already exist — skipping.")
    else:
        print("Loading occupations_en.csv...")
        occ_df = pd.read_csv(os.path.join(ESCO_DATA_DIR, "occupations_en.csv"))
        print(f"  {len(occ_df)} occupations loaded.")

        occ_texts = [build_occ_text(row) for _, row in occ_df.iterrows()]
        occ_meta = (
            occ_df[["conceptUri", "preferredLabel", "code", "iscoGroup"]]
            .fillna("")
            .to_dict("records")
        )

        print("Embedding occupations...")
        occ_emb = embed_texts(occ_texts, desc="Occupations")
        np.save(occ_emb_path, occ_emb)
        with open(occ_meta_path, "w", encoding="utf-8") as f:
            json.dump(occ_meta, f, ensure_ascii=False)
        print(f"  Saved: {occ_emb.shape} → {occ_emb_path}")

    # ── 2. Skills ─────────────────────────────────────────────────────────────
    skill_emb_path = os.path.join(OUTPUT_DIR, "skills_embeddings.npy")
    skill_meta_path = os.path.join(OUTPUT_DIR, "skills_meta.json")

    if os.path.exists(skill_emb_path) and os.path.exists(skill_meta_path):
        print("Skill embeddings already exist — skipping.")
    else:
        print("Loading skills_en.csv...")
        skill_df = pd.read_csv(os.path.join(ESCO_DATA_DIR, "skills_en.csv"))
        print(f"  {len(skill_df)} skills loaded.")

        skill_texts = [build_skill_text(row) for _, row in skill_df.iterrows()]
        skill_meta = (
            skill_df[["conceptUri", "preferredLabel", "skillType", "reuseLevel"]]
            .fillna("")
            .to_dict("records")
        )

        print("Embedding skills (this takes the longest)...")
        skill_emb = embed_texts(skill_texts, desc="Skills")
        np.save(skill_emb_path, skill_emb)
        with open(skill_meta_path, "w", encoding="utf-8") as f:
            json.dump(skill_meta, f, ensure_ascii=False)
        print(f"  Saved: {skill_emb.shape} → {skill_emb_path}")

    # ── 3. Occupation–Skill Relations ─────────────────────────────────────────
    rel_path = os.path.join(OUTPUT_DIR, "occ_skill_relations.json")

    if os.path.exists(rel_path):
        print("Occupation–skill relations already exist — skipping.")
    else:
        print("Loading occupationSkillRelations_en.csv...")
        rel_df = pd.read_csv(
            os.path.join(ESCO_DATA_DIR, "occupationSkillRelations_en.csv")
        )
        print(f"  {len(rel_df)} relations loaded.")

        relations: dict[str, list] = {}
        for _, row in tqdm(rel_df.iterrows(), total=len(rel_df), desc="Building relations"):
            uri = str(row["occupationUri"])
            relations.setdefault(uri, []).append(
                {
                    "skillUri": str(row["skillUri"]),
                    "skillLabel": safe_str(row["skillLabel"]),
                    "relationType": safe_str(row["relationType"]),
                    "skillType": safe_str(row["skillType"]),
                }
            )

        with open(rel_path, "w", encoding="utf-8") as f:
            json.dump(relations, f, ensure_ascii=False)
        print(f"  Saved relations for {len(relations)} occupations → {rel_path}")

    print("\nDone! All embedding files are ready in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
