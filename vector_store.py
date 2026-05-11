"""
FAISS vector store over SHL catalog.
Uses sentence-transformers/all-MiniLM-L6-v2 via langchain-huggingface.
"""
import json, os, pickle
from typing import List, Dict
import numpy as np

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "catalog.json")
INDEX_PATH   = os.path.join(os.path.dirname(__file__), "catalog_index.pkl")
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

TEST_TYPE_LABELS = {
    "A": "Ability Aptitude cognitive reasoning",
    "B": "Biodata background experience",
    "C": "Competency behavioural structured interview",
    "K": "Knowledge Skills technical job-specific",
    "P": "Personality Behaviour motivation values",
    "S": "Simulation work sample exercise",
}


def _load_catalog() -> List[Dict]:
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _doc_text(p: Dict) -> str:
    parts = [p["name"]]
    if p.get("test_types"):
        labels = [TEST_TYPE_LABELS.get(t, t) for t in p["test_types"]]
        parts.append("Test types: " + " | ".join(labels))
    if p.get("description"):
        parts.append(p["description"])
    if p.get("remote_testing"):
        parts.append("remote testing available")
    if p.get("adaptive"):
        parts.append("adaptive IRT")
    return " | ".join(parts)


def _get_embedder():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index():
    import faiss
    catalog  = _load_catalog()
    docs     = [_doc_text(p) for p in catalog]
    embedder = _get_embedder()

    print(f"Embedding {len(docs)} catalog items...")
    vecs = np.array(embedder.embed_documents(docs), dtype="float32")
    faiss.normalize_L2(vecs)

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"index": index, "catalog": catalog}, f)
    print(f"Index saved: {INDEX_PATH}  (dim={vecs.shape[1]}, n={len(catalog)})")


_store_cache    = None
_embedder_cache = None


def search(query: str, k: int = 10) -> List[Dict]:
    import faiss
    global _store_cache, _embedder_cache

    if _store_cache is None:
        with open(INDEX_PATH, "rb") as f:
            _store_cache = pickle.load(f)
    if _embedder_cache is None:
        _embedder_cache = _get_embedder()

    index, catalog = _store_cache["index"], _store_cache["catalog"]
    vec = np.array(_embedder_cache.embed_query(query), dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)

    scores, indices = index.search(vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        item = dict(catalog[idx])
        item["_score"] = float(score)
        results.append(item)
    return results


if __name__ == "__main__":
    build_index()
