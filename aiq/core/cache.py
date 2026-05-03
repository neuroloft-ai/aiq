"""Pipeline cache — save/load module outputs to disk.

Saves all module outputs for a phase so the user can skip re-running.
Each cache file is identified by source document hash.
"""
import hashlib
import pickle
import time
from pathlib import Path
from typing import Optional


CACHE_DIR = Path.home() / ".aiq" / "cache"


def _source_hash(raw_html: str) -> str:
    """Hash the source document to identify the cache."""
    return hashlib.md5(raw_html.encode()[:5000]).hexdigest()[:12]


def save_phase(phase: str, data: dict, raw_html: str):
    """Save module outputs for a phase.

    Args:
        phase: "phase1", "phase2", "phase3", "phase4"
        data: dict of {key: module_output} to save
        raw_html: source document HTML (for cache key)
    """
    CACHE_DIR.mkdir(exist_ok=True)
    doc_hash = _source_hash(raw_html)
    cache_path = CACHE_DIR / f"{doc_hash}_{phase}.pkl"

    payload = {
        "phase": phase,
        "timestamp": time.time(),
        "doc_hash": doc_hash,
        "data": data,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)

    return str(cache_path)


def load_phase(phase: str, raw_html: str) -> Optional[dict]:
    """Load cached module outputs for a phase.

    Returns dict of {key: module_output} or None if no cache.
    """
    doc_hash = _source_hash(raw_html)
    cache_path = CACHE_DIR / f"{doc_hash}_{phase}.pkl"

    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        if payload.get("doc_hash") != doc_hash:
            return None
        return payload.get("data")
    except Exception:
        return None


def has_cache(phase: str, raw_html: str) -> bool:
    """Check if cache exists for a phase."""
    doc_hash = _source_hash(raw_html)
    cache_path = CACHE_DIR / f"{doc_hash}_{phase}.pkl"
    return cache_path.exists()


def cache_age(phase: str, raw_html: str) -> Optional[float]:
    """Return cache age in seconds, or None if no cache."""
    doc_hash = _source_hash(raw_html)
    cache_path = CACHE_DIR / f"{doc_hash}_{phase}.pkl"
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        return time.time() - payload.get("timestamp", 0)
    except Exception:
        return None


def clear_cache(raw_html: str = None):
    """Clear cache files. If raw_html given, only that document. Otherwise all."""
    if not CACHE_DIR.exists():
        return
    if raw_html:
        doc_hash = _source_hash(raw_html)
        for f in CACHE_DIR.glob(f"{doc_hash}_*.pkl"):
            f.unlink()
    else:
        for f in CACHE_DIR.glob("*.pkl"):
            f.unlink()
