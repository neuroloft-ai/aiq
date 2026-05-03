"""AIQ File Loader — load documents from files into pipeline input format.

Supports: HTML, DOCX, PDF, plain text, Markdown.

Usage:
    from aiq.loader import load_file, load_directory

    # Single file
    doc = load_file("policy.html")
    result = pipeline.run(doc)

    # Multiple files
    docs = load_directory("kb_pages/")
    result = pipeline.run(docs)

    # With metadata
    doc = load_file("policy.docx", metadata={"author": "Jane", "status": "published"})
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def load_file(path: str, metadata: Optional[dict] = None) -> dict:
    """Load a single file into pipeline input format.

    Args:
        path: path to file (.html, .docx, .pdf, .txt, .md)
        metadata: optional metadata dict (author, last_modified, etc.)

    Returns:
        {"id": filename, "title": stem, "text": content, "metadata": {...}}
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = p.suffix.lower()
    doc_id = p.name
    title = p.stem

    if ext in (".html", ".htm"):
        text = _load_html(p)
    elif ext == ".docx":
        text = _load_docx(p)
    elif ext == ".pdf":
        text = _load_pdf(p)
    elif ext in (".txt", ".md", ".markdown"):
        text = _load_text(p)
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported: .html, .htm, .docx, .pdf, .txt, .md"
        )

    # Build metadata from file system + user-provided
    file_metadata = {
        "source_path": str(p.resolve()),
        "file_type": ext,
    }

    # Try to get file modification date
    try:
        import datetime
        mtime = os.path.getmtime(p)
        file_metadata["last_modified"] = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
    except Exception:
        pass

    # Merge user metadata (user values override file-system values)
    if metadata:
        file_metadata.update(metadata)

    return {
        "id": doc_id,
        "title": title,
        "text": text,
        "metadata": file_metadata,
    }


def load_directory(
    directory: str,
    extensions: Optional[list] = None,
    metadata: Optional[dict] = None,
) -> list[dict]:
    """Load all supported files from a directory.

    Args:
        directory: path to directory
        extensions: file extensions to include (default: all supported)
        metadata: metadata applied to all files (each file also gets file-level metadata)

    Returns:
        list of document dicts for pipeline.run()
    """
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    supported = {".html", ".htm", ".docx", ".pdf", ".txt", ".md", ".markdown"}
    if extensions:
        supported = {e if e.startswith(".") else f".{e}" for e in extensions}

    docs = []
    for f in sorted(d.iterdir()):
        if f.is_file() and f.suffix.lower() in supported:
            try:
                doc = load_file(str(f), metadata=metadata)
                docs.append(doc)
            except Exception as e:
                import logging
                logging.getLogger("aiq.loader").warning("Failed to load %s: %s", f, e)

    return docs


# ─────────────────────────────────────────────────────────────────────────
# Format-specific loaders
# ─────────────────────────────────────────────────────────────────────────

def _load_html(path: Path) -> str:
    """Load HTML file — returns raw HTML for A10/A12 to process."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _load_text(path: Path) -> str:
    """Load plain text or Markdown file."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _load_docx(path: Path) -> str:
    """Load DOCX file — extracts text with paragraph structure."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError(
            "python-docx not installed. Run: pip install python-docx"
        )

    doc = Document(str(path))
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        # Preserve heading structure as HTML-like tags for A12/A13
        style = para.style.name.lower() if para.style else ""
        if "heading 1" in style:
            paragraphs.append(f"<h1>{text}</h1>")
        elif "heading 2" in style:
            paragraphs.append(f"<h2>{text}</h2>")
        elif "heading 3" in style:
            paragraphs.append(f"<h3>{text}</h3>")
        elif "heading" in style:
            paragraphs.append(f"<h4>{text}</h4>")
        else:
            paragraphs.append(f"<p>{text}</p>")

    return "\n".join(paragraphs)


def _load_pdf(path: Path) -> str:
    """Load PDF file — extracts text page by page."""
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ImportError(
            "PyMuPDF not installed. Run: pip install pymupdf\n"
            "Or install AIQ with PDF support: pip install aiq[pdf]"
        )

    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        text = page.get_text().strip()
        if text:
            pages.append(text)
    doc.close()

    return "\n\n".join(pages)
