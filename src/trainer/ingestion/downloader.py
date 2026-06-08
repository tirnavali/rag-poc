"""Download utility for URL-based document sources.

Provides:
- is_url(): detect http(s) strings
- download_file(): fetch a URL to data_lake/downloads/{collection}/{doc_id}/
- validate_url(): lightweight HEAD request for --validate mode
- resolve_source(): adapter helper — turns URLs into local paths on demand

Design:
- Uses urllib (stdlib) — no extra dependency.
- Preserves original filename from URL.
- Skips re-download if file exists and Content-Length matches (best-effort).
- Stores ETag / Last-Modified so adapters can skip hash computation.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Tuple

from src.config import settings


def is_url(value: Optional[str]) -> bool:
    """Return True if value is an http(s) URL."""
    if not value:
        return False
    return value.startswith("http://") or value.startswith("https://")


def _url_to_filename(url: str) -> str:
    """Extract the last path component as filename; fall back to a hash."""
    # Strip query parameters and fragments for the filename
    clean = url.split("?")[0].split("#")[0]
    name = clean.rstrip("/").split("/")[-1]
    if name and "." in name:
        return name
    # No recognizable filename — use a short hash
    return hashlib.sha256(url.encode()).hexdigest()[:16] + ".bin"


def _download_dir(collection_name: str, document_id: str) -> Path:
    """Return the target directory for a downloaded file."""
    d = settings.DOWNLOADS_DIR / collection_name / document_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_file(
    url: str,
    collection_name: str,
    document_id: str,
    timeout: int = 30,
) -> Tuple[Path, Optional[str], Optional[str]]:
    """Download a file from URL to data_lake/downloads/.

    Returns:
        local_path: where the file was saved
        etag: ETag header value (or None)
        last_modified: Last-Modified header value (or None)

    Skips re-download if a local file exists with the same Content-Length.
    """
    target_dir = _download_dir(collection_name, document_id)
    filename = _url_to_filename(url)
    local_path = target_dir / filename

    # Lightweight HEAD to get headers before full download
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    remote_length: Optional[int] = None
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            etag = resp.headers.get("ETag")
            last_modified = resp.headers.get("Last-Modified")
            cl = resp.headers.get("Content-Length")
            if cl is not None:
                remote_length = int(cl)
    except Exception:
        # HEAD may fail on some servers; proceed with GET and see what happens
        pass

    # Best-effort skip: if file exists and size matches, trust it
    if local_path.exists() and remote_length is not None:
        if local_path.stat().st_size == remote_length:
            return local_path, etag, last_modified

    # Perform GET
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        # Re-read headers from GET response (may have more info than HEAD)
        etag = resp.headers.get("ETag") or etag
        last_modified = resp.headers.get("Last-Modified") or last_modified
        with open(local_path, "wb") as f:
            f.write(resp.read())

    return local_path, etag, last_modified


def validate_url(url: str, timeout: int = 10) -> Tuple[bool, Optional[str]]:
    """Quick HEAD request to check URL reachability.

    Returns:
        (ok, error_message)
    """
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return True, None
            return False, f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return False, f"URL error: {e.reason}"
    except Exception as e:
        return False, f"Network error: {e}"


def resolve_source(
    doc_source: Optional[str],
    collection_name: str,
    document_id: str,
) -> Tuple[str, Optional[str], Optional[str]]:
    """Resolve a document_source into a local path.

    If doc_source is a URL, download it and return the local path.
    If doc_source is already a local path, return as-is.

    Returns:
        local_path: absolute path usable by Docling / file readers
        etag: from server (if URL)
        last_modified: from server (if URL)
    """
    if not doc_source:
        return doc_source, None, None
    if is_url(doc_source):
        local_path, etag, last_modified = download_file(
            doc_source, collection_name, document_id
        )
        return str(local_path), etag, last_modified
    return doc_source, None, None
