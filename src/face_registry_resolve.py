"""Resolve enrolled-face registry root (default: project ``face_registry/``)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_face_registry_dir(face_registry_dir: str | None) -> str:
    """
    Face registry root: explicit path, or ``<repo>/face_registry`` when None/empty.
    Relative paths are resolved against the current working directory.
    """
    if face_registry_dir and str(face_registry_dir).strip():
        p = Path(face_registry_dir).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        else:
            p = p.resolve()
    else:
        p = (REPO_ROOT / "face_registry").resolve()
    if not p.is_dir():
        raise FileNotFoundError(
            f"Face registry folder not found: {p}\n"
            "Create it (see face_registry/README.txt) or pass face_registry_dir / --face-registry."
        )
    return str(p)
