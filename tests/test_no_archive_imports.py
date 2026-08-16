"""Live code must never import from archive/.

The archive exists because a banner does not stop code from being read or run. This test is the
wall: if anything under src/ or scripts/ ever imports an archived module, or a future refactor
drags an archived file back into the live path, this fails immediately rather than three weeks
later when a number turns out to have come from a retired lineage.

It also catches the reverse mistake — archiving a file that live code still needs — because that
shows up as an ImportError in the smoke test, not here.
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+[\w.]*\barchive\b", re.MULTILINE)


def test_no_live_code_imports_archive() -> None:
    offenders = []
    for area in ("src", "scripts"):
        for p in (ROOT / area).rglob("*.py"):
            if IMPORT_RE.search(p.read_text(encoding="utf-8", errors="ignore")):
                offenders.append(str(p.relative_to(ROOT)))
    assert not offenders, (
        "live code imports from archive/ — either the file was archived by mistake, or the "
        f"importer belongs in the archive too: {offenders}"
    )


def test_archive_is_not_a_package() -> None:
    """No __init__.py anywhere under archive/, so archived modules are not importable by
    accident. If one appears, `from archive...` starts silently working and the wall is gone."""
    inits = [str(p.relative_to(ROOT)) for p in (ROOT / "archive").rglob("__init__.py")]
    assert not inits, f"archive/ must not be a package; found: {inits}"
