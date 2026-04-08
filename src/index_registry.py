"""
index_registry.py

Tracks available indices on disk. Each index is identified by a course name
and maps to an artifacts directory + file prefix. The registry is persisted
as a JSON manifest at index/registry.json.
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

REQUIRED_SUFFIXES = [".faiss", "_bm25.pkl", "_chunks.pkl", "_sources.pkl", "_meta.pkl"]
REGISTRY_FILENAME = "registry.json"


@dataclass
class IndexEntry:
    """Metadata for a single registered index."""
    course: str
    artifacts_dir: str
    index_prefix: str
    pdf_dir: Optional[str] = None

    @property
    def artifacts_path(self) -> pathlib.Path:
        return pathlib.Path(self.artifacts_dir)

    def is_valid(self) -> bool:
        """Check that all 5 required artifact files exist on disk."""
        d = self.artifacts_path
        return all(
            (d / f"{self.index_prefix}{suffix}").exists()
            for suffix in REQUIRED_SUFFIXES
        )

    def missing_files(self) -> List[str]:
        """Return list of missing artifact filenames."""
        d = self.artifacts_path
        return [
            f"{self.index_prefix}{suffix}"
            for suffix in REQUIRED_SUFFIXES
            if not (d / f"{self.index_prefix}{suffix}").exists()
        ]


class IndexRegistry:
    """Discovers, stores, and looks up available course indices."""

    def __init__(self, index_root: str = "index"):
        self._root = pathlib.Path(index_root)
        self._manifest_path = self._root / REGISTRY_FILENAME
        self._entries: Dict[str, IndexEntry] = {}
        self._load_manifest()

    # ---- persistence ----

    def _load_manifest(self):
        """Load registry.json if it exists."""
        if self._manifest_path.exists():
            with open(self._manifest_path, "r") as f:
                data = json.load(f)
            for item in data:
                entry = IndexEntry(**item)
                self._entries[entry.course] = entry
            logger.info("Loaded %d entries from registry", len(self._entries))

    def _save_manifest(self):
        """Persist current entries to registry.json."""
        self._root.mkdir(parents=True, exist_ok=True)
        data = [asdict(e) for e in self._entries.values()]
        with open(self._manifest_path, "w") as f:
            json.dump(data, f, indent=2)

    # ---- registration ----

    def register(
        self,
        course: str,
        artifacts_dir: str,
        index_prefix: str,
        pdf_dir: Optional[str] = None,
    ) -> IndexEntry:
        """Register (or update) an index for a course."""
        entry = IndexEntry(
            course=course,
            artifacts_dir=str(artifacts_dir),
            index_prefix=index_prefix,
            pdf_dir=pdf_dir,
        )
        if not entry.is_valid():
            missing = entry.missing_files()
            raise FileNotFoundError(
                f"Cannot register '{course}': missing artifacts {missing}"
            )
        self._entries[course] = entry
        self._save_manifest()
        logger.info("Registered index '%s' (%s/%s)", course, artifacts_dir, index_prefix)
        return entry

    def unregister(self, course: str) -> bool:
        """Remove a course from the registry (does not delete files)."""
        if course in self._entries:
            del self._entries[course]
            self._save_manifest()
            return True
        return False

    # ---- lookup ----

    def get(self, course: str) -> Optional[IndexEntry]:
        """Look up a registered index by course name."""
        return self._entries.get(course)

    def list_courses(self) -> List[str]:
        """Return sorted list of registered course names."""
        return sorted(self._entries.keys())

    def list_entries(self) -> List[IndexEntry]:
        """Return all registered entries."""
        return list(self._entries.values())

    def __contains__(self, course: str) -> bool:
        return course in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    # ---- discovery ----

    def discover(self) -> List[IndexEntry]:
        """
        Scan the index root for valid index sets that aren't already
        registered. Returns newly discovered entries (does NOT auto-register
        them — caller decides).
        """
        discovered = []
        if not self._root.exists():
            return discovered

        registered_dirs = {
            (e.artifacts_dir, e.index_prefix) for e in self._entries.values()
        }

        for faiss_file in self._root.rglob("*.faiss"):
            prefix = faiss_file.stem  # e.g. "textbook_index"
            artifacts_dir = str(faiss_file.parent)
            if (artifacts_dir, prefix) in registered_dirs:
                continue
            entry = IndexEntry(
                course=prefix,  # default name; caller can rename
                artifacts_dir=artifacts_dir,
                index_prefix=prefix,
            )
            if entry.is_valid():
                discovered.append(entry)

        return discovered
