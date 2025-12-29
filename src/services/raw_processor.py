"""RAW Image Processing Service

Handles detection and processing of RAW image files.
When a RAW file is detected, finds the corresponding JPEG/PNG
for analysis and creates XMP sidecar for the RAW file.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class RawImagePair:
    """Represents a RAW image and its analysis source."""

    raw_path: Path
    source_path: Optional[Path] = None  # JPEG/PNG to analyze
    is_raw: bool = True
    raw_extension: str = ""

    def __post_init__(self):
        self.raw_extension = self.raw_path.suffix.lower()


@dataclass
class RawProcessingMapping:
    """Tracks mapping between shrink files and their target RAW files."""

    shrink_path: Path
    source_path: Path  # Original JPEG/PNG used for analysis
    raw_path: Path  # RAW file to receive XMP
    raw_extension: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "shrink_path": str(self.shrink_path),
            "source_path": str(self.source_path),
            "raw_path": str(self.raw_path),
            "raw_extension": self.raw_extension,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RawProcessingMapping":
        """Create from dictionary."""
        return cls(
            shrink_path=Path(data["shrink_path"]),
            source_path=Path(data["source_path"]),
            raw_path=Path(data["raw_path"]),
            raw_extension=data.get("raw_extension", ""),
        )


class RawImageProcessor:
    """Handles RAW image detection and pairing with analysis sources."""

    def __init__(
        self,
        raw_extensions: Set[str],
        analysis_source_priority: list[str],
        mapping_file: Optional[Path] = None,
    ):
        """Initialize the RAW image processor.

        Args:
            raw_extensions: Set of RAW file extensions (e.g., {".cr2", ".nef"}).
            analysis_source_priority: List of extensions to search for analysis.
            mapping_file: Path to store RAW processing mappings.
        """
        self.raw_extensions = {ext.lower() for ext in raw_extensions}
        self.analysis_source_priority = [ext.lower() for ext in analysis_source_priority]
        self.mapping_file = mapping_file or Path("temp/raw_mappings.json")
        self._mappings: dict[str, RawProcessingMapping] = {}

        # Load existing mappings
        self._load_mappings()

    def is_raw_file(self, file_path: Path) -> bool:
        """Check if a file is a RAW image.

        Args:
            file_path: Path to check.

        Returns:
            True if the file is a RAW image.
        """
        return file_path.suffix.lower() in self.raw_extensions

    def find_analysis_source(self, raw_path: Path) -> Optional[Path]:
        """Find a JPEG/PNG file to use for analyzing a RAW image.

        Searches for files with the same name but different extension
        in the same directory.

        Args:
            raw_path: Path to the RAW file.

        Returns:
            Path to analysis source, or None if not found.
        """
        parent = raw_path.parent
        stem = raw_path.stem

        for ext in self.analysis_source_priority:
            # Try lowercase
            candidate = parent / f"{stem}{ext}"
            if candidate.exists():
                logger.info(f"Found analysis source for {raw_path.name}: {candidate.name}")
                return candidate

            # Try uppercase
            candidate = parent / f"{stem}{ext.upper()}"
            if candidate.exists():
                logger.info(f"Found analysis source for {raw_path.name}: {candidate.name}")
                return candidate

        logger.warning(f"No analysis source found for RAW file: {raw_path}")
        return None

    def get_raw_image_pair(self, file_path: Path) -> RawImagePair:
        """Get RAW image pair information for a file.

        Args:
            file_path: Path to check (can be RAW or regular image).

        Returns:
            RawImagePair with source information.
        """
        if self.is_raw_file(file_path):
            source = self.find_analysis_source(file_path)
            return RawImagePair(
                raw_path=file_path,
                source_path=source,
                is_raw=True,
            )
        else:
            # Regular image - check if there's a corresponding RAW
            raw_path = self._find_raw_for_image(file_path)
            if raw_path:
                return RawImagePair(
                    raw_path=raw_path,
                    source_path=file_path,
                    is_raw=True,
                )
            else:
                return RawImagePair(
                    raw_path=file_path,
                    source_path=file_path,
                    is_raw=False,
                )

    def _find_raw_for_image(self, image_path: Path) -> Optional[Path]:
        """Find a RAW file corresponding to a regular image.

        Args:
            image_path: Path to the regular image.

        Returns:
            Path to RAW file, or None if not found.
        """
        parent = image_path.parent
        stem = image_path.stem

        for ext in self.raw_extensions:
            # Try lowercase
            candidate = parent / f"{stem}{ext}"
            if candidate.exists():
                return candidate

            # Try uppercase
            candidate = parent / f"{stem}{ext.upper()}"
            if candidate.exists():
                return candidate

        return None

    def find_all_raw_files(self, directory: Path, recursive: bool = True) -> list[Path]:
        """Find all RAW files in a directory.

        Args:
            directory: Directory to search.
            recursive: Whether to search subdirectories.

        Returns:
            List of RAW file paths.
        """
        raw_files = []

        for ext in self.raw_extensions:
            if recursive:
                raw_files.extend(directory.rglob(f"*{ext}"))
                raw_files.extend(directory.rglob(f"*{ext.upper()}"))
            else:
                raw_files.extend(directory.glob(f"*{ext}"))
                raw_files.extend(directory.glob(f"*{ext.upper()}"))

        return sorted(set(raw_files))

    def add_mapping(self, mapping: RawProcessingMapping):
        """Add a RAW processing mapping.

        Args:
            mapping: Mapping to add.
        """
        key = str(mapping.shrink_path)
        self._mappings[key] = mapping
        self._save_mappings()
        logger.debug(f"Added RAW mapping: {mapping.shrink_path} -> {mapping.raw_path}")

    def get_mapping(self, shrink_path: Path) -> Optional[RawProcessingMapping]:
        """Get mapping for a shrink file.

        Args:
            shrink_path: Path to the shrink file.

        Returns:
            Mapping if found, None otherwise.
        """
        return self._mappings.get(str(shrink_path))

    def remove_mapping(self, shrink_path: Path):
        """Remove a mapping after processing.

        Args:
            shrink_path: Path to the shrink file.
        """
        key = str(shrink_path)
        if key in self._mappings:
            del self._mappings[key]
            self._save_mappings()

    def get_all_mappings(self) -> list[RawProcessingMapping]:
        """Get all current mappings.

        Returns:
            List of all mappings.
        """
        return list(self._mappings.values())

    def _load_mappings(self):
        """Load mappings from file."""
        if self.mapping_file.exists():
            try:
                with open(self.mapping_file, "r") as f:
                    data = json.load(f)
                    self._mappings = {
                        k: RawProcessingMapping.from_dict(v)
                        for k, v in data.items()
                    }
                logger.debug(f"Loaded {len(self._mappings)} RAW mappings")
            except Exception as e:
                logger.warning(f"Failed to load RAW mappings: {e}")
                self._mappings = {}

    def _save_mappings(self):
        """Save mappings to file."""
        try:
            self.mapping_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.mapping_file, "w") as f:
                data = {k: v.to_dict() for k, v in self._mappings.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save RAW mappings: {e}")

    def get_xmp_path_for_raw(self, raw_path: Path) -> Path:
        """Get the XMP sidecar path for a RAW file.

        For RAW files, XMP is typically named like:
        - IMG_001.CR2 -> IMG_001.CR2.xmp (not IMG_001.xmp)
        This is the standard Adobe/Lightroom convention.

        Args:
            raw_path: Path to the RAW file.

        Returns:
            Path for the XMP sidecar file.
        """
        # Use full filename + .xmp for RAW files
        return raw_path.parent / f"{raw_path.name}.xmp"

    def clear_mappings(self):
        """Clear all mappings."""
        self._mappings = {}
        if self.mapping_file.exists():
            self.mapping_file.unlink()
