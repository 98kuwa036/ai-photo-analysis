"""Configuration Management

Loads and validates configuration from YAML files and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv


class Config:
    """Configuration manager for the application."""

    DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "settings.yaml"

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration.

        Args:
            config_path: Optional path to configuration file.
        """
        # Load environment variables from .env file
        load_dotenv()

        # Determine config file path
        self.config_path = config_path or Path(
            os.environ.get("CONFIG_FILE", self.DEFAULT_CONFIG_PATH)
        )

        # Load configuration
        self._config = self._load_config()

    def _load_config(self) -> dict:
        """Load configuration from YAML file.

        Returns:
            Configuration dictionary.
        """
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.

        Args:
            key: Configuration key (e.g., "onedrive.watch_folder").
            default: Default value if key not found.

        Returns:
            Configuration value.
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    @property
    def onedrive_local_path(self) -> Path:
        """Get OneDrive local sync path.

        Returns:
            Path to OneDrive local folder.
        """
        path = os.environ.get("ONEDRIVE_LOCAL_PATH")
        if path:
            return Path(path)
        return Path.home() / "OneDrive"

    @property
    def watch_folder(self) -> Path:
        """Get the folder to watch for new photos.

        Returns:
            Path to watch folder.
        """
        relative_folder = self.get("onedrive.watch_folder", "Pictures/Camera Roll")
        return self.onedrive_local_path / relative_folder

    @property
    def supported_extensions(self) -> set:
        """Get supported image extensions.

        Returns:
            Set of supported extensions.
        """
        extensions = self.get(
            "onedrive.supported_extensions",
            [".jpg", ".jpeg", ".png", ".heic", ".heif"],
        )
        return set(ext.lower() for ext in extensions)

    @property
    def shrink_size(self) -> int:
        """Get shrink image size.

        Returns:
            Shrink size in pixels.
        """
        return self.get("image_processing.shrink_size", 640)

    @property
    def jpeg_quality(self) -> int:
        """Get JPEG quality setting.

        Returns:
            JPEG quality (1-100).
        """
        return self.get("image_processing.jpeg_quality", 85)

    @property
    def temp_folder(self) -> Path:
        """Get temporary folder path.

        Returns:
            Path to temp folder.
        """
        folder = self.get("image_processing.temp_folder", "temp/shrink")
        return Path(folder)

    @property
    def vision_max_labels(self) -> int:
        """Get maximum labels for Vision API.

        Returns:
            Maximum number of labels.
        """
        return self.get("vision_api.max_labels", 20)

    @property
    def vision_min_confidence(self) -> float:
        """Get minimum confidence for Vision API.

        Returns:
            Minimum confidence score.
        """
        return self.get("vision_api.min_confidence", 0.7)

    @property
    def vision_features(self) -> list:
        """Get Vision API features.

        Returns:
            List of feature names.
        """
        return self.get(
            "vision_api.features",
            ["LABEL_DETECTION", "LANDMARK_DETECTION", "FACE_DETECTION", "OBJECT_LOCALIZATION"],
        )

    @property
    def deepl_api_key(self) -> Optional[str]:
        """Get DeepL API key.

        Returns:
            API key or None.
        """
        return os.environ.get("DEEPL_API_KEY")

    @property
    def deepl_target_lang(self) -> str:
        """Get DeepL target language.

        Returns:
            Target language code.
        """
        return self.get("deepl.target_lang", "JA")

    @property
    def deepl_source_lang(self) -> str:
        """Get DeepL source language.

        Returns:
            Source language code.
        """
        return self.get("deepl.source_lang", "EN")

    @property
    def xmp_creator_tool(self) -> str:
        """Get XMP creator tool name.

        Returns:
            Creator tool name.
        """
        return self.get("xmp.creator_tool", "AI Photo Analyzer")

    @property
    def xmp_include_original(self) -> bool:
        """Get whether to include original English labels.

        Returns:
            True if original labels should be included.
        """
        return self.get("xmp.include_original", True)

    @property
    def batch_start_hour(self) -> int:
        """Get batch processing start hour.

        Returns:
            Hour (0-23).
        """
        return self.get("scheduler.batch_start_hour", 2)

    @property
    def batch_start_minute(self) -> int:
        """Get batch processing start minute.

        Returns:
            Minute (0-59).
        """
        return self.get("scheduler.batch_start_minute", 0)

    @property
    def timezone(self) -> str:
        """Get timezone.

        Returns:
            Timezone string.
        """
        return self.get("scheduler.timezone", "Asia/Tokyo")

    @property
    def enable_immediate(self) -> bool:
        """Get whether immediate processing is enabled.

        Returns:
            True if immediate processing is enabled.
        """
        return self.get("scheduler.enable_immediate", False)

    @property
    def log_level(self) -> str:
        """Get logging level.

        Returns:
            Log level string.
        """
        return self.get("logging.level", "INFO")

    @property
    def log_file(self) -> Path:
        """Get log file path.

        Returns:
            Path to log file.
        """
        return Path(self.get("logging.file", "logs/photo_analyzer.log"))

    @property
    def debug(self) -> bool:
        """Check if debug mode is enabled.

        Returns:
            True if debug mode is enabled.
        """
        return os.environ.get("DEBUG", "").lower() in ("true", "1", "yes")

    # RAW Processing Settings
    @property
    def raw_processing_enabled(self) -> bool:
        """Check if RAW processing is enabled.

        Returns:
            True if RAW processing is enabled.
        """
        return self.get("raw_processing.enabled", True)

    @property
    def raw_extensions(self) -> set:
        """Get RAW file extensions.

        Returns:
            Set of RAW file extensions.
        """
        extensions = self.get(
            "raw_processing.raw_extensions",
            [".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2", ".dng"],
        )
        # Remove duplicates and normalize to lowercase
        return set(ext.lower() for ext in extensions)

    @property
    def analysis_source_priority(self) -> list:
        """Get priority order for analysis source files.

        When a RAW file exists, these extensions are searched in order
        to find a file to analyze instead.

        Returns:
            List of extensions in priority order.
        """
        return self.get(
            "raw_processing.analysis_source_priority",
            [".jpg", ".jpeg", ".png", ".heic", ".heif"],
        )

    @property
    def raw_labels_japanese(self) -> list:
        """Get Japanese labels for RAW images.

        Returns:
            List of Japanese RAW labels.
        """
        return self.get(
            "raw_processing.raw_labels.japanese",
            ["RAW", "RAW画像", "生データ"],
        )

    @property
    def raw_labels_english(self) -> list:
        """Get English labels for RAW images.

        Returns:
            List of English RAW labels.
        """
        return self.get(
            "raw_processing.raw_labels.english",
            ["RAW", "RAW Image", "Unprocessed"],
        )

    @property
    def all_watchable_extensions(self) -> set:
        """Get all extensions to watch (both regular and RAW).

        Returns:
            Set of all watchable extensions.
        """
        return self.supported_extensions | self.raw_extensions
