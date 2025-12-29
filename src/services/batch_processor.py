"""Batch Processor Service

Handles scheduled batch processing of shrink images during night hours.
Coordinates Vision API analysis, translation, and XMP generation.
Supports RAW image processing with paired JPEG analysis.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from .image_processor import ImageProcessor
from .vision_service import VisionService, VisionAnalysisResult
from .translation_service import TranslationService
from .xmp_generator import XMPGenerator
from .raw_processor import RawImageProcessor, RawProcessingMapping

logger = logging.getLogger(__name__)


class ProcessingResult:
    """Result of processing a single image."""

    def __init__(
        self,
        shrink_path: Path,
        original_path: Optional[Path] = None,
        success: bool = False,
        error: Optional[str] = None,
        labels_count: int = 0,
        is_raw: bool = False,
        raw_path: Optional[Path] = None,
    ):
        self.shrink_path = shrink_path
        self.original_path = original_path
        self.success = success
        self.error = error
        self.labels_count = labels_count
        self.is_raw = is_raw
        self.raw_path = raw_path


class BatchProcessor:
    """Processes images in batches during scheduled times."""

    def __init__(
        self,
        image_processor: ImageProcessor,
        vision_service: VisionService,
        translation_service: TranslationService,
        xmp_generator: XMPGenerator,
        original_folder: Path,
        batch_hour: int = 2,
        batch_minute: int = 0,
        timezone: str = "Asia/Tokyo",
        on_complete: Optional[Callable[[list[ProcessingResult]], None]] = None,
        raw_processor: Optional[RawImageProcessor] = None,
        raw_labels_ja: Optional[list[str]] = None,
        raw_labels_en: Optional[list[str]] = None,
    ):
        """Initialize the batch processor.

        Args:
            image_processor: ImageProcessor instance.
            vision_service: VisionService instance.
            translation_service: TranslationService instance.
            xmp_generator: XMPGenerator instance.
            original_folder: Folder containing original images.
            batch_hour: Hour to run batch processing (0-23).
            batch_minute: Minute to run batch processing (0-59).
            timezone: Timezone for scheduling.
            on_complete: Optional callback when batch completes.
            raw_processor: Optional RawImageProcessor instance for RAW support.
            raw_labels_ja: Japanese labels for RAW images.
            raw_labels_en: English labels for RAW images.
        """
        self.image_processor = image_processor
        self.vision_service = vision_service
        self.translation_service = translation_service
        self.xmp_generator = xmp_generator
        self.original_folder = original_folder
        self.batch_hour = batch_hour
        self.batch_minute = batch_minute
        self.timezone = timezone
        self.on_complete = on_complete
        self.raw_processor = raw_processor
        self.raw_labels_ja = raw_labels_ja or ["RAW", "RAW画像", "生データ"]
        self.raw_labels_en = raw_labels_en or ["RAW", "RAW Image", "Unprocessed"]

        self._scheduler: Optional[BackgroundScheduler] = None
        self._is_processing = False
        self._last_run: Optional[datetime] = None
        self._stats_file = Path("logs/batch_stats.json")

    def _find_original_image(self, shrink_path: Path) -> Optional[Path]:
        """Find the original image for a shrink file.

        Args:
            shrink_path: Path to the shrink image.

        Returns:
            Path to the original image, or None if not found.
        """
        original_stem = self.image_processor.get_original_name_from_shrink(shrink_path)

        # Search for original in the original folder
        for ext in self.image_processor.SUPPORTED_EXTENSIONS:
            for original_path in self.original_folder.rglob(f"{original_stem}{ext}"):
                return original_path
            for original_path in self.original_folder.rglob(
                f"{original_stem}{ext.upper()}"
            ):
                return original_path

        return None

    def process_single_image(
        self,
        shrink_path: Path,
        original_path: Optional[Path] = None,
        raw_mapping: Optional[RawProcessingMapping] = None,
    ) -> ProcessingResult:
        """Process a single shrink image.

        Args:
            shrink_path: Path to the shrink image.
            original_path: Optional path to original (if known).
            raw_mapping: Optional RAW mapping if this is for a RAW file.

        Returns:
            ProcessingResult with status and details.
        """
        # Check if this is a RAW processing task
        is_raw = raw_mapping is not None
        raw_path = raw_mapping.raw_path if raw_mapping else None

        result = ProcessingResult(
            shrink_path=shrink_path,
            is_raw=is_raw,
            raw_path=raw_path,
        )

        try:
            # Find original if not provided
            if original_path is None:
                if raw_mapping:
                    original_path = raw_mapping.source_path
                else:
                    original_path = self._find_original_image(shrink_path)

            if original_path is None:
                result.error = "Original image not found"
                logger.warning(
                    f"Original image not found for shrink: {shrink_path}"
                )
                return result

            result.original_path = original_path

            # Determine the target for XMP
            if is_raw and raw_path:
                xmp_target = raw_path
                # Skip if XMP already exists for RAW
                if self.xmp_generator.xmp_exists(raw_path, is_raw=True):
                    logger.info(f"XMP already exists for RAW: {raw_path}")
                    result.success = True
                    return result
            else:
                xmp_target = original_path
                # Skip if XMP already exists
                if self.xmp_generator.xmp_exists(original_path):
                    logger.info(f"XMP already exists for: {original_path}")
                    result.success = True
                    return result

            # Analyze with Vision API
            logger.info(f"Analyzing: {shrink_path}")
            vision_result = self.vision_service.analyze_image(shrink_path)

            if vision_result.error:
                result.error = f"Vision API error: {vision_result.error}"
                return result

            # Get all labels
            english_labels = vision_result.get_all_labels()

            if not english_labels:
                logger.warning(f"No labels detected for: {shrink_path}")
                result.error = "No labels detected"
                return result

            # Translate labels
            logger.info(f"Translating {len(english_labels)} labels")
            translated = self.translation_service.translate_labels(english_labels)
            japanese_labels = [t.translated for t in translated]

            # Extract landmarks
            landmarks = [l.description for l in vision_result.landmarks]

            # Generate and save XMP (different handling for RAW)
            if is_raw and raw_path:
                xmp_path = self.xmp_generator.save_xmp_for_raw(
                    raw_image_path=raw_path,
                    japanese_labels=japanese_labels,
                    english_labels=english_labels,
                    raw_labels_ja=self.raw_labels_ja,
                    raw_labels_en=self.raw_labels_en,
                    faces_detected=len(vision_result.faces),
                    landmarks=landmarks,
                )
                logger.info(
                    f"Successfully processed RAW: {raw_path.name} -> {xmp_path.name}"
                )
            else:
                xmp_path = self.xmp_generator.save_xmp(
                    original_image_path=original_path,
                    japanese_labels=japanese_labels,
                    english_labels=english_labels,
                    faces_detected=len(vision_result.faces),
                    landmarks=landmarks,
                )
                logger.info(
                    f"Successfully processed: {original_path.name} -> {xmp_path.name}"
                )

            result.success = True
            result.labels_count = len(japanese_labels)

        except Exception as e:
            result.error = str(e)
            logger.error(f"Error processing {shrink_path}: {e}")

        return result

    def process_batch(self) -> list[ProcessingResult]:
        """Process all pending shrink images.

        Returns:
            List of ProcessingResult objects.
        """
        if self._is_processing:
            logger.warning("Batch processing already in progress")
            return []

        self._is_processing = True
        self._last_run = datetime.now()
        results = []

        try:
            # Get pending shrink images
            shrink_images = self.image_processor.get_pending_shrink_images()
            logger.info(f"Found {len(shrink_images)} shrink images to process")

            for shrink_path in shrink_images:
                # Check if this is a RAW processing task
                raw_mapping = None
                if self.raw_processor:
                    raw_mapping = self.raw_processor.get_mapping(shrink_path)

                # Process image
                result = self.process_single_image(
                    shrink_path,
                    raw_mapping=raw_mapping,
                )
                results.append(result)

                # Clean up shrink image if successful
                if result.success:
                    self.image_processor.cleanup_shrink_image(shrink_path)
                    # Also remove the RAW mapping
                    if self.raw_processor and raw_mapping:
                        self.raw_processor.remove_mapping(shrink_path)

            # Save statistics
            self._save_stats(results)

            # Call completion callback
            if self.on_complete:
                self.on_complete(results)

        finally:
            self._is_processing = False

        # Log summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        raw_count = sum(1 for r in results if r.is_raw and r.success)
        logger.info(
            f"Batch processing complete: {successful} successful "
            f"({raw_count} RAW), {failed} failed"
        )

        return results

    def _save_stats(self, results: list[ProcessingResult]):
        """Save processing statistics to file.

        Args:
            results: List of processing results.
        """
        try:
            self._stats_file.parent.mkdir(parents=True, exist_ok=True)

            stats = {
                "timestamp": datetime.now().isoformat(),
                "total_processed": len(results),
                "successful": sum(1 for r in results if r.success),
                "failed": sum(1 for r in results if not r.success),
                "raw_processed": sum(1 for r in results if r.is_raw and r.success),
                "total_labels": sum(r.labels_count for r in results),
                "errors": [
                    {
                        "file": str(r.shrink_path),
                        "error": r.error,
                        "is_raw": r.is_raw,
                    }
                    for r in results
                    if not r.success
                ],
            }

            # Append to stats file
            history = []
            if self._stats_file.exists():
                try:
                    with open(self._stats_file, "r") as f:
                        history = json.load(f)
                except json.JSONDecodeError:
                    history = []

            history.append(stats)
            # Keep last 30 days of stats
            history = history[-30:]

            with open(self._stats_file, "w") as f:
                json.dump(history, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save stats: {e}")

    def start_scheduler(self):
        """Start the batch processing scheduler."""
        if self._scheduler is not None:
            logger.warning("Scheduler already running")
            return

        self._scheduler = BackgroundScheduler(timezone=self.timezone)

        # Schedule batch processing
        trigger = CronTrigger(
            hour=self.batch_hour,
            minute=self.batch_minute,
            timezone=self.timezone,
        )

        self._scheduler.add_job(
            self.process_batch,
            trigger=trigger,
            id="batch_processor",
            name="Night Batch Processing",
            replace_existing=True,
        )

        self._scheduler.start()
        logger.info(
            f"Scheduler started: batch processing at "
            f"{self.batch_hour:02d}:{self.batch_minute:02d} ({self.timezone})"
        )

    def stop_scheduler(self):
        """Stop the batch processing scheduler."""
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None
            logger.info("Scheduler stopped")

    def run_now(self) -> list[ProcessingResult]:
        """Run batch processing immediately.

        Returns:
            List of ProcessingResult objects.
        """
        logger.info("Running batch processing immediately")
        return self.process_batch()

    def get_status(self) -> dict:
        """Get current status of the batch processor.

        Returns:
            Dictionary with status information.
        """
        pending = self.image_processor.get_pending_shrink_images()

        return {
            "is_processing": self._is_processing,
            "scheduler_running": self._scheduler is not None
            and self._scheduler.running,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "next_run": (
                self._scheduler.get_job("batch_processor").next_run_time.isoformat()
                if self._scheduler and self._scheduler.get_job("batch_processor")
                else None
            ),
            "pending_images": len(pending),
            "scheduled_time": f"{self.batch_hour:02d}:{self.batch_minute:02d}",
            "timezone": self.timezone,
        }
