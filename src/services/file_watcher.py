"""File Watcher Service

Monitors OneDrive folder for new photo uploads and triggers
lightweight processing immediately.
"""

import logging
import time
from pathlib import Path
from typing import Callable, Optional, Set
from threading import Thread, Event

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent

logger = logging.getLogger(__name__)


class PhotoEventHandler(FileSystemEventHandler):
    """Handles file system events for photo uploads."""

    def __init__(
        self,
        on_new_photo: Callable[[Path], None],
        supported_extensions: Set[str],
        debounce_seconds: float = 2.0,
    ):
        """Initialize the event handler.

        Args:
            on_new_photo: Callback function when a new photo is detected.
            supported_extensions: Set of supported file extensions.
            debounce_seconds: Time to wait before processing (for upload completion).
        """
        super().__init__()
        self.on_new_photo = on_new_photo
        self.supported_extensions = supported_extensions
        self.debounce_seconds = debounce_seconds
        self._pending_files: dict[str, float] = {}
        self._processed_files: Set[str] = set()
        self._processing_thread: Optional[Thread] = None
        self._stop_event = Event()

    def _is_photo(self, path: Path) -> bool:
        """Check if the file is a supported photo format.

        Args:
            path: Path to the file.

        Returns:
            True if the file is a supported photo.
        """
        return path.suffix.lower() in self.supported_extensions

    def _is_shrink_file(self, path: Path) -> bool:
        """Check if the file is a shrink file (to avoid recursive processing).

        Args:
            path: Path to the file.

        Returns:
            True if the file is a shrink file.
        """
        return "_shrink" in path.stem

    def _process_pending(self):
        """Process pending files after debounce period."""
        while not self._stop_event.is_set():
            current_time = time.time()
            files_to_process = []

            # Find files ready for processing
            for file_path, timestamp in list(self._pending_files.items()):
                if current_time - timestamp >= self.debounce_seconds:
                    files_to_process.append(file_path)

            # Process ready files
            for file_path in files_to_process:
                del self._pending_files[file_path]
                if file_path not in self._processed_files:
                    self._processed_files.add(file_path)
                    path = Path(file_path)
                    if path.exists():
                        logger.info(f"Processing new photo: {path}")
                        try:
                            self.on_new_photo(path)
                        except Exception as e:
                            logger.error(f"Error processing {path}: {e}")
                            self._processed_files.discard(file_path)

            time.sleep(0.5)

    def on_created(self, event):
        """Handle file creation events.

        Args:
            event: File system event.
        """
        if event.is_directory:
            return

        path = Path(event.src_path)
        if self._is_photo(path) and not self._is_shrink_file(path):
            logger.debug(f"New file detected: {path}")
            self._pending_files[str(path)] = time.time()

    def on_modified(self, event):
        """Handle file modification events (for upload completion).

        Args:
            event: File system event.
        """
        if event.is_directory:
            return

        path = Path(event.src_path)
        file_key = str(path)

        # Update timestamp for pending files (upload still in progress)
        if file_key in self._pending_files:
            self._pending_files[file_key] = time.time()

    def start_processing(self):
        """Start the background processing thread."""
        self._stop_event.clear()
        self._processing_thread = Thread(target=self._process_pending, daemon=True)
        self._processing_thread.start()

    def stop_processing(self):
        """Stop the background processing thread."""
        self._stop_event.set()
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)


class FileWatcher:
    """Watches directories for new photo uploads."""

    def __init__(
        self,
        watch_paths: list[Path],
        on_new_photo: Callable[[Path], None],
        supported_extensions: Optional[Set[str]] = None,
        recursive: bool = True,
    ):
        """Initialize the file watcher.

        Args:
            watch_paths: List of directories to watch.
            on_new_photo: Callback function when a new photo is detected.
            supported_extensions: Set of supported file extensions.
            recursive: Whether to watch subdirectories.
        """
        self.watch_paths = watch_paths
        self.recursive = recursive
        self.supported_extensions = supported_extensions or {
            ".jpg",
            ".jpeg",
            ".png",
            ".heic",
            ".heif",
            ".webp",
        }

        self._event_handler = PhotoEventHandler(
            on_new_photo=on_new_photo,
            supported_extensions=self.supported_extensions,
        )
        self._observer = Observer()
        self._running = False

    def start(self):
        """Start watching for file changes."""
        if self._running:
            logger.warning("File watcher is already running")
            return

        for watch_path in self.watch_paths:
            if not watch_path.exists():
                logger.warning(f"Watch path does not exist: {watch_path}")
                continue

            self._observer.schedule(
                self._event_handler,
                str(watch_path),
                recursive=self.recursive,
            )
            logger.info(f"Watching directory: {watch_path}")

        self._event_handler.start_processing()
        self._observer.start()
        self._running = True
        logger.info("File watcher started")

    def stop(self):
        """Stop watching for file changes."""
        if not self._running:
            return

        self._observer.stop()
        self._event_handler.stop_processing()
        self._observer.join(timeout=5.0)
        self._running = False
        logger.info("File watcher stopped")

    def is_running(self) -> bool:
        """Check if the watcher is running.

        Returns:
            True if the watcher is running.
        """
        return self._running

    def scan_existing(self) -> list[Path]:
        """Scan for existing photos in watch directories.

        Returns:
            List of existing photo paths.
        """
        photos = []
        for watch_path in self.watch_paths:
            if not watch_path.exists():
                continue

            for ext in self.supported_extensions:
                if self.recursive:
                    photos.extend(watch_path.rglob(f"*{ext}"))
                    photos.extend(watch_path.rglob(f"*{ext.upper()}"))
                else:
                    photos.extend(watch_path.glob(f"*{ext}"))
                    photos.extend(watch_path.glob(f"*{ext.upper()}"))

        # Filter out shrink files
        photos = [p for p in photos if "_shrink" not in p.stem]
        return sorted(photos)
