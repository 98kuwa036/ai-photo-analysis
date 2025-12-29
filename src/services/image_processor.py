"""Image Processing Service

Handles image resizing and optimization for AI analysis.
Creates lightweight shrink copies to minimize API transfer costs.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ExifTags

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Processes images for AI analysis."""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
    SHRINK_SUFFIX = "_shrink"

    def __init__(
        self,
        shrink_size: int = 640,
        jpeg_quality: int = 85,
        temp_folder: Optional[Path] = None,
    ):
        """Initialize the image processor.

        Args:
            shrink_size: Target size for the short edge in pixels.
            jpeg_quality: JPEG compression quality (1-100).
            temp_folder: Folder to store shrink images.
        """
        self.shrink_size = shrink_size
        self.jpeg_quality = jpeg_quality
        self.temp_folder = temp_folder or Path("temp/shrink")
        self.temp_folder.mkdir(parents=True, exist_ok=True)

    def is_supported(self, file_path: Path) -> bool:
        """Check if the file is a supported image format.

        Args:
            file_path: Path to the image file.

        Returns:
            True if the file format is supported.
        """
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def get_shrink_path(self, original_path: Path) -> Path:
        """Generate the shrink file path for an original image.

        Args:
            original_path: Path to the original image.

        Returns:
            Path where the shrink image should be saved.
        """
        stem = original_path.stem
        suffix = ".jpg"  # Always save shrink images as JPEG
        shrink_name = f"{stem}{self.SHRINK_SUFFIX}{suffix}"
        return self.temp_folder / shrink_name

    def get_original_name_from_shrink(self, shrink_path: Path) -> str:
        """Extract the original filename from a shrink filename.

        Args:
            shrink_path: Path to the shrink image.

        Returns:
            Original filename without the _shrink suffix.
        """
        stem = shrink_path.stem
        if stem.endswith(self.SHRINK_SUFFIX):
            stem = stem[: -len(self.SHRINK_SUFFIX)]
        return stem

    def _calculate_new_size(self, width: int, height: int) -> Tuple[int, int]:
        """Calculate new dimensions maintaining aspect ratio.

        Args:
            width: Original width.
            height: Original height.

        Returns:
            Tuple of (new_width, new_height).
        """
        # Determine which side is shorter
        if width <= height:
            # Width is shorter or equal
            if width <= self.shrink_size:
                return width, height
            new_width = self.shrink_size
            new_height = int(height * (self.shrink_size / width))
        else:
            # Height is shorter
            if height <= self.shrink_size:
                return width, height
            new_height = self.shrink_size
            new_width = int(width * (self.shrink_size / height))

        return new_width, new_height

    def _get_exif_orientation(self, image: Image.Image) -> Optional[int]:
        """Get EXIF orientation value from image.

        Args:
            image: PIL Image object.

        Returns:
            EXIF orientation value or None.
        """
        try:
            exif = image._getexif()
            if exif is not None:
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag == "Orientation":
                        return value
        except (AttributeError, KeyError, IndexError):
            pass
        return None

    def _apply_exif_orientation(self, image: Image.Image) -> Image.Image:
        """Apply EXIF orientation to image.

        Args:
            image: PIL Image object.

        Returns:
            Correctly oriented image.
        """
        orientation = self._get_exif_orientation(image)
        if orientation is None:
            return image

        operations = {
            2: (Image.FLIP_LEFT_RIGHT,),
            3: (Image.ROTATE_180,),
            4: (Image.FLIP_TOP_BOTTOM,),
            5: (Image.FLIP_LEFT_RIGHT, Image.ROTATE_90),
            6: (Image.ROTATE_270,),
            7: (Image.FLIP_LEFT_RIGHT, Image.ROTATE_270),
            8: (Image.ROTATE_90,),
        }

        if orientation in operations:
            for op in operations[orientation]:
                image = image.transpose(op)

        return image

    def create_shrink_image(self, original_path: Path) -> Optional[Path]:
        """Create a shrink copy of an image for AI analysis.

        Args:
            original_path: Path to the original image.

        Returns:
            Path to the shrink image, or None if processing failed.
        """
        if not self.is_supported(original_path):
            logger.warning(f"Unsupported image format: {original_path}")
            return None

        shrink_path = self.get_shrink_path(original_path)

        try:
            with Image.open(original_path) as img:
                # Convert to RGB if necessary (for PNG with alpha, etc.)
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")

                # Apply EXIF orientation
                img = self._apply_exif_orientation(img)

                # Calculate new size
                new_width, new_height = self._calculate_new_size(img.width, img.height)

                # Resize using high-quality Lanczos resampling
                resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Save as JPEG
                resized.save(
                    shrink_path,
                    "JPEG",
                    quality=self.jpeg_quality,
                    optimize=True,
                )

                logger.info(
                    f"Created shrink image: {shrink_path} "
                    f"({img.width}x{img.height} -> {new_width}x{new_height})"
                )
                return shrink_path

        except Exception as e:
            logger.error(f"Failed to create shrink image for {original_path}: {e}")
            return None

    def cleanup_shrink_image(self, shrink_path: Path) -> bool:
        """Delete a shrink image after processing.

        Args:
            shrink_path: Path to the shrink image.

        Returns:
            True if deletion was successful.
        """
        try:
            if shrink_path.exists():
                shrink_path.unlink()
                logger.info(f"Deleted shrink image: {shrink_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to delete shrink image {shrink_path}: {e}")
        return False

    def get_pending_shrink_images(self) -> list[Path]:
        """Get list of shrink images waiting to be processed.

        Returns:
            List of paths to shrink images.
        """
        pattern = f"*{self.SHRINK_SUFFIX}.jpg"
        return list(self.temp_folder.glob(pattern))
