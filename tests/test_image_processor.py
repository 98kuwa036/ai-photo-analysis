"""Tests for ImageProcessor service."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.image_processor import ImageProcessor


class TestImageProcessor:
    """Tests for ImageProcessor class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def processor(self, temp_dir):
        """Create an ImageProcessor instance."""
        return ImageProcessor(
            shrink_size=640,
            jpeg_quality=85,
            temp_folder=temp_dir / "shrink",
        )

    def test_is_supported(self, processor):
        """Test file extension support checking."""
        assert processor.is_supported(Path("photo.jpg"))
        assert processor.is_supported(Path("photo.JPEG"))
        assert processor.is_supported(Path("photo.png"))
        assert processor.is_supported(Path("photo.heic"))
        assert not processor.is_supported(Path("document.pdf"))
        assert not processor.is_supported(Path("video.mp4"))

    def test_get_shrink_path(self, processor):
        """Test shrink path generation."""
        original = Path("/photos/IMG_001.jpg")
        shrink_path = processor.get_shrink_path(original)

        assert shrink_path.stem == "IMG_001_shrink"
        assert shrink_path.suffix == ".jpg"

    def test_get_original_name_from_shrink(self, processor):
        """Test original name extraction from shrink filename."""
        shrink = Path("/temp/IMG_001_shrink.jpg")
        original_name = processor.get_original_name_from_shrink(shrink)

        assert original_name == "IMG_001"

    def test_calculate_new_size_landscape(self, processor):
        """Test size calculation for landscape images."""
        # Width is longer, so height is the short edge
        new_width, new_height = processor._calculate_new_size(1920, 1080)

        assert new_height == 640
        assert new_width == int(1920 * (640 / 1080))

    def test_calculate_new_size_portrait(self, processor):
        """Test size calculation for portrait images."""
        # Height is longer, so width is the short edge
        new_width, new_height = processor._calculate_new_size(1080, 1920)

        assert new_width == 640
        assert new_height == int(1920 * (640 / 1080))

    def test_calculate_new_size_small_image(self, processor):
        """Test that small images are not enlarged."""
        # Image smaller than shrink size
        new_width, new_height = processor._calculate_new_size(400, 300)

        assert new_width == 400
        assert new_height == 300

    def test_get_pending_shrink_images(self, processor, temp_dir):
        """Test getting list of pending shrink images."""
        shrink_folder = temp_dir / "shrink"
        shrink_folder.mkdir(parents=True, exist_ok=True)

        # Create some test files
        (shrink_folder / "IMG_001_shrink.jpg").touch()
        (shrink_folder / "IMG_002_shrink.jpg").touch()
        (shrink_folder / "regular.jpg").touch()  # Should not be included

        pending = processor.get_pending_shrink_images()

        assert len(pending) == 2
        assert all("_shrink" in p.stem for p in pending)


class TestImageProcessorIntegration:
    """Integration tests requiring actual image files."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def processor(self, temp_dir):
        """Create an ImageProcessor instance."""
        return ImageProcessor(
            shrink_size=100,
            jpeg_quality=75,
            temp_folder=temp_dir / "shrink",
        )

    def test_create_shrink_image_with_pil(self, processor, temp_dir):
        """Test actual image shrinking with PIL."""
        try:
            from PIL import Image

            # Create a test image
            original_path = temp_dir / "test.jpg"
            img = Image.new("RGB", (800, 600), color="red")
            img.save(original_path, "JPEG")

            # Create shrink
            shrink_path = processor.create_shrink_image(original_path)

            assert shrink_path is not None
            assert shrink_path.exists()

            # Verify dimensions
            with Image.open(shrink_path) as shrink_img:
                # Short edge (height) should be 100 or less
                assert min(shrink_img.size) <= 100

        except ImportError:
            pytest.skip("PIL not installed")

    def test_cleanup_shrink_image(self, processor, temp_dir):
        """Test shrink image cleanup."""
        shrink_folder = temp_dir / "shrink"
        shrink_folder.mkdir(parents=True, exist_ok=True)

        shrink_path = shrink_folder / "test_shrink.jpg"
        shrink_path.touch()

        assert shrink_path.exists()

        result = processor.cleanup_shrink_image(shrink_path)

        assert result is True
        assert not shrink_path.exists()
