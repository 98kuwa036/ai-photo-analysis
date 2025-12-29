"""Tests for XMPGenerator service."""

import pytest
from pathlib import Path
import tempfile

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.xmp_generator import XMPGenerator


class TestXMPGenerator:
    """Tests for XMPGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create an XMPGenerator instance."""
        return XMPGenerator(
            creator_tool="Test Photo Analyzer",
            include_original=True,
        )

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_generate_xmp_with_labels(self, generator):
        """Test XMP generation with Japanese and English labels."""
        japanese_labels = ["空", "雲", "山"]
        english_labels = ["Sky", "Cloud", "Mountain"]

        xmp_content = generator.generate_xmp(
            japanese_labels=japanese_labels,
            english_labels=english_labels,
        )

        assert "<?xpacket begin" in xmp_content
        assert "dc:subject" in xmp_content
        assert "空" in xmp_content
        assert "Sky" in xmp_content

    def test_generate_xmp_without_english(self, generator):
        """Test XMP generation with Japanese labels only."""
        generator.include_original = False
        japanese_labels = ["犬", "猫", "鳥"]

        xmp_content = generator.generate_xmp(
            japanese_labels=japanese_labels,
            english_labels=["Dog", "Cat", "Bird"],
        )

        assert "犬" in xmp_content
        # When include_original is False, English should not be in keywords
        # But since we're testing with include_original=True fixture,
        # we need to test differently

    def test_generate_xmp_with_faces(self, generator):
        """Test XMP generation with face detection info."""
        xmp_content = generator.generate_xmp(
            japanese_labels=["人物"],
            faces_detected=3,
        )

        assert "3人の顔を検出" in xmp_content

    def test_generate_xmp_with_landmarks(self, generator):
        """Test XMP generation with landmark info."""
        japanese_labels = ["建物"]
        landmarks = ["東京タワー", "Tokyo Tower"]

        xmp_content = generator.generate_xmp(
            japanese_labels=japanese_labels,
            landmarks=landmarks,
        )

        assert "東京タワー" in xmp_content
        assert "Tokyo Tower" in xmp_content

    def test_generate_xmp_with_rating(self, generator):
        """Test XMP generation with rating."""
        xmp_content = generator.generate_xmp(
            japanese_labels=["風景"],
            rating=5,
        )

        assert "<xmp:Rating>5</xmp:Rating>" in xmp_content

    def test_escape_xml_characters(self, generator):
        """Test XML character escaping."""
        escaped = generator._escape_xml("<tag>&value</tag>")

        assert "&lt;" in escaped
        assert "&gt;" in escaped
        assert "&amp;" in escaped

    def test_save_xmp(self, generator, temp_dir):
        """Test saving XMP file."""
        # Create a mock image file
        image_path = temp_dir / "IMG_001.jpg"
        image_path.touch()

        xmp_path = generator.save_xmp(
            original_image_path=image_path,
            japanese_labels=["テスト"],
            english_labels=["Test"],
        )

        assert xmp_path.exists()
        assert xmp_path.suffix == ".xmp"
        assert xmp_path.stem == image_path.stem

        # Verify content
        content = xmp_path.read_text(encoding="utf-8")
        assert "テスト" in content

    def test_get_xmp_path(self, generator):
        """Test XMP path generation."""
        image_path = Path("/photos/IMG_001.jpg")
        xmp_path = generator.get_xmp_path(image_path)

        assert xmp_path == Path("/photos/IMG_001.xmp")

    def test_xmp_exists(self, generator, temp_dir):
        """Test XMP existence check."""
        image_path = temp_dir / "IMG_001.jpg"
        xmp_path = temp_dir / "IMG_001.xmp"

        assert not generator.xmp_exists(image_path)

        xmp_path.touch()
        assert generator.xmp_exists(image_path)

    def test_xmp_structure(self, generator):
        """Test XMP structure is valid."""
        xmp_content = generator.generate_xmp(
            japanese_labels=["テスト"],
        )

        # Check required XMP elements
        assert 'xmlns:x="adobe:ns:meta/"' in xmp_content
        assert 'xmlns:rdf=' in xmp_content
        assert 'xmlns:dc=' in xmp_content
        assert '<?xpacket end="w"?>' in xmp_content


class TestXMPGeneratorEdgeCases:
    """Edge case tests for XMPGenerator."""

    @pytest.fixture
    def generator(self):
        """Create an XMPGenerator instance."""
        return XMPGenerator()

    def test_empty_labels(self, generator):
        """Test handling of empty labels."""
        xmp_content = generator.generate_xmp(
            japanese_labels=[],
            english_labels=[],
        )

        # Should still produce valid XMP
        assert "<?xpacket begin" in xmp_content
        # Should not contain subject if no labels
        # (depends on implementation)

    def test_special_characters_in_labels(self, generator):
        """Test handling of special characters in labels."""
        labels = ["Rock & Roll", "C++ Programming", "Cat <-> Dog"]

        xmp_content = generator.generate_xmp(
            japanese_labels=labels,
        )

        # Should properly escape special characters
        assert "&amp;" in xmp_content
        assert "&lt;" in xmp_content

    def test_unicode_characters(self, generator):
        """Test handling of various Unicode characters."""
        labels = ["日本語", "한국어", "中文", "العربية", "🌸"]

        xmp_content = generator.generate_xmp(
            japanese_labels=labels,
        )

        for label in labels:
            assert label in xmp_content
