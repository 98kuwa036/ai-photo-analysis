"""XMP Metadata Generator

Generates XMP sidecar files with translated labels for Mylio Photos integration.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

# XMP namespace definitions
NAMESPACES = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "xmpMM": "http://ns.adobe.com/xap/1.0/mm/",
    "lr": "http://ns.adobe.com/lightroom/1.0/",
    "photoshop": "http://ns.adobe.com/photoshop/1.0/",
    "aux": "http://ns.adobe.com/exif/1.0/aux/",
}


class XMPGenerator:
    """Generates XMP sidecar files for photo metadata."""

    def __init__(
        self,
        creator_tool: str = "AI Photo Analyzer",
        include_original: bool = True,
    ):
        """Initialize the XMP generator.

        Args:
            creator_tool: Name of the tool creating the XMP.
            include_original: Whether to include original English labels.
        """
        self.creator_tool = creator_tool
        self.include_original = include_original

        # Register namespaces
        for prefix, uri in NAMESPACES.items():
            ET.register_namespace(prefix, uri)

    def _create_rdf_bag(self, parent: ET.Element, items: list[str]) -> ET.Element:
        """Create an RDF Bag element with items.

        Args:
            parent: Parent element.
            items: List of items to include.

        Returns:
            The Bag element.
        """
        bag = ET.SubElement(parent, f"{{{NAMESPACES['rdf']}}}Bag")
        for item in items:
            li = ET.SubElement(bag, f"{{{NAMESPACES['rdf']}}}li")
            li.text = item
        return bag

    def _create_rdf_alt(
        self, parent: ET.Element, text: str, lang: str = "x-default"
    ) -> ET.Element:
        """Create an RDF Alt element for language alternatives.

        Args:
            parent: Parent element.
            text: Text content.
            lang: Language code.

        Returns:
            The Alt element.
        """
        alt = ET.SubElement(parent, f"{{{NAMESPACES['rdf']}}}Alt")
        li = ET.SubElement(alt, f"{{{NAMESPACES['rdf']}}}li")
        li.set(f"{{{NAMESPACES['rdf']}}}lang", lang)
        li.text = text
        return alt

    def generate_xmp(
        self,
        japanese_labels: list[str],
        english_labels: Optional[list[str]] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        rating: Optional[int] = None,
        faces_detected: int = 0,
        landmarks: Optional[list[str]] = None,
    ) -> str:
        """Generate XMP content as a string.

        Args:
            japanese_labels: List of Japanese labels/keywords.
            english_labels: Optional list of original English labels.
            title: Optional title for the photo.
            description: Optional description.
            rating: Optional rating (1-5).
            faces_detected: Number of faces detected.
            landmarks: Optional list of detected landmarks.

        Returns:
            XMP content as a string.
        """
        # Create root xmpmeta element
        xmpmeta = ET.Element(f"{{{NAMESPACES['x']}}}xmpmeta")
        xmpmeta.set(f"{{{NAMESPACES['x']}}}xmptk", self.creator_tool)

        # Create RDF root
        rdf = ET.SubElement(xmpmeta, f"{{{NAMESPACES['rdf']}}}RDF")

        # Create Description element
        desc = ET.SubElement(rdf, f"{{{NAMESPACES['rdf']}}}Description")
        desc.set(f"{{{NAMESPACES['rdf']}}}about", "")

        # Add XMP basic metadata
        modify_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        ET.SubElement(desc, f"{{{NAMESPACES['xmp']}}}ModifyDate").text = modify_date
        ET.SubElement(desc, f"{{{NAMESPACES['xmp']}}}CreatorTool").text = self.creator_tool

        # Combine keywords
        all_keywords = list(japanese_labels)
        if self.include_original and english_labels:
            # Add unique English labels
            for label in english_labels:
                if label not in all_keywords:
                    all_keywords.append(label)

        # Add landmark names as keywords too
        if landmarks:
            for landmark in landmarks:
                if landmark not in all_keywords:
                    all_keywords.append(landmark)

        # Add dc:subject (keywords/tags) - This is what Mylio reads
        if all_keywords:
            subject = ET.SubElement(desc, f"{{{NAMESPACES['dc']}}}subject")
            self._create_rdf_bag(subject, all_keywords)

        # Add Lightroom hierarchical keywords for better organization
        if all_keywords:
            lr_keywords = ET.SubElement(
                desc, f"{{{NAMESPACES['lr']}}}hierarchicalSubject"
            )
            self._create_rdf_bag(lr_keywords, all_keywords)

        # Add title if provided
        if title:
            dc_title = ET.SubElement(desc, f"{{{NAMESPACES['dc']}}}title")
            self._create_rdf_alt(dc_title, title)

        # Add description if provided
        if description:
            dc_desc = ET.SubElement(desc, f"{{{NAMESPACES['dc']}}}description")
            self._create_rdf_alt(dc_desc, description)

        # Add rating if provided
        if rating is not None and 1 <= rating <= 5:
            ET.SubElement(desc, f"{{{NAMESPACES['xmp']}}}Rating").text = str(rating)

        # Add face count as supplementary info
        if faces_detected > 0:
            # Use photoshop:Headline for face info
            face_info = f"{faces_detected}人の顔を検出"
            ET.SubElement(desc, f"{{{NAMESPACES['photoshop']}}}Headline").text = face_info

        # Convert to string
        xml_str = ET.tostring(xmpmeta, encoding="unicode", method="xml")

        # Add XML declaration and proper formatting
        xmp_content = f'''<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="{self.creator_tool}">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:dc="http://purl.org/dc/elements/1.1/"
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmlns:lr="http://ns.adobe.com/lightroom/1.0/"
      xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"
      xmp:ModifyDate="{modify_date}"
      xmp:CreatorTool="{self.creator_tool}">
'''

        # Add keywords
        if all_keywords:
            xmp_content += "      <dc:subject>\n        <rdf:Bag>\n"
            for keyword in all_keywords:
                xmp_content += f"          <rdf:li>{self._escape_xml(keyword)}</rdf:li>\n"
            xmp_content += "        </rdf:Bag>\n      </dc:subject>\n"

            # Lightroom hierarchical subject
            xmp_content += "      <lr:hierarchicalSubject>\n        <rdf:Bag>\n"
            for keyword in all_keywords:
                xmp_content += f"          <rdf:li>{self._escape_xml(keyword)}</rdf:li>\n"
            xmp_content += "        </rdf:Bag>\n      </lr:hierarchicalSubject>\n"

        # Add title
        if title:
            xmp_content += f'''      <dc:title>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">{self._escape_xml(title)}</rdf:li>
        </rdf:Alt>
      </dc:title>
'''

        # Add description
        if description:
            xmp_content += f'''      <dc:description>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">{self._escape_xml(description)}</rdf:li>
        </rdf:Alt>
      </dc:description>
'''

        # Add rating
        if rating is not None and 1 <= rating <= 5:
            xmp_content += f"      <xmp:Rating>{rating}</xmp:Rating>\n"

        # Add face count
        if faces_detected > 0:
            xmp_content += f"      <photoshop:Headline>{faces_detected}人の顔を検出</photoshop:Headline>\n"

        xmp_content += '''    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>'''

        return xmp_content

    def _escape_xml(self, text: str) -> str:
        """Escape special XML characters.

        Args:
            text: Text to escape.

        Returns:
            Escaped text.
        """
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    def save_xmp(
        self,
        original_image_path: Path,
        japanese_labels: list[str],
        english_labels: Optional[list[str]] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        rating: Optional[int] = None,
        faces_detected: int = 0,
        landmarks: Optional[list[str]] = None,
    ) -> Path:
        """Generate and save XMP file next to the original image.

        Args:
            original_image_path: Path to the original image.
            japanese_labels: List of Japanese labels/keywords.
            english_labels: Optional list of original English labels.
            title: Optional title for the photo.
            description: Optional description.
            rating: Optional rating (1-5).
            faces_detected: Number of faces detected.
            landmarks: Optional list of detected landmarks.

        Returns:
            Path to the saved XMP file.
        """
        # Generate XMP path (same name as image, but .xmp extension)
        xmp_path = original_image_path.with_suffix(".xmp")

        # Generate XMP content
        xmp_content = self.generate_xmp(
            japanese_labels=japanese_labels,
            english_labels=english_labels,
            title=title,
            description=description,
            rating=rating,
            faces_detected=faces_detected,
            landmarks=landmarks,
        )

        # Save to file
        xmp_path.write_text(xmp_content, encoding="utf-8")

        logger.info(f"Saved XMP file: {xmp_path}")
        return xmp_path

    def get_xmp_path(self, image_path: Path, is_raw: bool = False) -> Path:
        """Get the expected XMP path for an image.

        Args:
            image_path: Path to the image.
            is_raw: Whether this is a RAW file.

        Returns:
            Expected path to the XMP sidecar file.
        """
        if is_raw:
            # RAW files use filename.CR2.xmp format (Adobe/Lightroom convention)
            return image_path.parent / f"{image_path.name}.xmp"
        else:
            return image_path.with_suffix(".xmp")

    def xmp_exists(self, image_path: Path, is_raw: bool = False) -> bool:
        """Check if XMP file already exists for an image.

        Args:
            image_path: Path to the image.
            is_raw: Whether this is a RAW file.

        Returns:
            True if XMP file exists.
        """
        return self.get_xmp_path(image_path, is_raw).exists()

    def save_xmp_for_raw(
        self,
        raw_image_path: Path,
        japanese_labels: list[str],
        english_labels: Optional[list[str]] = None,
        raw_labels_ja: Optional[list[str]] = None,
        raw_labels_en: Optional[list[str]] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        rating: Optional[int] = None,
        faces_detected: int = 0,
        landmarks: Optional[list[str]] = None,
    ) -> Path:
        """Generate and save XMP file for a RAW image.

        RAW files use a different naming convention: filename.CR2.xmp
        Also adds RAW-specific labels to indicate the file type.

        Args:
            raw_image_path: Path to the RAW image.
            japanese_labels: List of Japanese labels/keywords.
            english_labels: Optional list of original English labels.
            raw_labels_ja: Japanese labels for RAW (e.g., ["RAW", "RAW画像"]).
            raw_labels_en: English labels for RAW (e.g., ["RAW", "RAW Image"]).
            title: Optional title for the photo.
            description: Optional description.
            rating: Optional rating (1-5).
            faces_detected: Number of faces detected.
            landmarks: Optional list of detected landmarks.

        Returns:
            Path to the saved XMP file.
        """
        # Add RAW-specific labels at the beginning
        enhanced_ja_labels = list(raw_labels_ja or ["RAW", "RAW画像"])
        enhanced_en_labels = list(raw_labels_en or ["RAW", "RAW Image"])

        # Add the raw file extension as a label (e.g., "CR2", "NEF")
        raw_ext = raw_image_path.suffix.upper().lstrip(".")
        if raw_ext:
            enhanced_ja_labels.append(raw_ext)
            enhanced_en_labels.append(raw_ext)

        # Merge with detected labels
        enhanced_ja_labels.extend(japanese_labels)
        if english_labels:
            enhanced_en_labels.extend(english_labels)

        # Generate XMP path for RAW (filename.CR2.xmp format)
        xmp_path = self.get_xmp_path(raw_image_path, is_raw=True)

        # Generate XMP content
        xmp_content = self.generate_xmp(
            japanese_labels=enhanced_ja_labels,
            english_labels=enhanced_en_labels if self.include_original else None,
            title=title,
            description=description,
            rating=rating,
            faces_detected=faces_detected,
            landmarks=landmarks,
        )

        # Save to file
        xmp_path.write_text(xmp_content, encoding="utf-8")

        logger.info(f"Saved RAW XMP file: {xmp_path}")
        return xmp_path
