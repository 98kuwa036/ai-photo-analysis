#!/usr/bin/env python3
"""Cloud Runner for AI Photo Analysis

Simplified runner for cloud environments (GitHub Actions, Google Colab, etc.)
Processes images from a local directory and generates XMP files.

Usage:
    python -m src.cloud_runner --input-dir ./photos --output-dir ./output --cache-file ./config/translation_cache.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Set

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Verify required environment variables are set."""
    required_vars = ["GOOGLE_APPLICATION_CREDENTIALS", "DEEPL_API_KEY"]
    missing = [var for var in required_vars if not os.environ.get(var)]

    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        logger.info("Required variables:")
        logger.info("  GOOGLE_APPLICATION_CREDENTIALS: Path to Google Cloud service account JSON")
        logger.info("  DEEPL_API_KEY: DeepL API authentication key")
        return False

    return True


class DeepLTranslator:
    """Handles translation with local caching to save API usage."""

    def __init__(self, api_key: str, cache_file: Path):
        self.translator = None
        if api_key:
            import deepl
            self.translator = deepl.Translator(api_key)
        
        self.cache_file = cache_file
        self.cache: Dict[str, str] = self._load_cache()
        self.modified = False

    def _load_cache(self) -> Dict[str, str]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return {}
        return {}

    def save_cache(self):
        if self.modified:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logger.info("Translation cache saved.")

    def translate_text(self, text: str) -> str:
        """Translate text from English to Japanese using cache or API."""
        if not text:
            return ""
            
        # Check cache first
        if text in self.cache:
            return self.cache[text]

        if not self.translator:
            return text

        # Call DeepL API
        try:
            result = self.translator.translate_text(
                text,
                source_lang="EN",
                target_lang="JA"
            )
            translated_text = result.text
            
            # Update cache
            self.cache[text] = translated_text
            self.modified = True
            logger.info(f"DeepL Translated: {text} -> {translated_text}")
            
            return translated_text
        except Exception as e:
            logger.error(f"DeepL API Error: {e}")
            return text


class CloudPhotoProcessor:
    """Simplified photo processor for cloud environments."""

    # Supported extensions
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
    RAW_EXTENSIONS = {
        ".cr2", ".cr3", ".crw",  # Canon
        ".nef", ".nrw",  # Nikon
        ".arw", ".srf", ".sr2",  # Sony
        ".raf",  # Fujifilm
        ".orf",  # Olympus
        ".rw2",  # Panasonic
        ".pef", ".dng", ".rwl",  # Pentax, Leica, Adobe
        ".srw", ".x3f",  # Samsung, Sigma
    }

    RAW_LABELS_JA = ["RAW", "RAW画像", "生データ"]
    RAW_LABELS_EN = ["RAW", "RAW Image", "Unprocessed"]

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        temp_dir: Path,
        cache_file: Path,
        travel_keywords: str = "",
        shrink_size: int = 640,
        force_reprocess: bool = False,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.shrink_size = shrink_size
        self.force_reprocess = force_reprocess
        
        # Parse travel keywords
        self.travel_keywords = [k.strip().lower() for k in travel_keywords.split(',') if k.strip()]

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize services
        self._vision_client = None
        self.translator = DeepLTranslator(
            os.environ.get("DEEPL_API_KEY", ""),
            cache_file
        )

        # Statistics
        self.stats = {
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "raw_processed": 0,
            "labels_total": 0,
        }

    @property
    def vision_client(self):
        """Lazy initialization of Vision API client."""
        if self._vision_client is None:
            from google.cloud import vision
            self._vision_client = vision.ImageAnnotatorClient()
        return self._vision_client

    def is_raw_file(self, path: Path) -> bool:
        """Check if file is a RAW image."""
        return path.suffix.lower() in self.RAW_EXTENSIONS

    def is_image_file(self, path: Path) -> bool:
        """Check if file is a regular image."""
        return path.suffix.lower() in self.IMAGE_EXTENSIONS

    def is_travel_photo(self, path: Path) -> bool:
        """Check if photo is likely a travel/landscape photo based on path."""
        path_str = str(path).lower()
        return any(k in path_str for k in self.travel_keywords)

    def find_analysis_source(self, raw_path: Path) -> Optional[Path]:
        """Find JPEG/PNG source for RAW file analysis."""
        stem = raw_path.stem
        parent = raw_path.parent

        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            source = parent / f"{stem}{ext}"
            if source.exists():
                return source
        return None

    def get_xmp_path(self, image_path: Path, is_raw: bool = False) -> Path:
        """Get XMP output path for an image."""
        relative = image_path.relative_to(self.input_dir)

        if is_raw:
            xmp_name = f"{image_path.name}.xmp"
        else:
            xmp_name = f"{image_path.stem}.xmp"

        return self.output_dir / relative.parent / xmp_name

    def xmp_exists(self, image_path: Path, is_raw: bool = False) -> bool:
        """Check if XMP already exists."""
        xmp_path = self.get_xmp_path(image_path, is_raw)
        return xmp_path.exists()

    def create_shrink_image(self, image_path: Path) -> Optional[Path]:
        """Create shrink image for Vision API analysis."""
        from PIL import Image

        try:
            shrink_path = self.temp_dir / f"{image_path.stem}_shrink.jpg"

            with Image.open(image_path) as img:
                # Convert to RGB
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")

                # Calculate new size
                width, height = img.size
                if width <= height:
                    if width <= self.shrink_size:
                        new_size = (width, height)
                    else:
                        new_size = (self.shrink_size, int(height * self.shrink_size / width))
                else:
                    if height <= self.shrink_size:
                        new_size = (width, height)
                    else:
                        new_size = (int(width * self.shrink_size / height), self.shrink_size)

                # Resize and save
                resized = img.resize(new_size, Image.Resampling.LANCZOS)
                resized.save(shrink_path, "JPEG", quality=85, optimize=True)

            logger.debug(f"Created shrink: {shrink_path}")
            return shrink_path

        except Exception as e:
            logger.error(f"Failed to create shrink for {image_path}: {e}")
            return None

    def analyze_image(self, image_path: Path, enable_landmark: bool = False) -> dict:
        """Analyze image with Vision API and return results."""
        from google.cloud import vision

        try:
            with open(image_path, "rb") as f:
                content = f.read()

            image = vision.Image(content=content)

            # Define features
            features = [
                {"type_": vision.Feature.Type.LABEL_DETECTION, "max_results": 20},
                {"type_": vision.Feature.Type.TEXT_DETECTION},  # OCR is always enabled
            ]
            
            if enable_landmark:
                features.append({"type_": vision.Feature.Type.LANDMARK_DETECTION, "max_results": 5})
                logger.info("  -> Travel mode: LANDMARK_DETECTION enabled")

            response = self.vision_client.annotate_image({
                "image": image,
                "features": features,
            })

            result = {
                "labels": [],
                "landmarks": [],
                "text": ""
            }

            # Extract labels
            for label in response.label_annotations:
                if label.score >= 0.7:
                    result["labels"].append(label.description)

            # Extract landmarks
            for landmark in response.landmark_annotations:
                if landmark.score >= 0.7:
                    result["landmarks"].append(landmark.description)

            # Extract text (OCR)
            if response.text_annotations:
                # The first element contains the full text
                result["text"] = response.text_annotations[0].description.strip()

            return result

        except Exception as e:
            logger.error(f"Vision API error for {image_path}: {e}")
            return {}

    def generate_xmp(
        self,
        tags: List[str],
        description: str,
    ) -> str:
        """Generate XMP content."""
        modify_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Create XML safe tags
        safe_tags = []
        for tag in sorted(list(set(tags))): # Remove duplicates and sort
            safe_tags.append(tag.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            
        safe_desc = description.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # Build XMP manually to avoid extra dependencies like jinja2
        xmp = f'''<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="AI Photo Analyzer (Cloud)">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:dc="http://purl.org/dc/elements/1.1/"
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmlns:lr="http://ns.adobe.com/lightroom/1.0/"
      xmp:ModifyDate="{modify_date}"
      xmp:CreatorTool="AI Photo Analyzer (Cloud)">
      <dc:subject>
        <rdf:Bag>
'''
        for tag in safe_tags:
            xmp += f'          <rdf:li>{tag}</rdf:li>\n'

        xmp += '''        </rdf:Bag>
      </dc:subject>
      <dc:description>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">''' + safe_desc + '''</rdf:li>
        </rdf:Alt>
      </dc:description>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>'''

        return xmp

    def process_image(self, image_path: Path, is_raw: bool = False, raw_path: Optional[Path] = None):
        """Process a single image."""
        target_path = raw_path if is_raw else image_path

        # Check if already processed
        if not self.force_reprocess and self.xmp_exists(target_path, is_raw):
            logger.info(f"Skipping (XMP exists): {target_path.name}")
            self.stats["skipped"] += 1
            return

        logger.info(f"Processing: {target_path.name}")

        # Create shrink image
        shrink_path = self.create_shrink_image(image_path)
        if not shrink_path:
            self.stats["failed"] += 1
            return

        try:
            # Check for travel keywords
            is_travel = self.is_travel_photo(target_path)
            
            # Analyze with Vision API
            analysis_result = self.analyze_image(shrink_path, enable_landmark=is_travel)
            
            if not analysis_result:
                self.stats["failed"] += 1
                return

            # Prepare tags list
            all_tags = []
            
            # Process Labels (Translate)
            labels = analysis_result.get("labels", [])
            for label in labels:
                all_tags.append(label) # English
                all_tags.append(self.translator.translate_text(label)) # Japanese

            # Process Landmarks (Translate)
            landmarks = analysis_result.get("landmarks", [])
            for landmark in landmarks:
                all_tags.append(landmark) # English
                all_tags.append(self.translator.translate_text(landmark)) # Japanese

            # Process RAW tags
            raw_ext = raw_path.suffix if raw_path else ""
            if is_raw:
                all_tags.extend(self.RAW_LABELS_JA)
                all_tags.extend(self.RAW_LABELS_EN)
                if raw_ext:
                    all_tags.append(raw_ext.upper().lstrip("."))

            # Process OCR Text (Do NOT Translate, put in description)
            ocr_text = analysis_result.get("text", "")
            
            # Generate XMP
            xmp_content = self.generate_xmp(
                tags=all_tags,
                description=ocr_text
            )

            # Save XMP
            xmp_path = self.get_xmp_path(target_path, is_raw)
            xmp_path.parent.mkdir(parents=True, exist_ok=True)
            xmp_path.write_text(xmp_content, encoding="utf-8")

            logger.info(f"Generated: {xmp_path.name} ({len(labels)} labels, {len(landmarks)} landmarks)")

            self.stats["processed"] += 1
            self.stats["labels_total"] += len(all_tags)
            if is_raw:
                self.stats["raw_processed"] += 1

        finally:
            # Cleanup shrink
            if shrink_path and shrink_path.exists():
                shrink_path.unlink()

    def find_raw_pair(self, image_path: Path) -> Optional[Path]:
        """Find RAW file paired with this image."""
        stem = image_path.stem
        parent = image_path.parent

        for ext in self.RAW_EXTENSIONS:
            for case in [ext, ext.upper()]:
                raw_path = parent / f"{stem}{case}"
                if raw_path.exists():
                    return raw_path
        return None

    def run(self):
        """Run the processing pipeline."""
        logger.info(f"Scanning: {self.input_dir}")
        logger.info(f"Travel keywords: {self.travel_keywords}")
        
        # Collect all files
        all_files = list(self.input_dir.rglob("*"))
        images = [f for f in all_files if self.is_image_file(f)]
        raws = [f for f in all_files if self.is_raw_file(f)]

        logger.info(f"Found {len(images)} images, {len(raws)} RAW files")

        processed_raws = set()

        try:
            # Process RAW files first
            for raw_path in raws:
                source = self.find_analysis_source(raw_path)
                if source:
                    self.process_image(source, is_raw=True, raw_path=raw_path)
                    processed_raws.add(raw_path.stem.lower())
                else:
                    logger.warning(f"No source for RAW: {raw_path.name}")

            # Process regular images
            for image_path in images:
                if image_path.stem.lower() in processed_raws:
                    continue

                raw_pair = self.find_raw_pair(image_path)
                if raw_pair:
                    if raw_pair.stem.lower() not in processed_raws:
                        self.process_image(image_path, is_raw=True, raw_path=raw_pair)
                        processed_raws.add(raw_pair.stem.lower())
                else:
                    self.process_image(image_path)
        
        finally:
            # Save translation cache before exiting
            self.translator.save_cache()

        # Print summary
        logger.info("=" * 50)
        logger.info("Processing Complete")
        logger.info(f"  Processed: {self.stats['processed']}")
        logger.info(f"  RAW files: {self.stats['raw_processed']}")
        logger.info(f"  Skipped:   {self.stats['skipped']}")
        logger.info(f"  Failed:    {self.stats['failed']}")
        logger.info(f"  Total tags: {self.stats['labels_total']}")

        return self.stats


def main():
    parser = argparse.ArgumentParser(description="Cloud Photo Processor")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory containing photos",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for XMP files",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("./temp"),
        help="Temporary directory for shrink images",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path("./translation_cache.json"),
        help="Path to DeepL translation cache file",
    )
    parser.add_argument(
        "--travel-keywords",
        type=str,
        default="",
        help="Comma separated keywords to enable landmark detection",
    )
    parser.add_argument(
        "--shrink-size",
        type=int,
        default=640,
        help="Shrink image size (short edge)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocess all images",
    )

    args = parser.parse_args()

    # Check environment
    if not setup_environment():
        sys.exit(1)

    # Run processor
    processor = CloudPhotoProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        temp_dir=args.temp_dir,
        cache_file=args.cache_file,
        travel_keywords=args.travel_keywords,
        shrink_size=args.shrink_size,
        force_reprocess=args.force,
    )

    stats = processor.run()

    # Exit with error if all failed
    if stats["processed"] == 0 and stats["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
