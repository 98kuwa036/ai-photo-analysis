#!/usr/bin/env python3
"""Cloud Runner for AI Photo Analysis

Usage:
    python -m src.cloud_runner --input-dir ./photos --output-dir ./output --cache-file ./config/translation_cache.json --history-file ./config/processing_history.json --home-location "35.689,139.691"
"""

import argparse
import json
import logging
import os
import sys
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Set, Any
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# JST timezone definition
JST = timezone(timedelta(hours=9), 'JST')


def setup_environment():
    """Verify required environment variables are set."""
    required_vars = ["GOOGLE_APPLICATION_CREDENTIALS", "DEEPL_API_KEY"]
    missing = [var for var in required_vars if not os.environ.get(var)]

    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        return False
    return True


class DeepLTranslator:
    """Handles translation with local caching."""
    def __init__(self, api_key: str, cache_file: Path):
        self.translator = None
        if api_key:
            try:
                import deepl
                self.translator = deepl.Translator(api_key)
            except ImportError:
                logger.warning("deepl library not installed. Translation disabled.")
        
        self.cache_file = cache_file
        self.cache: Dict[str, str] = self._load_cache()
        self.modified = False

    def _load_cache(self) -> Dict[str, str]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_cache(self):
        if self.modified:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def translate_text(self, text: str) -> str:
        if not text: return ""
        if text in self.cache: return self.cache[text]
        if not self.translator: return text

        try:
            result = self.translator.translate_text(text, source_lang="EN", target_lang="JA")
            self.cache[text] = result.text
            self.modified = True
            return result.text
        except Exception as e:
            logger.error(f"DeepL API Error: {e}")
            return text


class ProcessingHistory:
    """Manages the history of processed files."""
    def __init__(self, history_file: Path):
        self.history_file = history_file
        self.history: Dict[str, str] = self._load_history()
        self.modified = False

    def _load_history(self) -> Dict[str, str]:
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_history(self):
        if self.modified:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
            logger.info("Processing history saved.")

    def is_processed(self, file_stem: str) -> bool:
        return file_stem in self.history

    def add(self, file_stem: str):
        now_jst = datetime.now(JST).isoformat()
        self.history[file_stem] = now_jst
        self.modified = True


class CloudPhotoProcessor:
    """Simplified photo processor for cloud environments."""
    
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
    RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".dng"}
    RAW_LABELS = ["RAW", "RAW画像", "Raw Image"]

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        temp_dir: Path,
        cache_file: Path,
        history_file: Path,
        home_location: str = "",
        shrink_size: int = 640,
        force_reprocess: bool = False,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.shrink_size = shrink_size
        self.force_reprocess = force_reprocess
        
        self.home_lat = None
        self.home_lon = None
        if home_location:
            try:
                parts = home_location.split(',')
                self.home_lat = float(parts[0].strip())
                self.home_lon = float(parts[1].strip())
                logger.info(f"Home location set: {self.home_lat}, {self.home_lon}")
            except Exception:
                logger.warning("Invalid home location format.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self._vision_client = None
        self.translator = DeepLTranslator(os.environ.get("DEEPL_API_KEY", ""), cache_file)
        self.history = ProcessingHistory(history_file)

        self.stats = {"processed": 0, "skipped": 0, "failed": 0, "labels_total": 0}

    @property
    def vision_client(self):
        if self._vision_client is None:
            from google.cloud import vision
            self._vision_client = vision.ImageAnnotatorClient()
        return self._vision_client

    # 【復活・修正】正しいXMPパスを生成するメソッド
    def get_xmp_path(self, image_path: Path, is_raw: bool = False) -> Path:
        """Get XMP output path. Uses .stem (no ext) for normal images, .name for RAW."""
        relative = image_path.relative_to(self.input_dir)

        if is_raw:
            # RAWの場合: DSC001.ARW -> DSC001.ARW.xmp
            xmp_name = f"{image_path.name}.xmp"
        else:
            # JPEG/PNGの場合: IMG_001.jpg -> IMG_001.xmp (拡張子なし)
            xmp_name = f"{image_path.stem}.xmp"

        return self.output_dir / relative.parent / xmp_name

    def _get_exif_gps(self, image_path: Path) -> Optional[tuple[float, float]]:
        try:
            with Image.open(image_path) as img:
                exif = img.getexif()
                if not exif: return None
                
                geotags = {}
                for (idx, tag) in TAGS.items():
                    if tag == 'GPSInfo':
                        if idx not in exif: return None
                        gps_info = exif.get_ifd(idx)
                        for (key, val) in GPSTAGS.items():
                            if key in gps_info: geotags[val] = gps_info[key]

                if 'GPSLatitude' in geotags and 'GPSLongitude' in geotags:
                    lat = self._convert_to_degrees(geotags['GPSLatitude'])
                    lon = self._convert_to_degrees(geotags['GPSLongitude'])
                    
                    lat_ref = geotags.get('GPSLatitudeRef')
                    if lat_ref == 'S' or lat_ref == b'S': lat = -lat
                    
                    lon_ref = geotags.get('GPSLongitudeRef')
                    if lon_ref == 'W' or lon_ref == b'W': lon = -lon
                    
                    return (lat, lon)
        except Exception:
            pass
        return None

    def _convert_to_degrees(self, value):
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        return d + (m / 60.0) + (s / 3600.0)

    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2) * math.sin(dlat/2) + \
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
            math.sin(dlon/2) * math.sin(dlon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

    def is_travel_photo(self, image_path: Path) -> bool:
        if self.home_lat is None or self.home_lon is None: return False
        gps = self._get_exif_gps(image_path)
        if gps:
            lat, lon = gps
            dist = self._calculate_distance(self.home_lat, self.home_lon, lat, lon)
            if dist > 20.0:
                logger.info(f"  -> Travel detected! Distance: {dist:.1f}km")
                return True
            else:
                logger.info(f"  -> Local photo. Distance: {dist:.1f}km")
                return False
        return False

    def create_shrink_image(self, image_path: Path) -> Optional[Path]:
        from PIL import Image
        try:
            shrink_path = self.temp_dir / f"{image_path.stem}_shrink.jpg"
            with Image.open(image_path) as img:
                if img.mode in ("RGBA", "P"): img = img.convert("RGB")
                
                width, height = img.size
                scale = self.shrink_size / min(width, height)
                if scale < 1:
                    new_size = (int(width * scale), int(height * scale))
                    resized = img.resize(new_size, Image.Resampling.LANCZOS)
                    resized.save(shrink_path, "JPEG", quality=85)
                else:
                    img.save(shrink_path, "JPEG", quality=85)
            return shrink_path
        except Exception as e:
            logger.error(f"Shrink failed: {e}")
            return None

    def analyze_image(self, image_path: Path, enable_landmark: bool = False) -> dict:
        from google.cloud import vision
        try:
            with open(image_path, "rb") as f:
                content = f.read()
            image = vision.Image(content=content)

            features = [
                {"type_": vision.Feature.Type.LABEL_DETECTION, "max_results": 20},
                {"type_": vision.Feature.Type.TEXT_DETECTION},
            ]
            if enable_landmark:
                features.append({"type_": vision.Feature.Type.LANDMARK_DETECTION, "max_results": 5})

            response = self.vision_client.annotate_image({"image": image, "features": features})
            
            result = {"labels": [], "landmarks": [], "text": ""}
            for label in response.label_annotations:
                if label.score >= 0.7: result["labels"].append(label.description)
            for landmark in response.landmark_annotations:
                if landmark.score >= 0.7: result["landmarks"].append(landmark.description)
            if response.text_annotations:
                result["text"] = response.text_annotations[0].description.strip()
            return result
        except Exception as e:
            logger.error(f"Vision API error: {e}")
            return {}

    def generate_xmp(self, tags: List[str], description: str) -> str:
        modify_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        safe_tags = [t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") for t in sorted(list(set(tags)))]
        safe_desc = description.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        xmp = f'''<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="AI Photo Analyzer">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:dc="http://purl.org/dc/elements/1.1/"
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmlns:lr="http://ns.adobe.com/lightroom/1.0/"
      xmp:ModifyDate="{modify_date}">
      <dc:subject>
        <rdf:Bag>
'''
        for tag in safe_tags:
            xmp += f'          <rdf:li>{tag}</rdf:li>\n'
        xmp += f'''        </rdf:Bag>
      </dc:subject>
      <dc:description>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">{safe_desc}</rdf:li>
        </rdf:Alt>
      </dc:description>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>'''
        return xmp

    def process_image(self, image_path: Path, is_raw: bool = False, raw_path: Optional[Path] = None):
        target_path = raw_path if is_raw else image_path
        
        # 【重要】スキップ判定を一番最初に行う（無駄な処理防止）
        
        # 1. 履歴台帳チェック (ファイル名で判断)
        if not self.force_reprocess and self.history.is_processed(target_path.stem):
            logger.info(f"Skipping (In History): {target_path.name}")
            self.stats["skipped"] += 1
            return

        # 2. XMP存在チェック (正しいパスを使用)
        xmp_path = self.get_xmp_path(target_path, is_raw)
        if not self.force_reprocess and xmp_path.exists():
            logger.info(f"Skipping (XMP exists): {target_path.name}")
            self.history.add(target_path.stem)
            self.stats["skipped"] += 1
            return

        logger.info(f"Processing: {target_path.name}")
        
        # --- ここから重い処理が始まります ---
        
        # 1. 縮小画像作成
        shrink_path = self.create_shrink_image(image_path)
        if not shrink_path: return

        try:
            # 2. GPSによる旅行判定
            is_travel = self.is_travel_photo(image_path)

            # 3. Vision API 分析
            result = self.analyze_image(shrink_path, enable_landmark=is_travel)
            if not result:
                self.stats["failed"] += 1
                return

            # 4. タグ生成 & 翻訳
            all_tags = []
            if is_raw: all_tags.extend(self.RAW_LABELS)
            
            for lbl in result["labels"] + result["landmarks"]:
                all_tags.append(lbl)
                all_tags.append(self.translator.translate_text(lbl))

            # 5. XMP保存
            xmp_content = self.generate_xmp(all_tags, result["text"])
            
            xmp_path.parent.mkdir(parents=True, exist_ok=True)
            xmp_path.write_text(xmp_content, encoding="utf-8")
            
            # 完了したら履歴に追加
            self.history.add(target_path.stem)
            
            self.stats["processed"] += 1
            self.stats["labels_total"] += len(all_tags)

        finally:
            if shrink_path.exists(): shrink_path.unlink()

    def run(self):
        all_files = list(self.input_dir.rglob("*"))
        images = [f for f in all_files if f.suffix.lower() in self.IMAGE_EXTENSIONS]
        raws = [f for f in all_files if f.suffix.lower() in self.RAW_EXTENSIONS]
        
        # RAWペア処理
        processed_stems = set()
        for raw in raws:
            src = next((f for f in images if f.stem == raw.stem and f.parent == raw.parent), None)
            if src:
                self.process_image(src, is_raw=True, raw_path=raw)
                processed_stems.add(raw.stem)

        # JPEG単独処理
        for img in images:
            if img.stem not in processed_stems:
                self.process_image(img)

        self.translator.save_cache()
        self.history.save_history()
        
        return self.stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temp-dir", type=Path, default=Path("./temp"))
    parser.add_argument("--cache-file", type=Path, default=Path("./cache.json"))
    parser.add_argument("--history-file", type=Path, default=Path("./history.json"))
    parser.add_argument("--home-location", type=str, default="")
    parser.add_argument("--force", action="store_true")
    
    args = parser.parse_args()
    if not setup_environment(): sys.exit(1)
    
    CloudPhotoProcessor(
        args.input_dir, args.output_dir, args.temp_dir, 
        args.cache_file, args.history_file,
        args.home_location, force_reprocess=args.force
    ).run()

if __name__ == "__main__":
    main()
