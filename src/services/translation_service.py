"""DeepL Translation Service

Translates English labels from Vision API to Japanese
for better searchability in Mylio Photos.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import deepl

logger = logging.getLogger(__name__)


@dataclass
class TranslatedLabel:
    """Represents a translated label."""

    original: str
    translated: str


class TranslationService:
    """Service for translating labels using DeepL API."""

    # Common photo-related terms that should be preserved or have specific translations
    TERM_MAPPING = {
        # Nature
        "sky": "空",
        "cloud": "雲",
        "tree": "木",
        "flower": "花",
        "mountain": "山",
        "ocean": "海",
        "beach": "ビーチ",
        "sunset": "夕日",
        "sunrise": "日の出",
        "forest": "森",
        "river": "川",
        "lake": "湖",
        "waterfall": "滝",
        "snow": "雪",
        "rain": "雨",
        # Animals
        "dog": "犬",
        "cat": "猫",
        "bird": "鳥",
        "fish": "魚",
        # People
        "person": "人物",
        "people": "人々",
        "man": "男性",
        "woman": "女性",
        "child": "子供",
        "baby": "赤ちゃん",
        "family": "家族",
        "group": "グループ",
        # Places
        "city": "都市",
        "street": "通り",
        "building": "建物",
        "house": "家",
        "temple": "寺院",
        "shrine": "神社",
        "castle": "城",
        "park": "公園",
        "garden": "庭園",
        # Food
        "food": "食べ物",
        "meal": "食事",
        "restaurant": "レストラン",
        "coffee": "コーヒー",
        # Activities
        "travel": "旅行",
        "sport": "スポーツ",
        "party": "パーティー",
        "wedding": "結婚式",
        "birthday": "誕生日",
        # Objects
        "car": "車",
        "train": "電車",
        "airplane": "飛行機",
        "boat": "船",
        "bicycle": "自転車",
        # Photography terms
        "portrait": "ポートレート",
        "landscape": "風景",
        "macro": "マクロ",
        "night photography": "夜景",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        target_lang: str = "JA",
        source_lang: str = "EN",
        formality: str = "default",
    ):
        """Initialize the translation service.

        Args:
            api_key: DeepL API key. If not provided, reads from DEEPL_API_KEY env.
            target_lang: Target language code.
            source_lang: Source language code.
            formality: Formality level for translations.
        """
        self.api_key = api_key or os.environ.get("DEEPL_API_KEY")
        self.target_lang = target_lang
        self.source_lang = source_lang
        self.formality = formality
        self._translator: Optional[deepl.Translator] = None
        self._cache: dict[str, str] = {}

        # Pre-populate cache with known mappings
        self._cache.update(self.TERM_MAPPING)

    @property
    def translator(self) -> deepl.Translator:
        """Get or create the DeepL translator.

        Returns:
            DeepL Translator instance.
        """
        if self._translator is None:
            if not self.api_key:
                raise ValueError(
                    "DeepL API key not provided. "
                    "Set DEEPL_API_KEY environment variable or pass api_key parameter."
                )
            self._translator = deepl.Translator(self.api_key)
        return self._translator

    def _normalize_key(self, text: str) -> str:
        """Normalize text for cache lookup.

        Args:
            text: Text to normalize.

        Returns:
            Normalized text.
        """
        return text.lower().strip()

    def translate_label(self, label: str) -> TranslatedLabel:
        """Translate a single label to Japanese.

        Args:
            label: English label to translate.

        Returns:
            TranslatedLabel with original and translated text.
        """
        normalized = self._normalize_key(label)

        # Check cache first
        if normalized in self._cache:
            return TranslatedLabel(
                original=label,
                translated=self._cache[normalized],
            )

        try:
            # Use DeepL API for translation
            result = self.translator.translate_text(
                label,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
            )
            translated = result.text

            # Cache the result
            self._cache[normalized] = translated

            logger.debug(f"Translated: {label} -> {translated}")

            return TranslatedLabel(
                original=label,
                translated=translated,
            )

        except Exception as e:
            logger.warning(f"Failed to translate '{label}': {e}")
            # Return original as fallback
            return TranslatedLabel(
                original=label,
                translated=label,
            )

    def translate_labels(self, labels: list[str]) -> list[TranslatedLabel]:
        """Translate multiple labels to Japanese.

        Uses batch translation for efficiency when possible.

        Args:
            labels: List of English labels to translate.

        Returns:
            List of TranslatedLabel objects.
        """
        results = []
        labels_to_translate = []
        indices_to_translate = []

        # Check cache for each label
        for i, label in enumerate(labels):
            normalized = self._normalize_key(label)
            if normalized in self._cache:
                results.append(
                    TranslatedLabel(
                        original=label,
                        translated=self._cache[normalized],
                    )
                )
            else:
                results.append(None)  # Placeholder
                labels_to_translate.append(label)
                indices_to_translate.append(i)

        # Batch translate remaining labels
        if labels_to_translate:
            try:
                api_results = self.translator.translate_text(
                    labels_to_translate,
                    source_lang=self.source_lang,
                    target_lang=self.target_lang,
                )

                for idx, api_result in zip(indices_to_translate, api_results):
                    original = labels[idx]
                    translated = api_result.text

                    # Cache the result
                    normalized = self._normalize_key(original)
                    self._cache[normalized] = translated

                    results[idx] = TranslatedLabel(
                        original=original,
                        translated=translated,
                    )

                    logger.debug(f"Translated: {original} -> {translated}")

            except Exception as e:
                logger.error(f"Batch translation failed: {e}")
                # Fall back to individual translation
                for idx in indices_to_translate:
                    if results[idx] is None:
                        results[idx] = self.translate_label(labels[idx])

        return results

    def get_usage(self) -> dict:
        """Get DeepL API usage statistics.

        Returns:
            Dictionary with usage information.
        """
        try:
            usage = self.translator.get_usage()
            return {
                "character_count": usage.character.count if usage.character else 0,
                "character_limit": usage.character.limit if usage.character else 0,
            }
        except Exception as e:
            logger.error(f"Failed to get usage: {e}")
            return {"error": str(e)}

    def clear_cache(self):
        """Clear the translation cache."""
        self._cache.clear()
        # Re-populate with known mappings
        self._cache.update(self.TERM_MAPPING)
        logger.info("Translation cache cleared")
