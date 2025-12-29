"""Google Cloud Vision API Service

Analyzes images using Google Cloud Vision API to detect labels,
landmarks, faces, and objects.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from google.cloud import vision

logger = logging.getLogger(__name__)


@dataclass
class DetectedLabel:
    """Represents a detected label from Vision API."""

    description: str
    score: float
    topicality: float = 0.0


@dataclass
class DetectedLandmark:
    """Represents a detected landmark from Vision API."""

    description: str
    score: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@dataclass
class DetectedFace:
    """Represents a detected face from Vision API."""

    joy_likelihood: str
    sorrow_likelihood: str
    anger_likelihood: str
    surprise_likelihood: str
    confidence: float


@dataclass
class DetectedObject:
    """Represents a detected object from Vision API."""

    name: str
    score: float


@dataclass
class VisionAnalysisResult:
    """Complete analysis result from Vision API."""

    labels: list[DetectedLabel] = field(default_factory=list)
    landmarks: list[DetectedLandmark] = field(default_factory=list)
    faces: list[DetectedFace] = field(default_factory=list)
    objects: list[DetectedObject] = field(default_factory=list)
    web_entities: list[str] = field(default_factory=list)
    dominant_colors: list[str] = field(default_factory=list)
    safe_search: dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None

    def get_all_labels(self) -> list[str]:
        """Get all detected labels as strings.

        Returns:
            List of label descriptions.
        """
        all_labels = []

        # Add labels
        all_labels.extend([l.description for l in self.labels])

        # Add landmarks
        all_labels.extend([l.description for l in self.landmarks])

        # Add objects
        all_labels.extend([o.name for o in self.objects])

        # Add web entities
        all_labels.extend(self.web_entities)

        # Remove duplicates while preserving order
        seen = set()
        unique_labels = []
        for label in all_labels:
            label_lower = label.lower()
            if label_lower not in seen:
                seen.add(label_lower)
                unique_labels.append(label)

        return unique_labels


class VisionService:
    """Service for analyzing images with Google Cloud Vision API."""

    def __init__(
        self,
        max_labels: int = 20,
        min_confidence: float = 0.7,
        features: Optional[list[str]] = None,
    ):
        """Initialize the Vision service.

        Args:
            max_labels: Maximum number of labels to return.
            min_confidence: Minimum confidence score (0.0-1.0).
            features: List of features to detect.
        """
        self.max_labels = max_labels
        self.min_confidence = min_confidence
        self.features = features or [
            "LABEL_DETECTION",
            "LANDMARK_DETECTION",
            "FACE_DETECTION",
            "OBJECT_LOCALIZATION",
        ]
        self._client: Optional[vision.ImageAnnotatorClient] = None

    @property
    def client(self) -> vision.ImageAnnotatorClient:
        """Get or create the Vision API client.

        Returns:
            Vision API client instance.
        """
        if self._client is None:
            self._client = vision.ImageAnnotatorClient()
        return self._client

    def _build_features(self) -> list[vision.Feature]:
        """Build the list of features to request.

        Returns:
            List of Vision API features.
        """
        feature_types = {
            "LABEL_DETECTION": vision.Feature.Type.LABEL_DETECTION,
            "LANDMARK_DETECTION": vision.Feature.Type.LANDMARK_DETECTION,
            "FACE_DETECTION": vision.Feature.Type.FACE_DETECTION,
            "OBJECT_LOCALIZATION": vision.Feature.Type.OBJECT_LOCALIZATION,
            "WEB_DETECTION": vision.Feature.Type.WEB_DETECTION,
            "IMAGE_PROPERTIES": vision.Feature.Type.IMAGE_PROPERTIES,
            "SAFE_SEARCH_DETECTION": vision.Feature.Type.SAFE_SEARCH_DETECTION,
        }

        features = []
        for feature_name in self.features:
            if feature_name in feature_types:
                features.append(
                    vision.Feature(
                        type_=feature_types[feature_name],
                        max_results=self.max_labels,
                    )
                )

        return features

    def analyze_image(self, image_path: Path) -> VisionAnalysisResult:
        """Analyze an image using Vision API.

        Args:
            image_path: Path to the image file.

        Returns:
            VisionAnalysisResult containing detected elements.
        """
        result = VisionAnalysisResult()

        try:
            # Read image content
            with open(image_path, "rb") as f:
                content = f.read()

            image = vision.Image(content=content)
            features = self._build_features()

            # Make API request
            response = self.client.annotate_image(
                {"image": image, "features": features}
            )

            if response.error.message:
                result.error = response.error.message
                logger.error(f"Vision API error: {response.error.message}")
                return result

            # Process labels
            for label in response.label_annotations:
                if label.score >= self.min_confidence:
                    result.labels.append(
                        DetectedLabel(
                            description=label.description,
                            score=label.score,
                            topicality=label.topicality,
                        )
                    )

            # Process landmarks
            for landmark in response.landmark_annotations:
                if landmark.score >= self.min_confidence:
                    lat = None
                    lon = None
                    if landmark.locations:
                        loc = landmark.locations[0].lat_lng
                        lat = loc.latitude
                        lon = loc.longitude

                    result.landmarks.append(
                        DetectedLandmark(
                            description=landmark.description,
                            score=landmark.score,
                            latitude=lat,
                            longitude=lon,
                        )
                    )

            # Process faces
            for face in response.face_annotations:
                if face.detection_confidence >= self.min_confidence:
                    result.faces.append(
                        DetectedFace(
                            joy_likelihood=vision.Likelihood(face.joy_likelihood).name,
                            sorrow_likelihood=vision.Likelihood(
                                face.sorrow_likelihood
                            ).name,
                            anger_likelihood=vision.Likelihood(
                                face.anger_likelihood
                            ).name,
                            surprise_likelihood=vision.Likelihood(
                                face.surprise_likelihood
                            ).name,
                            confidence=face.detection_confidence,
                        )
                    )

            # Process localized objects
            for obj in response.localized_object_annotations:
                if obj.score >= self.min_confidence:
                    result.objects.append(
                        DetectedObject(
                            name=obj.name,
                            score=obj.score,
                        )
                    )

            # Process web entities if available
            if hasattr(response, "web_detection") and response.web_detection:
                for entity in response.web_detection.web_entities:
                    if entity.score >= self.min_confidence and entity.description:
                        result.web_entities.append(entity.description)

            # Process safe search if available
            if hasattr(response, "safe_search_annotation"):
                safe = response.safe_search_annotation
                result.safe_search = {
                    "adult": vision.Likelihood(safe.adult).name,
                    "spoof": vision.Likelihood(safe.spoof).name,
                    "medical": vision.Likelihood(safe.medical).name,
                    "violence": vision.Likelihood(safe.violence).name,
                    "racy": vision.Likelihood(safe.racy).name,
                }

            logger.info(
                f"Analyzed {image_path}: "
                f"{len(result.labels)} labels, "
                f"{len(result.landmarks)} landmarks, "
                f"{len(result.faces)} faces, "
                f"{len(result.objects)} objects"
            )

        except Exception as e:
            result.error = str(e)
            logger.error(f"Failed to analyze {image_path}: {e}")

        return result

    def analyze_batch(
        self, image_paths: list[Path]
    ) -> dict[Path, VisionAnalysisResult]:
        """Analyze multiple images.

        Args:
            image_paths: List of image paths to analyze.

        Returns:
            Dictionary mapping paths to analysis results.
        """
        results = {}
        for path in image_paths:
            results[path] = self.analyze_image(path)
        return results
