from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.schemas.classification import ClassificationResult, InputKind
from app.services.routing_service import RoutingService


class ImageClassifier:
    def classify(self, file_path: str, mime_type: str) -> ClassificationResult:
        features: dict[str, Any] = {
            "mime_type": mime_type,
        }

        warnings: list[str] = []

        exif_features = self._extract_exif_features(file_path)
        cv_features = self._extract_cv_features(file_path)

        features.update(exif_features)
        features.update(cv_features)

        screenshot_score, camera_score = self._score(features)

        total_score = screenshot_score + camera_score

        if total_score <= 0:
            kind = InputKind.UNKNOWN
            confidence = 0.0
            warnings.append("image_classification_failed")
        elif screenshot_score > camera_score:
            kind = InputKind.SCREENSHOT
            confidence = screenshot_score / total_score
        else:
            kind = InputKind.CAMERA_PHOTO
            confidence = camera_score / total_score

        if confidence < 0.65:
            warnings.append("low_confidence_image_classification")

        if features.get("blur_score", 0) < 80 and kind == InputKind.CAMERA_PHOTO:
            warnings.append("possible_blurry_camera_photo")

        routing_key = RoutingService.get_routing_key(kind)

        return ClassificationResult(
            kind=kind,
            confidence=round(float(confidence), 4),
            needs_ocr=RoutingService.needs_ocr(kind),
            needs_preprocessing=RoutingService.needs_preprocessing(kind),
            routing_key=routing_key,
            features=features,
            warnings=warnings,
        )

    def _extract_exif_features(self, file_path: str) -> dict[str, Any]:
        try:
            image = Image.open(file_path)
            exif = image.getexif()

            exif_present = bool(exif)

            camera_related_tags = [
                271,  # Make
                272,  # Model
                33434,  # ExposureTime
                33437,  # FNumber
                34855,  # ISO
                37386,  # FocalLength
            ]

            camera_tag_count = 0

            if exif_present:
                for tag in camera_related_tags:
                    if tag in exif:
                        camera_tag_count += 1

            return {
                "exif_present": exif_present,
                "camera_exif_tag_count": camera_tag_count,
                "image_width": image.width,
                "image_height": image.height,
                "aspect_ratio": round(image.width / image.height, 4)
                if image.height
                else None,
            }

        except Exception as exc:
            return {
                "exif_present": False,
                "camera_exif_tag_count": 0,
                "exif_error": str(exc),
            }

    def _extract_cv_features(self, file_path: str) -> dict[str, Any]:
        image = cv2.imread(file_path)

        if image is None:
            return {
                "cv_error": "OpenCV could not read image",
                "blur_score": 0,
                "edge_density": 0,
                "has_document_contour": False,
                "max_contour_area_ratio": 0,
            }

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        height, width = gray.shape[:2]
        image_area = width * height

        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        edges = cv2.Canny(gray, 80, 200)
        edge_density = np.count_nonzero(edges) / edges.size

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        max_contour_area = 0.0
        has_document_contour = False
        quadrilateral_count = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            max_contour_area = max(max_contour_area, area)

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                quadrilateral_count += 1

                if area > 0.25 * image_area:
                    has_document_contour = True

        max_contour_area_ratio = max_contour_area / image_area if image_area else 0.0

        horizontal_vertical_line_score = self._estimate_axis_aligned_line_score(edges)

        return {
            "cv_width": width,
            "cv_height": height,
            "blur_score": round(float(blur_score), 4),
            "edge_density": round(float(edge_density), 4),
            "max_contour_area_ratio": round(float(max_contour_area_ratio), 4),
            "has_document_contour": has_document_contour,
            "quadrilateral_count": quadrilateral_count,
            "axis_aligned_line_score": round(float(horizontal_vertical_line_score), 4),
        }

    def _estimate_axis_aligned_line_score(self, edges: np.ndarray) -> float:
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)

        line_pixels = np.count_nonzero(horizontal_lines) + np.count_nonzero(vertical_lines)
        total_edge_pixels = max(np.count_nonzero(edges), 1)

        return line_pixels / total_edge_pixels

    def _score(self, features: dict[str, Any]) -> tuple[float, float]:
        screenshot_score = 0.0
        camera_score = 0.0

        mime_type = features.get("mime_type")
        exif_present = features.get("exif_present", False)
        camera_exif_tag_count = features.get("camera_exif_tag_count", 0)

        blur_score = features.get("blur_score", 0)
        edge_density = features.get("edge_density", 0)
        has_document_contour = features.get("has_document_contour", False)
        max_contour_area_ratio = features.get("max_contour_area_ratio", 0)
        axis_aligned_line_score = features.get("axis_aligned_line_score", 0)

        image_width = features.get("image_width") or features.get("cv_width")
        image_height = features.get("image_height") or features.get("cv_height")
        aspect_ratio = features.get("aspect_ratio")

        # PNG/WebP screenshots are very common.
        # JPEG photos are more common for camera captures.
        if mime_type in {"image/png", "image/webp"}:
            screenshot_score += 1.5

        if mime_type in {"image/jpeg", "image/heic", "image/heif"}:
            camera_score += 0.8

        # EXIF existing alone is NOT enough.
        # Only camera-specific EXIF tags should strongly indicate camera photo.
        if camera_exif_tag_count >= 2:
            camera_score += 3.0
        elif exif_present and camera_exif_tag_count == 0:
            screenshot_score += 0.5

        if not exif_present:
            screenshot_score += 0.7

        # Very sharp images are usually screenshots.
        # Camera photos can also be sharp, but screenshots are typically extremely sharp.
        if blur_score >= 1000:
            screenshot_score += 2.0
        elif blur_score >= 500:
            screenshot_score += 1.2
        elif blur_score < 150:
            camera_score += 1.0

        # Large document contour strongly suggests camera photo of a document.
        if has_document_contour:
            camera_score += 2.5

        if max_contour_area_ratio >= 0.35:
            camera_score += 1.0

        # Screenshots often have many axis-aligned UI/table lines.
        if axis_aligned_line_score >= 0.20:
            screenshot_score += 1.5
        elif axis_aligned_line_score >= 0.08:
            screenshot_score += 0.8

        # Low edge density should not automatically mean camera photo.
        # It can be a clean mobile app screenshot.
        if edge_density >= 0.10:
            screenshot_score += 0.7

        # Common mobile screenshot aspect ratios.
        # Your example: 828x1792 => 0.462, very typical phone screenshot.
        if aspect_ratio:
            if 0.40 <= aspect_ratio <= 0.60:
                screenshot_score += 1.2
            elif 1.7 <= aspect_ratio <= 2.4:
                screenshot_score += 1.0

        # Common iPhone screenshot dimensions or similar mobile screenshots.
        if image_width and image_height:
            short_side = min(image_width, image_height)
            long_side = max(image_width, image_height)

            if short_side in range(720, 1300) and long_side in range(1400, 3000):
                screenshot_score += 0.8

        return screenshot_score, camera_score