"""Deterministic, ROS-independent perception helpers for ERC phase 1.

The module deliberately uses only OpenCV and NumPy.  In particular, digit
recognition is template based and never downloads a model at run time.  All
bounding boxes use OpenCV's ``(x, y, width, height)`` convention and all input
colour images are expected to be BGR unless documented otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import cv2
import numpy as np


BBox = tuple[int, int, int, int]
Point = tuple[float, float]
ImageSource = Union[np.ndarray, str, PathLike[str]]
TemplateSources = Union[
    Mapping[int, ImageSource],
    Sequence[ImageSource],
    str,
    PathLike[str],
]


@dataclass(frozen=True)
class MarkerDetection:
    """One detected square number marker."""

    bbox: BBox
    digit: int
    confidence: float
    quadrilateral: Optional[tuple[Point, Point, Point, Point]] = None

    @property
    def center(self) -> Point:
        """Return the bounding-box centre in image coordinates."""

        x, y, width, height = self.bbox
        return (x + width / 2.0, y + height / 2.0)


@dataclass(frozen=True)
class BookDetection:
    """One colour-segmented book candidate."""

    color: str
    bbox: BBox
    area: float
    confidence: float
    row: Optional[int] = None

    @property
    def center(self) -> Point:
        """Return the bounding-box centre in image coordinates."""

        x, y, width, height = self.bbox
        return (x + width / 2.0, y + height / 2.0)


@dataclass(frozen=True)
class BinDetection:
    """The large red collection-bin candidate."""

    bbox: BBox
    area: float
    confidence: float

    @property
    def center(self) -> Point:
        """Return the bounding-box centre in image coordinates."""

        x, y, width, height = self.bbox
        return (x + width / 2.0, y + height / 2.0)


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera parameters in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float


# Gazebo's nominal colours are highly saturated, while these deliberately
# broad ranges retain detections under moderate lighting and compression.
HSV_RANGES: dict[str, tuple[tuple[np.ndarray, np.ndarray], ...]] = {
    "red": (
        (np.array((0, 80, 45), dtype=np.uint8),
         np.array((12, 255, 255), dtype=np.uint8)),
        (np.array((168, 80, 45), dtype=np.uint8),
         np.array((179, 255, 255), dtype=np.uint8)),
    ),
    "yellow": (
        (np.array((16, 75, 60), dtype=np.uint8),
         np.array((37, 255, 255), dtype=np.uint8)),
    ),
    "green": (
        (np.array((38, 65, 40), dtype=np.uint8),
         np.array((90, 255, 255), dtype=np.uint8)),
    ),
    "blue": (
        (np.array((91, 70, 35), dtype=np.uint8),
         np.array((138, 255, 255), dtype=np.uint8)),
    ),
}

DRAW_COLORS: dict[str, tuple[int, int, int]] = {
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "green": (0, 200, 0),
    "yellow": (0, 220, 255),
    "marker": (255, 255, 255),
    "bin": (255, 0, 255),
    "target": (255, 255, 0),
}


def _validate_image(image: np.ndarray, name: str = "image") -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if image.size == 0 or image.ndim not in (2, 3):
        raise ValueError(f"{name} must be a non-empty 2-D or 3-D image")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(f"{name} must have 1, 3, or 4 channels")
    return image


def _as_bgr(image: np.ndarray) -> np.ndarray:
    image = _validate_image(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _as_gray(image: np.ndarray) -> np.ndarray:
    image = _validate_image(image)
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    conversion = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(image, conversion)


def _read_image(source: ImageSource, *, grayscale: bool = False) -> np.ndarray:
    if isinstance(source, np.ndarray):
        return _as_gray(source) if grayscale else _validate_image(source)
    if not isinstance(source, (str, PathLike)):
        raise TypeError("template sources must be image arrays or file paths")
    path = Path(source).expanduser()
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(f"could not read digit template: {path}")
    return image


def _normalize_digit(
    image: np.ndarray,
    output_size: tuple[int, int] = (64, 96),
) -> np.ndarray:
    """Turn a black-on-light digit crop into a centred binary ink mask."""

    gray = _as_gray(image)
    if gray.dtype != np.uint8:
        finite = np.isfinite(gray)
        if not np.any(finite):
            raise ValueError("digit image contains no finite pixels")
        low = float(np.min(gray[finite]))
        high = float(np.max(gray[finite]))
        if high > low:
            gray = np.clip((gray - low) * (255.0 / (high - low)), 0, 255)
        else:
            gray = np.zeros_like(gray)
        gray = gray.astype(np.uint8)

    if min(gray.shape[:2]) >= 3:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, dark_ink = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    # In the normal case the border is the light marker background.  If an
    # explicitly supplied template is white-on-black, swap polarity using the
    # border occupancy rather than the global mean (digits can fill the image).
    border = np.concatenate(
        (dark_ink[0, :], dark_ink[-1, :], dark_ink[:, 0], dark_ink[:, -1])
    )
    if float(np.mean(border > 0)) > 0.5:
        dark_ink = cv2.bitwise_not(dark_ink)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (dark_ink > 0).astype(np.uint8), connectivity=8
    )
    if count <= 1:
        raise ValueError("digit image does not contain foreground ink")

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.max(component_areas))
    keep_labels = [
        index + 1
        for index, area in enumerate(component_areas)
        if int(area) >= max(2, int(round(largest * 0.025)))
    ]
    ink = np.isin(labels, keep_labels).astype(np.uint8) * 255
    locations = cv2.findNonZero(ink)
    if locations is None:
        raise ValueError("digit image does not contain usable foreground ink")
    x, y, width, height = cv2.boundingRect(locations)
    glyph = ink[y:y + height, x:x + width]

    canvas_width, canvas_height = output_size
    if canvas_width < 8 or canvas_height < 8:
        raise ValueError("output_size must be at least 8 by 8 pixels")
    margin = max(2, int(round(min(canvas_width, canvas_height) * 0.08)))
    scale = min(
        (canvas_width - 2 * margin) / max(width, 1),
        (canvas_height - 2 * margin) / max(height, 1),
    )
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    glyph = cv2.resize(
        glyph, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST
    )
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    x_offset = (canvas_width - resized_width) // 2
    y_offset = (canvas_height - resized_height) // 2
    canvas[
        y_offset:y_offset + resized_height,
        x_offset:x_offset + resized_width,
    ] = glyph
    return canvas


def load_digit_templates(
    templates: TemplateSources,
    *,
    digits: Iterable[int] = range(1, 6),
    output_size: tuple[int, int] = (64, 96),
) -> dict[int, np.ndarray]:
    """Load and normalize templates for digits 1 through 5.

    ``templates`` can be a mapping from digit to image array/path, a sequence
    ordered like ``digits``, or a directory containing ``1.png`` ... ``5.png``.
    The returned masks are suitable for repeated calls to :func:`classify_digit`.
    """

    required_digits = tuple(int(digit) for digit in digits)
    if not required_digits or any(digit < 1 or digit > 5 for digit in required_digits):
        raise ValueError("digits must be a non-empty subset of 1 through 5")
    if len(set(required_digits)) != len(required_digits):
        raise ValueError("digits must not contain duplicates")

    sources: dict[int, ImageSource]
    if isinstance(templates, (str, PathLike)):
        directory = Path(templates).expanduser()
        if not directory.is_dir():
            raise FileNotFoundError(f"template directory does not exist: {directory}")
        sources = {digit: directory / f"{digit}.png" for digit in required_digits}
    elif isinstance(templates, Mapping):
        sources = {}
        for raw_digit, source in templates.items():
            digit = int(raw_digit)
            if digit in required_digits:
                sources[digit] = source
    elif isinstance(templates, Sequence):
        if len(templates) != len(required_digits):
            raise ValueError("template sequence length must match digits")
        sources = dict(zip(required_digits, templates))
    else:
        raise TypeError("templates must be a mapping, sequence, or directory path")

    missing = sorted(set(required_digits) - set(sources))
    if missing:
        raise ValueError(f"missing templates for digits: {missing}")
    return {
        digit: _normalize_digit(
            _read_image(sources[digit], grayscale=True), output_size=output_size
        )
        for digit in required_digits
    }


def _prepared_templates(
    templates: Mapping[int, ImageSource],
    output_size: tuple[int, int],
) -> dict[int, np.ndarray]:
    prepared: dict[int, np.ndarray] = {}
    for raw_digit, source in templates.items():
        digit = int(raw_digit)
        if digit < 1 or digit > 5:
            continue
        prepared[digit] = _normalize_digit(
            _read_image(source, grayscale=True), output_size=output_size
        )
    if not prepared:
        raise ValueError("at least one template for a digit from 1 through 5 is required")
    return prepared


def _coerce_template_sources(
    templates: TemplateSources,
    output_size: tuple[int, int],
) -> dict[int, np.ndarray]:
    if isinstance(templates, Mapping):
        return _prepared_templates(templates, output_size)
    return load_digit_templates(templates, output_size=output_size)


def _binary_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first_on = first > 0
    second_on = second > 0
    union = int(np.count_nonzero(first_on | second_on))
    intersection = int(np.count_nonzero(first_on & second_on))
    iou = intersection / union if union else 0.0

    # A small distance tolerance makes the score resistant to interpolation,
    # perspective warping, and slightly different font rasterisation.
    distance_to_second = cv2.distanceTransform(
        (~second_on).astype(np.uint8), cv2.DIST_L2, 3
    )
    distance_to_first = cv2.distanceTransform(
        (~first_on).astype(np.uint8), cv2.DIST_L2, 3
    )
    first_count = max(1, int(np.count_nonzero(first_on)))
    second_count = max(1, int(np.count_nonzero(second_on)))
    forward = float(np.sum(np.exp(-distance_to_second[first_on] / 2.0))) / first_count
    backward = float(np.sum(np.exp(-distance_to_first[second_on] / 2.0))) / second_count
    chamfer_similarity = (forward + backward) / 2.0
    return float(np.clip(0.55 * iou + 0.45 * chamfer_similarity, 0.0, 1.0))


def classify_digit(
    image: np.ndarray,
    templates: TemplateSources,
    *,
    output_size: tuple[int, int] = (64, 96),
    allow_right_angle_rotations: bool = True,
) -> tuple[int, float]:
    """Classify a marker crop, returning ``(digit, confidence)``.

    Confidence is a deterministic template-similarity score in ``[0, 1]``.
    Trying right-angle rotations is useful for markers viewed from above.  The
    original orientation wins exact ties, followed by the lowest digit.
    """

    prepared = _coerce_template_sources(templates, output_size)
    rotation_codes: tuple[Optional[int], ...]
    if allow_right_angle_rotations:
        rotation_codes = (
            None,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_180,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        )
    else:
        rotation_codes = (None,)

    best_digit = min(prepared)
    best_score = -1.0
    best_rotation = len(rotation_codes)
    for rotation_index, rotation_code in enumerate(rotation_codes):
        candidate_image = image if rotation_code is None else cv2.rotate(image, rotation_code)
        try:
            candidate = _normalize_digit(candidate_image, output_size=output_size)
        except ValueError:
            continue
        for digit in sorted(prepared):
            score = _binary_similarity(candidate, prepared[digit])
            if (
                score > best_score + 1e-12
                or (
                    abs(score - best_score) <= 1e-12
                    and (rotation_index, digit) < (best_rotation, best_digit)
                )
            ):
                best_digit = digit
                best_score = score
                best_rotation = rotation_index

    if best_score < 0.0:
        raise ValueError("marker crop does not contain a classifiable digit")
    return best_digit, float(np.clip(best_score, 0.0, 1.0))


def _order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    result = np.empty((4, 2), dtype=np.float32)
    coordinate_sum = points.sum(axis=1)
    coordinate_difference = np.diff(points, axis=1).ravel()
    result[0] = points[np.argmin(coordinate_sum)]  # top-left
    result[2] = points[np.argmax(coordinate_sum)]  # bottom-right
    result[1] = points[np.argmin(coordinate_difference)]  # top-right
    result[3] = points[np.argmax(coordinate_difference)]  # bottom-left
    return result


def _warp_quad(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    ordered = _order_quad(quad)
    top_width = np.linalg.norm(ordered[1] - ordered[0])
    bottom_width = np.linalg.norm(ordered[2] - ordered[3])
    left_height = np.linalg.norm(ordered[3] - ordered[0])
    right_height = np.linalg.norm(ordered[2] - ordered[1])
    width = max(8, int(round(max(top_width, bottom_width))))
    height = max(8, int(round(max(left_height, right_height))))
    destination = np.array(
        ((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)),
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _classify_upright_glyph(
    glyph: np.ndarray,
    templates: Mapping[int, ImageSource],
) -> tuple[int, float]:
    """Classify one upright glyph while tolerating shelf-view foreshortening.

    The competition cards remain upright, but an oblique camera view can make
    their glyphs substantially wider than the frontal templates.  Trying a
    small, deterministic set of vertical stretches restores that lost aspect
    ratio without introducing the 90-degree ambiguities of the general marker
    classifier.
    """

    best_digit = min(int(digit) for digit in templates)
    best_score = -1.0
    best_scale_index = 0
    for scale_index, vertical_scale in enumerate(
        (1.0, 1.2, 1.4, 1.6, 1.8, 0.85)
    ):
        candidate = glyph
        if vertical_scale != 1.0:
            candidate = cv2.resize(
                glyph,
                None,
                fx=1.0,
                fy=vertical_scale,
                interpolation=cv2.INTER_NEAREST,
            )
        digit, score = classify_digit(
            candidate,
            templates,
            allow_right_angle_rotations=False,
        )
        if (
            score > best_score + 1e-12
            or (
                abs(score - best_score) <= 1e-12
                and (scale_index, digit) < (best_scale_index, best_digit)
            )
        ):
            best_digit = digit
            best_score = score
            best_scale_index = scale_index
    return best_digit, best_score


def _detect_markers_from_dark_glyphs(
    bgr: np.ndarray,
    templates: Mapping[int, ImageSource],
    *,
    min_area: float,
    max_area_ratio: float,
    min_confidence: float,
) -> list[MarkerDetection]:
    """Recover markers whose white cards merge into a bright shelf.

    Gazebo's marker cards can have nearly the same colour as the shelf and sky,
    so a white-card contour is not always isolated.  The printed black glyphs
    remain distinct.  This fallback extracts those glyphs, verifies that each
    is surrounded by a bright low-saturation patch, and infers the card box.
    """

    height, width = bgr.shape[:2]
    image_area = float(height * width)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    dark_neutral = cv2.inRange(
        hsv,
        np.array((0, 0, 0), dtype=np.uint8),
        np.array((179, 80, 110), dtype=np.uint8),
    )
    dark_neutral = cv2.morphologyEx(
        dark_neutral,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        dark_neutral,
        connectivity=8,
    )

    minimum_glyph_area = max(8, int(round(min_area * 0.08)))
    maximum_glyph_area = max(
        minimum_glyph_area,
        int(round(image_area * max_area_ratio * 0.65)),
    )
    detections: list[MarkerDetection] = []
    for label_index in range(1, component_count):
        x, y, glyph_width, glyph_height, glyph_area = (
            int(value) for value in stats[label_index]
        )
        if not minimum_glyph_area <= glyph_area <= maximum_glyph_area:
            continue
        if glyph_width < 3 or glyph_height < 6:
            continue
        glyph_aspect = glyph_width / max(glyph_height, 1)
        glyph_fill = glyph_area / float(glyph_width * glyph_height)
        if not 0.22 <= glyph_aspect <= 1.75 or not 0.12 <= glyph_fill <= 0.82:
            continue
        # Marker-search head poses keep the column numbers in the upper part
        # of the image.  This rejects dark floor seams and robot geometry.
        if y + glyph_height / 2.0 > height * 0.70:
            continue

        padding = max(3, int(round(glyph_height * 0.25)))
        card_x1 = max(0, x - padding)
        card_y1 = max(0, y - padding)
        card_x2 = min(width, x + glyph_width + padding)
        card_y2 = min(height, y + glyph_height + padding)
        card_width = card_x2 - card_x1
        card_height = card_y2 - card_y1
        card_area = float(card_width * card_height)
        if card_area < min_area or card_area > image_area * max_area_ratio:
            continue

        local_hsv = hsv[card_y1:card_y2, card_x1:card_x2]
        surround = np.ones(local_hsv.shape[:2], dtype=bool)
        surround[
            y - card_y1:y - card_y1 + glyph_height,
            x - card_x1:x - card_x1 + glyph_width,
        ] = False
        surrounding_pixels = local_hsv[surround]
        if surrounding_pixels.size == 0:
            continue
        bright_neutral = (
            (surrounding_pixels[:, 1] <= 100)
            & (surrounding_pixels[:, 2] >= 150)
        )
        bright_fraction = float(np.mean(bright_neutral))
        if bright_fraction < 0.55:
            continue

        glyph = np.full((glyph_height + 8, glyph_width + 8), 255, dtype=np.uint8)
        component = labels[y:y + glyph_height, x:x + glyph_width] == label_index
        glyph[4:4 + glyph_height, 4:4 + glyph_width][component] = 0
        try:
            digit, template_score = _classify_upright_glyph(glyph, templates)
        except ValueError:
            continue
        confidence = float(
            np.clip(0.92 * template_score + 0.08 * bright_fraction, 0.0, 1.0)
        )
        if confidence < min_confidence:
            continue

        quad = (
            (float(card_x1), float(card_y1)),
            (float(card_x2 - 1), float(card_y1)),
            (float(card_x2 - 1), float(card_y2 - 1)),
            (float(card_x1), float(card_y2 - 1)),
        )
        detections.append(
            MarkerDetection(
                bbox=(card_x1, card_y1, card_width, card_height),
                digit=digit,
                confidence=confidence,
                quadrilateral=quad,
            )
        )
    return detections


def detect_number_markers(
    image: np.ndarray,
    templates: TemplateSources,
    *,
    min_area: float = 250.0,
    max_area_ratio: float = 0.35,
    min_rectangularity: float = 0.68,
    aspect_ratio_range: tuple[float, float] = (0.55, 1.8),
    min_confidence: float = 0.25,
) -> list[MarkerDetection]:
    """Detect and classify every visible white number marker.

    The detector segments low-saturation, bright square cards, rectifies each
    candidate with a perspective transform, then compares its digit with the
    supplied 1--5 templates.  Results are sorted top-to-bottom then left-to-right.
    """

    bgr = _as_bgr(image)
    height, width = bgr.shape[:2]
    image_area = float(height * width)
    if min_area <= 0 or not 0 < max_area_ratio <= 1:
        raise ValueError("min_area and max_area_ratio must be positive")
    if not 0 < min_rectangularity <= 1:
        raise ValueError("min_rectangularity must be in (0, 1]")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be in [0, 1]")

    prepared = _coerce_template_sources(templates, (64, 96))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(
        hsv,
        np.array((0, 0, 125), dtype=np.uint8),
        np.array((179, 72, 255), dtype=np.uint8),
    )
    kernel_size = max(3, int(round(min(height, width) * 0.007)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(kernel_size, 11)
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, close_kernel)
    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    contours, _ = cv2.findContours(
        white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detections: list[MarkerDetection] = []
    min_aspect, max_aspect = aspect_ratio_range
    if min_aspect <= 0 or max_aspect < min_aspect:
        raise ValueError("aspect_ratio_range is invalid")

    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < min_area or contour_area > image_area * max_area_ratio:
            continue
        rectangle = cv2.minAreaRect(contour)
        rectangle_width, rectangle_height = rectangle[1]
        rectangle_area = float(rectangle_width * rectangle_height)
        if rectangle_area <= 0:
            continue
        rectangularity = contour_area / rectangle_area
        aspect = max(rectangle_width, rectangle_height) / max(
            min(rectangle_width, rectangle_height), 1e-6
        )
        # Convert the caller's possibly asymmetric interval to the equivalent
        # orientation-independent maximum ratio.
        allowed_aspect = max(max_aspect, 1.0 / min_aspect)
        if rectangularity < min_rectangularity or aspect > allowed_aspect:
            continue

        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        polygon = cv2.approxPolyDP(hull, 0.025 * perimeter, True)
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            quad = _order_quad(polygon.reshape(4, 2))
        else:
            quad = _order_quad(cv2.boxPoints(rectangle))
        marker_crop = _warp_quad(bgr, quad)
        try:
            digit, template_score = classify_digit(
                marker_crop,
                prepared,
                allow_right_angle_rotations=True,
            )
        except ValueError:
            continue
        geometry_score = float(np.clip(rectangularity, 0.0, 1.0))
        confidence = float(np.clip(0.9 * template_score + 0.1 * geometry_score, 0, 1))
        if confidence < min_confidence:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        detections.append(
            MarkerDetection(
                bbox=(int(x), int(y), int(box_width), int(box_height)),
                digit=digit,
                confidence=confidence,
                quadrilateral=tuple(
                    (float(point[0]), float(point[1])) for point in quad
                ),  # type: ignore[arg-type]
            )
        )

    # A card can merge with the bright shelf/background and disappear from the
    # external white-contour set.  Add digit-first candidates that are not
    # already covered by a card detection.
    for candidate in _detect_markers_from_dark_glyphs(
        bgr,
        prepared,
        min_area=min_area,
        max_area_ratio=max_area_ratio,
        min_confidence=min_confidence,
    ):
        center_x, center_y = candidate.center
        if any(
            existing.bbox[0] <= center_x <= existing.bbox[0] + existing.bbox[2]
            and existing.bbox[1] <= center_y <= existing.bbox[1] + existing.bbox[3]
            for existing in detections
        ):
            continue
        detections.append(candidate)

    detections.sort(key=lambda detection: (detection.bbox[1], detection.bbox[0]))
    return detections


def hsv_color_mask(
    image: np.ndarray,
    color: str,
    *,
    open_kernel: int = 3,
    close_kernel: int = 7,
) -> np.ndarray:
    """Return a cleaned binary HSV mask for one competition colour."""

    normalized_color = color.lower().strip()
    if normalized_color not in HSV_RANGES:
        raise ValueError(f"unsupported colour {color!r}; choose {sorted(HSV_RANGES)}")
    if open_kernel < 0 or close_kernel < 0:
        raise ValueError("morphology kernel sizes cannot be negative")

    hsv = cv2.cvtColor(_as_bgr(image), cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in HSV_RANGES[normalized_color]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    if open_kernel > 1:
        if open_kernel % 2 == 0:
            open_kernel += 1
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (open_kernel, open_kernel)),
        )
    if close_kernel > 1:
        if close_kernel % 2 == 0:
            close_kernel += 1
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel)),
        )
    return mask


def _component_detections(
    mask: np.ndarray,
    *,
    min_area: float,
    max_area: float,
    min_rectangularity: float,
    aspect_ratio_range: tuple[float, float],
) -> list[tuple[BBox, float, float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    components: list[tuple[BBox, float, float]] = []
    minimum_aspect, maximum_aspect = aspect_ratio_range
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width <= 0 or height <= 0:
            continue
        rectangularity = area / float(width * height)
        aspect = width / float(height)
        if rectangularity < min_rectangularity:
            continue
        if not minimum_aspect <= aspect <= maximum_aspect:
            continue
        components.append(
            ((int(x), int(y), int(width), int(height)), area, rectangularity)
        )
    components.sort(key=lambda component: (component[0][1], component[0][0]))
    return components


def detect_colored_books(
    image: np.ndarray,
    *,
    colors: Sequence[str] = ("red", "blue", "green", "yellow"),
    min_area: float = 120.0,
    max_area_ratio: float = 0.12,
    min_rectangularity: float = 0.55,
    aspect_ratio_range: tuple[float, float] = (0.35, 3.5),
    large_red_area_factor: Optional[float] = 3.0,
    assign_rows: bool = True,
    row_count: int = 4,
) -> list[BookDetection]:
    """Detect coloured rectangular book faces using HSV segmentation.

    The relative upper-area limit and a cross-colour red-area outlier check
    reject the much larger collection bin while retaining a red target book.
    Set ``large_red_area_factor=None`` to disable the outlier check.
    """

    bgr = _as_bgr(image)
    image_area = float(bgr.shape[0] * bgr.shape[1])
    if min_area <= 0 or not 0 < max_area_ratio <= 1:
        raise ValueError("min_area and max_area_ratio must be positive")
    if not 0 < min_rectangularity <= 1:
        raise ValueError("min_rectangularity must be in (0, 1]")
    if large_red_area_factor is not None and large_red_area_factor <= 1:
        raise ValueError("large_red_area_factor must exceed 1 or be None")
    minimum_aspect, maximum_aspect = aspect_ratio_range
    if minimum_aspect <= 0 or maximum_aspect < minimum_aspect:
        raise ValueError("aspect_ratio_range is invalid")

    normalized_colors = tuple(color.lower().strip() for color in colors)
    unknown = sorted(set(normalized_colors) - set(HSV_RANGES))
    if unknown:
        raise ValueError(f"unsupported colours: {unknown}")

    books: list[BookDetection] = []
    for color in normalized_colors:
        mask = hsv_color_mask(bgr, color)
        components = _component_detections(
            mask,
            min_area=min_area,
            max_area=image_area * max_area_ratio,
            min_rectangularity=min_rectangularity,
            aspect_ratio_range=aspect_ratio_range,
        )
        for bbox, area, rectangularity in components:
            x, y, width, height = bbox
            patch = mask[y:y + height, x:x + width]
            purity = float(np.count_nonzero(patch)) / float(max(1, width * height))
            confidence = float(
                np.clip(0.55 * rectangularity + 0.45 * purity, 0.0, 1.0)
            )
            books.append(
                BookDetection(
                    color=color,
                    bbox=bbox,
                    area=area,
                    confidence=confidence,
                )
            )

    if large_red_area_factor is not None:
        red_books = [book for book in books if book.color == "red"]
        non_red_areas = [book.area for book in books if book.color != "red"]
        reference_areas = non_red_areas
        if not reference_areas and len(red_books) > 1:
            # With only red objects visible, treat every component except the
            # largest as a possible book population.  This handles a red target
            # and red bin without assuming either one's image position.
            reference_areas = sorted(book.area for book in red_books)[:-1]
        if reference_areas:
            typical_book_area = float(np.median(reference_areas))
            maximum_red_book_area = typical_book_area * large_red_area_factor
            books = [
                book
                for book in books
                if book.color != "red" or book.area <= maximum_red_book_area
            ]

    books.sort(key=lambda book: (book.center[1], book.center[0], book.color))
    if assign_rows and books:
        books = assign_book_rows(books, row_count=row_count)
    return books


def detect_shelf_books(
    image: np.ndarray,
    *,
    min_area: float = 70.0,
) -> list[BookDetection]:
    """Detect shelf books with the live camera's near and far view limits.

    A book seen nearly edge-on at the grasp standoff can be only 10--15%
    as wide as it is tall.  Keeping this production profile in the pure
    vision module makes the close-range acceptance rule independently
    testable without importing ROS.
    """

    return detect_colored_books(
        image,
        min_area=min_area,
        max_area_ratio=0.10,
        aspect_ratio_range=(0.10, 3.0),
    )


def resolve_book_row(
    target: BookDetection,
    confirmed_row: Optional[int] = None,
) -> Optional[int]:
    """Use an overview-confirmed row when tracking a close shelf crop."""

    if confirmed_row is None:
        return target.row
    if not 1 <= int(confirmed_row) <= 4:
        raise ValueError("confirmed_row must be from 1 through 4")
    return int(confirmed_row)


def _natural_row_groups(
    books: Sequence[BookDetection],
    *,
    tolerance: Optional[float],
) -> tuple[list[list[int]], np.ndarray]:
    if not books:
        return [], np.asarray([], dtype=np.float64)

    centers = np.array([book.center[1] for book in books], dtype=np.float64)
    heights = np.array([book.bbox[3] for book in books], dtype=np.float64)
    if tolerance is None:
        tolerance = max(4.0, float(np.median(heights)) * 0.65)
    if tolerance <= 0:
        raise ValueError("row tolerance must be positive")

    order = np.argsort(centers, kind="stable")
    groups: list[list[int]] = []
    for raw_index in order:
        index = int(raw_index)
        if not groups:
            groups.append([index])
            continue
        group_center = float(np.median(centers[groups[-1]]))
        if abs(float(centers[index]) - group_center) <= tolerance:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups, centers


def _merge_nearest_row_groups(
    groups: Sequence[Sequence[int]],
    centers: np.ndarray,
    row_count: int,
) -> list[list[int]]:
    """Merge segmentation splits until no more than ``row_count`` remain."""
    merged = [list(group) for group in groups]
    while len(merged) > row_count:
        group_centers = [float(np.median(centers[group])) for group in merged]
        merge_index = int(np.argmin(np.diff(group_centers)))
        merged[merge_index].extend(merged.pop(merge_index + 1))
    return merged


def _row_labels(
    books: Sequence[BookDetection],
    *,
    row_count: int,
    tolerance: Optional[float],
) -> list[int]:
    if row_count < 1:
        raise ValueError("row_count must be at least 1")
    groups, centers = _natural_row_groups(books, tolerance=tolerance)
    if not groups:
        return []

    # If segmentation noise splits a physical row, merge the nearest adjacent
    # clusters until at most the known number of shelf rows remains.
    groups = _merge_nearest_row_groups(groups, centers, row_count)

    labels = [0] * len(books)
    groups.sort(key=lambda group: float(np.median(centers[group])))
    for row, group in enumerate(groups, start=1):
        for index in group:
            labels[index] = row
    return labels


def assign_book_rows(
    books: Sequence[BookDetection],
    *,
    row_count: int = 4,
    tolerance: Optional[float] = None,
) -> list[BookDetection]:
    """Return copies of detections labelled top-to-bottom from row 1."""

    labels = _row_labels(books, row_count=row_count, tolerance=tolerance)
    return [replace(book, row=row) for book, row in zip(books, labels)]


def has_complete_book_row_layout(
    books: Sequence[BookDetection],
    *,
    row_count: int = 4,
    tolerance: Optional[float] = None,
) -> bool:
    """Return whether an overview contains every physical shelf row."""

    if row_count < 1:
        raise ValueError("row_count must be at least 1")
    groups, centers = _natural_row_groups(books, tolerance=tolerance)
    if len(groups) < row_count:
        return False
    groups = _merge_nearest_row_groups(groups, centers, row_count)
    group_centers = np.asarray(
        [float(np.median(centers[group])) for group in groups],
        dtype=float,
    )
    gaps = np.diff(group_centers)
    if not len(gaps):
        return True
    median_gap = float(np.median(gaps))
    return bool(
        median_gap > 0.0
        and float(np.min(gaps)) >= 0.55 * median_gap
        and float(np.max(gaps)) <= 1.70 * median_gap
    )


def infer_book_row(
    target: BookDetection,
    all_books: Sequence[BookDetection],
    *,
    row_count: int = 4,
    tolerance: Optional[float] = None,
) -> Optional[int]:
    """Infer a target's top-to-bottom shelf row from all coloured candidates."""

    if target.row is not None and 1 <= target.row <= row_count:
        return target.row
    if not all_books:
        return None
    labelled = assign_book_rows(all_books, row_count=row_count, tolerance=tolerance)

    # Identity is preferred, but equality and nearest centre make the helper
    # convenient when detections have been copied or deserialized.
    for original, assigned in zip(all_books, labelled):
        if original is target or original == target:
            return assigned.row
    target_x, target_y = target.center
    nearest = min(
        labelled,
        key=lambda book: (
            (book.center[0] - target_x) ** 2 + (book.center[1] - target_y) ** 2,
            book.bbox[1],
            book.bbox[0],
        ),
    )
    return nearest.row


def select_target_book(
    books: Sequence[BookDetection],
    target_color: str,
    *,
    reference_point: Optional[Point] = None,
) -> Optional[BookDetection]:
    """Choose one target-colour book deterministically.

    With a reference point (for example the image centre), the closest candidate
    wins.  Otherwise confidence, then visible area, then top-left position are
    used.  ``None`` is returned when the colour is absent.
    """

    normalized_color = target_color.lower().strip()
    if normalized_color not in HSV_RANGES:
        raise ValueError(f"unsupported target colour: {target_color!r}")
    candidates = [book for book in books if book.color == normalized_color]
    if not candidates:
        return None
    if reference_point is not None:
        reference_x, reference_y = map(float, reference_point)
        return min(
            candidates,
            key=lambda book: (
                (book.center[0] - reference_x) ** 2
                + (book.center[1] - reference_y) ** 2,
                -book.confidence,
                -book.area,
                book.bbox[1],
                book.bbox[0],
            ),
        )
    return min(
        candidates,
        key=lambda book: (
            -book.confidence,
            -book.area,
            book.bbox[1],
            book.bbox[0],
        ),
    )


def bbox_iou(first: BBox, second: BBox) -> float:
    """Return intersection-over-union for two ``(x, y, w, h)`` boxes."""

    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    intersection_x1 = max(first_x, second_x)
    intersection_y1 = max(first_y, second_y)
    intersection_x2 = min(first_x + first_width, second_x + second_width)
    intersection_y2 = min(first_y + first_height, second_y + second_height)
    intersection = max(0, intersection_x2 - intersection_x1) * max(
        0, intersection_y2 - intersection_y1
    )
    first_area = max(0, first_width) * max(0, first_height)
    second_area = max(0, second_width) * max(0, second_height)
    union = first_area + second_area - intersection
    return float(intersection / union) if union else 0.0


def detect_red_bin(
    image: np.ndarray,
    *,
    book_detections: Sequence[BookDetection] = (),
    min_area: float = 1000.0,
    min_area_ratio: float = 0.01,
    min_rectangularity: float = 0.18,
    exclusion_iou: float = 0.55,
    min_book_area_factor: float = 1.8,
) -> Optional[BinDetection]:
    """Find the large red collection bin without consuming a red target book.

    Red components matching supplied red-book boxes are excluded.  When books
    are supplied, a remaining component must also be materially larger than a
    typical book.  The bin mesh need not be a perfect rectangle.
    """

    bgr = _as_bgr(image)
    image_area = float(bgr.shape[0] * bgr.shape[1])
    if min_area <= 0 or not 0 <= min_area_ratio <= 1:
        raise ValueError("bin area thresholds are invalid")
    if not 0 <= exclusion_iou <= 1:
        raise ValueError("exclusion_iou must be in [0, 1]")
    if min_book_area_factor < 1:
        raise ValueError("min_book_area_factor must be at least 1")

    red_mask = hsv_color_mask(bgr, "red", open_kernel=3, close_kernel=9)
    contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    red_books = [book for book in book_detections if book.color == "red"]
    typical_book_area = (
        float(np.median([book.area for book in book_detections]))
        if book_detections
        else 0.0
    )
    required_area = max(min_area, min_area_ratio * image_area)
    if typical_book_area > 0:
        required_area = max(required_area, typical_book_area * min_book_area_factor)

    candidates: list[BinDetection] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < required_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        bbox = (int(x), int(y), int(width), int(height))
        if any(bbox_iou(bbox, book.bbox) >= exclusion_iou for book in red_books):
            continue
        rectangularity = area / float(max(1, width * height))
        if rectangularity < min_rectangularity:
            continue
        patch = red_mask[y:y + height, x:x + width]
        purity = float(np.count_nonzero(patch)) / float(max(1, width * height))
        size_score = min(1.0, area / max(required_area * 2.0, 1.0))
        confidence = float(
            np.clip(
                0.35 * rectangularity + 0.35 * purity + 0.30 * size_score,
                0.0,
                1.0,
            )
        )
        candidates.append(BinDetection(bbox=bbox, area=area, confidence=confidence))

    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: (
            -candidate.area,
            -candidate.confidence,
            candidate.bbox[1],
            candidate.bbox[0],
        ),
    )


def median_valid_depth(
    depth_image: np.ndarray,
    pixel: tuple[float, float],
    *,
    radius: int = 2,
    min_depth: float = 0.05,
    max_depth: float = 10.0,
    depth_scale: float = 1.0,
) -> Optional[float]:
    """Return the median valid depth around ``pixel`` in metres (or caller units).

    Invalid zero, negative, NaN, and infinite samples are discarded.  Integer
    millimetre images should pass ``depth_scale=0.001``.  The sampling window is
    clipped at image borders and ``None`` indicates that no valid sample exists.
    """

    depth = _validate_image(depth_image, "depth_image")
    if depth.ndim == 3:
        if depth.shape[2] != 1:
            raise ValueError("depth_image must be single-channel")
        depth = depth[:, :, 0]
    if radius < 0:
        raise ValueError("radius cannot be negative")
    if depth_scale <= 0 or min_depth < 0 or max_depth <= min_depth:
        raise ValueError("depth limits and scale are invalid")
    u, v = map(float, pixel)
    if not np.isfinite((u, v)).all():
        raise ValueError("pixel coordinates must be finite")
    center_x = int(round(u))
    center_y = int(round(v))
    height, width = depth.shape
    if center_x < 0 or center_x >= width or center_y < 0 or center_y >= height:
        return None
    x1 = max(0, center_x - radius)
    x2 = min(width, center_x + radius + 1)
    y1 = max(0, center_y - radius)
    y2 = min(height, center_y + radius + 1)
    samples = depth[y1:y2, x1:x2].astype(np.float64, copy=False) * depth_scale
    valid = samples[
        np.isfinite(samples) & (samples >= min_depth) & (samples <= max_depth)
    ]
    if valid.size == 0:
        return None
    return float(np.median(valid))


# A descriptive alias used by some callers.
sample_median_depth = median_valid_depth


def _coerce_intrinsics(intrinsics: Any) -> CameraIntrinsics:
    if isinstance(intrinsics, CameraIntrinsics):
        return intrinsics
    if isinstance(intrinsics, Mapping):
        try:
            return CameraIntrinsics(
                fx=float(intrinsics["fx"]),
                fy=float(intrinsics["fy"]),
                cx=float(intrinsics["cx"]),
                cy=float(intrinsics["cy"]),
            )
        except KeyError as error:
            raise ValueError(f"missing camera intrinsic {error.args[0]!r}") from error
    if all(hasattr(intrinsics, name) for name in ("fx", "fy", "cx", "cy")):
        return CameraIntrinsics(
            fx=float(intrinsics.fx),
            fy=float(intrinsics.fy),
            cx=float(intrinsics.cx),
            cy=float(intrinsics.cy),
        )
    # sensor_msgs/CameraInfo stores the row-major camera matrix in k/K.
    matrix = getattr(intrinsics, "k", getattr(intrinsics, "K", None))
    if matrix is not None and len(matrix) >= 6:
        return CameraIntrinsics(
            fx=float(matrix[0]),
            fy=float(matrix[4]),
            cx=float(matrix[2]),
            cy=float(matrix[5]),
        )
    raise TypeError(
        "intrinsics must be CameraIntrinsics, a mapping, an fx/fy/cx/cy object, "
        "or a CameraInfo-like object"
    )


def deproject_pixel(
    pixel: tuple[float, float],
    depth: float,
    intrinsics: Any,
) -> tuple[float, float, float]:
    """Deproject one ``(u, v)`` pixel using a pinhole camera model."""

    camera = _coerce_intrinsics(intrinsics)
    u, v = map(float, pixel)
    z = float(depth)
    values = np.array((u, v, z, camera.fx, camera.fy, camera.cx, camera.cy))
    if not np.isfinite(values).all():
        raise ValueError("pixel, depth, and camera intrinsics must be finite")
    if z <= 0:
        raise ValueError("depth must be positive")
    if camera.fx <= 0 or camera.fy <= 0:
        raise ValueError("camera focal lengths must be positive")
    x = (u - camera.cx) * z / camera.fx
    y = (v - camera.cy) * z / camera.fy
    return (float(x), float(y), z)


def deproject_detection_center(
    detection: Union[MarkerDetection, BookDetection, BinDetection],
    depth_image: np.ndarray,
    intrinsics: Any,
    **depth_options: Any,
) -> Optional[tuple[float, float, float]]:
    """Sample depth and deproject a detection centre, or return ``None``."""

    depth = median_valid_depth(depth_image, detection.center, **depth_options)
    if depth is None:
        return None
    return deproject_pixel(detection.center, depth, intrinsics)


def _draw_label(
    image: np.ndarray,
    bbox: BBox,
    label: str,
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
) -> None:
    x, y, width, height = bbox
    cv2.rectangle(
        image,
        (x, y),
        (x + max(0, width - 1), y + max(0, height - 1)),
        color,
        thickness,
        cv2.LINE_AA,
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_thickness = max(1, thickness - 1)
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, text_thickness
    )
    text_x = max(0, x)
    text_y = y - 5
    if text_y - text_height < 0:
        text_y = min(image.shape[0] - baseline - 1, y + text_height + 5)
    cv2.rectangle(
        image,
        (text_x, max(0, text_y - text_height - 2)),
        (
            min(image.shape[1] - 1, text_x + text_width + 4),
            min(image.shape[0] - 1, text_y + baseline),
        ),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        image,
        label,
        (text_x + 2, text_y),
        font,
        font_scale,
        (0, 0, 0),
        text_thickness,
        cv2.LINE_AA,
    )


def annotate_markers(
    image: np.ndarray,
    detections: Sequence[MarkerDetection],
    *,
    copy: bool = True,
) -> np.ndarray:
    """Draw marker boxes, digits, and confidences."""

    canvas = _as_bgr(image).copy() if copy else _as_bgr(image)
    for detection in detections:
        _draw_label(
            canvas,
            detection.bbox,
            f"marker {detection.digit} {detection.confidence:.2f}",
            DRAW_COLORS["marker"],
        )
    return canvas


def annotate_books(
    image: np.ndarray,
    detections: Sequence[BookDetection],
    *,
    target: Optional[BookDetection] = None,
    copy: bool = True,
) -> np.ndarray:
    """Draw colour/row labels, highlighting an optional target."""

    canvas = _as_bgr(image).copy() if copy else _as_bgr(image)
    for detection in detections:
        is_target = target is not None and (
            detection is target
            or (detection.color, detection.bbox) == (target.color, target.bbox)
        )
        color = DRAW_COLORS["target"] if is_target else DRAW_COLORS[detection.color]
        row_text = f" r{detection.row}" if detection.row is not None else ""
        prefix = "target " if is_target else ""
        _draw_label(
            canvas,
            detection.bbox,
            f"{prefix}{detection.color}{row_text} {detection.confidence:.2f}",
            color,
            thickness=3 if is_target else 2,
        )
    return canvas


def annotate_bin(
    image: np.ndarray,
    detection: Optional[BinDetection],
    *,
    copy: bool = True,
) -> np.ndarray:
    """Draw the collection-bin result when one is present."""

    canvas = _as_bgr(image).copy() if copy else _as_bgr(image)
    if detection is not None:
        _draw_label(
            canvas,
            detection.bbox,
            f"red bin {detection.confidence:.2f}",
            DRAW_COLORS["bin"],
            thickness=3,
        )
    return canvas


def annotate_scene(
    image: np.ndarray,
    *,
    markers: Sequence[MarkerDetection] = (),
    books: Sequence[BookDetection] = (),
    target_book: Optional[BookDetection] = None,
    red_bin: Optional[BinDetection] = None,
) -> np.ndarray:
    """Return one annotated copy containing all available perception results."""

    canvas = _as_bgr(image).copy()
    annotate_markers(canvas, markers, copy=False)
    annotate_books(canvas, books, target=target_book, copy=False)
    annotate_bin(canvas, red_bin, copy=False)
    return canvas


__all__ = [
    "BBox",
    "BinDetection",
    "BookDetection",
    "CameraIntrinsics",
    "HSV_RANGES",
    "MarkerDetection",
    "TemplateSources",
    "annotate_bin",
    "annotate_books",
    "annotate_markers",
    "annotate_scene",
    "assign_book_rows",
    "bbox_iou",
    "classify_digit",
    "deproject_detection_center",
    "deproject_pixel",
    "detect_colored_books",
    "detect_number_markers",
    "detect_red_bin",
    "detect_shelf_books",
    "hsv_color_mask",
    "has_complete_book_row_layout",
    "infer_book_row",
    "load_digit_templates",
    "median_valid_depth",
    "resolve_book_row",
    "sample_median_depth",
    "select_target_book",
]
