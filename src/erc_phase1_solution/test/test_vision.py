"""Unit tests for the pure OpenCV/NumPy phase-1 perception library."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from erc_phase1_solution.vision import (
    BinDetection,
    BookDetection,
    CameraIntrinsics,
    MarkerDetection,
    annotate_scene,
    assign_book_rows,
    bbox_iou,
    classify_digit,
    deproject_detection_center,
    deproject_pixel,
    detect_colored_books,
    detect_number_markers,
    detect_red_bin,
    detect_shelf_books,
    hsv_color_mask,
    has_complete_book_row_layout,
    infer_book_row,
    load_digit_templates,
    median_valid_depth,
    resolve_book_row,
    select_target_book,
)


def _digit_card(digit: int, size: int = 112) -> np.ndarray:
    """Create a high-contrast square marker card for deterministic tests."""

    card = np.full((size, size, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = size / 42.0
    thickness = max(3, int(round(size / 16.0)))
    (width, height), _ = cv2.getTextSize(str(digit), font, scale, thickness)
    origin = ((size - width) // 2, (size + height) // 2)
    cv2.putText(
        card,
        str(digit),
        origin,
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
    return card


@pytest.fixture()
def digit_templates() -> dict[int, np.ndarray]:
    return {digit: _digit_card(digit) for digit in range(1, 6)}


def test_load_templates_from_arrays_sequence_and_directory(
    tmp_path, digit_templates
):
    from_mapping = load_digit_templates(digit_templates)
    assert set(from_mapping) == {1, 2, 3, 4, 5}
    assert all(mask.shape == (96, 64) for mask in from_mapping.values())
    assert all(mask.dtype == np.uint8 for mask in from_mapping.values())

    from_sequence = load_digit_templates(
        [digit_templates[digit] for digit in range(1, 6)]
    )
    assert all(
        np.array_equal(from_mapping[digit], from_sequence[digit])
        for digit in range(1, 6)
    )

    for digit, image in digit_templates.items():
        assert cv2.imwrite(str(tmp_path / f"{digit}.png"), image)
    from_directory = load_digit_templates(tmp_path)
    assert all(
        np.array_equal(from_mapping[digit], from_directory[digit])
        for digit in range(1, 6)
    )


@pytest.mark.parametrize("digit", range(1, 6))
@pytest.mark.parametrize(
    "rotation_code",
    [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180],
)
def test_classify_digits_one_through_five_at_right_angle_rotations(
    digit_templates, digit, rotation_code
):
    crop = digit_templates[digit]
    if rotation_code is not None:
        crop = cv2.rotate(crop, rotation_code)
    result, confidence = classify_digit(crop, digit_templates)
    assert result == digit
    assert confidence > 0.90


def test_classify_uses_paths_and_rejects_blank_crop(tmp_path, digit_templates):
    paths = {}
    for digit, image in digit_templates.items():
        path = tmp_path / f"template_{digit}.png"
        assert cv2.imwrite(str(path), image)
        paths[digit] = path
    digit, confidence = classify_digit(digit_templates[4], paths)
    assert digit == 4
    assert confidence > 0.90
    sequence_digit, _ = classify_digit(
        digit_templates[2], [paths[digit] for digit in range(1, 6)]
    )
    assert sequence_digit == 2
    with pytest.raises(ValueError, match="classifiable"):
        classify_digit(np.full((80, 80), 255, dtype=np.uint8), paths)


def test_detects_and_returns_all_number_markers(digit_templates):
    scene = np.full((420, 620, 3), (55, 85, 45), dtype=np.uint8)
    placements = [
        (35, 40, 1, None),
        (245, 60, 4, cv2.ROTATE_90_CLOCKWISE),
        (445, 235, 5, cv2.ROTATE_180),
    ]
    for x, y, digit, rotation in placements:
        card = digit_templates[digit]
        if rotation is not None:
            card = cv2.rotate(card, rotation)
        scene[y:y + card.shape[0], x:x + card.shape[1]] = card

    detections = detect_number_markers(scene, digit_templates, min_area=1000)
    assert len(detections) == 3
    assert [detection.digit for detection in detections] == [1, 4, 5]
    assert all(isinstance(detection, MarkerDetection) for detection in detections)
    assert all(detection.confidence > 0.75 for detection in detections)
    for detection, (expected_x, expected_y, _, _) in zip(detections, placements):
        x, y, width, height = detection.bbox
        assert abs(x - expected_x) <= 2
        assert abs(y - expected_y) <= 2
        assert width >= 108 and height >= 108
        assert detection.quadrilateral is not None


def test_marker_detector_rectifies_a_perspective_card(digit_templates):
    scene = np.full((230, 280, 3), (45, 80, 35), dtype=np.uint8)
    card = digit_templates[2]
    source = np.array(
        ((0, 0), (card.shape[1] - 1, 0),
         (card.shape[1] - 1, card.shape[0] - 1), (0, card.shape[0] - 1)),
        dtype=np.float32,
    )
    destination = np.array(
        ((52, 28), (173, 47), (155, 188), (35, 162)), dtype=np.float32
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(card, transform, (scene.shape[1], scene.shape[0]))
    alpha = cv2.warpPerspective(
        np.full(card.shape[:2], 255, dtype=np.uint8),
        transform,
        (scene.shape[1], scene.shape[0]),
    )
    scene[alpha > 0] = warped[alpha > 0]

    detections = detect_number_markers(scene, digit_templates, min_area=1000)
    assert len(detections) == 1
    assert detections[0].digit == 2
    assert detections[0].confidence > 0.65


def test_marker_detector_recovers_glyphs_when_cards_merge_with_bright_shelf(
    digit_templates,
):
    scene = np.full((260, 520, 3), 180, dtype=np.uint8)
    placements = [(42, 35, 2), (205, 52, 1), (355, 69, 4)]
    expected_centers = []
    for x, y, digit in placements:
        # Oblique shelf views vertically compress the otherwise square cards.
        card = cv2.resize(
            digit_templates[digit],
            (112, 70),
            interpolation=cv2.INTER_AREA,
        )
        scene[y:y + card.shape[0], x:x + card.shape[1]] = card
        expected_centers.append((x + card.shape[1] / 2, y + card.shape[0] / 2))

    detections = detect_number_markers(scene, digit_templates, min_area=120)

    assert [detection.digit for detection in detections] == [2, 1, 4]
    for detection, expected_center in zip(detections, expected_centers):
        assert detection.center == pytest.approx(expected_center, abs=8)
        assert detection.confidence > 0.60


def _add_sparse_colour_noise(image: np.ndarray) -> None:
    rng = np.random.default_rng(20260827)
    colors = np.array(
        [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255)],
        dtype=np.uint8,
    )
    for x, y, color_index in zip(
        rng.integers(0, image.shape[1], 180),
        rng.integers(0, image.shape[0], 180),
        rng.integers(0, 4, 180),
    ):
        image[int(y), int(x)] = colors[int(color_index)]


def _book_scene(include_bin: bool = False) -> np.ndarray:
    scene = np.full((470, 660, 3), (35, 35, 35), dtype=np.uint8)
    # Four shelf levels; two books share row 2 to exercise clustering.
    rectangles = [
        ((42, 35), (126, 70), (0, 0, 255)),
        ((58, 130), (150, 167), (255, 0, 0)),
        ((230, 134), (316, 169), (0, 255, 255)),
        ((75, 228), (161, 264), (0, 255, 0)),
        ((90, 330), (178, 367), (255, 0, 0)),
    ]
    for top_left, bottom_right, color in rectangles:
        cv2.rectangle(scene, top_left, bottom_right, color, cv2.FILLED)
        # A narrow dark seam should be repaired by the close operation.
        seam_x = (top_left[0] + bottom_right[0]) // 2
        cv2.line(
            scene,
            (seam_x, top_left[1] + 4),
            (seam_x, bottom_right[1] - 4),
            (25, 25, 25),
            2,
        )
    if include_bin:
        # This is under detect_colored_books' relative area cap, so the
        # cross-colour red outlier logic (not only the cap) must reject it.
        cv2.rectangle(scene, (450, 270), (640, 440), (0, 0, 245), cv2.FILLED)
        # Hollow lip makes the candidate less perfectly rectangular, like a bin.
        cv2.rectangle(scene, (475, 290), (615, 325), (35, 35, 35), cv2.FILLED)
    _add_sparse_colour_noise(scene)
    return scene


def test_hsv_masks_remove_sparse_noise_and_bridge_book_seam():
    scene = _book_scene()
    red_mask = hsv_color_mask(scene, "red")
    contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    substantial = [contour for contour in contours if cv2.contourArea(contour) > 100]
    assert len(substantial) == 1
    assert red_mask[52, 84] == 255
    with pytest.raises(ValueError, match="unsupported"):
        hsv_color_mask(scene, "purple")


def test_detect_books_assign_rows_and_choose_target():
    scene = _book_scene()
    books = detect_colored_books(scene)
    assert len(books) == 5
    assert [book.row for book in books] == [1, 2, 2, 3, 4]
    assert {book.color for book in books} == {"red", "blue", "green", "yellow"}
    assert all(book.confidence > 0.80 for book in books)

    row_four_blue = select_target_book(books, "blue", reference_point=(120, 350))
    assert row_four_blue is not None
    assert row_four_blue.row == 4
    assert infer_book_row(row_four_blue, books) == 4
    assert select_target_book(books, "green").row == 3
    assert select_target_book(books, "yellow").row == 2


def test_live_profile_accepts_narrow_close_range_book():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (326, 78), (338, 165), (0, 0, 255), -1)

    books = detect_shelf_books(image)

    assert len(books) == 1
    assert books[0].color == "red"
    assert books[0].row == 1
    assert books[0].bbox[2] / books[0].bbox[3] < 0.15


@pytest.mark.parametrize('local_row', [1, 2, 3, 4])
@pytest.mark.parametrize('confirmed_row', [1, 2, 3, 4])
def test_close_range_tracking_preserves_confirmed_global_row(
    local_row,
    confirmed_row,
):
    target = BookDetection('red', (10, 20, 12, 88), 900, 0.92, row=local_row)

    assert resolve_book_row(target, confirmed_row) == confirmed_row


def test_overview_tracking_uses_detected_row_without_confirmation():
    target = BookDetection('blue', (10, 20, 30, 80), 2000, 0.90, row=3)

    assert resolve_book_row(target) == 3


def test_row_assignment_merges_split_clusters_deterministically():
    books = [
        BookDetection("blue", (10, y, 30, 20), 600, 0.9)
        for y in (8, 15, 80, 145, 210, 217)
    ]
    labelled = assign_book_rows(books, row_count=4, tolerance=3)
    assert [book.row for book in labelled] == [1, 1, 2, 3, 4, 4]
    detached_target = BookDetection("blue", (11, 211, 30, 20), 600, 0.9)
    assert infer_book_row(detached_target, labelled) == 4


@pytest.mark.parametrize('missing_y', [20, 120, 220, 320])
def test_overview_row_layout_rejects_any_missing_physical_band(missing_y):
    books = [
        BookDetection('blue', (10, y, 30, 70), 1800, 0.9)
        for y in (20, 120, 220, 320)
        if y != missing_y
    ]
    labelled = assign_book_rows(books, row_count=4, tolerance=20)

    assert not has_complete_book_row_layout(labelled)


def test_overview_row_layout_accepts_all_four_bands():
    books = [
        BookDetection('blue', (10, y, 30, 70), 1800, 0.9)
        for y in (20, 120, 220, 320)
    ]

    assert has_complete_book_row_layout(assign_book_rows(books))


def test_overview_row_layout_accepts_split_component_with_all_four_rows():
    books = [
        BookDetection('blue', (10, y, 30, 20), 600, 0.9)
        for y in (20, 45, 120, 220, 320)
    ]

    assert has_complete_book_row_layout(assign_book_rows(books))


def test_overview_row_layout_rejects_split_cluster_masking_a_missing_row():
    books = [
        BookDetection('blue', (10, y, 30, 20), 600, 0.9)
        for y in (120, 165, 220, 320)
    ]

    assert not has_complete_book_row_layout(assign_book_rows(books))


def test_large_red_bin_and_red_target_book_are_both_retained():
    scene = _book_scene(include_bin=True)
    books = detect_colored_books(scene)
    red_books = [book for book in books if book.color == "red"]
    assert len(red_books) == 1
    assert red_books[0].row == 1

    collection_bin = detect_red_bin(scene, book_detections=books)
    assert isinstance(collection_bin, BinDetection)
    assert collection_bin.area > red_books[0].area * 8
    x, y, width, height = collection_bin.bbox
    assert x <= 451 and y <= 271
    assert width >= 188 and height >= 168
    assert bbox_iou(collection_bin.bbox, red_books[0].bbox) == 0.0

    # Supplying the red book by itself must not make it masquerade as a bin.
    book_only_scene = np.full((200, 260, 3), 30, dtype=np.uint8)
    cv2.rectangle(book_only_scene, (50, 60), (130, 95), (0, 0, 255), cv2.FILLED)
    book = BookDetection("red", (50, 60, 81, 36), 2800, 1.0, row=1)
    assert detect_red_bin(
        book_only_scene, book_detections=[book], min_area=300
    ) is None


def test_median_valid_depth_discards_invalid_samples_and_clips_window():
    depth = np.zeros((7, 7), dtype=np.float32)
    depth[2:5, 2:5] = np.array(
        [
            [np.nan, 1.0, np.inf],
            [0.0, 2.0, -1.0],
            [3.0, 12.0, 4.0],
        ],
        dtype=np.float32,
    )
    assert median_valid_depth(depth, (3, 3), radius=1) == pytest.approx(2.5)
    assert median_valid_depth(depth, (0, 0), radius=3) == pytest.approx(1.5)
    assert median_valid_depth(depth, (6, 6), radius=0) is None
    assert median_valid_depth(depth, (-1, 2)) is None

    millimetres = np.array([[0, 1500], [2500, 0]], dtype=np.uint16)
    assert median_valid_depth(
        millimetres,
        (1, 0),
        radius=1,
        depth_scale=0.001,
    ) == pytest.approx(2.0)


@dataclass
class _CameraInfoLike:
    k: list[float]


def test_deprojection_accepts_intrinsics_mapping_and_camera_info():
    intrinsics = CameraIntrinsics(fx=500, fy=400, cx=320, cy=240)
    assert deproject_pixel((420, 200), 2.0, intrinsics) == pytest.approx(
        (0.4, -0.2, 2.0)
    )
    assert deproject_pixel(
        (420, 200), 2.0, {"fx": 500, "fy": 400, "cx": 320, "cy": 240}
    ) == pytest.approx((0.4, -0.2, 2.0))
    info = _CameraInfoLike([500, 0, 320, 0, 400, 240, 0, 0, 1])
    assert deproject_pixel((420, 200), 2.0, info) == pytest.approx(
        (0.4, -0.2, 2.0)
    )
    with pytest.raises(ValueError, match="positive"):
        deproject_pixel((1, 2), 0.0, intrinsics)


def test_deproject_detection_center_and_annotations_do_not_mutate_input():
    image = np.zeros((180, 260, 3), dtype=np.uint8)
    original = image.copy()
    marker = MarkerDetection((10, 20, 60, 60), 3, 0.91)
    book = BookDetection("green", (90, 80, 70, 30), 2000, 0.88, row=2)
    bin_detection = BinDetection((175, 55, 70, 100), 6500, 0.82)
    annotated = annotate_scene(
        image,
        markers=[marker],
        books=[book],
        target_book=book,
        red_bin=bin_detection,
    )
    assert np.array_equal(image, original)
    assert annotated.shape == image.shape
    assert np.count_nonzero(annotated) > 0

    depth = np.full((180, 260), 2.0, dtype=np.float32)
    point = deproject_detection_center(
        book,
        depth,
        CameraIntrinsics(fx=100, fy=100, cx=0, cy=0),
    )
    assert point == pytest.approx((2.5, 1.9, 2.0))
