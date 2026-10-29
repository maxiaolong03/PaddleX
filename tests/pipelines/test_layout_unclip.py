# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""V4 unclipping must reach recognition crops, including page rectification."""

import numpy as np
import pytest

from paddlex.inference.models.layout_analysis.processors import LayoutAnalysisProcess
from paddlex.inference.pipelines.components.common.crop_image_regions import CropByBoxes
from paddlex.inference.pipelines.paddleocr_vl.pipeline import _PaddleOCRVLPipeline


@pytest.fixture(
    params=[
        "off",
        "gated",
        "low_confidence",
        "quarter_turn",
        "homography",
        "rotated_homography",
    ]
)
def correction_mode(request):
    return request.param


def _layout(quads, ratio, shape_mode="auto", classes=None, order=None):
    points = np.asarray(quads)
    count = len(points)
    classes = np.zeros(count) if classes is None else classes
    order = np.arange(count) if order is None else order
    boxes = np.column_stack(
        [classes, np.full(count, 0.9), points.min(axis=1), points.max(axis=1), order]
    ).astype(np.float32)
    return LayoutAnalysisProcess(labels=["text", "table"], scale_size=[180, 240])(
        [{"boxes": boxes, "quad": points.reshape(count, 8)}],
        [{"ori_img_size": (240, 180)}],
        threshold=0.5,
        layout_nms=False,
        layout_unclip_ratio=ratio,
        layout_merge_bboxes_mode=None,
        layout_shape_mode=shape_mode,
        filter_overlap_boxes=False,
        skip_order_labels=[],
    )[0]


def _crop(image, boxes, mode, shape_mode="auto"):
    warp_params = {
        "off": (7.0, 1.0, -3.0, 0.1, -0.05),
        "gated": (0.2, 1.0, 0.1, 0.005, 0.0),
        "low_confidence": (90.0, 0.5, 5.0, 0.2, 0.2),
        "quarter_turn": (90.4, 1.0, 0.0, 0.0, 0.0),
        "homography": (7.0, 1.0, -3.0, 0.1, -0.05),
        "rotated_homography": (-86.0, 1.0, 3.0, 0.05, -0.08),
    }
    correction = _PaddleOCRVLPipeline._page_correction_for_crop(
        image,
        {"boxes": boxes, "warp_params": warp_params[mode]},
        {
            "use_layout_deform": mode != "off",
            "layout_deform_conf_thresh": 0.9,
            "layout_deform_min_roll": 2.0,
            "layout_deform_min_shear": 2.0,
            "layout_deform_min_keystone": 2.0,
        },
    )
    if "homography" in mode:
        assert correction.homography is not None
    elif mode == "quarter_turn":
        assert correction.homography is None and correction.rot90_k == 3
    else:
        assert correction is None
    return CropByBoxes()(
        image, boxes, shape_mode, **_PaddleOCRVLPipeline._crop_kwargs(correction)
    )[0]


def _paint(image, point, color):
    x, y = point
    image[y - 3 : y + 4, x - 3 : x + 4] = color


def _pixel_at_page_point(crop, point):
    if "crop_matrix" in crop:
        projected = crop["crop_matrix"] @ [*point, 1.0]
        xy = projected[:2] / projected[2]
    else:
        xy = np.asarray(point) - crop["box"][:2]
    x, y = np.rint(xy).astype(int)
    assert 0 <= x < crop["img"].shape[1]
    assert 0 <= y < crop["img"].shape[0]
    return crop["img"][y, x]


@pytest.mark.parametrize("shape_mode", ["auto", "rect"])
@pytest.mark.parametrize(
    "ratio, expected_box",
    [
        (2.0, [70, 50, 150, 130]),
        ([2.0, 1.5], [70, 60, 150, 120]),
        ((2.0, 1.5), [70, 60, 150, 120]),
        ({0: (2.0, 1.5)}, [70, 60, 150, 120]),
    ],
)
def test_unclip_reveals_pixels_outside_original_quad(
    ratio, expected_box, shape_mode, correction_mode
):
    image = np.full((180, 240, 3), 255, dtype=np.uint8)
    quad = np.array([[[90, 70], [130, 70], [130, 110], [90, 110]]], dtype=np.float32)
    markers = [
        ((78, 90), [0, 0, 200]),
        ((142, 90), [0, 150, 0]),
        ((110, 65), [150, 0, 0]),
        ((110, 115), [0, 100, 200]),
    ]
    for point, color in markers:
        _paint(image, point, color)
    result = _layout(quad, ratio, shape_mode)
    assert result[0]["coordinate"] == expected_box
    crop = _crop(image, result, correction_mode, shape_mode)
    assert crop["box"] == expected_box
    for point, color in markers:
        # Verify actual source pixels reach recognition, not just larger metadata.
        np.testing.assert_allclose(_pixel_at_page_point(crop, point), color, atol=2)


def test_skew_quad_expands_without_filling_enclosing_rectangle(correction_mode):
    image = np.full((180, 240, 3), 255, dtype=np.uint8)
    quad = np.array([[[90, 70], [130, 80], [120, 110], [80, 100]]], dtype=np.float32)
    _paint(image, (65, 100), [10, 40, 200])
    _paint(image, (60, 65), [0, 200, 0])
    result = _layout(quad, [2.0, 1.5])
    assert result[0]["coordinate"] == [55, 60, 155, 120]
    crop = _crop(image, result, correction_mode)
    np.testing.assert_allclose(
        _pixel_at_page_point(crop, (65, 100)), [10, 40, 200], atol=2
    )
    assert not np.any(
        np.all(np.abs(crop["img"].astype(int) - [0, 200, 0]) < 10, axis=-1)
    )
    np.testing.assert_array_equal(
        result[0]["quad"], [[75, 60], [155, 75], [135, 120], [55, 105]]
    )


def test_class_specific_unclip_follows_sorted_boxes():
    quads = np.array(
        [
            [[90, 70], [130, 70], [130, 110], [90, 110]],
            [[10, 10], [30, 10], [30, 30], [10, 30]],
        ],
        dtype=np.float32,
    )
    result = _layout(quads, {0: (2.0, 1.5)}, classes=[0, 1], order=[1, 0])
    assert [box["cls_id"] for box in result] == [1, 0]
    np.testing.assert_array_equal(result[0]["quad"], quads[1])
    np.testing.assert_array_equal(
        result[1]["quad"], [[70, 60], [150, 60], [150, 120], [70, 120]]
    )


@pytest.mark.parametrize("ratio", [1.0, [1.0, 1.0], {0: (1.0, 1.0)}, {1: (2.0, 2.0)}])
def test_identity_unclip_preserves_quad_and_crop_exactly(ratio, correction_mode):
    image = np.random.default_rng(123).integers(0, 256, (180, 240, 3), dtype=np.uint8)
    quads = np.array(
        [[[90.25, 70.5], [130.25, 80.5], [120.25, 110.5], [80.25, 100.5]]],
        dtype=np.float32,
    )
    original = quads.copy()
    plain = _layout(quads, None)
    expanded = _layout(quads, ratio)
    assert expanded == plain
    np.testing.assert_array_equal(
        _crop(image, expanded, correction_mode)["img"],
        _crop(image, plain, correction_mode)["img"],
    )
    np.testing.assert_array_equal(quads, original)


def test_unclip_does_not_mutate_or_reexpand_model_quads():
    quads = np.array([[[90, 70], [130, 70], [130, 110], [90, 110]]], dtype=np.float32)
    original = quads.copy()
    first = _layout(quads, 2.0)
    second = _layout(quads, 2.0)
    assert first == second
    np.testing.assert_array_equal(quads, original)


def test_unclip_supports_shrinking_quad(correction_mode):
    image = np.full((180, 240, 3), 255, dtype=np.uint8)
    _paint(image, (110, 90), [0, 0, 0])
    _paint(image, (94, 90), [0, 200, 0])
    quads = np.array([[[90, 70], [130, 70], [130, 110], [90, 110]]], dtype=np.float32)
    result = _layout(quads, 0.5)
    assert result[0]["coordinate"] == [100, 80, 120, 100]
    np.testing.assert_array_equal(
        result[0]["quad"], [[100, 80], [120, 80], [120, 100], [100, 100]]
    )
    crop = _crop(image, result, correction_mode)
    np.testing.assert_allclose(_pixel_at_page_point(crop, (110, 90)), [0, 0, 0], atol=2)
    assert not np.any(
        np.all(np.abs(crop["img"].astype(int) - [0, 200, 0]) < 10, axis=-1)
    )


def test_unclip_at_page_edge_clips_boxes_but_preserves_quad_shape(correction_mode):
    image = np.full((180, 240, 3), [17, 31, 49], dtype=np.uint8)
    quads = np.array([[[0, 0], [30, 0], [30, 20], [0, 20]]], dtype=np.float32)
    result = _layout(quads, 2.0)
    assert result[0]["coordinate"] == [0, 0, 45, 30]
    np.testing.assert_array_equal(
        result[0]["quad"], [[-15, -10], [45, -10], [45, 30], [-15, 30]]
    )
    crop = _crop(image, result, correction_mode)
    np.testing.assert_allclose(
        _pixel_at_page_point(crop, (10, 10)), [17, 31, 49], atol=2
    )
    if "homography" in correction_mode:
        np.testing.assert_array_equal(
            _pixel_at_page_point(crop, (-10, -5)), [255, 255, 255]
        )
    else:
        assert np.all(crop["img"] == [17, 31, 49])
