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

"""Historical default geometry and durable pre-unclip filter boundaries."""

import json

import numpy as np
import pytest

from paddlex.inference.models.layout_analysis import processors
from paddlex.inference.models.layout_analysis.result import LayoutAnalysisResult
from paddlex.inference.pipelines.components.common.crop_image_regions import CropByBoxes
from paddlex.inference.pipelines.components.common.page_rectify import (
    plan_page_correction,
)
from paddlex.inference.pipelines.layout_parsing.utils import remove_overlap_blocks
from paddlex.inference.pipelines.paddleocr_vl.uilts import filter_overlap_boxes


def _process(output, ratio=None, shape="rect", filtering=False, **kwargs):
    options = dict(
        threshold=0.5,
        layout_nms=False,
        layout_unclip_ratio=ratio,
        layout_merge_bboxes_mode=None,
        layout_shape_mode=shape,
        filter_overlap_boxes=filtering,
        skip_order_labels=[],
    )
    options.update(kwargs)
    return processors.LayoutAnalysisProcess(
        labels=["text", "inline_formula", "image"], scale_size=[400, 400]
    )([output], [{"ori_img_size": (400, 400)}], **options)[0]


@pytest.mark.parametrize("ratio", [None, 1.0, [1.0, 1.0], {0: (1.0, 1.0)}, {1: 2.0}])
@pytest.mark.parametrize("kind", ["rect", "quad", "mask"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_default_coordinate_and_schema_match_historical_rounding(ratio, kind, dtype):
    boxes = np.array([[0, 0.9, 10.6, 40.6, 20.4, 60.4]], dtype=dtype)
    before = boxes.copy()
    output = {"boxes": boxes}
    quad = np.array([[10.6, 40.6, 20.4, 40.6, 20.4, 60.4, 10.6, 60.4]], dtype=dtype)
    if kind != "rect":
        output["quad"] = quad
    if kind == "mask":
        output["masks"] = np.ones((1, 100, 100), dtype=np.uint8)
    box = _process(output, ratio=ratio, shape="auto")[0]
    # Fixed snapshots from the pre-PR processor, rather than comparing two
    # executions of the new processor with None and 1 as equivalent settings.
    assert box["coordinate"] == [11, 41, 20, 60]
    assert box["score"] == float(before[0, 1])
    assert type(box) is dict
    assert "original_geometry" not in box
    if kind != "rect":
        np.testing.assert_array_equal(box["quad"], quad.reshape(4, 2))
    np.testing.assert_array_equal(boxes, before)


@pytest.mark.parametrize("ratio", [None, 1.0, 3.0])
@pytest.mark.parametrize(
    "coordinates,classes,settings,expected_scores",
    [
        ([[10.4, 40, 15.9, 60]], [0], {"filtering": True}, [0.9]),
        (
            [[10.4, 40, 30.4, 60], [15.6, 40, 35.4, 60]],
            [0, 0],
            {"layout_nms": True},
            [0.9, 0.8],
        ),
        (
            [[10.4, 40, 30.4, 60], [0.2, 40, 28.2, 60]],
            [0, 0],
            {"layout_merge_bboxes_mode": "large"},
            [0.8],
        ),
        (
            [[10.4, 40, 30.4, 60], [16.2, 40, 36.6, 60]],
            [0, 0],
            {"filtering": True},
            [0.9, 0.8],
        ),
        (
            [[10.4, 40, 30.4, 60], [20.2, 40, 40.6, 60]],
            [0, 1],
            {"filtering": True},
            [0.9, 0.8],
        ),
    ],
    ids=["minimum-size", "nms-06", "containment-09", "overlap-07", "inline-05"],
)
def test_filter_thresholds_keep_historical_rounded_aabbs(
    ratio, coordinates, classes, settings, expected_scores
):
    output = {
        "boxes": np.column_stack([classes, [0.9, 0.8][: len(classes)], coordinates])
    }
    boxes = _process(output, ratio=ratio, **settings)
    assert [box["score"] for box in boxes] == expected_scores
    if settings.get("filtering"):
        unfiltered = _process(output, ratio=ratio)
        pipeline_boxes = filter_overlap_boxes({"boxes": unfiltered}, "rect")["boxes"]
        assert [box["score"] for box in pipeline_boxes] == expected_scores


@pytest.mark.parametrize("kind", ["quad", "polygon_points"])
@pytest.mark.parametrize(
    "ratio", [None, 1.0, [1.0, 1.0], {0: 1.0, 1: (1.0, 1.0)}, {2: 2.0}]
)
def test_identity_skips_scaling_and_geometry_copies(kind, ratio, monkeypatch):
    def unexpected_transform(*args, **kwargs):
        pytest.fail("Identity settings must not allocate scaled geometry")

    monkeypatch.setattr(processors, "_scale_points", unexpected_transform)
    monkeypatch.setattr(processors, "_scale_boxes", unexpected_transform)
    points = np.array([[[10.6, 40.6], [20.4, 40.6], [20.4, 60.4], [10.6, 60.4]]])
    output = {"boxes": np.array([[0, 0.9, 10.6, 40.6, 20.4, 60.4]])}
    output[kind] = points.reshape(1, 8) if kind == "quad" else list(points)
    box = _process(output, ratio=ratio, shape="poly")[0]
    assert type(box) is dict
    assert "original_geometry" not in box
    assert box["coordinate"] == [11, 41, 20, 60]


def test_filtered_scaling_classes_also_take_identity_path(monkeypatch):
    monkeypatch.setattr(
        processors, "_scale_points", lambda *args: pytest.fail("unexpected scaling")
    )
    output = {
        "boxes": np.array([[0, 0.9, 10.6, 40.6, 20.4, 60.4], [1, 0.1, 40, 40, 60, 60]])
    }
    box = _process(output, ratio={1: 2.0})[0]
    assert box["coordinate"] == [11, 41, 20, 60]
    assert "original_geometry" not in box


def test_mixed_classes_only_scaled_detections_change_schema_and_bounds():
    quads = np.array(
        [
            [[40.6, 40.6], [60.4, 40.6], [60.4, 60.4], [40.6, 60.4]],
            [[100.6, 100.6], [120.4, 100.6], [120.4, 120.4], [100.6, 120.4]],
        ]
    )
    output = {
        "boxes": np.array(
            [[0, 0.9, 40.6, 40.6, 60.4, 60.4], [1, 0.8, 100.6, 100.6, 120.4, 120.4]]
        ),
        "quad": quads.reshape(2, 8),
    }
    first, second = _process(output, ratio={1: 2.0}, shape="quad")
    assert first["coordinate"] == [41, 41, 60, 60]
    assert "original_geometry" not in first
    np.testing.assert_array_equal(first["quad"], quads[0])
    assert second["coordinate"] == [90, 90, 131, 131]
    assert second["original_geometry"]["coordinate"] == [101, 101, 120, 120]
    np.testing.assert_allclose(
        second["quad"], [[90.7, 90.7], [130.3, 90.7], [130.3, 130.3], [90.7, 130.3]]
    )


@pytest.mark.parametrize("shape", ["quad", "poly"])
@pytest.mark.parametrize("transport", ["dict", "json", "result_json"])
@pytest.mark.parametrize("filter_kind", ["model", "vl", "structure"])
def test_original_geometry_survives_external_result_boundaries(
    shape, transport, filter_kind
):
    points = np.array(
        [
            [[100, 120], [140, 120], [140, 160], [100, 160]],
            [[144, 120], [184, 120], [184, 160], [144, 160]],
        ]
    )
    key = "quad" if shape == "quad" else "polygon_points"
    output = {
        "boxes": np.array([[0, 0.9, 100, 120, 140, 160], [0, 0.8, 144, 120, 184, 160]]),
        key: points.reshape(2, 8) if shape == "quad" else list(points),
    }
    expanded = _process(output, ratio=4.0, shape=shape)
    if transport == "dict":
        boxes = [dict(box) for box in expanded]
    elif transport == "json":
        boxes = json.loads(json.dumps(expanded, default=lambda array: array.tolist()))
    else:
        result = LayoutAnalysisResult(
            {"input_img": np.zeros((400, 400, 3), dtype=np.uint8), "boxes": expanded}
        )
        boxes = json.loads(json.dumps(result.json))["res"]["boxes"]
    assert [box["original_geometry"]["coordinate"] for box in boxes] == [
        [100, 120, 140, 160],
        [144, 120, 184, 160],
    ]
    if filter_kind == "model":
        after = processors.filter_boxes(boxes, shape)
    elif filter_kind == "vl":
        after = filter_overlap_boxes({"boxes": boxes}, shape)["boxes"]
    else:
        after = remove_overlap_blocks({"boxes": boxes}, threshold=0.5)["boxes"]
    assert len(after) == 2
    assert [box["coordinate"] for box in after] == [
        box["coordinate"] for box in expanded
    ]


@pytest.mark.parametrize("shape", ["poly", "auto", "quad"])
def test_fractional_mask_polygon_matches_historical_pixel_origin(shape):
    output = {
        "boxes": np.array([[0, 0.9, 10.6, 40.6, 30.4, 60.4]]),
        "quad": np.array([[11.0, 41.0, 30.0, 42.0, 29.0, 60.0, 11.0, 60.0]]),
        "masks": np.ones((1, 100, 100), dtype=np.uint8),
    }
    box = _process(output, shape=shape)[0]
    expected = (
        [[11, 41], [11, 59], [29, 59], [29, 41]]
        if shape == "poly"
        else output["quad"].reshape(4, 2)
    )
    np.testing.assert_array_equal(box["polygon_points"], expected)
    # The extractor is also safe when called directly with fractional boxes.
    direct = processors.extract_polygon_points_by_masks(
        output["boxes"], output["masks"], [1, 1], shape, quad=output["quad"]
    )
    np.testing.assert_array_equal(direct[0], expected)


@pytest.mark.parametrize(
    "quad,expected",
    [
        (
            [[40.6, 41.1], [79.6, 40.6], [80.4, 79.7], [41.3, 80.4]],
            [[41, 41], [80, 41], [80, 80], [41, 80]],
        ),
        (
            [[40.6, 46.1], [73.5, 40.6], [80.4, 70.9], [41.5, 80.4]],
            [[40.6, 46.1], [73.5, 40.6], [80.4, 70.9], [41.5, 80.4]],
        ),
    ],
    ids=["rect-iou-095", "polygon-quad-iou-08"],
)
def test_auto_shape_thresholds_match_historical_mask_decisions(quad, expected):
    output = {
        "boxes": np.array([[0, 0.9, 40.6, 40.6, 80.4, 80.4]]),
        "quad": np.asarray(quad).reshape(1, 8),
        "masks": np.ones((1, 100, 100), dtype=np.uint8),
    }
    box = _process(output, shape="auto")[0]
    np.testing.assert_array_equal(box["polygon_points"], expected)


@pytest.mark.parametrize("shape", ["poly", "auto", "quad"])
@pytest.mark.parametrize("has_quad", [False, True])
@pytest.mark.parametrize(
    "reason", ["zero_mask", "empty_crop", "no_contour", "decoded_missing"]
)
def test_missing_contour_preserves_a_valid_detection(
    shape, has_quad, reason, monkeypatch
):
    quad = np.array([[[50, 40], [60, 50], [50, 60], [40, 50]]])
    output = {"boxes": np.array([[0, 0.9, 40, 40, 60, 60]])}
    if has_quad:
        output["quad"] = quad.reshape(1, 8)
    if reason == "decoded_missing":
        output["polygon_points"] = [None]
    else:
        size = 1 if reason == "empty_crop" else 100
        output["masks"] = np.full(
            (1, size, size), reason != "zero_mask", dtype=np.uint8
        )
        if reason == "no_contour":
            monkeypatch.setattr(processors, "mask2polygon", lambda *args: None)
    boxes = _process(output, shape=shape)
    assert len(boxes) == 1
    expected = quad[0] if has_quad else [[40, 40], [60, 40], [60, 60], [40, 60]]
    np.testing.assert_array_equal(boxes[0]["polygon_points"], expected)
    crop = CropByBoxes()(np.zeros((400, 400, 3), dtype=np.uint8), boxes, shape)[0][
        "img"
    ]
    np.testing.assert_array_equal(crop[10, 10], [0, 0, 0])


def test_missing_contour_quad_still_expands_for_recognition(monkeypatch):
    monkeypatch.setattr(processors, "mask2polygon", lambda *args: None)
    output = {
        "boxes": np.array([[0, 0.9, 40, 40, 60, 60]]),
        "quad": np.array([[50, 40, 60, 50, 50, 60, 40, 50]]),
        "masks": np.ones((1, 100, 100), dtype=np.uint8),
    }
    box = _process(output, ratio=2.0, shape="auto")[0]
    np.testing.assert_array_equal(box["quad"], [[50, 30], [70, 50], [50, 70], [30, 50]])
    np.testing.assert_array_equal(box["polygon_points"], box["quad"])
    assert box["coordinate"] == [30, 30, 70, 70]


def test_mask_vertex_distance_uses_width_not_xmax_minus_ymin(monkeypatch):
    distances = []

    def contour(mask, max_allowed_dist):
        distances.append(max_allowed_dist)
        return None

    monkeypatch.setattr(processors, "mask2polygon", contour)
    output = {
        "boxes": np.array([[0, 0.9, 100, 10, 120, 30], [0, 0.8, 5, 250, 45, 290]]),
        "masks": np.ones((2, 100, 100), dtype=np.uint8),
    }
    assert len(_process(output, shape="poly")) == 2
    assert distances == [40, 40]


@pytest.mark.parametrize("shape", ["poly", "auto", "quad"])
@pytest.mark.parametrize(
    "ratio,low,high,bounds",
    [
        (3.0, [170, 170], [227, 227], [170, 170, 231, 231]),
        (5.0, [150, 150], [245, 245], [150, 150, 251, 251]),
        ({0: (5.0, 3.0)}, [150, 170], [245, 227], [150, 170, 251, 231]),
    ],
)
@pytest.mark.parametrize("correction", ["plain", "quarter_turn", "homography"])
def test_fractional_mask_scales_in_its_rounded_pixel_frame(
    shape, ratio, low, high, bounds, correction
):
    output = {
        "boxes": np.array([[0, 0.9, 190.4, 190.4, 210.4, 210.4]]),
        "masks": np.ones((1, 100, 100), dtype=np.uint8),
    }
    box = _process(output, ratio=ratio, shape=shape)[0]
    points = np.asarray(box["polygon_points"])
    expected = np.array(
        [[low[0], low[1]], [high[0], low[1]], [high[0], high[1]], [low[0], high[1]]]
    )
    np.testing.assert_array_equal(
        points[np.lexsort(points.T)], expected[np.lexsort(expected.T)]
    )
    assert box["coordinate"] == bounds
    local = points.astype(np.int32) - bounds[:2]
    assert np.all(local >= 0)
    image = np.random.default_rng(9).integers(0, 256, (400, 400, 3), dtype=np.uint8)
    kwargs = {}
    if correction != "plain":
        params = (
            (90, 1, 0, 0, 0) if correction == "quarter_turn" else (7, 1, -3, 0.1, -0.05)
        )
        plan = plan_page_correction(image.shape, params)
        kwargs = {"page_homography": plan.homography, "rot90_k": plan.rot90_k}
    actual = CropByBoxes()(image, [box], shape, **kwargs)[0]["img"]
    reference = CropByBoxes()(
        image, [dict(box, polygon_points=expected)], shape, **kwargs
    )[0]["img"]
    np.testing.assert_array_equal(actual, reference)


def test_rounded_mask_width_is_enclosed_after_scaling():
    box = _process(
        {
            "boxes": np.array([[0, 0.9, 190.4, 190.4, 209.6, 209.6]]),
            "masks": np.ones((1, 100, 100), dtype=np.uint8),
        },
        ratio=5.0,
        shape="poly",
    )[0]
    assert box["coordinate"] == [150, 150, 248, 248]
    points = np.asarray(box["polygon_points"])
    np.testing.assert_array_equal(points.min(0), [150, 150])
    np.testing.assert_array_equal(points.max(0), [245, 245])


@pytest.mark.parametrize("shape", ["auto", "quad"])
@pytest.mark.parametrize("empty_mask", [False, True])
@pytest.mark.parametrize("ratio", [3.0, 5.0])
def test_mask_selected_model_quad_keeps_float_center(shape, empty_mask, ratio):
    points = np.array([[190.4, 190.4], [210.4, 190.4], [210.4, 210.4], [190.4, 210.4]])
    box = _process(
        {
            "boxes": np.array([[0, 0.9, 190.4, 190.4, 210.4, 210.4]]),
            "quad": points.reshape(1, 8),
            "masks": np.full((1, 100, 100), not empty_mask, dtype=np.uint8),
        },
        ratio=ratio,
        shape=shape,
    )[0]
    expected = (points - 200.4) * ratio + 200.4
    np.testing.assert_allclose(box["quad"], expected, atol=1e-10)
    np.testing.assert_array_equal(box["polygon_points"], box["quad"])


def test_auto_rectangle_with_model_quad_uses_rounded_frame():
    points = np.array([[150.4, 150.4], [250.4, 150.4], [250.4, 250.4], [150.4, 250.4]])
    box = _process(
        {
            "boxes": np.array([[0, 0.9, 150.4, 150.4, 250.4, 250.4]]),
            "quad": points.reshape(1, 8),
            "masks": np.ones((1, 100, 100), dtype=np.uint8),
        },
        ratio=3.0,
        shape="auto",
    )[0]
    np.testing.assert_array_equal(
        box["polygon_points"], [[50, 50], [350, 50], [350, 350], [50, 350]]
    )
    assert box["coordinate"] == [50, 50, 351, 351]
