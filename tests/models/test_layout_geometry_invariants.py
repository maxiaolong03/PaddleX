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

"""Keep filtering independent of crop expansion and all geometry in sync."""

import copy
import json

import numpy as np
import pytest

from paddlex.inference.models.layout_analysis.predictor import (
    LayoutAnalysisRunnerPredictor,
    LayoutAnalysisTransformersPredictor,
)
from paddlex.inference.models.layout_analysis.processors import (
    LayoutAnalysisProcess,
    filter_boxes,
)
from paddlex.inference.models.layout_analysis.result import LayoutAnalysisResult
from paddlex.inference.pipelines.components.common.crop_image_regions import CropByBoxes
from paddlex.inference.pipelines.components.common.page_rectify import (
    plan_page_correction,
)
from paddlex.inference.pipelines.paddleocr_vl.uilts import filter_overlap_boxes


def _process(
    output, ratio=None, shape="poly", threshold=0.5, filtering=False, **kwargs
):
    options = dict(
        threshold=threshold,
        layout_nms=False,
        layout_unclip_ratio=ratio,
        layout_merge_bboxes_mode=None,
        layout_shape_mode=shape,
        filter_overlap_boxes=filtering,
        skip_order_labels=[],
    )
    options.update(kwargs)
    return LayoutAnalysisProcess(
        labels=["text", "inline_formula", "image"], scale_size=[400, 400]
    )([output], [{"ori_img_size": (400, 400)}], **options,)[0]


def _square_output(kind):
    quad = np.array([[40, 40, 60, 40, 60, 60, 40, 60]], dtype=np.float64)
    output = {"boxes": np.array([[0, 0.9, 40, 40, 60, 60]], dtype=np.float64)}
    if kind == "mask_quad":
        output = LayoutAnalysisRunnerPredictor._format_output(
            None,
            [
                np.column_stack([output["boxes"][:, :2], quad]),
                np.array([1]),
                np.ones((1, 100, 100), dtype=np.uint8),
                np.array([[0]]),
                np.zeros((1, 1, 1)),
                np.zeros((1, 1, 1)),
            ],
        )[0]
    elif kind == "polygon":
        output["polygon_points"] = [quad.reshape(4, 2)]
    else:
        output["quad"] = quad
    return output


@pytest.mark.parametrize("kind", ["mask_quad", "polygon", "quad"])
@pytest.mark.parametrize("shape", ["quad", "auto", "poly"])
@pytest.mark.parametrize("correction", ["plain", "quarter_turn", "homography"])
def test_every_crop_boundary_expands(kind, shape, correction):
    image = np.full((400, 400, 3), 255, dtype=np.uint8)
    image[47:54, 32:39] = [10, 30, 200]
    result = _process(_square_output(kind), ratio=2.0, shape=shape)
    kwargs = {}
    if correction != "plain":
        params = (
            (90, 1, 0, 0, 0) if correction == "quarter_turn" else (7, 1, -3, 0.1, -0.05)
        )
        plan = plan_page_correction(image.shape, params)
        kwargs = {"page_homography": plan.homography, "rot90_k": plan.rot90_k}
    crop = CropByBoxes()(image, result, shape, **kwargs)[0]["img"]
    assert np.any(np.all(np.abs(crop.astype(int) - [10, 30, 200]) < 3, axis=-1))
    if kind == "mask_quad" and shape == "quad":
        np.testing.assert_array_equal(result[0]["polygon_points"], result[0]["quad"])


@pytest.mark.parametrize("kind", ["polygon", "quad"])
@pytest.mark.parametrize("labels,ratio", [([0, 0], 4.0), ([0, 1], 3.0)])
@pytest.mark.parametrize("shape", ["poly", "rect"])
def test_expansion_does_not_remove_adjacent_detections(kind, labels, ratio, shape):
    quads = np.array(
        [
            [[100, 120], [140, 120], [140, 160], [100, 160]],
            [[144, 120], [184, 120], [184, 160], [144, 160]],
        ],
        dtype=np.float64,
    )

    def output():
        geometry = (
            quads.reshape(2, 8).copy() if kind == "quad" else [p.copy() for p in quads]
        )
        return {
            "boxes": np.array(
                [
                    [labels[0], 0.9, 100, 120, 140, 160],
                    [labels[1], 0.85, 144, 120, 184, 160],
                ]
            ),
            "quad" if kind == "quad" else "polygon_points": geometry,
        }

    plain = _process(output(), shape=shape, filtering=True)
    expanded = _process(output(), ratio=ratio, shape=shape, filtering=True)
    assert len(plain) == len(expanded) == 2
    assert [b["label"] for b in expanded] == [b["label"] for b in plain]
    for candidate in [
        expanded,
        copy.deepcopy(expanded),
        [box.copy() for box in expanded],
    ]:
        assert len(filter_boxes(candidate, shape)) == 2
        after = filter_overlap_boxes({"boxes": candidate}, shape)["boxes"]
        assert len(after) == 2
        assert [b["coordinate"] for b in after] == [b["coordinate"] for b in expanded]


@pytest.mark.parametrize("field", ["polygon_points", "quad", "masks", "qi"])
@pytest.mark.parametrize("threshold", [0.5, {0: 0.5}])
@pytest.mark.parametrize("extra", [-1, 1])
@pytest.mark.parametrize("shape", ["poly", "rect"])
def test_geometry_count_mismatch_is_rejected_before_filtering(
    field, threshold, extra, shape
):
    count = 3 + extra
    output = {
        "boxes": np.array(
            [
                [0, 0.9, 40, 40, 60, 60],
                [0, 0.9, 80, 40, 100, 60],
                [0, 0.1, 120, 40, 140, 60],
            ]
        )
    }
    polygon = np.array([[40, 40], [60, 40], [60, 60], [40, 60]])
    output[field] = {
        "polygon_points": [polygon.copy() for _ in range(count)],
        "quad": np.tile(polygon.reshape(1, 8), (count, 1)),
        "masks": np.ones((count, 100, 100), dtype=np.uint8),
        "qi": np.arange(count),
    }[field]
    with pytest.raises(ValueError, match=field):
        _process(output, threshold=threshold, shape=shape)


class _Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


@pytest.mark.parametrize("model", ["PP-DocLayoutV3", "PP-DocLayoutV4"])
@pytest.mark.parametrize("count", [0, 2])
def test_transformers_adapter_validates_polygon_count(model, count):
    predictor = object.__new__(LayoutAnalysisTransformersPredictor)
    predictor.model_name = model
    polygons = np.zeros((count, 4, 2))
    with pytest.raises(ValueError, match="polygon_points"):
        predictor._format_layout_transformers_output(
            {
                "boxes": _Tensor([[40, 40, 60, 60]]),
                "scores": _Tensor([0.9]),
                "labels": _Tensor([0]),
                "polygon_points": (
                    _Tensor(polygons) if model.endswith("V4") else list(polygons)
                ),
            }
        )


@pytest.mark.parametrize("kind", ["polygon", "quad"])
@pytest.mark.parametrize("ratio", [2.0, [4.0, 1.5], {0: (4.0, 1.5)}])
def test_fractional_geometry_uses_one_float_center(kind, ratio):
    points = np.array([[100.4, 100.4], [120.4, 100.4], [120.4, 120.4], [100.4, 120.4]])
    output = {"boxes": np.array([[0, 0.9, 100.4, 100.4, 120.4, 120.4]])}
    key = "quad" if kind == "quad" else "polygon_points"
    output[key] = points.reshape(1, 8) if kind == "quad" else [points]
    before = copy.deepcopy(output)
    pair = (
        [ratio, ratio]
        if isinstance(ratio, float)
        else ratio[0] if isinstance(ratio, dict) else ratio
    )
    expected = [110.4, 110.4] + (points - [110.4, 110.4]) * pair
    result = _process(output, ratio=ratio)[0]
    np.testing.assert_allclose(result[key], expected, atol=1e-10)
    bounds = result["coordinate"]
    assert np.all(np.asarray(bounds[:2]) <= expected.min(0))
    assert np.all(np.asarray(bounds[2:]) >= expected.max(0))
    np.testing.assert_array_equal(output["boxes"], before["boxes"])
    np.testing.assert_array_equal(output[key], before[key])


def test_internal_filter_geometry_does_not_leak_into_json():
    boxes = _process(_square_output("quad"), ratio=2.0)
    result = LayoutAnalysisResult(
        {"input_img": np.zeros((400, 400, 3), dtype=np.uint8), "boxes": boxes}
    )
    encoded = json.loads(json.dumps(result.json))
    assert set(encoded["res"]["boxes"][0]) == {
        "cls_id",
        "label",
        "score",
        "coordinate",
        "order",
        "quad",
    }
    assert encoded["res"]["boxes"][0]["coordinate"] == [30, 30, 70, 70]


@pytest.mark.parametrize("ratio", [{0: 2.0}, {0: [2.0, 1.5]}, {0: (2.0, 1.5)}])
def test_class_specific_ratio_resolution_is_shared_by_all_shapes(ratio):
    result = _process(_square_output("mask_quad"), ratio=ratio, shape="quad")[0]
    np.testing.assert_array_equal(result["polygon_points"], result["quad"])
    bounds = np.asarray(result["quad"])
    np.testing.assert_array_equal(
        result["coordinate"], [*bounds.min(0), *bounds.max(0)]
    )


@pytest.mark.parametrize("kind", ["polygon", "mask_quad"])
@pytest.mark.parametrize("threshold", [0.5, {0: 0.75, 1: 0.9, 2: 0.9}])
@pytest.mark.parametrize(
    "merge,expected", [(None, [1, 0, 5]), ("large", [1, 0]), ({0: "small"}, [1, 5])]
)
def test_retained_indices_keep_ragged_polygons_and_masks_aligned(
    kind, threshold, merge, expected
):
    boxes = np.array(
        [
            [0, 0.8, 100, 40, 140, 80, 2],
            [1, 0.95, 20, 40, 60, 80, 0],
            [0, 0.7, 100, 40, 140, 80, 9],
            [0, 0.2, 20, 240, 60, 280, 1],
            [2, 0.99, 0, 0, 400, 400, 8],
            [0, 0.85, 110, 50, 120, 60, 3],
        ]
    )
    quads = [
        np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        for x1, y1, x2, y2 in boxes[:, 2:6]
    ]
    output = {"boxes": boxes}
    if kind == "polygon":
        # Different vertex counts catch accidental rectangular/object-array reshaping.
        output["polygon_points"] = [
            np.concatenate(
                [
                    np.column_stack(
                        [
                            np.linspace(p[0, 0], p[1, 0], i + 2)[:-1],
                            np.full(i + 1, p[0, 1]),
                        ]
                    ),
                    p[1:],
                ]
            )
            for i, p in enumerate(quads)
        ]
    else:
        output["quad"] = np.asarray(quads).reshape(-1, 8)
        masks = np.zeros((len(boxes), 100, 100), dtype=np.uint8)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = (box[2:6] / 4).astype(int)
            masks[i, y1:y2, x1 : (x1 + x2) // 2] = 1
        output["masks"] = masks
    ratio = {0: (2.0, 1.5), 1: 0.75}
    result = _process(
        output,
        ratio=ratio,
        threshold=threshold,
        layout_nms=True,
        layout_merge_bboxes_mode=merge,
    )
    assert [box["score"] for box in result] == boxes[expected, 1].tolist()
    for actual, source_index in zip(result, expected):
        single = {
            name: (
                [values[source_index]]
                if name == "polygon_points"
                else values[source_index : source_index + 1]
            )
            for name, values in output.items()
        }
        independent = _process(single, ratio=ratio)[0]
        assert actual["coordinate"] == independent["coordinate"]
        for key in ["polygon_points", "quad"]:
            if key in independent:
                np.testing.assert_array_equal(actual[key], independent[key])


@pytest.mark.parametrize("kind", ["polygon", "quad", "rect"])
@pytest.mark.parametrize("ratio", [0.2, 3.0, {0: 0.2, 1: 3.0}])
def test_original_overlap_and_minimum_size_rules_still_apply(kind, ratio):
    points = np.array(
        [
            [[40, 40], [60, 40], [60, 60], [40, 60]],
            [[40, 40], [60, 40], [60, 60], [40, 60]],
            [[100, 40], [104, 40], [104, 60], [100, 60]],
        ]
    )
    output = {
        "boxes": np.array(
            [
                [0, 0.9, 40, 40, 60, 60],
                [1, 0.85, 40, 40, 60, 60],
                [0, 0.8, 100, 40, 104, 60],
            ]
        )
    }
    if kind == "quad":
        output["quad"] = points.reshape(-1, 8)
    elif kind == "polygon":
        output["polygon_points"] = list(points)
    boxes = _process(output, ratio=ratio)
    for filtered in [
        filter_boxes(boxes, kind),
        filter_overlap_boxes({"boxes": boxes}, kind)["boxes"],
    ]:
        assert len(filtered) == 1
        assert filtered[0]["label"] == "text"
        assert filtered[0]["score"] == 0.9
        assert filtered[0]["coordinate"] == boxes[0]["coordinate"]
    # Plain dictionaries from external callers still use their own geometry.
    unscaled = [dict(box) for box in _process(output)]
    assert len(filter_boxes(unscaled, kind)) == 1
    assert len(filter_overlap_boxes({"boxes": unscaled}, kind)["boxes"]) == 1


@pytest.mark.parametrize("kind", ["mask_quad", "polygon"])
@pytest.mark.parametrize("shape", ["quad", "auto", "poly"])
@pytest.mark.parametrize("correction", ["plain", "quarter_turn", "homography"])
def test_identity_preserves_mask_and_polygon_crops_exactly(kind, shape, correction):
    image = np.random.default_rng(42).integers(0, 256, (400, 400, 3), dtype=np.uint8)
    original = _process(_square_output(kind), shape=shape)
    identity = _process(_square_output(kind), shape=shape, ratio=1.0)
    kwargs = {}
    if correction != "plain":
        params = (
            (90, 1, 0, 0, 0) if correction == "quarter_turn" else (7, 1, -3, 0.1, -0.05)
        )
        plan = plan_page_correction(image.shape, params)
        kwargs = {"page_homography": plan.homography, "rot90_k": plan.rot90_k}
    before = CropByBoxes()(image, original, shape, **kwargs)[0]["img"]
    after = CropByBoxes()(image, identity, shape, **kwargs)[0]["img"]
    np.testing.assert_array_equal(after, before)
    np.testing.assert_array_equal(
        identity[0]["polygon_points"], original[0]["polygon_points"]
    )
