# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


"""Exercise V4 geometry through the shared transformers predictor result path."""

from types import SimpleNamespace

import numpy as np
import paddle
import pytest

from paddlex.inference.models.layout_analysis.predictor import (
    LayoutAnalysisTransformersPredictor,
)
from paddlex.inference.models.layout_analysis.processors import LayoutAnalysisProcess


def prediction(empty=False, geometry=True):
    n = 0 if empty else 2
    points = np.array(
        [
            [[2, 3], [24, 1], [22, 20], [1, 18]],
            [[40, 40], [60, 40], [60, 60], [40, 60]],
        ],
        dtype="float32",
    )[:n]
    result = dict(
        boxes=paddle.to_tensor(
            [[1.0, 1.0, 24.0, 20.0], [40.0, 40.0, 60.0, 60.0]][:n]
        ).reshape([n, 4]),
        scores=paddle.to_tensor([0.9, 0.2][:n]),
        labels=paddle.to_tensor([0, 0][:n]),
        polygon_points=paddle.to_tensor(points),
        order_seq=paddle.to_tensor([0, 1][:n]),
    )
    if geometry:
        result["warp_params"] = paddle.to_tensor([90.0, 0.99, 2.0, 0.1, -0.2])
    return result


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("geometry", [False, True])
def test_v4_transformers_geometry_reaches_public_result(empty, geometry):
    p = object.__new__(LayoutAnalysisTransformersPredictor)
    p.model_name = "PP-DocLayoutV4"
    p.image_processor = SimpleNamespace(post_process_object_detection=True)
    p.threshold = 0.3
    p.layout_nms = False
    p.layout_unclip_ratio = None
    p.layout_merge_bboxes_mode = None
    data = {
        "img": np.zeros((80, 80, 3), dtype=np.uint8),
        "ori_img": np.zeros((80, 80, 3), dtype=np.uint8),
        "ori_img_size": (80, 80),
    }
    p.read_op = lambda _: [data, data]
    p.preprocess_images = lambda **_: None
    p.forward = lambda _: None
    preds = [prediction(empty, geometry), prediction(True, geometry)]
    p.postprocess = lambda *args, **kwargs: preds
    p.layout_postprocess = LayoutAnalysisProcess(labels=["text"], scale_size=[800, 800])
    batch = SimpleNamespace(
        instances=[None, None],
        input_paths=["one.png", "two.png"],
        page_indexes=[None, None],
    )
    result = p.process(
        batch, threshold=0.3, layout_shape_mode="quad", filter_overlap_boxes=False
    )
    assert len(result["boxes"]) == 2
    assert len(result["boxes"][0]) == (0 if empty else 1)
    assert len(result["boxes"][1]) == 0
    if not empty:
        np.testing.assert_allclose(
            np.asarray(result["boxes"][0][0]["quad"]).reshape(4, 2),
            preds[0]["polygon_points"][0].numpy(),
        )
    assert ("warp_params" in result) == geometry
    if geometry:
        for row in result["warp_params"]:
            np.testing.assert_allclose(row, [90, 0.99, 2, 0.1, -0.2])


def test_v4_transformers_rejects_invalid_page_geometry():
    p = object.__new__(LayoutAnalysisTransformersPredictor)
    p.model_name = "PP-DocLayoutV4"
    item = prediction()
    item["warp_params"] = paddle.zeros([2, 5])
    with pytest.raises(ValueError, match="warp_params shape"):
        p._format_layout_transformers_output(item)
