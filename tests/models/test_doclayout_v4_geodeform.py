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


"""Regression coverage for V4 checkpoint conversion and batched warp outputs."""
import numpy as np
import pytest

from paddlex.modules.base.weight_converter import (
    WeightConverter,
    _preprocess_tensors,
    _derive_doclayoutv4_deform_config,
)
from paddlex.modules.base.utils.pdparams2safetensors import (
    PP_DOCLAYOUTV4_MAPPING,
    apply_key_mapping,
)
from paddlex.inference.models.layout_analysis.predictor import (
    LayoutAnalysisRunnerPredictor,
)


def test_safetensors_roundtrip_noncontiguous_weights(tmp_path):
    from safetensors.numpy import load_file

    original = np.arange(24, dtype=np.float32).reshape(4, 6)
    weights = {
        "linear": original.T,
        "qkv_slice": original.T[:2],
        "scalar": np.array(3, dtype=np.int64),
    }
    converter = object.__new__(WeightConverter)
    converter.output_dir = str(tmp_path)
    converter._save_safetensors(weights)
    restored = load_file(str(tmp_path / "model.safetensors"))
    for key, value in weights.items():
        np.testing.assert_array_equal(restored[key], value)
        assert restored[key].shape == value.shape


@pytest.mark.parametrize(
    "old,new",
    [
        ("dec_rel_head.0", "decoder_order_head.0"),
        ("rel_scorer.dense", "decoder_global_pointer.dense"),
        ("dec_succ_head.0", "decoder_roor_order_head.0"),
        ("succ_scorer.dense", "decoder_roor_global_pointer.dense"),
        ("dec_order_head.0", "decoder_order_head.0"),
        ("dec_roor_order_head.0", "decoder_roor_order_head.0"),
    ],
)
def test_order_weights_are_renamed_and_transposed(old, new):
    # Non-symmetric square matrix catches silent missing transposes.
    value = np.arange(16, dtype=np.float32).reshape(4, 4)
    converted = apply_key_mapping(
        _preprocess_tensors({f"transformer.{old}.weight": value}),
        PP_DOCLAYOUTV4_MAPPING,
    )
    np.testing.assert_array_equal(converted[f"model.{new}.weight"], value.T)


def test_geometry_config_legacy_and_incomplete():
    assert _derive_doclayoutv4_deform_config({}, {})["deform_pred"] is False
    with pytest.raises(ValueError, match="Incomplete"):
        _derive_doclayoutv4_deform_config(
            {}, {"model.deform_head.body.0.weight": np.zeros((128, 258, 3, 3))}
        )


def test_batched_warp_output_including_empty_page():
    counts = np.array([1, 0, 1], dtype=np.int32)
    boxes = np.array(
        [[0, 0.9, 2, 3, 9, 3, 9, 8, 2, 8], [1, 0.8, 1, 2, 5, 2, 5, 6, 1, 6]], np.float32
    )
    qi = np.zeros((3, 1), np.int64)
    logits = np.zeros((3, 1, 1), np.float32)
    warp = np.array(
        [[90, 0.99, 0, 0, 0], [0, 0.8, 2, 0.1, 0], [-90, 0.95, 0, 0, 0.2]], np.float32
    )
    outputs = [boxes, counts, qi, logits, logits, warp]
    results = LayoutAnalysisRunnerPredictor._format_output(None, outputs)
    assert [len(row["boxes"]) for row in results] == [1, 0, 1]
    for row, expected in zip(results, warp):
        np.testing.assert_array_equal(row["warp_params"], expected)
    np.testing.assert_array_equal(results[2]["boxes"][0, 2:], [1, 2, 5, 6])
    legacy = LayoutAnalysisRunnerPredictor._format_output(None, outputs[:5])
    assert all("warp_params" not in row for row in legacy)
    with pytest.raises(ValueError, match="warp_params"):
        LayoutAnalysisRunnerPredictor._format_output(None, outputs[:5] + [warp[:, :4]])


def test_default_training_config_has_geometry_and_roundtrips(tmp_path):
    from paddlex.repo_apis.PaddleDetection_api.object_det.config import DetConfig

    config = DetConfig("PP-DocLayoutV4")
    assert config.DINOHeadV2["deform_pred"] and config.DINOHeadV2["deform_out"]
    assert config.deform_only_train
    assert {"NormalizeQuad": {"deform_labels": True}} in config.TrainReader[
        "batch_transforms"
    ]
    target = tmp_path / "config.yaml"
    config.dump(str(target))
    assert (
        DetConfig("PP-DocLayoutV4", config_path=str(target)).DINOHeadV2
        == config.DINOHeadV2
    )


@pytest.mark.parametrize(
    "weights,resume", [(None, None), ("122.pdparams", None), (None, "checkpoint")]
)
def test_geometry_training_requires_trained_detector(monkeypatch, weights, resume):
    from types import SimpleNamespace
    from paddlex.repo_apis.PaddleDetection_api.object_det.config import DetConfig
    from paddlex.modules.layout_analysis.trainer import LayoutAnalysisTrainer
    from paddlex.modules.object_detection.trainer import DetTrainer

    monkeypatch.setattr(DetTrainer, "update_config", lambda self: None)
    trainer = object.__new__(LayoutAnalysisTrainer)
    trainer.global_config = SimpleNamespace(model="PP-DocLayoutV4")
    trainer.pdx_config = DetConfig("PP-DocLayoutV4")
    trainer.pdx_config["pretrain_weights"] = weights
    trainer.train_config = {"resume_path": resume}
    if weights or resume:
        trainer.update_config()
    else:
        with pytest.raises(ValueError, match="Train.pretrain_weight_path"):
            trainer.update_config()


@pytest.mark.parametrize(
    "warp,enabled,corrected",
    [
        (None, True, False),
        ([90, 0.99, 0, 0, 0], False, False),
        ([90, 0.5, 0, 0, 0], True, False),
        ([90, 0.99, 0, 0, 0], True, True),
        ([4, 0.99, -3, 0.1, 0.05], True, True),
    ],
)
def test_serial_and_threaded_preparation_forward_corrected_crops(
    warp, enabled, corrected
):
    from types import SimpleNamespace
    from paddlex.inference.pipelines.paddleocr_vl.pipeline import _PaddleOCRVLPipeline
    from paddlex.inference.pipelines.components.common.crop_image_regions import (
        CropByBoxes,
    )

    pipeline = object.__new__(_PaddleOCRVLPipeline)
    pipeline.crop_by_boxes = CropByBoxes()
    pipeline.vl_rec_model = SimpleNamespace(model_name="PaddleOCR-VL-1.8-0.9B")
    image = np.arange(80 * 100 * 3, dtype=np.uint8).reshape(80, 100, 3)
    box = {"cls_id": 22, "label": "text", "coordinate": [10, 10, 90, 60], "score": 0.99}
    result = {"boxes": [box]}
    if warp is not None:
        result["warp_params"] = warp
    config = dict(
        use_layout_deform=enabled,
        layout_deform_conf_thresh=0.9,
        layout_deform_min_roll=2.0,
        layout_deform_min_shear=2.0,
        layout_deform_min_keystone=2.0,
        layout_shape_mode="rect",
        merge_layout_blocks=False,
        image_labels=["image"],
        ocr_min_pixels=112896,
        ocr_max_pixels=1000000,
        use_chart_recognition=False,
        use_seal_recognition=False,
    )
    payload = (0, image, result, [], config)
    serial = pipeline._paddleocr_vl_prepare_page_serial_benchmarked(payload)
    threaded = pipeline._paddleocr_vl_prepare_page_core(payload)
    a, b = serial[1][0], threaded[1][0]
    np.testing.assert_array_equal(a["img"], b["img"])
    np.testing.assert_array_equal(serial[2][0][2], a["img"])
    np.testing.assert_array_equal(threaded[2][0][2], b["img"])
    assert ("crop_matrix" in a) == corrected
    if corrected:
        np.testing.assert_allclose(a["crop_matrix"], b["crop_matrix"])
    else:
        np.testing.assert_array_equal(a["img"], image[10:60, 10:90])


def test_pipeline_reads_deformation_confidence_config(monkeypatch):
    from types import SimpleNamespace
    from paddlex.inference.pipelines.paddleocr_vl.pipeline import _PaddleOCRVLPipeline

    monkeypatch.setattr(
        _PaddleOCRVLPipeline,
        "create_model",
        lambda *args, **kwargs: SimpleNamespace(close=lambda: None),
    )
    pipeline = _PaddleOCRVLPipeline(
        dict(
            use_doc_preprocessor=False,
            use_layout_detection=False,
            use_layout_deform=True,
            layout_deform_conf_thresh=0.75,
        ),
        device="cpu",
    )
    assert pipeline.use_layout_deform is True
    assert pipeline.layout_deform_conf_thresh == 0.75
    pipeline.close()
