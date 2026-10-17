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


"""Real Hub checkpoint compatibility. Run with RUN_SLOW=1 and HF access."""

import os
import shutil

import numpy as np
import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_SLOW") != "1",
    reason="Downloads real Hub checkpoints; set RUN_SLOW=1",
)
def test_doclayout_v4_hub_weight_roundtrip(tmp_path):
    import requests
    from huggingface_hub import HfApi, snapshot_download
    from safetensors.numpy import load_file, save_file
    from paddlex import create_model
    from paddlex.inference.models.object_detection.modeling.pp_doclayout_v4 import (
        PPDocLayoutV4,
    )
    from transformers import PPDocLayoutV4ForObjectDetection

    # Resolve main once per repository, so each test run uses its latest commit.
    api = HfApi()
    directories = {}
    for name in ("PP-DocLayoutV4", "PP-DocLayoutV4_safetensors"):
        repo = f"PaddlePaddle/{name}"
        revision = api.model_info(repo).sha
        directories[name] = snapshot_download(repo, revision=revision)

    source = directories["PP-DocLayoutV4_safetensors"]
    image = tmp_path / "document.jpg"
    response = requests.get(
        "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg",
        timeout=60,
    )
    response.raise_for_status()
    image.write_bytes(response.content)

    model = PPDocLayoutV4.from_pretrained(source, convert_from_hf=True)
    roundtrip = tmp_path / "roundtrip"
    roundtrip.mkdir()
    for name in ("config.json", "preprocessor_config.json", "inference.yml"):
        shutil.copyfile(os.path.join(source, name), roundtrip / name)
    transpose = set(model.get_transpose_weight_keys())
    exported = {}
    for key, tensor in model.get_hf_state_dict().items():
        array = tensor.numpy()
        exported[key] = np.ascontiguousarray(array.T if key in transpose else array)
    save_file(exported, str(roundtrip / "model.safetensors"))

    # A roundtrip must preserve every actual parameter/buffer, not just produce
    # plausible boxes. In particular, preserve the exact perspective centers.
    original = load_file(os.path.join(source, "model.safetensors"))
    for key, array in exported.items():
        np.testing.assert_array_equal(array, original[key], err_msg=key)
    assert set(original) - set(exported) == {
        k for k in original if k.endswith(".num_batches_tracked")
    }
    hf_model, info = PPDocLayoutV4ForObjectDetection.from_pretrained(
        str(roundtrip), output_loading_info=True
    )
    assert (
        not info["missing_keys"]
        and not info["unexpected_keys"]
        and not info["mismatched_keys"]
    )
    del model, hf_model

    # Legacy PaddleX packages used fused dense projections. Test real loading,
    # including the non-square transpose, rather than mocking set_state_dict.
    legacy = tmp_path / "legacy"
    shutil.copytree(roundtrip, legacy)
    fused = dict(exported)
    for head in ("decoder_global_pointer", "decoder_roor_global_pointer"):
        for suffix in ("weight", "bias"):
            prefix = f"model.{head}."
            fused[prefix + "dense." + suffix] = np.concatenate(
                [
                    fused.pop(prefix + "q_proj." + suffix),
                    fused.pop(prefix + "k_proj." + suffix),
                ],
                axis=0,
            )
    fused["model.deform_bin_centers"] = fused.pop("model.deform_head.bin_centers")
    save_file(fused, str(legacy / "model.safetensors"))

    results = []
    for engine, directory in (
        ("paddle_static", directories["PP-DocLayoutV4"]),
        ("paddle_dynamic", source),
        ("paddle_dynamic", str(legacy)),
        ("transformers", source),
        ("transformers", str(roundtrip)),
    ):
        predictor = create_model(
            "PP-DocLayoutV4", model_dir=directory, engine=engine, device="cpu"
        )
        try:
            pages = list(
                predictor.predict(
                    [str(image), str(image)],
                    batch_size=2,
                    threshold=0.5,
                    layout_nms=False,
                    layout_unclip_ratio=1.0,
                    filter_overlap_boxes=False,
                    skip_order_labels=[],
                )
            )
            assert len(pages) == 2
            results.extend(pages)
            empty = list(predictor.predict(str(image), threshold=1.0))[0]
            assert empty["boxes"] == []
            assert len(empty["warp_params"]) == 5
        finally:
            predictor.close()

    reference = results[0]
    assert len(reference["boxes"]) == 13
    for result in results[1:]:
        assert len(result["boxes"]) == len(reference["boxes"])
        assert [b["label"] for b in result["boxes"]] == [
            b["label"] for b in reference["boxes"]
        ]
        assert [b.get("order") for b in result["boxes"]] == [
            b.get("order") for b in reference["boxes"]
        ]
        for actual, expected in zip(result["boxes"], reference["boxes"]):
            np.testing.assert_allclose(
                actual["coordinate"], expected["coordinate"], atol=1, rtol=0
            )
            np.testing.assert_allclose(
                actual["score"], expected["score"], atol=0.01, rtol=0
            )
            np.testing.assert_allclose(actual["quad"], expected["quad"], atol=1, rtol=0)
        # Different default image interpolation backends are not bit-identical.
        np.testing.assert_allclose(
            result["warp_params"], reference["warp_params"], atol=0.002, rtol=0
        )
