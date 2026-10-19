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

# from typing import Any, List, Optional, Tuple, Union
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from ..object_detection.predictor import DetRunnerPredictor, DetTransformersPredictor
from ..object_detection.processors import Resize, ToBatch
from .processors import LayoutAnalysisProcess
from .result import LayoutAnalysisResult
from .utils import STATIC_SHAPE_MODEL_LIST

LAYOUT_ANALYSIS_TRANSFORMERS_MODELS = [
    "PP-DocLayoutV2",
    "PP-DocLayoutV3",
    "PP-DocLayoutV4",
]


class LayoutAnalysisRunnerPredictor(DetRunnerPredictor):
    """Layout analysis predictor."""

    def __init__(
        self,
        *args,
        img_size: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs,
    ):
        """Initializes LayoutAnalysisPredictor.
        Args:
            *args: Arbitrary positional arguments passed to the superclass.
            img_size (Optional[Union[int, Tuple[int, int]]], optional): The input image size (w, h). Defaults to None.
            threshold (Optional[float], optional): The threshold for filtering out low-confidence predictions.
                Defaults to None.
            layout_nms (bool, optional): Whether to use layout-aware NMS. Defaults to False.
            layout_unclip_ratio (Optional[Union[float, Tuple[float, float]]], optional): The ratio of unclipping the bounding box.
                Defaults to None.
                If it's a single number, then both width and height are used.
                If it's a tuple of two numbers, then they are used separately for width and height respectively.
                If it's None, then no unclipping will be performed.
            layout_merge_bboxes_mode (Optional[Union[str, dict]], optional): The mode for merging bounding boxes. Defaults to None.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        if img_size is not None:
            assert (
                self.model_name not in STATIC_SHAPE_MODEL_LIST
            ), f"The model {self.model_name} is not supported set input shape"
            if isinstance(img_size, int):
                img_size = (img_size, img_size)
            elif isinstance(img_size, (tuple, list)):
                assert len(img_size) == 2, f"The length of `img_size` should be 2."
            else:
                raise ValueError(
                    f"The type of `img_size` must be int or Tuple[int, int], but got {type(img_size)}."
                )
        super().__init__(*args, **kwargs)

    def _get_result_class(self):
        return LayoutAnalysisResult

    def _format_output(self, pred: Sequence[Any]) -> List[dict]:
        """
        Transform batch outputs into a list of single image output.

        Args:
            pred (Sequence[Any]): The input predictions, which can be either a list of 3 or 4 elements.
                - When len(pred) == 6 and pred[2] is 3-D, it is expected to be in the format [bbox_pred, bbox_num, mask_pred, qi, rel_logits, roor_logits],
                  compatible with Modeling_V2 output.
                - When len(pred) == 5, it is expected to be in the format [bbox_pred, bbox_num, qi, rel_logits, roor_logits],
                  compatible with DocLayoutV2 (no mask) output.
                - When len(pred) == 6 and pred[2] is 2-D, it is the same as the
                  5-output case plus a trailing `warp_params` tensor of shape
                  [batch, 5] = (roll_deg, quadrant_conf, shear_deg, kx, ky).
                - When len(pred) == 3, it is expected to be in the format [boxes, box_nums, masks],
                  compatible with Instance Segmentation output.
                - Otherwise it is treated as plain detection output [bbox_pred, bbox_num].
                  In this case bbox_pred may have 6 columns (2-point box) or 10 columns
                  (4-point / quad box); the quad is detected by column count and extracted
                  automatically when present, so both models stay compatible.

        Returns:
            List[dict]: A list of dictionaries, each containing either 'class_id' and 'masks' (for SOLOv2),
                or 'boxes' and 'masks' (for Instance Segmentation), or just 'boxes' if no masks are provided.
        """
        box_idx_start = 0
        pred_box = []

        # DocLayoutV3 V2: 6 outputs (bbox_pred, bbox_num, mask_pred, qi, rel_logits, roor_logits)
        # `pred[2].ndim == 3` separates this from the maskless + angle model
        # below, which also has 6 outputs but carries `qi` ([B, K], 2-D) in that
        # slot instead of a mask batch ([N, H, W], 3-D).
        if len(pred) == 6 and pred[2].ndim == 3:
            bbox_pred, bbox_num, mask_pred, qi, rel_logits, roor_logits = pred

            # Detect 4-point model output (10 columns: label, score, x1,y1,x2,y2,x3,y3,x4,y4)
            quad_all = None
            if bbox_pred.shape[1] == 10:
                quad_all = bbox_pred[:, 2:10].copy()  # [B*K, 8]
                xs = quad_all[:, 0::2]  # x1, x2, x3, x4
                ys = quad_all[:, 1::2]  # y1, y2, y3, y4
                bbox_pred = np.column_stack(
                    [
                        bbox_pred[:, 0],
                        bbox_pred[:, 1],
                        xs.min(axis=1),
                        ys.min(axis=1),
                        xs.max(axis=1),
                        ys.max(axis=1),
                    ]
                )

            results = []
            box_idx_start = 0
            for idx in range(len(bbox_num)):
                K = int(bbox_num[idx])
                box_idx_end = box_idx_start + K
                result_dict = {
                    "boxes": bbox_pred[box_idx_start:box_idx_end],
                    "masks": mask_pred[box_idx_start:box_idx_end],
                    "rel_logits": rel_logits[idx],
                    "roor_logits": roor_logits[idx],
                    "qi": qi[idx, :K],
                }
                if quad_all is not None:
                    result_dict["quad"] = quad_all[box_idx_start:box_idx_end]
                else:
                    result_dict["quad"] = None
                results.append(result_dict)
                box_idx_start = box_idx_end
            return results

        # DocLayoutV2 without mask: 5 outputs (bbox_pred, bbox_num, qi, rel_logits, roor_logits),
        # or 6 when the model also predicts the page deformation: a trailing
        # [B, 5] `warp_params` = (roll_deg, quadrant_conf, shear_deg, kx, ky),
        # passed through per image untouched (see components/common/page_rectify.py).
        if len(pred) == 5 or len(pred) == 6:
            bbox_pred, bbox_num, qi, rel_logits, roor_logits = pred[:5]
            warp_params = pred[5] if len(pred) == 6 else None
            if warp_params is not None and (
                warp_params.ndim != 2 or warp_params.shape != (len(bbox_num), 5)
            ):
                raise ValueError(
                    f"Expected the 6th output to be warp_params of shape "
                    f"[batch, 5], got {warp_params.shape}."
                )

            # Detect 4-point model output (10 columns: label, score, x1,y1,x2,y2,x3,y3,x4,y4)
            quad_all = None
            if bbox_pred.shape[1] == 10:
                quad_all = bbox_pred[:, 2:10].copy()  # [B*K, 8]
                xs = quad_all[:, 0::2]
                ys = quad_all[:, 1::2]
                bbox_pred = np.column_stack(
                    [
                        bbox_pred[:, 0],
                        bbox_pred[:, 1],
                        xs.min(axis=1),
                        ys.min(axis=1),
                        xs.max(axis=1),
                        ys.max(axis=1),
                    ]
                )

            results = []
            box_idx_start = 0
            for idx in range(len(bbox_num)):
                K = int(bbox_num[idx])
                box_idx_end = box_idx_start + K
                result_dict = {
                    "boxes": bbox_pred[box_idx_start:box_idx_end],
                    "rel_logits": rel_logits[idx],
                    "roor_logits": roor_logits[idx],
                    "qi": qi[idx, :K],
                }
                if quad_all is not None:
                    result_dict["quad"] = quad_all[box_idx_start:box_idx_end]
                else:
                    result_dict["quad"] = None
                if warp_params is not None:
                    result_dict["warp_params"] = [float(v) for v in warp_params[idx]]

                results.append(result_dict)
                box_idx_start = box_idx_end
            return results

        if len(pred) == 3:
            # Adapt to Instance Segmentation
            pred_mask = []

        # Detect 4-point (quad) box output by column count instead of by
        # len(pred), so any plain-detection output remains compatible.
        # Quad models put the 8 corner coords at columns 2:10. There may be
        # trailing metadata columns after the quad (e.g. a pre-decoded
        # reading-order column -> 11 cols); those are kept and appended after
        # the 6-column AABB so the post-processor can sort by them
        # (it recognizes 7-/8-column ordered-detection boxes).
        bbox_all = pred[0]
        quad_all = None
        if (
            len(pred) != 3
            and hasattr(bbox_all, "ndim")
            and bbox_all.ndim == 2
            and bbox_all.shape[1] >= 10
        ):
            quad_all = bbox_all[:, 2:10].copy()  # [B*K, 8]
            xs = quad_all[:, 0::2]  # x1, x2, x3, x4
            ys = quad_all[:, 1::2]  # y1, y2, y3, y4
            cols = [
                bbox_all[:, 0],
                bbox_all[:, 1],
                xs.min(axis=1),
                ys.min(axis=1),
                xs.max(axis=1),
                ys.max(axis=1),
            ]
            # Preserve any trailing columns (reading order, etc.) after the quad.
            if bbox_all.shape[1] > 10:
                cols.append(bbox_all[:, 10:])
            bbox_all = np.column_stack(cols)

        pred_quad = []
        for idx in range(len(pred[1])):
            np_boxes_num = pred[1][idx]
            box_idx_end = box_idx_start + np_boxes_num
            np_boxes = bbox_all[box_idx_start:box_idx_end]
            pred_box.append(np_boxes)
            if quad_all is not None:
                pred_quad.append(quad_all[box_idx_start:box_idx_end])
            if len(pred) == 3:
                np_masks = pred[2][box_idx_start:box_idx_end]
                pred_mask.append(np_masks)
            box_idx_start = box_idx_end

        if len(pred) == 3:
            return [
                {"boxes": np.asarray(pred_box[i]), "masks": np.asarray(pred_mask[i])}
                for i in range(len(pred_box))
            ]
        elif quad_all is not None:
            return [
                {"boxes": np.array(pred_box[i]), "quad": pred_quad[i]}
                for i in range(len(pred_box))
            ]
        else:
            return [{"boxes": np.array(res)} for res in pred_box]

    def process(
        self,
        batch_data: List[Any],
        threshold: Optional[Union[float, dict]] = None,
        layout_nms: bool = False,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float], dict]] = None,
        layout_merge_bboxes_mode: Optional[Union[str, dict]] = None,
        layout_shape_mode: Optional[str] = "auto",
        filter_overlap_boxes: Optional[bool] = True,
        skip_order_labels: Optional[List[str]] = None,
    ):
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[Union[str, np.ndarray], ...]): A batch of input data (e.g., image file paths).
            threshold (Optional[float, dict], optional): The threshold for filtering out low-confidence predictions.
            layout_nms (bool, optional): Whether to use layout-aware NMS. Defaults to None.
            layout_unclip_ratio (Optional[Union[float, Tuple[float, float]]], optional): The ratio of unclipping the bounding box.
            layout_merge_bboxes_mode (Optional[Union[str, dict]], optional): The mode for merging bounding boxes. Defaults to None.
            layout_shape_mode (Optional[str], optional): The mode for layout shape. Defaults to "auto", [ "rect", "quad","poly", "auto"]. are supported.
            filter_overlap_boxes (Optional[bool], optional): Whether to filter out overlap boxes. Defaults to True.
            skip_order_labels (Optional[List[str]], optional): The labels to skip order. Defaults to None.

        Returns:
            dict: A dictionary containing the input path, raw image, class IDs, scores, and label names
                for every instance of the batch. Keys include 'input_path', 'input_img', 'class_ids', 'scores', and 'label_names'.
        """
        datas = batch_data.instances
        # preprocess
        for pre_op in self.pre_ops[:-1]:
            datas = pre_op(datas)

        # use `ToBatch` format batch inputs
        batch_inputs = self.pre_ops[-1](datas)

        # do infer
        batch_preds = self.runner(batch_inputs)

        # process a batch of predictions into a list of single image result
        preds_list = self._format_output(batch_preds)
        # postprocess
        boxes = self.post_op(
            preds_list,
            datas,
            threshold=threshold if threshold is not None else self.threshold,
            layout_nms=layout_nms or self.layout_nms,
            layout_unclip_ratio=layout_unclip_ratio or self.layout_unclip_ratio,
            layout_merge_bboxes_mode=layout_merge_bboxes_mode
            or self.layout_merge_bboxes_mode,
            layout_shape_mode=layout_shape_mode,
            filter_overlap_boxes=filter_overlap_boxes,
            skip_order_labels=skip_order_labels,
        )

        result = {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": [data["ori_img"] for data in datas],
            "boxes": boxes,
        }
        # Image-level, so it never goes through the box post-processor.  Added
        # only for models that predict it, to leave every other model's result
        # (and its serialized JSON) unchanged.  `BasePredictor.apply` splits this
        # dict of lists by index, so each single-image result gets its own row.
        if any("warp_params" in pred for pred in preds_list):
            result["warp_params"] = [pred.get("warp_params") for pred in preds_list]
        return result

    @DetRunnerPredictor.register("Resize")
    def build_resize(self, target_size, keep_ratio=False, interp=2):
        assert target_size
        self.target_size = target_size
        if isinstance(interp, int):
            interp = {
                0: "NEAREST",
                1: "LINEAR",
                2: "BICUBIC",
                3: "AREA",
                4: "LANCZOS4",
            }[interp]
        op = Resize(target_size=target_size[::-1], keep_ratio=keep_ratio, interp=interp)
        return op

    def build_to_batch(self):
        models_required_imgsize = [
            "PP-DocLayoutV2",
            "PP-DocLayoutV3",
            "PP-DocLayoutV4",
        ]
        if any(name in self.model_name for name in models_required_imgsize):
            ordered_required_keys = (
                "img_size",
                "img",
                "scale_factors",
            )
        else:
            ordered_required_keys = ("img", "scale_factors")

        return ToBatch(ordered_required_keys=ordered_required_keys)

    def build_postprocess(self):
        if self.threshold is None:
            self.threshold = self.config.get("draw_threshold", 0.5)
        if not self.layout_nms:
            self.layout_nms = self.config.get("layout_nms", None)
        if self.layout_unclip_ratio is None:
            self.layout_unclip_ratio = self.config.get("layout_unclip_ratio", None)
        if self.layout_merge_bboxes_mode is None:
            self.layout_merge_bboxes_mode = self.config.get(
                "layout_merge_bboxes_mode", None
            )
        scale_size = getattr(self, "target_size", [800, 800])
        return LayoutAnalysisProcess(
            labels=self.config["label_list"], scale_size=scale_size
        )


class LayoutAnalysisTransformersPredictor(DetTransformersPredictor):
    """Layout analysis predictor backed by HuggingFace transformers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layout_postprocess = LayoutAnalysisProcess(
            labels=self.labels,
            scale_size=self._resolve_scale_size(),
        )

    def _get_result_class(self):
        return LayoutAnalysisResult

    def _resolve_scale_size(self) -> List[int]:
        for cfg in self.model_config.get("Preprocess", []):
            if cfg.get("type") != "Resize":
                continue
            target_size = cfg.get("target_size")
            if isinstance(target_size, int):
                return [target_size, target_size]
            if isinstance(target_size, (tuple, list)) and len(target_size) == 2:
                return list(target_size)
        return [800, 800]

    def preprocess_images(self, images, **kwargs):
        if self.model_name == "PP-DocLayoutV4" and kwargs.get("do_resize", True):
            # Match the Paddle engines exactly: tensor bicubic interpolation has
            # different uint8 rounding, which changes the page geometry output.
            resize = Resize(
                target_size=self._resolve_scale_size()[::-1], interp="BICUBIC"
            )
            images = [Image.fromarray(resize.resize(np.asarray(im))) for im in images]
            kwargs["do_resize"] = False
        return super().preprocess_images(images, **kwargs)

    def _format_layout_transformers_output(self, prediction):
        formatted = self._format_transformers_output(prediction)
        if "order_seq" in prediction:
            order_seq = (
                prediction["order_seq"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            if len(formatted) == len(order_seq):
                formatted = np.concatenate([formatted, order_seq[:, None]], axis=1)

        output = {"boxes": formatted}
        if "polygon_points" in prediction:
            points = prediction["polygon_points"]
            if self.model_name == "PP-DocLayoutV4":
                # V4 predicts quads in original-image coordinates. Keep them in
                # the same filtering/sorting path as the Paddle engine outputs.
                points = points.detach().cpu().numpy()
                output["quad"] = points.reshape(len(formatted), 8)
            else:
                output["polygon_points"] = [np.asarray(p) for p in points]
        if "warp_params" in prediction:
            warp = prediction["warp_params"].detach().cpu().numpy()
            if warp.shape != (5,):
                raise ValueError(
                    f"Expected page warp_params shape (5,), got {warp.shape}"
                )
            output["warp_params"] = warp
        return output

    def process(
        self,
        batch_data: List[Any],
        threshold: Optional[Union[float, dict]] = None,
        layout_nms: bool = False,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float], dict]] = None,
        layout_merge_bboxes_mode: Optional[Union[str, dict]] = None,
        layout_shape_mode: Optional[str] = "auto",
        filter_overlap_boxes: Optional[bool] = True,
        skip_order_labels: Optional[List[str]] = None,
    ):
        if not hasattr(self.image_processor, "post_process_object_detection"):
            raise RuntimeError(
                f"{type(self.image_processor).__name__} does not support "
                "`post_process_object_detection`."
            )

        datas = self.read_op(batch_data.instances)
        images = [Image.fromarray(data["img"]) for data in datas]
        effective_threshold, hf_threshold = self._get_hf_threshold(threshold)

        model_inputs = self.preprocess_images(images=images)
        outputs = self.forward(model_inputs)
        predictions = self.postprocess(outputs, datas=datas, threshold=hf_threshold)

        batch_outputs = [
            self._format_layout_transformers_output(prediction)
            for prediction in predictions
        ]
        boxes = self.layout_postprocess(
            batch_outputs,
            datas,
            threshold=effective_threshold,
            layout_nms=layout_nms or self.layout_nms,
            layout_unclip_ratio=layout_unclip_ratio or self.layout_unclip_ratio,
            layout_merge_bboxes_mode=layout_merge_bboxes_mode
            or self.layout_merge_bboxes_mode,
            layout_shape_mode=layout_shape_mode,
            filter_overlap_boxes=filter_overlap_boxes,
            skip_order_labels=skip_order_labels,
        )

        result = {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": [data["ori_img"] for data in datas],
            "boxes": boxes,
        }
        if any("warp_params" in output for output in batch_outputs):
            result["warp_params"] = [
                output.get("warp_params") for output in batch_outputs
            ]
        return result

    def postprocess(self, outputs, *, datas, threshold, **kwargs):
        predictions = self.image_processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=self._get_target_sizes(datas),
        )

        return predictions
