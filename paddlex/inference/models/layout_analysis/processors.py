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

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray

from ....utils import logging
from ....utils.deps import function_requires_deps, is_dep_available
from ...utils.benchmark import benchmark
from ..object_detection.processors import check_containment, nms
from .utils import compute_reading_order

if is_dep_available("opencv-contrib-python"):
    import cv2

Boxes = List[dict]
Number = Union[int, float]


SKIP_ORDER_LABELS = [
    "figure_title",
    "vision_footnote",
    "image",
    "chart",
    "table",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "footnote",
    "aside_text",
]


def is_convex(p_prev, p_curr, p_next):
    """
    Calculate if the polygon is convex.
    """
    v1 = p_curr - p_prev
    v2 = p_next - p_curr
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return cross < 0


def angle_between_vectors(v1, v2):
    """
    Calculate the angle between two vectors.
    """

    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_prod = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    angle_rad = np.arccos(dot_prod)
    return np.degrees(angle_rad)


def calc_new_point(p_curr, v1, v2, distance=20):
    """
    Calculate the new point based on the direction of two vectors.
    """
    dir_vec = v1 / np.linalg.norm(v1) + v2 / np.linalg.norm(v2)
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    p_new = p_curr + dir_vec * distance
    return p_new


def extract_custom_vertices(
    polygon, max_allowed_dist, sharp_angle_thresh=45, max_dist_ratio=0.3
):
    poly = np.array(polygon)
    n = len(poly)
    max_allowed_dist *= max_dist_ratio

    point_info = []
    for i in range(n):
        p_prev, p_curr, p_next = poly[(i - 1) % n], poly[i], poly[(i + 1) % n]
        v1, v2 = p_prev - p_curr, p_next - p_curr
        is_convex_point = is_convex(p_prev, p_curr, p_next)
        angle = angle_between_vectors(v1, v2)
        point_info.append(
            {
                "index": i,
                "is_convex": is_convex_point,
                "angle": angle,
                "v1": v1,
                "v2": v2,
            }
        )

    concave_indices = [i for i, info in enumerate(point_info) if not info["is_convex"]]
    preserve_concave = set()

    if concave_indices:
        groups = []
        current_group = [concave_indices[0]]

        for i in range(1, len(concave_indices)):
            if concave_indices[i] - concave_indices[i - 1] == 1 or (
                concave_indices[i - 1] == n - 1 and concave_indices[i] == 0
            ):
                current_group.append(concave_indices[i])
            else:
                if len(current_group) >= 2:
                    groups.extend(current_group)
                current_group = [concave_indices[i]]

        if len(current_group) >= 2:
            groups.extend(current_group)

        if (
            len(concave_indices) >= 2
            and concave_indices[0] == 0
            and concave_indices[-1] == n - 1
        ):
            if 0 in groups and n - 1 in groups:
                preserve_concave.update(groups)
        else:
            preserve_concave.update(groups)

    kept_points = [
        i
        for i, info in enumerate(point_info)
        if info["is_convex"] or (i in preserve_concave and info["angle"] >= 120)
    ]

    final_points = []
    for idx in range(len(kept_points)):
        current_idx = kept_points[idx]
        next_idx = kept_points[(idx + 1) % len(kept_points)]
        final_points.append(current_idx)

        dist = np.linalg.norm(poly[current_idx] - poly[next_idx])
        if dist > max_allowed_dist:
            intermediate = (
                list(range(current_idx + 1, next_idx))
                if next_idx > current_idx
                else list(range(current_idx + 1, n)) + list(range(0, next_idx))
            )

            if intermediate:
                num_needed = int(np.ceil(dist / max_allowed_dist)) - 1
                if len(intermediate) <= num_needed:
                    final_points.extend(intermediate)
                else:
                    step = len(intermediate) / num_needed
                    final_points.extend(
                        [intermediate[int(i * step)] for i in range(num_needed)]
                    )

    final_points = sorted(set(final_points))
    res = []

    for i in final_points:
        info = point_info[i]
        p_curr = poly[i]

        if info["is_convex"] and abs(info["angle"] - sharp_angle_thresh) < 1:
            v1_norm = info["v1"] / np.linalg.norm(info["v1"])
            v2_norm = info["v2"] / np.linalg.norm(info["v2"])
            dir_vec = v1_norm + v2_norm
            dir_vec /= np.linalg.norm(dir_vec)
            d = (np.linalg.norm(info["v1"]) + np.linalg.norm(info["v2"])) / 2
            res.append(tuple(p_curr + dir_vec * d))
        else:
            res.append(tuple(p_curr))

    return res


@function_requires_deps("opencv-contrib-python")
def mask2polygon(mask, max_allowed_dist, epsilon_ratio=0.004, extract_custom=True):
    """
    Postprocess mask by removing small noise.
    Args:
        mask (ndarray): The input mask of shape [H, W].
        epsilon_ratio (float): The ratio of epsilon.
    Returns:
        ndarray: The output mask after postprocessing.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)
    epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
    approx_cnt = cv2.approxPolyDP(cnt, epsilon, True)
    polygon_points = approx_cnt.squeeze()
    polygon_points = np.atleast_2d(polygon_points)
    if extract_custom:
        polygon_points = extract_custom_vertices(polygon_points, max_allowed_dist)

    return polygon_points


def extract_polygon_points_by_masks(
    boxes, masks, scale_ratio, layout_shape_mode, quad=None
):
    """
    修改后的提取函数：auto 模式下信任几何决策
    """
    scale_w, scale_h = scale_ratio[0] / 4, scale_ratio[1] / 4
    h_m, w_m = masks.shape[1:]
    polygon_points = []

    max_box_w = max(boxes[:, 4] - boxes[:, 3])

    for i in range(len(boxes)):
        x_min, y_min, x_max, y_max = boxes[i, 2:6].astype(np.int32)
        box_w, box_h = x_max - x_min, y_max - y_min
        rect = _rect_from_box(boxes[i, 2:6])

        if box_w <= 0 or box_h <= 0:
            polygon_points.append(rect)
            continue

        # crop mask
        x_s = np.clip(
            [int(round(x_min * scale_w)), int(round(x_max * scale_w))], 0, w_m
        )
        y_s = np.clip(
            [int(round(y_min * scale_h)), int(round(y_max * scale_h))], 0, h_m
        )

        cropped = masks[i, y_s[0] : y_s[1], x_s[0] : x_s[1]]
        if cropped.size == 0 or np.sum(cropped) == 0:
            polygon_points.append(rect)
            continue

        # resize mask to match box size
        resized_mask = cv2.resize(
            cropped.astype(np.uint8), (box_w, box_h), interpolation=cv2.INTER_NEAREST
        )

        if box_w > max_box_w * 0.6:
            max_allowed_dist = box_w
        else:
            max_allowed_dist = max_box_w

        polygon = mask2polygon(resized_mask, max_allowed_dist)
        if polygon is not None and len(polygon) > 0:
            polygon = polygon + np.array([x_min, y_min])
        model_quad = None
        if quad is not None:
            model_quad = quad[i].reshape(4, 2)
        polygon_points.append(
            _normalize_layout_polygon(
                box=boxes[i, 2:6],
                polygon=polygon,
                layout_shape_mode=layout_shape_mode,
                previous_polygon=(
                    polygon_points[-1] if len(polygon_points) > 0 else None
                ),
                model_quad=model_quad,
            )
        )

    return polygon_points


def normalize_polygon_points_by_boxes(
    boxes, polygon_points, layout_shape_mode, quad=None
):
    normalized_points = []

    for i, (polygon, box) in enumerate(zip(polygon_points, boxes)):
        model_quad = quad[i].reshape(4, 2) if quad is not None else None
        normalized_points.append(
            _normalize_layout_polygon(
                box=box[2:6],
                polygon=polygon,
                layout_shape_mode=layout_shape_mode,
                previous_polygon=(
                    normalized_points[-1] if len(normalized_points) > 0 else None
                ),
                model_quad=model_quad,
            )
        )

    return normalized_points


def _rect_from_box(box):
    x_min, y_min, x_max, y_max = np.asarray(box).astype(np.int32)
    return np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
        dtype=np.float32,
    )


def _normalize_layout_polygon(
    box,
    polygon,
    layout_shape_mode,
    previous_polygon=None,
    model_quad=None,
):
    rect = _rect_from_box(box)

    if polygon is None and model_quad is None:
        return rect

    if polygon is not None:
        polygon = np.asarray(polygon, dtype=np.float32)
        if polygon.ndim == 1:
            polygon = polygon.reshape(-1, 2)

    if polygon is not None and len(polygon) < 4 and model_quad is None:
        return rect

    if layout_shape_mode == "rect":
        return rect

    if layout_shape_mode == "poly":
        return polygon

    # Use model_quad directly if available, otherwise compute from polygon
    if model_quad is not None:
        quad = model_quad
    else:
        quad = convert_polygon_to_quad(polygon)

    if layout_shape_mode == "quad":
        return quad if quad is not None else rect

    if layout_shape_mode == "auto":
        rect_list = rect.tolist()
        if quad is not None:
            quad_list = quad.tolist() if isinstance(quad, np.ndarray) else quad
            iou_rect_quad = calculate_polygon_overlap_ratio(
                rect_list, quad_list, mode="union"
            )
            if iou_rect_quad >= 0.95:
                return rect

            poly_list = polygon.tolist()
            iou_poly_quad = calculate_polygon_overlap_ratio(
                poly_list, quad_list, mode="union"
            )

            iou_pre = 0
            if previous_polygon is not None:
                iou_pre = calculate_polygon_overlap_ratio(
                    previous_polygon.tolist(),
                    rect_list,
                    mode="small",
                )

            if iou_poly_quad >= 0.8 and iou_pre < 0.01:
                return quad

        return polygon

    raise ValueError(
        "layout_shape_mode must be one of ['rect', 'poly', 'quad', 'auto']"
    )


def convert_polygon_to_quad(polygon):
    """
    Convert polygon to minimum bounding rectangle (quad).
    Args:
        polygon (ndarray): The polygon points of shape [N, 2].
    Returns:
        quad (ndarray): The 4-point quad, clockwise from top-left, or None if invalid.
    """
    if polygon is None or len(polygon) < 3:
        return None

    points = np.array(polygon, dtype=np.float32)
    if len(points.shape) == 1:
        points = points.reshape(-1, 2)

    min_rect = cv2.minAreaRect(points)
    quad = cv2.boxPoints(min_rect)

    center = quad.mean(axis=0)
    angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    quad = quad[sorted_indices]
    sums = quad[:, 0] + quad[:, 1]
    top_left_idx = np.argmin(sums)
    quad = np.roll(quad, -top_left_idx, axis=0)

    return quad


def restructured_boxes(
    boxes: ndarray,
    labels: List[str],
    img_size: Tuple[int, int],
    polygon_points: ndarray = None,
    quad: ndarray = None,
) -> Boxes:
    """
    Restructure the given bounding boxes and labels based on the image size.

    Args:
        boxes (ndarray): A 2D array of bounding boxes with each box represented as [cls_id, score, xmin, ymin, xmax, ymax].
        labels (List[str]): A list of class labels corresponding to the class ids.
        img_size (Tuple[int, int]): A tuple representing the width and height of the image.
        polygon_points (ndarray): A 2D array of polygon points with each point represented as [x, y].
        quad (ndarray, optional): Quad coordinates [N, 8] for 4-point box models.
    Returns:
        Boxes: A list of dictionaries, each containing 'cls_id', 'label', 'score', and 'coordinate' keys.
    """
    box_list = []
    w, h = img_size

    for idx, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box[2:]
        xmin = int(max(0, xmin))
        ymin = int(max(0, ymin))
        xmax = int(min(w, xmax))
        ymax = int(min(h, ymax))
        if xmax <= xmin or ymax <= ymin:
            continue
        res = {
            "cls_id": int(box[0]),
            "label": labels[int(box[0])],
            "score": float(box[1]),
            "coordinate": [xmin, ymin, xmax, ymax],
            "order": idx + 1,
        }
        if polygon_points is not None:
            polygon_point = polygon_points[idx]
            if polygon_point is None:
                continue
            res["polygon_points"] = polygon_point
        if quad is not None:
            res["quad"] = quad[idx].reshape(4, 2).tolist()
        box_list.append(res)

    return box_list


def unclip_boxes(boxes, unclip_ratio=None):
    """
    Expand bounding boxes from (x1, y1, x2, y2) format using an unclipping ratio.

    Parameters:
    - boxes: np.ndarray of shape (N, 4), where each row is (x1, y1, x2, y2).
    - unclip_ratio: tuple of (width_ratio, height_ratio), optional.

    Returns:
    - expanded_boxes: np.ndarray of shape (N, 4), where each row is (x1, y1, x2, y2).
    """
    if unclip_ratio is None:
        return boxes

    if isinstance(unclip_ratio, dict):
        expanded_boxes = []
        for box in boxes:
            class_id, score, x1, y1, x2, y2 = box
            if class_id in unclip_ratio:
                width_ratio, height_ratio = unclip_ratio[class_id]

                width = x2 - x1
                height = y2 - y1

                new_w = width * width_ratio
                new_h = height * height_ratio
                center_x = x1 + width / 2
                center_y = y1 + height / 2

                new_x1 = center_x - new_w / 2
                new_y1 = center_y - new_h / 2
                new_x2 = center_x + new_w / 2
                new_y2 = center_y + new_h / 2

                expanded_boxes.append([class_id, score, new_x1, new_y1, new_x2, new_y2])
            else:
                expanded_boxes.append(box)
        return np.array(expanded_boxes)

    else:
        widths = boxes[:, 4] - boxes[:, 2]
        heights = boxes[:, 5] - boxes[:, 3]

        new_w = widths * unclip_ratio[0]
        new_h = heights * unclip_ratio[1]
        center_x = boxes[:, 2] + widths / 2
        center_y = boxes[:, 3] + heights / 2

        new_x1 = center_x - new_w / 2
        new_y1 = center_y - new_h / 2
        new_x2 = center_x + new_w / 2
        new_y2 = center_y + new_h / 2
        expanded_boxes = np.column_stack(
            (boxes[:, 0], boxes[:, 1], new_x1, new_y1, new_x2, new_y2)
        )
        return expanded_boxes


def _unclip_quads(quad, boxes, unclip_ratio):
    """Scale quads about the same AABB centers as ``unclip_boxes``.

    Ratios are in the original page's x/y axes, normalized to a pair or a
    per-class dictionary by ``LayoutAnalysisProcess.apply``. Scale the actual
    crop boundary as well as its enclosing rectangle, so masking and page
    rectification retain the added pixels. Keep out-of-page corners intact:
    clipping them would distort the quad used by the rectifier.
    """
    if isinstance(unclip_ratio, dict):
        ratios = np.asarray(
            [unclip_ratio.get(int(class_id), (1.0, 1.0)) for class_id in boxes[:, 0]]
        )
    else:
        ratios = np.broadcast_to(unclip_ratio, (len(boxes), 2))
    centers = boxes[:, 2:4] + (boxes[:, 4:6] - boxes[:, 2:4]) / 2
    points = quad.reshape(-1, 4, 2)
    # Add only the displacement: identity ratios preserve fractional coordinates
    # exactly, and the model's output array is never modified in place.
    return (
        points + (points - centers[:, None, :]) * (ratios[:, None, :] - 1.0)
    ).reshape(-1, 8)


def make_valid(poly):
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def calculate_polygon_overlap_ratio(
    polygon1: List[Tuple[int, int]],
    polygon2: List[Tuple[int, int]],
    mode: str = "union",
) -> float:
    """
    Calculate the overlap ratio between two polygons.

    Args:
        polygon1 (List[Tuple[int, int]]): First polygon represented as a list of points.
        polygon2 (List[Tuple[int, int]]): Second polygon represented as a list of points.
        mode (str, optional): Overlap calculation mode. Defaults to "union".

    Returns:
        float: Overlap ratio value between 0 and 1.
    """
    try:
        from shapely.geometry import Polygon
    except ImportError:
        raise ImportError("Please install Shapely library.")
    poly1 = Polygon(polygon1)
    poly2 = Polygon(polygon2)
    poly1 = make_valid(poly1)
    poly2 = make_valid(poly2)
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    if mode == "union":
        return intersection / union
    elif mode == "small":
        small_area = min(poly1.area, poly2.area)
        return intersection / small_area
    elif mode == "large":
        large_area = max(poly1.area, poly2.area)
        return intersection / large_area
    else:
        raise ValueError(f"Unknown mode: {mode}")


def calculate_bbox_area(bbox):
    """Calculate bounding box area"""
    x1, y1, x2, y2 = map(float, bbox)
    area = abs((x2 - x1) * (y2 - y1))
    return area


def calculate_overlap_ratio(
    bbox1: Union[np.ndarray, list, tuple],
    bbox2: Union[np.ndarray, list, tuple],
    mode="union",
) -> float:
    """
    Calculate the overlap ratio between two bounding boxes using NumPy.

    Args:
        bbox1 (np.ndarray, list or tuple): The first bounding box, format [x_min, y_min, x_max, y_max]
        bbox2 (np.ndarray, list or tuple): The second bounding box, format [x_min, y_min, x_max, y_max]
        mode (str): The mode of calculation, either 'union', 'small', or 'large'.

    Returns:
        float: The overlap ratio value between the two bounding boxes
    """
    bbox1 = np.array(bbox1)
    bbox2 = np.array(bbox2)

    x_min_inter = np.maximum(bbox1[0], bbox2[0])
    y_min_inter = np.maximum(bbox1[1], bbox2[1])
    x_max_inter = np.minimum(bbox1[2], bbox2[2])
    y_max_inter = np.minimum(bbox1[3], bbox2[3])

    inter_width = np.maximum(0, x_max_inter - x_min_inter)
    inter_height = np.maximum(0, y_max_inter - y_min_inter)

    inter_area = inter_width * inter_height

    bbox1_area = calculate_bbox_area(bbox1)
    bbox2_area = calculate_bbox_area(bbox2)

    if mode == "union":
        ref_area = bbox1_area + bbox2_area - inter_area
    elif mode == "small":
        ref_area = np.minimum(bbox1_area, bbox2_area)
    elif mode == "large":
        ref_area = np.maximum(bbox1_area, bbox2_area)
    else:
        raise ValueError(
            f"Invalid mode {mode}, must be one of ['union', 'small', 'large']."
        )

    if ref_area == 0:
        return 0.0

    return inter_area / ref_area


def quad_iou(quad1, quad2):
    """Compute IoU between two quad polygons.
    Corresponds to iou(box1, box2): inter / (area1 + area2 - inter)
    Args:
        quad1, quad2: ndarray of shape [8], i.e. [x1,y1,x2,y2,x3,y3,x4,y4]
    Returns:
        float: IoU value
    """
    from shapely.geometry import Polygon as ShapelyPolygon

    p1 = ShapelyPolygon(quad1.reshape(4, 2))
    p2 = ShapelyPolygon(quad2.reshape(4, 2))
    if not p1.is_valid:
        p1 = p1.buffer(0)
    if not p2.is_valid:
        p2 = p2.buffer(0)
    inter = p1.intersection(p2).area
    union_area = p1.area + p2.area - inter
    return inter / union_area if union_area > 0 else 0.0


def quad_overlap_ratio(quad1, quad2, mode="union"):
    """Compute overlap ratio between two quad polygons.
    Corresponds to calculate_overlap_ratio(bbox1, bbox2, mode).
    Args:
        quad1, quad2: ndarray of shape [8]
        mode: "union", "small", or "large"
    Returns:
        float: overlap ratio
    """
    from shapely.geometry import Polygon as ShapelyPolygon

    p1 = ShapelyPolygon(quad1.reshape(4, 2))
    p2 = ShapelyPolygon(quad2.reshape(4, 2))
    if not p1.is_valid:
        p1 = p1.buffer(0)
    if not p2.is_valid:
        p2 = p2.buffer(0)
    inter = p1.intersection(p2).area
    if mode == "union":
        ref = p1.area + p2.area - inter
    elif mode == "small":
        ref = min(p1.area, p2.area)
    elif mode == "large":
        ref = max(p1.area, p2.area)
    else:
        raise ValueError(
            f"Invalid mode {mode}, must be one of ['union', 'small', 'large']."
        )
    return inter / ref if ref > 0 else 0.0


def quad_area(quad):
    """Compute area of a quad polygon using Shoelace formula.
    Corresponds to calculate_bbox_area(bbox).
    Args:
        quad: ndarray of shape [8], i.e. [x1,y1,x2,y2,x3,y3,x4,y4]
    Returns:
        float: area
    """
    pts = quad.reshape(4, 2)
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def quad_is_contained(quad1, quad2):
    """Check if quad1 is contained within quad2.
    Corresponds to is_contained(box1, box2): intersect_area / box1_area >= 0.9
    Args:
        quad1, quad2: ndarray of shape [8]
    Returns:
        bool
    """
    from shapely.geometry import Polygon as ShapelyPolygon

    p1 = ShapelyPolygon(quad1.reshape(4, 2))
    p2 = ShapelyPolygon(quad2.reshape(4, 2))
    if not p1.is_valid:
        p1 = p1.buffer(0)
    if not p2.is_valid:
        p2 = p2.buffer(0)
    if p1.area <= 0:
        return False
    return p1.intersection(p2).area / p1.area >= 0.9


def _prepare_quad_geoms(quads):
    """Precompute shapely polygons, areas and axis-aligned bounding boxes once.

    Building shapely polygons is expensive, so we do it a single time per box
    instead of rebuilding two polygons for every O(N^2) pair. The returned AABBs
    enable a cheap disjointness pre-filter: if two AABBs do not overlap, the
    polygons cannot intersect, so their IoU / overlap is exactly 0 and the
    shapely intersection can be skipped entirely (numerically identical result).

    Args:
        quads: ndarray [N, 8], each row [x1,y1,x2,y2,x3,y3,x4,y4].
    Returns:
        (polys, areas[N], aabb[N, 4]) where aabb rows are [xmin,ymin,xmax,ymax].
        Polygon validity is repaired with buffer(0), matching quad_iou/
        quad_overlap_ratio/quad_is_contained; areas are taken from the repaired
        polygon so ratios match the original implementations exactly.
    """
    from shapely.geometry import Polygon as ShapelyPolygon

    n = len(quads)
    polys = [None] * n
    areas = np.zeros(n, dtype=np.float64)
    aabb = np.zeros((n, 4), dtype=np.float64)
    for k in range(n):
        pts = np.asarray(quads[k], dtype=np.float64).reshape(4, 2)
        p = ShapelyPolygon(pts)
        if not p.is_valid:
            p = p.buffer(0)
        polys[k] = p
        areas[k] = p.area
        aabb[k, 0] = pts[:, 0].min()
        aabb[k, 1] = pts[:, 1].min()
        aabb[k, 2] = pts[:, 0].max()
        aabb[k, 3] = pts[:, 1].max()
    return polys, areas, aabb


def _aabb_disjoint(a, b):
    """True if two AABBs [xmin,ymin,xmax,ymax] do not overlap (touch counts as
    disjoint since a shared edge has zero intersection area)."""
    return a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def nms_quad(boxes, quad, iou_same=0.6, iou_diff=0.95):
    """NMS using quad polygon IoU.
    Corresponds to nms(boxes, iou_same, iou_diff) with identical logic:
    1. Sort by score descending
    2. Greedy selection with per-pair IoU check
    3. Same-class threshold iou_same, cross-class threshold iou_diff
    Args:
        boxes: ndarray [N, 6+], columns [class, score, ...]
        quad: ndarray [N, 8]
    Returns:
        list of selected indices
    """
    quad = np.asarray(quad)
    n = len(quad)
    polys, areas, aabb = _prepare_quad_geoms(quad.reshape(n, 8))

    scores = boxes[:, 1]
    indices = np.argsort(scores)[::-1].tolist()
    selected = []

    while indices:
        current = indices.pop(0)
        selected.append(current)
        remaining = []
        current_class = boxes[current, 0]
        cp = polys[current]
        ca = areas[current]
        cbox = aabb[current]
        for i in indices:
            threshold = iou_same if current_class == boxes[i, 0] else iou_diff
            if _aabb_disjoint(cbox, aabb[i]):
                # Disjoint AABBs => IoU 0 < threshold => box is kept.
                remaining.append(i)
                continue
            inter = cp.intersection(polys[i]).area
            union_area = ca + areas[i] - inter
            iou_val = inter / union_area if union_area > 0 else 0.0
            if iou_val < threshold:
                remaining.append(i)
        indices = remaining
    return selected


def check_containment_quad(
    quad, class_ids, formula_index=None, category_index=None, mode=None
):
    """Check containment relationships among quad polygons.
    Corresponds to check_containment(boxes, formula_index, category_index, mode)
    with identical logic, using quad_is_contained instead of is_contained.
    Args:
        quad: ndarray [N, 8]
        class_ids: ndarray [N], class label for each box
        formula_index, category_index, mode: same as check_containment
    Returns:
        (contains_other, contained_by_other): ndarray [N] each
    """
    n = len(quad)
    quad = np.asarray(quad)
    polys, areas, aabb = _prepare_quad_geoms(quad.reshape(n, 8))
    contains_other = np.zeros(n, dtype=int)
    contained_by_other = np.zeros(n, dtype=int)

    def _contained(i, j):
        # quad[i] contained in quad[j]: inter_area / area_i >= 0.9
        if areas[i] <= 0:
            return False
        if _aabb_disjoint(aabb[i], aabb[j]):
            return False
        inter = polys[i].intersection(polys[j]).area
        return inter / areas[i] >= 0.9

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if formula_index is not None:
                if class_ids[i] == formula_index and class_ids[j] != formula_index:
                    continue
            if category_index is not None and mode is not None:
                if mode == "large" and class_ids[j] == category_index:
                    if _contained(i, j):
                        contained_by_other[i] = 1
                        contains_other[j] = 1
                if mode == "small" and class_ids[i] == category_index:
                    if _contained(i, j):
                        contained_by_other[i] = 1
                        contains_other[j] = 1
            else:
                if _contained(i, j):
                    contained_by_other[i] = 1
                    contains_other[j] = 1
    return contains_other, contained_by_other


def filter_boxes(
    src_boxes: Dict[str, List[Dict]], layout_shape_mode: str
) -> Dict[str, List[Dict]]:
    """
    Remove overlapping boxes from layout detection results based on a given overlap ratio.

    Args:
        boxes (Dict[str, List[Dict]]): Layout detection result dict containing a 'boxes' list.

    Returns:
        Dict[str, List[Dict]]: Filtered dict with overlapping boxes removed.
    """
    boxes = [box for box in src_boxes if box["label"] != "reference"]
    dropped_indexes = set()

    # Precompute quad polygons/areas/AABBs once (instead of rebuilding two
    # shapely polygons per O(N^2) pair). Entries are None for boxes without quad.
    n_boxes = len(boxes)
    _qpoly = [None] * n_boxes
    _qarea = [0.0] * n_boxes
    _qaabb = [None] * n_boxes
    if any("quad" in box for box in boxes):
        from shapely.geometry import Polygon as ShapelyPolygon

        for k in range(n_boxes):
            if "quad" in boxes[k]:
                pts = np.array(boxes[k]["quad"], dtype=np.float64).reshape(4, 2)
                p = ShapelyPolygon(pts)
                if not p.is_valid:
                    p = p.buffer(0)
                _qpoly[k] = p
                _qarea[k] = p.area
                _qaabb[k] = (
                    pts[:, 0].min(),
                    pts[:, 1].min(),
                    pts[:, 0].max(),
                    pts[:, 1].max(),
                )

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]["coordinate"]
        w, h = x2 - x1, y2 - y1
        if w < 6 or h < 6:
            dropped_indexes.add(i)
        for j in range(i + 1, len(boxes)):
            if i in dropped_indexes or j in dropped_indexes:
                continue
            # Use quad overlap if both boxes have quad
            if "quad" in boxes[i] and "quad" in boxes[j]:
                # AABB pre-filter: disjoint AABBs => overlap ratio is exactly 0.
                if _aabb_disjoint(_qaabb[i], _qaabb[j]):
                    overlap_ratio = 0.0
                else:
                    inter = _qpoly[i].intersection(_qpoly[j]).area
                    ref = min(_qarea[i], _qarea[j])
                    overlap_ratio = inter / ref if ref > 0 else 0.0
            else:
                overlap_ratio = calculate_overlap_ratio(
                    boxes[i]["coordinate"], boxes[j]["coordinate"], "small"
                )
            if (
                boxes[i]["label"] == "inline_formula"
                or boxes[j]["label"] == "inline_formula"
            ):
                if overlap_ratio > 0.5:
                    if boxes[i]["label"] == "inline_formula":
                        dropped_indexes.add(i)
                    if boxes[j]["label"] == "inline_formula":
                        dropped_indexes.add(j)
                    continue
            if overlap_ratio > 0.7:
                if layout_shape_mode != "rect" and "polygon_points" in boxes[i]:
                    poly_overlap_ratio = calculate_polygon_overlap_ratio(
                        boxes[i]["polygon_points"], boxes[j]["polygon_points"], "small"
                    )
                    if poly_overlap_ratio < 0.7:
                        continue
                # Use quad area if available
                if "quad" in boxes[i]:
                    box_area_i = quad_area(np.array(boxes[i]["quad"]).flatten())
                else:
                    box_area_i = calculate_bbox_area(boxes[i]["coordinate"])
                if "quad" in boxes[j]:
                    box_area_j = quad_area(np.array(boxes[j]["quad"]).flatten())
                else:
                    box_area_j = calculate_bbox_area(boxes[j]["coordinate"])
                labels = {boxes[i]["label"], boxes[j]["label"]}
                if labels & {"image", "table", "seal", "chart"} and len(labels) > 1:
                    if "table" not in labels or labels <= {
                        "table",
                        "image",
                        "seal",
                        "chart",
                    }:
                        continue
                if box_area_i >= box_area_j:
                    dropped_indexes.add(j)
                else:
                    dropped_indexes.add(i)
    out_boxes = [box for idx, box in enumerate(boxes) if idx not in dropped_indexes]
    return out_boxes


def update_order_index(boxes: List[Dict], skip_order_labels: List[str]):
    """
    Update the 'order_index' field of each box in the provided list of boxes.

    Args:
        boxes (List[Dict]): A list of boxes, where each box is represented as a dictionary with an 'order_index' field.

    Returns:
        None. The  function updates the 'order_index' field of each box in the input list.
    """
    order_index = 1
    for box in boxes:
        label = box["label"]
        if label not in skip_order_labels:
            box["order"] = order_index
            order_index += 1
        else:
            box["order"] = None
    return boxes


def find_label_position(box, polygon_points, text_w, text_h, max_shift=50):
    try:
        from shapely.geometry import Polygon
    except ImportError:
        raise ImportError("Please install Shapely library.")
    poly = Polygon(polygon_points)
    min_x = min([p[0] for p in polygon_points])
    min_y = min([p[1] for p in polygon_points])
    for dy in range(max_shift):
        x1, y1 = min_x, min_y + dy
        x2, y2 = x1 + text_w, y1 + text_h
        label_rect = box(x1, y1, x2, y2)
        if poly.intersects(label_rect):
            return int(x1), int(y1)

    return int(min_x), int(min_y)


@benchmark.timeit
class LayoutAnalysisProcess:
    """Save Result Transform

    This class is responsible for post-processing detection results, including
    thresholding, non-maximum suppression (NMS), and restructuring the boxes
    based on the input type (normal or rotated object detection).
    """

    def __init__(
        self, labels: Optional[List[str]] = None, scale_size: Optional[List[int]] = None
    ) -> None:
        """Initialize the DetPostProcess class.

        Args:
            threshold (float, optional): The threshold to apply to the detection scores. Defaults to 0.5.
            labels (Optional[List[str]], optional): The list of labels for the detection categories. Defaults to None.
            layout_postprocess (bool, optional): Whether to apply layout post-processing. Defaults to False.
        """
        super().__init__()
        self.labels = labels
        self.scale_size = scale_size

    def apply(
        self,
        boxes: ndarray,
        img_size: Tuple[int, int],
        threshold: Union[float, dict],
        layout_nms: Optional[bool],
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float], dict]],
        layout_merge_bboxes_mode: Optional[Union[str, dict]],
        masks: Optional[ndarray] = None,
        layout_shape_mode: Optional[str] = "auto",
        quad: Optional[ndarray] = None,
        rel_logits: Optional[ndarray] = None,
        roor_logits: Optional[ndarray] = None,
        qi: Optional[ndarray] = None,
        polygon_points: Optional[List[ndarray]] = None,
    ) -> Boxes:
        """Apply post-processing to the detection boxes.

        Args:
            boxes (ndarray): The input detection boxes with scores.
            img_size (tuple): The original image size.
            quad (ndarray, optional): Quad coordinates [N, 8] for 4-point box models.
            polygon_points (List[ndarray], optional): Per-box polygons supplied by
                Transformers mask models; vertex counts may differ between boxes.

        Returns:
            Boxes: The post-processed detection boxes.
        """
        if layout_shape_mode == "rect":
            masks = None
            quad = None
            polygon_points = None
        has_order_logits = (
            rel_logits is not None and roor_logits is not None and qi is not None
        )
        boxes[:, 2:6] = np.round(boxes[:, 2:6]).astype(int)
        if isinstance(threshold, float):
            expect_boxes = (boxes[:, 1] > threshold) & (boxes[:, 0] > -1)
            boxes = boxes[expect_boxes, :]
            if masks is not None:
                masks = masks[expect_boxes, ...]
            if quad is not None:
                quad = quad[expect_boxes]
            if polygon_points is not None:
                polygon_points = [
                    polygon_points[i] for i, keep in enumerate(expect_boxes) if keep
                ]
            if has_order_logits:
                qi = qi[expect_boxes]
        elif isinstance(threshold, dict):
            category_filtered_boxes = []
            if masks is not None:
                category_filtered_masks = []
            if quad is not None:
                category_filtered_quad = []
            if polygon_points is not None:
                category_filtered_polygon_points = []
            if has_order_logits:
                category_filtered_qi = []
            for cat_id in np.unique(boxes[:, 0]):
                cat_mask = boxes[:, 0] == cat_id
                category_boxes = boxes[cat_mask]
                if masks is not None:
                    category_masks = masks[cat_mask]
                if quad is not None:
                    category_quad = quad[cat_mask]
                if has_order_logits:
                    category_qi = qi[cat_mask]
                category_threshold = threshold.get(int(cat_id), 0.5)
                selected_indices = (category_boxes[:, 1] > category_threshold) & (
                    category_boxes[:, 0] > -1
                )
                if masks is not None:
                    category_masks = category_masks[selected_indices]
                    category_filtered_masks.append(category_masks)
                if quad is not None:
                    category_filtered_quad.append(category_quad[selected_indices])
                if polygon_points is not None:
                    category_filtered_polygon_points.extend(
                        polygon_points[i]
                        for i in np.flatnonzero(cat_mask)[selected_indices]
                    )
                if has_order_logits:
                    category_filtered_qi.append(category_qi[selected_indices])
                category_filtered_boxes.append(category_boxes[selected_indices])
            boxes = (
                np.vstack(category_filtered_boxes)
                if category_filtered_boxes
                else boxes[:0]
            )
            if masks is not None:
                masks = (
                    np.concatenate(category_filtered_masks)
                    if category_filtered_masks
                    else np.array([])
                )
            if quad is not None:
                quad = (
                    np.vstack(category_filtered_quad)
                    if category_filtered_quad
                    else np.array([]).reshape(0, 8)
                )
            if polygon_points is not None:
                polygon_points = category_filtered_polygon_points
            if has_order_logits:
                qi = (
                    np.concatenate(category_filtered_qi)
                    if category_filtered_qi
                    else np.array([], dtype=np.int64)
                )

        # Compute reading order right after threshold filtering
        if has_order_logits and len(boxes) > 0:
            order = compute_reading_order(boxes, rel_logits, roor_logits, qi)
            boxes = np.concatenate(
                [boxes, order.reshape(-1, 1).astype(np.float32)], axis=1
            )

        if layout_nms:
            if quad is not None:
                selected_indices = nms_quad(boxes, quad, iou_same=0.6, iou_diff=0.98)
            else:
                selected_indices = nms(boxes[:, :6], iou_same=0.6, iou_diff=0.98)
            boxes = np.array(boxes[selected_indices])
            if masks is not None:
                masks = [masks[i] for i in selected_indices]
            if quad is not None:
                quad = quad[selected_indices]
            if polygon_points is not None:
                polygon_points = [polygon_points[i] for i in selected_indices]

        filter_large_image = True
        # boxes.shape[1] == 6 is object detection, 7 is new ordered object detection, 8 is ordered object detection
        if filter_large_image and len(boxes) > 1 and boxes.shape[1] in [6, 7, 8]:
            if img_size[0] > img_size[1]:
                area_thres = 0.82
            else:
                area_thres = 0.93
            image_index = self.labels.index("image") if "image" in self.labels else None
            img_area = img_size[0] * img_size[1]
            filtered_boxes = []
            filtered_masks = []
            filtered_quad = []
            filtered_polygon_points = []
            for idx, box in enumerate(boxes):
                (
                    label_index,
                    score,
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                ) = box[:6]
                if label_index == image_index:
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(img_size[0], xmax)
                    ymax = min(img_size[1], ymax)
                    if quad is not None:
                        box_area = quad_area(quad[idx])
                    else:
                        box_area = (xmax - xmin) * (ymax - ymin)
                    if box_area <= area_thres * img_area:
                        filtered_boxes.append(box)
                        if masks is not None:
                            filtered_masks.append(masks[idx])
                        if quad is not None:
                            filtered_quad.append(quad[idx])
                        if polygon_points is not None:
                            filtered_polygon_points.append(polygon_points[idx])
                else:
                    filtered_boxes.append(box)
                    if masks is not None:
                        filtered_masks.append(masks[idx])
                    if quad is not None:
                        filtered_quad.append(quad[idx])
                    if polygon_points is not None:
                        filtered_polygon_points.append(polygon_points[idx])
            if len(filtered_boxes) == 0:
                filtered_boxes = boxes
                if masks is not None:
                    filtered_masks = masks
                if quad is not None:
                    filtered_quad = list(quad)
                if polygon_points is not None:
                    filtered_polygon_points = polygon_points
            boxes = np.array(filtered_boxes)
            if masks is not None:
                masks = filtered_masks
            if quad is not None:
                quad = np.array(filtered_quad)
            if polygon_points is not None:
                polygon_points = filtered_polygon_points

        if layout_merge_bboxes_mode:
            formula_index = (
                self.labels.index("formula") if "formula" in self.labels else None
            )
            if isinstance(layout_merge_bboxes_mode, str):
                assert layout_merge_bboxes_mode in [
                    "union",
                    "large",
                    "small",
                ], f"The value of `layout_merge_bboxes_mode` must be one of ['union', 'large', 'small'], but got {layout_merge_bboxes_mode}"

                if layout_merge_bboxes_mode == "union":
                    pass
                else:
                    if quad is not None:
                        contains_other, contained_by_other = check_containment_quad(
                            quad, boxes[:, 0], formula_index
                        )
                    else:
                        contains_other, contained_by_other = check_containment(
                            boxes[:, :6], formula_index
                        )
                    if layout_merge_bboxes_mode == "large":
                        keep_mask = contained_by_other == 0
                        boxes = boxes[keep_mask]
                        if masks is not None:
                            masks = [
                                mask for i, mask in enumerate(masks) if keep_mask[i]
                            ]
                        if quad is not None:
                            quad = quad[keep_mask]
                        if polygon_points is not None:
                            polygon_points = [
                                p for p, keep in zip(polygon_points, keep_mask) if keep
                            ]
                    elif layout_merge_bboxes_mode == "small":
                        keep_mask = (contains_other == 0) | (contained_by_other == 1)
                        boxes = boxes[keep_mask]
                        if masks is not None:
                            masks = [
                                mask for i, mask in enumerate(masks) if keep_mask[i]
                            ]
                        if quad is not None:
                            quad = quad[keep_mask]
                        if polygon_points is not None:
                            polygon_points = [
                                p for p, keep in zip(polygon_points, keep_mask) if keep
                            ]
            elif isinstance(layout_merge_bboxes_mode, dict):
                keep_mask = np.ones(len(boxes), dtype=bool)
                for category_index, layout_mode in layout_merge_bboxes_mode.items():
                    assert layout_mode in [
                        "union",
                        "large",
                        "small",
                    ], f"The value of `layout_merge_bboxes_mode` must be one of ['union', 'large', 'small'], but got {layout_mode}"
                    if layout_mode == "union":
                        pass
                    else:
                        if layout_mode == "large":
                            if quad is not None:
                                contains_other, contained_by_other = (
                                    check_containment_quad(
                                        quad,
                                        boxes[:, 0],
                                        formula_index,
                                        category_index,
                                        mode=layout_mode,
                                    )
                                )
                            else:
                                contains_other, contained_by_other = check_containment(
                                    boxes[:, :6],
                                    formula_index,
                                    category_index,
                                    mode=layout_mode,
                                )
                            # Remove boxes that are contained by other boxes
                            keep_mask &= contained_by_other == 0
                        elif layout_mode == "small":
                            if quad is not None:
                                contains_other, contained_by_other = (
                                    check_containment_quad(
                                        quad,
                                        boxes[:, 0],
                                        formula_index,
                                        category_index,
                                        mode=layout_mode,
                                    )
                                )
                            else:
                                contains_other, contained_by_other = check_containment(
                                    boxes[:, :6],
                                    formula_index,
                                    category_index,
                                    mode=layout_mode,
                                )
                            # Keep boxes that do not contain others or are contained by others
                            keep_mask &= (contains_other == 0) | (
                                contained_by_other == 1
                            )
                boxes = boxes[keep_mask]
                if masks is not None:
                    masks = [mask for i, mask in enumerate(masks) if keep_mask[i]]
                if quad is not None:
                    quad = quad[keep_mask]
                if polygon_points is not None:
                    polygon_points = [
                        p for p, keep in zip(polygon_points, keep_mask) if keep
                    ]

        if boxes.size == 0:
            return np.array([])

        if boxes.shape[1] == 8:
            # Sort boxes by their order
            sorted_idx = np.lexsort((-boxes[:, 7], boxes[:, 6]))
            sorted_boxes = boxes[sorted_idx]
            boxes = sorted_boxes[:, :6]
            if masks is not None:
                sorted_masks = [masks[i] for i in sorted_idx]
                masks = sorted_masks
            if quad is not None:
                quad = quad[sorted_idx]
            if polygon_points is not None:
                polygon_points = [polygon_points[i] for i in sorted_idx]

        if boxes.shape[1] == 7:
            # Sort boxes by their order
            sorted_idx = np.argsort(boxes[:, 6])
            sorted_boxes = boxes[sorted_idx]
            boxes = sorted_boxes[:, :6]
            if masks is not None:
                sorted_masks = [masks[i] for i in sorted_idx]
                masks = sorted_masks
            if quad is not None:
                quad = quad[sorted_idx]
            if polygon_points is not None:
                polygon_points = [polygon_points[i] for i in sorted_idx]

        if polygon_points is None and masks is not None:
            scale_ratio = [h / s for h, s in zip(self.scale_size, img_size)]
            polygon_points = extract_polygon_points_by_masks(
                boxes, np.array(masks), scale_ratio, layout_shape_mode, quad=quad
            )
        elif polygon_points is not None:
            polygon_points = normalize_polygon_points_by_boxes(
                boxes, polygon_points, layout_shape_mode, quad=quad
            )

        if layout_unclip_ratio:
            if isinstance(layout_unclip_ratio, float):
                layout_unclip_ratio = (layout_unclip_ratio, layout_unclip_ratio)
            elif isinstance(layout_unclip_ratio, (tuple, list)):
                assert (
                    len(layout_unclip_ratio) == 2
                ), f"The length of `layout_unclip_ratio` should be 2."
            elif isinstance(layout_unclip_ratio, dict):
                pass
            else:
                raise ValueError(
                    f"The type of `layout_unclip_ratio` must be float, Tuple[float, float] or  Dict[int, Tuple[float, float]], but got {type(layout_unclip_ratio)}."
                )
            if quad is not None:
                quad = _unclip_quads(quad, boxes, layout_unclip_ratio)
            boxes = unclip_boxes(boxes, layout_unclip_ratio)

        if boxes.shape[1] == 6:
            """For Normal Object Detection"""
            boxes = restructured_boxes(
                boxes, self.labels, img_size, polygon_points, quad=quad
            )
        else:
            """Unexpected Input Box Shape"""
            raise ValueError(
                f"The shape of boxes should be 6 or 10, instead of {boxes.shape[1]}"
            )
        return boxes

    def __call__(
        self,
        batch_outputs: List[dict],
        datas: List[dict],
        threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float]]] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
        layout_shape_mode: Optional[str] = None,
        filter_overlap_boxes: Optional[bool] = None,
        skip_order_labels: Optional[List[str]] = None,
    ) -> List[Boxes]:
        """Apply the post-processing to a batch of outputs.

        Args:
            batch_outputs (List[dict]): The list of detection outputs.
            datas (List[dict]): The list of input data.

        Returns:
            List[Boxes]: The list of post-processed detection boxes.
        """
        outputs = []
        for idx, (data, output) in enumerate(zip(datas, batch_outputs)):
            current_layout_shape_mode = layout_shape_mode
            masks = output.get("masks", None)
            quad = output.get("quad", None)
            # Paddle mask models supply masks; Transformers supplies decoded
            # polygons instead. Keep V4's independent quad path in both cases.
            polygon_points = output.get("polygon_points") if masks is None else None
            if masks is None and quad is None and polygon_points is None:
                current_layout_shape_mode = "rect"
                if idx == 0 and layout_shape_mode not in ["rect", "auto"]:
                    logging.warning(
                        f"The model you are using does not support polygon output, but the layout_shape_mode is specified as {layout_shape_mode}, which will be set to 'rect'"
                    )

            boxes = self.apply(
                output["boxes"],
                data["ori_img_size"],
                threshold,
                layout_nms,
                layout_unclip_ratio,
                layout_merge_bboxes_mode,
                masks,
                current_layout_shape_mode,
                quad=quad,
                rel_logits=output.get("rel_logits", None),
                roor_logits=output.get("roor_logits", None),
                qi=output.get("qi", None),
                polygon_points=polygon_points,
            )
            if filter_overlap_boxes:
                boxes = filter_boxes(boxes, current_layout_shape_mode)
            skip_order_labels = (
                skip_order_labels
                if skip_order_labels is not None
                else SKIP_ORDER_LABELS
            )
            boxes = update_order_index(boxes, skip_order_labels)
            outputs.append(boxes)
        return outputs
