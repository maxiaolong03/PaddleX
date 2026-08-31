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

import heapq
from collections import defaultdict

import numpy as np

STATIC_SHAPE_MODEL_LIST = [
    "PP-DocLayoutV2",
    "PP-DocLayoutV3",
    "PP-DocLayoutV4",
]


def _find_cycle(n, edges):
    """
    Detect the first cycle in a directed graph using iterative DFS.

    Args:
        n (int): Number of nodes.
        edges (dict): Edge dict {(i, j): confidence, ...}.

    Returns:
        list|None: List of nodes forming the first found cycle, or None.
    """
    graph = defaultdict(list)
    for i, j in edges.keys():
        graph[i].append(j)
    visited = [False] * n
    rec_stack = [False] * n

    def dfs(vertex, path):
        visited[vertex] = True
        rec_stack[vertex] = True
        path.append(vertex)
        for neighbor in graph[vertex]:
            if not visited[neighbor]:
                ret = dfs(neighbor, path)
                if ret is not None:
                    return ret
            elif rec_stack[neighbor]:
                cycle_start = path.index(neighbor)
                return path[cycle_start:]
        path.pop()
        rec_stack[vertex] = False
        return None

    for node in range(n):
        if not visited[node]:
            cycle = dfs(node, [])
            if cycle:
                return cycle
    return None


def _remove_cycles(n, edges):
    """
    Greedily remove cycles from a directed graph to obtain a DAG.

    Args:
        n (int): Number of nodes.
        edges (dict): Edge dict {(i, j): confidence, ...}.

    Returns:
        list: List of remaining DAG edges as (i, j) tuples.
    """
    edges = dict(edges)
    while True:
        cycle = _find_cycle(n, edges)
        if cycle is None:
            break
        min_edge, min_conf = None, 1e8
        for i, j in zip(cycle, cycle[1:] + cycle[:1]):
            if (i, j) in edges and edges[(i, j)] < min_conf:
                min_edge = (i, j)
                min_conf = edges[(i, j)]
        if min_edge is not None:
            del edges[min_edge]
    return list(edges.keys())


def _sort_by_rel_matrix(candidates, rel_matrix):
    """
    Iteratively sort candidates using Borda-count from a relative order matrix.

    Args:
        candidates (list[int]): Node indices to sort.
        rel_matrix (np.ndarray): Relative order score matrix [N, N].

    Returns:
        list[int]: Nodes sorted from earliest to latest in reading order.
    """
    candidates = list(candidates)
    result = []
    while candidates:
        if len(candidates) == 1:
            result.append(candidates[0])
            break
        best, best_score = None, -1.0
        for c in candidates:
            score = sum(rel_matrix[c][o] for o in candidates if o != c)
            if score > best_score or (
                score == best_score and (best is None or c < best)
            ):
                best_score = score
                best = c
        result.append(best)
        candidates.remove(best)
    return result


def _topological_sort(n, edges, tiebreaker=None, rel_matrix=None, roor_matrix=None):
    """
    Topological sort with three tie-breaking modes.

    Args:
        n (int): Number of nodes.
        edges (list[tuple]): DAG edges as (i, j) pairs.
        tiebreaker (np.ndarray|None): Per-node tie-breaking scores.
        rel_matrix (np.ndarray|None): Relative order scores [n, n].
        roor_matrix (np.ndarray|None): ROOR confidence scores [n, n].

    Returns:
        list[int]: Topologically sorted node indices.
    """
    in_degree = [0] * n
    graph = defaultdict(list)
    for i, j in edges:
        graph[i].append(j)
        in_degree[j] += 1

    if roor_matrix is not None and rel_matrix is not None:
        # Mode 1: stack-based DFS with ROOR branch priority
        initial = [i for i in range(n) if in_degree[i] == 0]
        if len(initial) > 1:
            initial_sorted = _sort_by_rel_matrix(initial, rel_matrix)
        else:
            initial_sorted = initial

        stack = list(reversed(initial_sorted))
        result = []
        while stack:
            node = stack.pop()
            result.append(node)
            new_candidates = []
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    new_candidates.append(neighbor)
            if len(new_candidates) > 1:
                new_candidates.sort(key=lambda c: roor_matrix[node][c])
            for c in new_candidates:
                stack.append(c)

    elif rel_matrix is not None:
        # Mode 2: Borda-count iterative selection
        candidates = [i for i in range(n) if in_degree[i] == 0]
        result = []
        while candidates:
            if len(candidates) == 1:
                best = candidates[0]
            else:
                best, best_score = None, -1.0
                for c in candidates:
                    score = sum(rel_matrix[c][o] for o in candidates if o != c)
                    if score > best_score or (
                        score == best_score and (best is None or c < best)
                    ):
                        best_score = score
                        best = c
            candidates.remove(best)
            result.append(best)
            for neighbor in graph[best]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    candidates.append(neighbor)
    else:
        # Mode 3: min-heap Kahn's algorithm
        queue = []
        for i in range(n):
            if in_degree[i] == 0:
                key = (tiebreaker[i], i) if tiebreaker is not None else (i,)
                heapq.heappush(queue, key)
        result = []
        while queue:
            node = heapq.heappop(queue)[-1]
            result.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    key = (
                        (tiebreaker[neighbor], neighbor)
                        if tiebreaker is not None
                        else (neighbor,)
                    )
                    heapq.heappush(queue, key)

    # Fallback: append any nodes not yet covered
    if len(result) < n:
        in_result = set(result)
        for i in range(n):
            if i not in in_result:
                result.append(i)

    return result


def _find_connected_components(n, edges):
    """
    Find connected components of an undirected view of a directed graph.

    Args:
        n (int): Number of nodes.
        edges (list[tuple]): DAG edges as (i, j) pairs.

    Returns:
        list[list[int]]: List of connected components.
    """
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in edges:
        union(i, j)

    comp_map = defaultdict(list)
    for i in range(n):
        comp_map[find(i)].append(i)

    return list(comp_map.values())


def _get_order_single_image(roor_b, rel_b):
    """
    Decode reading order for a single image.

    Args:
        roor_b (np.ndarray): ROOR logits [N, N].
        rel_b (np.ndarray): Relative order logits [N, N].

    Returns:
        np.ndarray: Position indices [N]. pos[i] = rank of element i.
    """
    N = roor_b.shape[0]

    # Step 1: Threshold ROOR logits to get candidate edges
    edges = {}
    for i in range(N):
        for j in range(N):
            if i != j and roor_b[i][j] > 0:
                edges[(i, j)] = float(roor_b[i][j])

    # Step 2: Remove cycles greedily
    dag_edges = _remove_cycles(N, edges)

    # Step 3: Find connected components
    components = _find_connected_components(N, dag_edges)

    # Step 4: Compute node votes from relative scores
    # Numerically stable sigmoid: the raw logits reach +-1e4 (masked entries),
    # which overflows `exp(-x)` for the naive form.
    rel_scores = np.where(
        rel_b >= 0,
        1.0 / (1.0 + np.exp(-np.abs(rel_b))),
        np.exp(-np.abs(rel_b)) / (1.0 + np.exp(-np.abs(rel_b))),
    )
    np.fill_diagonal(rel_scores, 0.0)
    node_votes = np.sum(rel_scores, axis=0)  # [N]

    # Step 5: Topological sort within each component
    comp_sequences = {}
    for ci, comp in enumerate(components):
        if len(comp) == 1:
            comp_sequences[ci] = list(comp)
            continue
        comp_set = set(comp)
        sorted_comp = sorted(comp)
        old2new = {old: new for new, old in enumerate(sorted_comp)}
        new2old = {new: old for old, new in old2new.items()}
        mapped_edges = [
            (old2new[u], old2new[v])
            for u, v in dag_edges
            if u in comp_set and v in comp_set
        ]
        m = len(comp)
        local_rel = np.array(
            [[rel_scores[oi][oj] for oj in sorted_comp] for oi in sorted_comp]
        )
        local_roor = np.array(
            [[roor_b[oi][oj] for oj in sorted_comp] for oi in sorted_comp]
        )
        local_order = _topological_sort(
            m, mapped_edges, tiebreaker=None, rel_matrix=local_rel, roor_matrix=None
        )
        comp_sequences[ci] = [new2old[idx] for idx in local_order]

    # Step 6: Sort components by mean node_votes
    comp_avg = []
    for ci, comp in enumerate(components):
        avg_rank = np.mean([node_votes[node] for node in comp])
        comp_avg.append((avg_rank, ci))
    comp_avg.sort(key=lambda x: x[0])

    order_list = []
    for _, ci in comp_avg:
        order_list.extend(comp_sequences[ci])

    # Step 7: Convert to position-index array
    pos = np.zeros(N, dtype=np.int64)
    for rank, node in enumerate(order_list):
        pos[node] = rank

    return pos


def decode_reading_order(
    bbox_pred,
    bbox_num,
    mask_pred,
    rel_order_logits,
    roor_order_logits,
    qi,
    order_score_thr=0.5,
):
    """
    Decode reading order from model outputs and append to bbox_pred.

    This function takes the outputs of the exported static model and
    computes reading order using the DAG-based algorithm, producing
    the final 7-column bbox_pred that matches the original model output.

    Args:
        bbox_pred (np.ndarray): Detection results [B*K, 6].
            Fields: [label, score, x1, y1, x2, y2]
        bbox_num (np.ndarray): Number of detections per image [B].
        mask_pred (np.ndarray|None): Post-processed masks.
        rel_order_logits (np.ndarray): Relative order logits [B, N, N].
        roor_order_logits (np.ndarray): ROOR order logits [B, N, N].
        qi (np.ndarray): Query indices [B, K].
        order_score_thr (float): Score threshold for reading order filtering.

    Returns:
        tuple: (bbox_pred_7, bbox_num, mask_pred)
            - bbox_pred_7 (np.ndarray): [B*K, 7] with reading order appended.
            - bbox_num (np.ndarray): unchanged [B].
            - mask_pred (np.ndarray|None): unchanged.
    """
    B = len(bbox_num)
    offset = 0
    order_list = []

    for b in range(B):
        K = int(bbox_num[b])
        bbox_b = bbox_pred[offset : offset + K]  # [K, 6]
        qi_b = qi[b, :K]
        scores_b = bbox_b[:, 1]  # score column

        # Filter high-confidence queries
        high_mask = scores_b >= order_score_thr
        high_indices = np.where(high_mask)[0]

        if len(high_indices) <= 1:
            seq = np.zeros(K, dtype=np.int64)
        else:
            qi_high = qi_b[high_indices]
            unique_qi, inv_map = np.unique(qi_high, return_inverse=True)

            # Extract sub-matrices
            sub_roor = roor_order_logits[b][unique_qi][:, unique_qi]
            sub_rel = rel_order_logits[b][unique_qi][:, unique_qi]

            # Run DAG decode on the sub-matrix
            sub_pos = _get_order_single_image(sub_roor, sub_rel)

            # Map back to top_k length
            max_rank = len(unique_qi)
            seq = np.full(K, max_rank, dtype=np.int64)
            seq[high_indices] = sub_pos[inv_map]

        order_list.append(seq)
        offset += K

    # Concatenate order as 7th column
    all_order = np.concatenate([s.reshape(-1) for s in order_list])
    bbox_pred_7 = np.concatenate(
        [bbox_pred, all_order.reshape(-1, 1).astype(np.float32)], axis=1
    )

    return bbox_pred_7, bbox_num, mask_pred


def compute_reading_order(boxes, rel_logits, roor_logits, qi):
    """
    Compute reading order on already-filtered boxes (no score threshold needed).

    Args:
        boxes (np.ndarray): Detection boxes [K, 6], columns: [label, score, x1, y1, x2, y2].
        rel_logits (np.ndarray): Relative order logits [N, N].
        roor_logits (np.ndarray): ROOR order logits [N, N].
        qi (np.ndarray): Query indices [K].

    Returns:
        np.ndarray: Reading order ranks [K], 0-based.
    """
    K = len(boxes)
    if K <= 1:
        return np.zeros(K, dtype=np.int64)

    qi_high = qi[:K]
    unique_qi, inv_map = np.unique(qi_high, return_inverse=True)
    sub_roor = roor_logits[unique_qi][:, unique_qi]
    sub_rel = rel_logits[unique_qi][:, unique_qi]
    sub_pos = _get_order_single_image(sub_roor, sub_rel)
    return sub_pos[inv_map]
