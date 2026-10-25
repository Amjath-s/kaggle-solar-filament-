"""V12 consensus 4-flip TTA for YOLO-seg (friend V17-style).

Merge rule: keep a detection only if it has IoU>=match_iou with a detection
from at least one other flip view. Then score-order NMS at nms_iou.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

FLIP_VIEWS = ("original", "horizontal", "vertical", "horizontal_vertical")


def flip_image(arr: np.ndarray, view: str) -> np.ndarray:
    if view == "original":
        return arr
    if view == "horizontal":
        return np.ascontiguousarray(arr[:, ::-1])
    if view == "vertical":
        return np.ascontiguousarray(arr[::-1, :])
    if view == "horizontal_vertical":
        return np.ascontiguousarray(arr[::-1, ::-1])
    raise ValueError(view)


def inverse_flip_boxes(boxes: np.ndarray, width: int, height: int, view: str) -> np.ndarray:
    mapped = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    if len(mapped) == 0:
        return mapped
    if view in ("horizontal", "horizontal_vertical"):
        x1 = width - mapped[:, 2]
        x2 = width - mapped[:, 0]
        mapped[:, 0], mapped[:, 2] = x1, x2
    if view in ("vertical", "horizontal_vertical"):
        y1 = height - mapped[:, 3]
        y2 = height - mapped[:, 1]
        mapped[:, 1], mapped[:, 3] = y1, y2
    mapped[:, [0, 2]] = np.clip(mapped[:, [0, 2]], 0, width)
    mapped[:, [1, 3]] = np.clip(mapped[:, [1, 3]], 0, height)
    return mapped


def inverse_flip_mask(mask: np.ndarray, view: str) -> np.ndarray:
    return flip_image(mask, view)  # same transforms are involutions


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in a (N,4) vs every box in b (M,4) -> (N,M)."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    area_a = np.maximum(ax2 - ax1, 0) * np.maximum(ay2 - ay1, 0)
    area_b = np.maximum(bx2 - bx1, 0) * np.maximum(by2 - by1, 0)
    return (inter / np.maximum(area_a + area_b - inter, 1e-6)).astype(np.float32)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return (mask > 0).astype(np.uint8)
    struct = ndi.generate_binary_structure(2, 1)
    return ndi.binary_dilation(mask > 0, structure=struct, iterations=int(radius)).astype(np.uint8)


def consensus_merge(
    view_boxes: dict[str, np.ndarray],
    view_confs: dict[str, np.ndarray],
    view_masks: dict[str, list[np.ndarray]],
    *,
    match_iou: float = 0.5,
    nms_iou: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Keep dets that match another view; then score-order NMS."""
    # Flatten all candidates with view tags
    boxes_list: list[np.ndarray] = []
    confs_list: list[float] = []
    masks_list: list[np.ndarray] = []
    views_list: list[str] = []

    for view in FLIP_VIEWS:
        boxes = np.asarray(view_boxes.get(view, np.zeros((0, 4))), dtype=np.float32).reshape(-1, 4)
        confs = np.asarray(view_confs.get(view, np.zeros((0,))), dtype=np.float32).reshape(-1)
        masks = view_masks.get(view, [])
        for i in range(len(boxes)):
            boxes_list.append(boxes[i])
            confs_list.append(float(confs[i]) if i < len(confs) else 0.0)
            masks_list.append(masks[i] if i < len(masks) else np.zeros((1, 1), np.uint8))
            views_list.append(view)

    if not boxes_list:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), []

    boxes = np.stack(boxes_list, axis=0)
    confs = np.asarray(confs_list, dtype=np.float32)
    views = np.asarray(views_list)

    # Consensus: must match at least one candidate from a different view
    iou = box_iou_matrix(boxes, boxes)
    keep_cons: list[int] = []
    for i in range(len(boxes)):
        others = [
            j
            for j in range(len(boxes))
            if j != i and views[j] != views[i] and iou[i, j] >= match_iou
        ]
        if others:
            keep_cons.append(i)

    if not keep_cons:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), []

    boxes = boxes[keep_cons]
    confs = confs[keep_cons]
    masks_list = [masks_list[i] for i in keep_cons]

    # Score-order NMS
    order = np.argsort(-confs, kind="stable")
    kept: list[int] = []
    while order.size:
        chosen = int(order[0])
        kept.append(chosen)
        if order.size == 1:
            break
        remaining = order[1:]
        overlaps = box_iou_matrix(boxes[chosen : chosen + 1], boxes[remaining]).reshape(-1)
        order = remaining[overlaps <= nms_iou]

    out_boxes = boxes[kept].astype(np.float32)
    out_confs = confs[kept].astype(np.float32)
    out_masks = [masks_list[i] for i in kept]
    return out_boxes, out_confs, out_masks


def yolo_seg_predict(
    model,
    image: np.ndarray,
    *,
    imgsz: int,
    conf: float,
    iou: float,
    device,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Run YOLO-seg on HxW uint8 (or float) image; return boxes, confs, full-res masks."""
    result = model.predict(
        source=image,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )[0]
    h, w = image.shape[:2]
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), []

    boxes = result.boxes.xyxy.detach().float().cpu().numpy().astype(np.float32)
    confs = result.boxes.conf.detach().float().cpu().numpy().astype(np.float32)
    masks: list[np.ndarray] = []
    if result.masks is not None and result.masks.data is not None:
        raw = result.masks.data.detach().float().cpu().numpy()
        for i in range(raw.shape[0]):
            m = raw[i]
            if m.shape != (h, w):
                from PIL import Image

                m = np.asarray(
                    Image.fromarray((m > 0.5).astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
                )
                masks.append((m > 127).astype(np.uint8))
            else:
                masks.append((m > 0.5).astype(np.uint8))
    else:
        # Fall back: fill box rectangles (should be rare for -seg weights)
        for box in boxes:
            m = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            m[y1:y2, x1:x2] = 1
            masks.append(m)
    # Align lengths
    n = min(len(boxes), len(confs), len(masks))
    return boxes[:n], confs[:n], masks[:n]


def detect_consensus_4flip(
    model,
    image: np.ndarray,
    *,
    imgsz: int = 1024,
    conf: float = 0.30,
    iou: float = 0.7,
    match_iou: float = 0.5,
    nms_iou: float = 0.5,
    device="cuda",
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Four-flip consensus TTA on one HxW grayscale/RGB array."""
    h, w = image.shape[:2]
    view_boxes: dict[str, np.ndarray] = {}
    view_confs: dict[str, np.ndarray] = {}
    view_masks: dict[str, list[np.ndarray]] = {}

    for view in FLIP_VIEWS:
        flipped = flip_image(image, view)
        boxes, confs, masks = yolo_seg_predict(
            model, flipped, imgsz=imgsz, conf=conf, iou=iou, device=device
        )
        view_boxes[view] = inverse_flip_boxes(boxes, w, h, view)
        view_confs[view] = confs
        view_masks[view] = [inverse_flip_mask(m, view) for m in masks]

    return consensus_merge(
        view_boxes, view_confs, view_masks, match_iou=match_iou, nms_iou=nms_iou
    )


def detect_single(
    model,
    image: np.ndarray,
    *,
    imgsz: int = 1024,
    conf: float = 0.30,
    iou: float = 0.7,
    device="cuda",
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    return yolo_seg_predict(model, image, imgsz=imgsz, conf=conf, iou=iou, device=device)
