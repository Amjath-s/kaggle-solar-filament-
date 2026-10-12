"""V62: make Ultralytics' segmentation validator work when mask prototypes are not at stride 4.

ultralytics 8.4.138 `SegmentationValidator.postprocess` hard-codes `imgsz = 4 * proto_size` and
`_prepare_batch` hard-codes `mask_size = imgsz // 4`. In V62 `p2first` the prototypes are at
stride 2 (512x512 for a 1024 input), so validation computes GT masks at 256 and predictions at 512
and crashes in `mask_iou` (mat1 and mat2 shapes cannot be multiplied). The trainer runs a final
validation even with val=False, BEFORE saving that epoch's checkpoint, so a full run would die at
the last epoch.

`install(factor)` replaces those two methods with copies that use `factor` (= input / proto size):
factor 4 for the stock graph and `p2last`, 2 for `p2first`. For factor 4 the replacement is
behaviourally identical to upstream (tested against the original on a toy batch). Validation is
not used for any model selection here (fixed-duration final checkpoints); this only prevents the
crash and keeps the logged metrics meaningful.
"""
from __future__ import annotations

_ORIG = {}


def install(factor: int = 4) -> int:
    import torch.nn.functional as F
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.models.yolo.segment import val as V
    from ultralytics.utils import ops

    cls = V.SegmentationValidator
    _ORIG.setdefault("postprocess", cls.postprocess)
    _ORIG.setdefault("_prepare_batch", cls._prepare_batch)

    def postprocess(self, preds):
        proto = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        preds = DetectionValidator.postprocess(self, preds[0])
        imgsz = [factor * x for x in proto.shape[2:]]
        for i, pred in enumerate(preds):
            coefficient = pred.pop("extra")
            pred["masks"] = self.process(proto[i], coefficient, pred["bboxes"], shape=imgsz)
        return preds

    def _prepare_batch(self, si, batch):
        import torch

        prepared = DetectionValidator._prepare_batch(self, si, batch)
        nl = prepared["cls"].shape[0]
        if self.args.overlap_mask:
            masks = batch["masks"][si]
            index = torch.arange(1, nl + 1, device=masks.device).view(nl, 1, 1)
            masks = (masks == index).float()
        else:
            masks = batch["masks"][batch["batch_idx"] == si]
        if nl:
            mask_size = [s if self.process is ops.process_mask_native else s // factor
                         for s in prepared["imgsz"]]
            if list(masks.shape[1:]) != list(mask_size):
                masks = F.interpolate(masks[None], mask_size, mode="bilinear",
                                      align_corners=False)[0]
                masks = masks.gt_(0.5)
        prepared["masks"] = masks
        return prepared

    cls.postprocess = postprocess
    cls._prepare_batch = _prepare_batch
    return factor


def uninstall() -> None:
    from ultralytics.models.yolo.segment import val as V

    for name, fn in _ORIG.items():
        setattr(V.SegmentationValidator, name, fn)
