"""V62 architecture: extend YOLO11s-seg with a P2/4 head level (two variants) + probes.

Base graph is EXACTLY ultralytics yolo11-seg.yaml (layers 0-22 keep their indices, so backbone and
neck weights of yolo11s-seg.pt transfer unchanged). New layers are APPENDED after layer 22:

  23  Upsample(layer 16 = head P3/8)            -> stride 4
  24  Concat(23, layer 2 = backbone P2/4 C3k2)   -> 128 ch at s-scale
  25  C3k2 [128]  "P2/4-xsmall" fusion block      -> 64 ch at s-scale
  26  Segment over
        p2first : [25, 16, 19, 22]  -> x[0] = P2  => Proto reads P2: mask prototypes at stride 2
        p2last  : [16, 19, 22, 25]  -> x[0] = P3  => Proto unchanged (stride 4); P2 only adds a
                                                    detection/coefficient scale

Why both: the Segment head builds its prototypes from x[0] ONLY (ultralytics 8.4.138
Segment.forward: `proto = self.proto(x[0])`), and cv2/cv3/cv4 are per-input ModuleLists. So
input order decides (a) whether the prototype pathway sees the high-resolution feature and
(b) which COCO head weights can transfer. p2last isolates "detection at stride 4"; p2first
adds "prototypes at stride 2".

  python scripts/v62_arch.py write    # configs/yolo11s-p2first-seg.yaml, -p2last-
  CUDA_VISIBLE_DEVICES=0 python scripts/v62_arch.py probe
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
BASE_HEAD_SEGMENT = "  - [[16, 19, 22], 1, Segment, [nc, 32, 256]] # Segment(P3, P4, P5)"
NEW_LAYERS = """  - [16, 1, nn.Upsample, [None, 2, "nearest"]] # 23: head P3/8 -> stride 4
  - [[-1, 2], 1, Concat, [1]] # 24: cat backbone P2/4
  - [-1, 2, C3k2, [128, False]] # 25: P2/4-xsmall fusion
"""
SEGMENT = {
    "p2first": "  - [[25, 16, 19, 22], 1, Segment, [nc, 32, 256]] # Segment(P2, P3, P4, P5)\n",
    "p2last": "  - [[16, 19, 22, 25], 1, Segment, [nc, 32, 256]] # Segment(P3, P4, P5, P2)\n",
}


def build_yaml(base_text: str, variant: str) -> str:
    """Base YOLO11-seg YAML text -> variant YAML text (pure string transform, tested)."""
    if BASE_HEAD_SEGMENT not in base_text:
        raise ValueError("base yolo11-seg.yaml Segment line not found (Ultralytics changed?)")
    head, _, tail = base_text.partition(BASE_HEAD_SEGMENT)
    return head + NEW_LAYERS + "\n" + SEGMENT[variant] + tail.lstrip("\n")


def base_yaml_path() -> Path:
    import ultralytics
    return Path(ultralytics.__file__).parent / "cfg/models/11/yolo11-seg.yaml"


def write_all() -> list[Path]:
    text = base_yaml_path().read_text()
    CONFIGS.mkdir(exist_ok=True)
    out = []
    for v in SEGMENT:
        p = CONFIGS / f"yolo11s-{v}-seg.yaml"   # "11s" in stem => scale s
        p.write_text(build_yaml(text, v))
        out.append(p)
    return out


def probe() -> None:
    import torch
    from ultralytics import YOLO

    base = YOLO("yolo11s-seg.pt")
    sd_base = base.model.state_dict()
    rows = []
    for name, cfg in (("baseline", None), ("p2last", CONFIGS / "yolo11s-p2last-seg.yaml"),
                      ("p2first", CONFIGS / "yolo11s-p2first-seg.yaml")):
        model = YOLO("yolo11s-seg.pt") if cfg is None else YOLO(str(cfg)).load("yolo11s-seg.pt")
        net = model.model
        m = net.model[-1]
        n_params = sum(p.numel() for p in net.parameters())
        strides = [float(s) for s in m.stride]
        x = torch.zeros(1, 3, 1024, 1024)
        net.eval()
        with torch.no_grad():
            net(x)
        feats = None
        net.train()
        with torch.no_grad():
            tr = net(x)
        feats = tr["feats"] if isinstance(tr, dict) else None
        proto = tr["proto"] if isinstance(tr, dict) and "proto" in tr else None
        anchors = sum(f.shape[-1] * f.shape[-2] for f in feats) if feats else None
        # how many pretrained tensors transfer (same key AND shape)
        sd = net.state_dict()
        same = sum(1 for k, v in sd.items() if k in sd_base and sd_base[k].shape == v.shape)
        rows.append({"name": name, "params_M": n_params / 1e6, "strides": strides,
                     "anchors@1024": anchors,
                     "proto_shape": tuple(proto.shape) if proto is not None else None,
                     "tensors": len(sd), "same_key_shape_as_coco": same})
        del model, net
    for r in rows:
        print(r)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "write":
        for p in write_all():
            print("wrote", p)
    elif len(sys.argv) > 1 and sys.argv[1] == "probe":
        probe()
    else:
        print(__doc__)
