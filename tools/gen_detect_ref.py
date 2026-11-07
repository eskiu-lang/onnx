#!/usr/bin/env python3
"""Generate tests/data/detect.bin, the numpy reference tests/detect.esk checks
<onnx/detect> against.

Usage: python3 -I tools/gen_detect_ref.py
Needs numpy. The checked-in file comes from numpy 2.5.4 (Python 3.15, macOS arm64).

The math follows two detectors' Python post-processing, written here from their formulas
in float32 as numpy evaluates them:

- An SSD in TensorFlow's convention (the face detector of the face-pasive-liveness
  service, face_liveness/face_engines.py): 5118 anchors (cy, cx, h, w) over feature maps
  32, 16, 8, 4, 2, 1 with scales np.linspace(0.2, 0.95, 6); encodings (ty, tx, th, tw)
  divided by 10, 10, 5, 5; boxes clipped to 0..1; scores >= threshold; greedy NMS.
- RetinaFace's PriorBox and decode (Pytorch_Retinaface): priors (cx, cy, w, h) from
  feature map sizes, steps and min sizes, clipped; locations times the variances 0.1, 0.2
  (in float32 as torch computes them on the CPU).

NMS orders by score with np.argsort(scores, kind="stable")[::-1]: higher scores first,
equal scores the later candidate first, NaN before everything.

Layout (little-endian):
    uint64 FNV-1a of the SSD anchors as float32 (cy, cx, h, w), int32 their count
    int32 n, float32[n][4] RetinaFace priors (cx, cy, w, h)
    int32 n, float32[n][4] SSD encodings, float32[n][4] decoded boxes (x1, y1, x2, y2),
        float32[n][2] scores (background, face); encoding i goes with anchor (i * 997) % 5118
    int32 n, float32[n][4] RetinaFace locations, float32[n][4] decoded boxes
    then NMS cases: int32 threshold mode (1: candidates are scores >= 0.5), float32 iou,
        int32 k, int32[k] kept indices
"""
import math
import os
import struct

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "tests", "data", "detect.bin")


def ssd_anchors():
    feature_maps = (32, 16, 8, 4, 2, 1)
    scales = np.linspace(0.2, 0.95, len(feature_maps))
    anchors = []
    for layer, fm in enumerate(feature_maps):
        if layer == 0:
            dims = ((0.1, 0.1), (0.2 / np.sqrt(2), 0.2 * np.sqrt(2)), (0.2 * np.sqrt(2), 0.2 / np.sqrt(2)))
        else:
            scale = scales[layer]
            nxt = scales[layer + 1] if layer + 1 < len(scales) else 1.0
            dims = [(scale, scale), (np.sqrt(scale * nxt), np.sqrt(scale * nxt))]
            for ratio in (2.0, 0.5, 3.0, 1.0 / 3.0):
                dims.append((scale / np.sqrt(ratio), scale * np.sqrt(ratio)))
        for row in range(fm):
            for col in range(fm):
                for h, w in dims:
                    anchors.append(((row + 0.5) / fm, (col + 0.5) / fm, h, w))
    return np.asarray(anchors, dtype=np.float32)


def retina_priors(image_size=(48, 64), steps=(8, 16, 32), min_sizes=((8, 16), (32,), (64, 96))):
    anchors = []
    for k, step in enumerate(steps):
        fm = (math.ceil(image_size[0] / step), math.ceil(image_size[1] / step))
        for i in range(fm[0]):
            for j in range(fm[1]):
                for m in min_sizes[k]:
                    anchors += [(j + 0.5) * step / image_size[1], (i + 0.5) * step / image_size[0],
                                m / image_size[1], m / image_size[0]]
    return np.clip(np.asarray(anchors, dtype=np.float32).reshape(-1, 4), 0.0, 1.0)


def decode_ssd(enc, anchors):
    # face_engines.decode_ssd_boxes
    cy = enc[:, 0] / 10.0 * anchors[:, 2] + anchors[:, 0]
    cx = enc[:, 1] / 10.0 * anchors[:, 3] + anchors[:, 1]
    h = np.exp(enc[:, 2] / 5.0) * anchors[:, 2]
    w = np.exp(enc[:, 3] / 5.0) * anchors[:, 3]
    return np.column_stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))


def decode_retina(loc, priors, variances=(0.1, 0.2)):
    # Pytorch_Retinaface's box_utils.decode, in float32.
    v0 = np.float32(variances[0])
    v1 = np.float32(variances[1])
    centers = priors[:, :2] + loc[:, :2] * v0 * priors[:, 2:]
    sizes = priors[:, 2:] * np.exp(loc[:, 2:] * v1)
    mins = centers - sizes / 2
    return np.column_stack((mins, sizes + mins))


def nms(boxes, scores, threshold):
    # face_engines.non_maximum_suppression with a stable sort.
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores, kind="stable")[::-1]
    keep = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        ix1 = np.maximum(x1[current], x1[order[1:]])
        iy1 = np.maximum(y1[current], y1[order[1:]])
        ix2 = np.minimum(x2[current], x2[order[1:]])
        iy2 = np.minimum(y2[current], y2[order[1:]])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        union = areas[current] + areas[order[1:]] - inter
        overlap = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = order[1:][overlap <= np.float32(threshold)]
    return keep


def portable(values, arg):
    """Redraws the values whose exp argument has a float32 exp that is not the correctly
    rounded one. numpy's exp on macOS arm64 is the C library's expf, which rounds about
    one value in 160 the other way; other C libraries (glibc, mingw) round those
    differently, so the test keeps to values every expf gives alike."""
    rng = np.random.default_rng(5)
    for i in np.ndindex(values.shape):
        while True:
            a = arg(values[i].reshape(1))
            if not np.isfinite(a[0]) or np.exp(a)[0] == np.float32(math.exp(float(a[0]))):
                break
            values[i] = np.float32(rng.standard_normal() * 2.0)


def fnv(data):
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def main():
    rng = np.random.default_rng(11)
    out = bytearray()
    anchors = ssd_anchors()
    assert anchors.shape == (5118, 4)
    out += struct.pack("<Qi", fnv(anchors.tobytes()), len(anchors))
    priors = retina_priors()
    out += struct.pack("<i", len(priors)) + priors.tobytes()

    n = 256
    enc = (rng.standard_normal((n, 4)) * 2.0).astype(np.float32)
    enc[:8] = np.array([[0, 0, 0, 0], [30, -30, 60, -60], [-1e3, 1e3, 1e3, -1e3], [0.5, 0.5, 450, 450],
                        [0, 0, -450, 1], [1e-7, -1e-7, 1e-7, -1e-7], [5, 5, 5, 5], [-5, -5, -5, -5]], np.float32)
    portable(enc[8:, 2:], lambda v: v / 5.0)
    idx = np.array([(i * 997) % 5118 for i in range(n)])
    with np.errstate(over="ignore", invalid="ignore"):
        boxes = decode_ssd(enc, anchors[idx]).astype(np.float32)
    scores = np.zeros((n, 2), np.float32)
    scores[:, 1] = np.floor(rng.random(n) * 32) / 32
    scores[:, 1][rng.random(n) < 0.3] = 0.75
    scores[17, 1] = np.nan
    scores[:, 0] = 1 - scores[:, 1]
    out += struct.pack("<i", n) + enc.tobytes() + boxes.tobytes() + scores.tobytes()

    loc = (rng.standard_normal((len(priors), 4)) * 1.5).astype(np.float32)
    portable(loc[:, 2:], lambda v: v * np.float32(0.2))
    rboxes = decode_retina(loc, priors).astype(np.float32)
    out += struct.pack("<i", len(priors)) + loc.tobytes() + rboxes.tobytes()

    with np.errstate(invalid="ignore"):
        clipped = np.clip(boxes, 0.0, 1.0)
    face = scores[:, 1]
    for mode, iou in [(1, 0.3), (1, 0.5), (0, 0.45), (0, 0.0), (1, 1.0)]:
        with np.errstate(invalid="ignore"):
            cand = np.flatnonzero(face >= 0.5) if mode == 1 else np.arange(n)
            kept = [int(cand[k]) for k in nms(clipped[cand], face[cand], iou)]
        out += struct.pack("<ifi", mode, iou, len(kept)) + struct.pack("<%di" % len(kept), *kept)

    with open(PATH, "wb") as f:
        f.write(out)
    print("wrote", PATH, len(out), "bytes, numpy", np.__version__)


if __name__ == "__main__":
    main()
