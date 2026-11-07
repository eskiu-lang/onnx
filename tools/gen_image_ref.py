#!/usr/bin/env python3
"""Generate the reference outputs tests/image_exact.esk compares <onnx/image> against.

Usage:
    python3 -I tools/gen_image_ref.py pillow    # tests/data/pillow<MAJOR>.bin, with the installed Pillow
    python3 -I tools/gen_image_ref.py cv2       # tests/data/cv2.bin, with OpenCV and numpy

Run the pillow mode once with Pillow 11 (the pass order of 11 and 12) and once with
Pillow 13 (which may run the vertical pass first). The checked-in files come from:

    pillow11.bin  Pillow 11.3.0, numpy 2.0.2, Python 3.9
    pillow13.bin  Pillow 13.0.0, numpy 2.5.4, Python 3.15
    cv2.bin       opencv-python-headless 5.0.0, numpy 2.5.4, Python 3.15

all on macOS arm64. The source images are made here from a fixed integer generator that
the test repeats, so only the outputs are stored. Each record is little-endian:

    int32 kind, int32 source, int32 n, int32[n], int32 m, float64[m], int32 ow, int32 oh, uint8[ow * oh * 3]

kind 1: Pillow resize, ints (filter, nw, nh), doubles the box (float32 values)
kind 2: Pillow crop with a float box, doubles the box
kind 3: cv2.resize INTER_LINEAR, ints (nw, nh)
kind 4: the MiniFASNet crop rule alone, ints (src_w, src_h, x1, y1, x2, y2), doubles (x, y, w, h, scale)
kind 5: the MiniFASNet patch (crop rule + cv2.resize), ints (out_w, out_h), doubles (x, y, w, h, scale)
"""
import os
import random
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "data")

# (kind, width, height, seed): kind 0 is noise, 1 a gradient with a little noise.
SOURCES = [(0, 37, 23, 1), (1, 64, 48, 2), (0, 9, 7, 3), (1, 10, 8, 4)]


def source(i):
    kind, w, h, seed = SOURCES[i]
    s = seed
    out = bytearray()
    for y in range(h):
        for x in range(w):
            for c in range(3):
                s = (s * 1103515245 + 12345) & 0xFFFFFFFF
                n = (s >> 16) & 255
                if kind == 0:
                    out.append(n)
                else:
                    out.append((x * 200 // w + y * 50 // h + c * 30 + (n >> 3)) & 255)
    return w, h, bytes(out)


def record(f, kind, src, ints, doubles, ow, oh, pixels):
    f.write(struct.pack("<3i", kind, src, len(ints)))
    f.write(struct.pack("<%di" % len(ints), *ints))
    f.write(struct.pack("<i", len(doubles)))
    f.write(struct.pack("<%dd" % len(doubles), *doubles))
    f.write(struct.pack("<2i", ow, oh))
    assert len(pixels) == ow * oh * 3
    f.write(pixels)


def f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


def pillow():
    import PIL
    from PIL import Image

    major = int(PIL.__version__.split(".")[0])
    images = [Image.frombytes("RGB", source(i)[:2], source(i)[2]) for i in range(len(SOURCES))]
    # (source, nw, nh, box): shrinking and enlarging, odd sizes, other aspect ratios.
    cases = [
        (0, 24, 15, None), (2, 20, 17, None), (1, 21, 16, None), (1, 11, 30, None),
        (0, 7, 19, None), (2, 3, 11, None), (1, 17, 13, (3.25, 5.5, 50.75, 40.125)),
        (0, 9, 9, (0.3, 0.0, 37.0, 22.9)),
    ]
    # Shapes whose height shrinks by more than twice as many pixels as the width, where
    # Pillow 13 runs the vertical pass first.
    tall = [(1, 60, 10, None), (0, 35, 5, None)]
    if major >= 13:
        cases = tall + [(1, 21, 16, None)]
    else:
        cases = cases + tall
    filters = [2, 3, 1, 4, 5]
    path = os.path.join(DATA, "pillow%d.bin" % major)
    with open(path, "wb") as f:
        for flt in filters:
            for src, nw, nh, box in cases:
                # Lanczos, box and Hamming share the machinery: three shapes each.
                if flt not in (2, 3) and (src, nw, nh) not in ((0, 24, 15), (1, 21, 16), (1, 60, 10)):
                    continue
                im = images[src]
                b = box if box is not None else (0.0, 0.0, float(im.width), float(im.height))
                out = im.resize((nw, nh), resample=flt, box=box)
                record(f, 1, src, [flt, nw, nh], [f32(v) for v in b], nw, nh, out.tobytes())
        if major < 13:
            # Float boxes round with Python's round(): halves go to the even integer.
            for src, box in [(0, (2.5, 3.5, 20.5, 15.5)), (0, (-3.5, -0.5, 10.5, 30.5)),
                             (1, (0.49, 1.51, 63.5, 47.5)), (2, (1.5, 2.5, 1.5, 6.5))]:
                out = images[src].crop(box)
                record(f, 2, src, [], list(box), out.width, out.height, out.tobytes())
    print("wrote", path, os.path.getsize(path), "bytes, Pillow", PIL.__version__)


def kleidicv_takes(w, h, nw, nh, cn=3):
    """KleidiCV's resize_linear_u8_is_implemented (26.03) for cn <= 3: the cases arm64
    OpenCV builds with KleidiCV compute with an approximation instead of OpenCV's code."""
    if cn <= 3 and nw * 3 >= w and nw < w and nh < h:
        if nw * cn < 8:
            return False
        if nw * 2 >= w and cn != 3:
            return w * cn >= 16
        return w * cn >= 32
    return False


def cv_port(img, nw, nh):
    """OpenCV's own INTER_LINEAR on 8 bits (resize.cpp, resizeGeneric_ with HResizeLinear
    and the VResizeLinear specialization), the arithmetic onnx_image_resize_cv2 repeats."""
    import math
    import numpy as np

    h, w = img.shape[:2]

    def axis(n, nn, x):
        scale = 1.0 / (nn / n)
        ofs, a0, a1 = [], [], []
        for i in range(nn):
            f = np.float32((i + 0.5) * scale - 0.5)
            s = math.floor(f)
            f = np.float32(f - np.float32(s))
            if x and s < 0:
                s, f = 0, np.float32(0)
            if x and s >= n - 1:
                s, f = n - 1, np.float32(0)
            ofs.append(s)
            a0.append(int(np.rint(np.float32(np.float32(1) - f) * np.float32(2048))))
            a1.append(int(np.rint(f * np.float32(2048))))
        return ofs, a0, a1

    xo, xa, xb = axis(w, nw, True)
    yo, ya, yb = axis(h, nh, False)
    s = img.astype(np.int64)
    rows = np.stack([s[:, xo[i]] * xa[i] + s[:, min(xo[i] + 1, w - 1)] * xb[i] for i in range(nw)], 1)
    out = np.zeros((nh, nw) + img.shape[2:], np.int64)
    for j in range(nh):
        r0 = min(max(yo[j], 0), h - 1)
        r1 = min(max(yo[j] + 1, 0), h - 1)
        out[j] = ((((rows[r0] >> 4) * ya[j]) >> 16) + (((rows[r1] >> 4) * yb[j]) >> 16) + 2) >> 2
    return np.clip(out, 0, 255).astype(np.uint8)


def get_new_box(src_w, src_h, bbox, scale):
    # Minivision's CropImage._get_new_box (MiniFASNet src/generate_patches.py).
    x, y, box_w, box_h = bbox
    scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
    new_width = box_w * scale
    new_height = box_h * scale
    center_x, center_y = box_w / 2 + x, box_h / 2 + y
    left_top_x = center_x - new_width / 2
    left_top_y = center_y - new_height / 2
    right_bottom_x = center_x + new_width / 2
    right_bottom_y = center_y + new_height / 2
    if left_top_x < 0:
        right_bottom_x -= left_top_x
        left_top_x = 0
    if left_top_y < 0:
        right_bottom_y -= left_top_y
        left_top_y = 0
    if right_bottom_x > src_w - 1:
        left_top_x -= right_bottom_x - src_w + 1
        right_bottom_x = src_w - 1
    if right_bottom_y > src_h - 1:
        left_top_y -= right_bottom_y - src_h + 1
        right_bottom_y = src_h - 1
    return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)


def opencv():
    import cv2
    import numpy as np

    arrays = []
    for i in range(len(SOURCES)):
        w, h, px = source(i)
        arrays.append(np.frombuffer(px, np.uint8).reshape(h, w, 3))
    # (source, nw, nh). None is one KleidiCV takes over (see kleidicv_takes): enlarging,
    # one side only, more than 3 times smaller, or too narrow for its vectors.
    cases = [(2, 32, 30), (0, 45, 26), (0, 16, 23), (1, 64, 20), (1, 15, 11), (2, 4, 3),
             (3, 5, 4), (0, 11, 40), (1, 80, 9), (2, 9, 7), (0, 1, 1), (2, 2, 30)]
    path = os.path.join(DATA, "cv2.bin")
    with open(path, "wb") as f:
        for src, nw, nh in cases:
            img = arrays[src]
            h, w = img.shape[:2]
            assert not kleidicv_takes(w, h, nw, nh), (src, nw, nh)
            out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            if (nw, nh) != (w, h):
                assert np.array_equal(out, cv_port(img, nw, nh)), (src, nw, nh)
            record(f, 3, src, [nw, nh], [], nw, nh, out.tobytes())
        # The crop rule: random boxes and the edges (a box at a border, one larger than
        # the image allows, the scale clamped by width or by height).
        rng = random.Random(7)
        boxes = [(640, 480, (300, 200, 120, 150), 2.7), (640, 480, (0, 0, 100, 120), 4.0),
                 (640, 480, (590, 400, 50, 80), 2.7), (640, 480, (10, 10, 300, 400), 4.0),
                 (100, 300, (20, 100, 60, 50), 4.0), (300, 100, (100, 20, 50, 60), 4.0),
                 (64, 48, (60, 44, 4, 4), 2.7), (64, 48, (-5, -3, 20, 20), 2.7),
                 (1280, 720, (639, 359, 1, 1), 2.7), (99, 77, (0, 0, 98, 76), 1.0)]
        for _ in range(40):
            sw = rng.randint(40, 1920)
            sh = rng.randint(40, 1080)
            bw = rng.randint(1, sw)
            bh = rng.randint(1, sh)
            boxes.append((sw, sh, (rng.randint(-bw // 2, sw - 1), rng.randint(-bh // 2, sh - 1), bw, bh),
                          rng.choice([2.7, 4.0, 1.0, 0.8, 1.5])))
        for sw, sh, bbox, scale in boxes:
            r = get_new_box(sw, sh, bbox, scale)
            record(f, 4, -1, [sw, sh] + list(r), [float(v) for v in bbox] + [scale], 0, 0, b"")
        # Whole patches from the 64x48 source.
        for bbox, scale, ow, oh in [((20, 15, 12, 14), 2.7, 36, 36), ((30, 20, 20, 22), 4.0, 48, 48),
                                    ((50, 30, 10, 12), 2.7, 33, 37), ((5, 5, 30, 30), 4.0, 12, 12)]:
            x0, y0, x1, y1 = get_new_box(64, 48, bbox, scale)
            crop = arrays[1][y0:y1 + 1, x0:x1 + 1]
            assert not kleidicv_takes(crop.shape[1], crop.shape[0], ow, oh)
            out = cv2.resize(crop, (ow, oh))
            record(f, 5, 1, [ow, oh], [float(v) for v in bbox] + [scale], ow, oh, out.tobytes())
    print("wrote", path, os.path.getsize(path), "bytes, OpenCV", cv2.__version__, "numpy", np.__version__)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("pillow", "cv2"):
        sys.exit(__doc__)
    pillow() if sys.argv[1] == "pillow" else opencv()
