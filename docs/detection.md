# Detection

`<onnx/detect>`: SSD-style post-processing (anchors, box decoding, NMS) that matches numpy bit for bit. Every function and type is in the [API reference](api.md).

`<onnx/detect>` is the post-processing of SSD-style detectors, with no model built in: the
anchors come from the model's configuration. Each step is computed with the same float32
operations, in the same order, as the Python code such models ship with, so decoded boxes
and the boxes NMS keeps are the same bits (`tests/detect.esk` against numpy, from
`tools/gen_detect_ref.py`).

| Function | |
|---|---|
| `onnx_anchors_grid` | appends one feature map's anchors (`OnnxAnchor`: `cx`, `cy`, `w`, `h`): per row, per column, one per shape, centered at `((column + 0.5) * step / size_x, (row + 0.5) * step / size_y)` |
| `onnx_anchors_clip` | clamps anchors into 0..1 (PriorBox's `clip`) |
| `onnx_linspace` | `numpy.linspace`, for anchor scales |
| `onnx_boxes_decode` | encodings against their anchors into `OnnxBox` corners, with an `OnnxBoxCoder` |
| `onnx_boxes_clamp` | clamps boxes into `0..w` and `0..h` (`np.clip`) |
| `onnx_scores_above` | the indices whose score is at least a threshold (`np.flatnonzero(scores >= t)`), with a stride for `[N, classes]` scores |
| `onnx_nms` | greedy non-maximum suppression over candidate indices; the kept indices, best first |
| `onnx_box_iou` | intersection over union, as `onnx_nms` computes it |

`onnx_coder_tf()` is TensorFlow's SSD encoding: `(ty, tx, th, tw)` divided by 10, 10, 5, 5,
corners `center -+ size / 2`. `onnx_coder_variance(0.1, 0.2)` is the PyTorch SSD and
RetinaFace one: `(tx, ty, tw, th)` times the variances, `x2 = x1 + w`. The exponential is the
C library's `expf`, which numpy also calls on macOS arm64; numpy on x86-64 CPUs with AVX2 or
AVX-512 has its own float32 `exp`, which rounds some values one unit in the last place
differently, so a decoded box can differ in its last bit there.

`onnx_nms` orders candidates as `np.argsort(scores, kind="stable")[::-1]`: higher scores
first, equal scores the later candidate first, a NaN score before any number. numpy's
default `argsort` agrees whenever no two scores are equal, and for up to 16 candidates.

The face detector of the face-pasive-liveness PAD service (`face_engines.py`) maps onto it
like this: anchors over feature maps 32, 16, 8, 4, 2, 1 with scales
`np.linspace(0.2, 0.95, 6)`, three shapes in the first layer and six in the others; boxes
decoded with `onnx_coder_tf()`, clipped to 0..1; the face scores (column 1 of `[N, 2]`) at
or above the threshold; NMS at 0.3.

```eskiu
List<OnnxAnchor> anchors;
int[6] maps = {32, 16, 8, 4, 2, 1};
double[6] scales;
onnx_linspace(&scales[0], 0.2, 0.95, 6);
for (l in 0..6) {
    double[6] hs;                                   // heights and widths, as (h, w) pairs
    double[6] ws;
    int n = 0;
    if (l == 0) {
        hs[0] = 0.1;              ws[0] = 0.1;
        hs[1] = 0.2 / sqrt(2.0);  ws[1] = 0.2 * sqrt(2.0);
        hs[2] = 0.2 * sqrt(2.0);  ws[2] = 0.2 / sqrt(2.0);
        n = 3;
    } else {
        double s = scales[l];
        double next = l + 1 < 6 ? scales[l + 1] : 1.0;
        hs[0] = s;                ws[0] = s;
        hs[1] = sqrt(s * next);   ws[1] = hs[1];
        double[4] ratios = {2.0, 0.5, 3.0, 1.0 / 3.0};
        for (k in 0..4) {
            hs[k + 2] = s / sqrt(ratios[k]);
            ws[k + 2] = s * sqrt(ratios[k]);
        }
        n = 6;
    }
    double m = (double)maps[l];
    onnx_anchors_grid(&anchors, maps[l], maps[l], 1.0, m, m, &ws[0], &hs[0], n);   // 5118 in all
}

// After the run: encodings [1, 5118, 4] and scores [1, 5118, 2].
*OnnxBox boxes = alloc<OnnxBox>(anchors.size);
OnnxBoxCoder coder = onnx_coder_tf();
onnx_boxes_decode(enc.f32(), anchors.data, anchors.size, &coder, boxes);
onnx_boxes_clamp(boxes, anchors.size, 1.0, 1.0);
List<int> found;
onnx_scores_above(scores.f32() + 1, anchors.size, 2, 0.5, &found);
// (the service drops implausible boxes from found here, by size and aspect)
List<int> keep;
onnx_nms(boxes, scores.f32() + 1, 2, found.data, found.size, 0.3, &keep);
```

RetinaFace's PriorBox is `onnx_anchors_grid` per feature map with the layer's step, the
input size as `size_x` and `size_y`, and `min_size / width`, `min_size / height` as the
shape, then `onnx_anchors_clip` when `clip` is set.
