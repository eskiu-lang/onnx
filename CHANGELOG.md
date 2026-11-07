# Changelog

All notable changes to this package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-11-07

### Added

- `<onnx/image>`: `onnx_image_resize_pillow` and `onnx_image_resize_pillow_box`, Pillow's
  `Image.resize` byte for byte (BILINEAR, BICUBIC, LANCZOS, BOX and HAMMING, with an
  optional source box), in Pillow 11 and 12's pass order or, with `ONNX_PILLOW_13`,
  Pillow 13's.
- `<onnx/image>`: `onnx_image_resize_cv2`, OpenCV's `cv2.resize` with `INTER_LINEAR` on
  8-bit images byte for byte (OpenCV's own code; see the README for the arm64 builds
  where KleidiCV or Carotene replace it).
- `<onnx/image>`: `onnx_image_crop_pillow` (Pillow's `crop` with a float box, rounded half
  to even) and `onnx_round_even`.
- `<onnx/image>`: `onnx_minifasnet_box` and `onnx_minifasnet_crop`, MiniFASNet's crop rule
  and input patch, with `OnnxRect`.
- `<onnx/detect>`: SSD post-processing: `onnx_anchors_grid`, `onnx_anchors_clip`,
  `onnx_linspace`, `onnx_boxes_decode` with `OnnxBoxCoder` (`onnx_coder_tf`,
  `onnx_coder_variance`), `onnx_boxes_clamp`, `onnx_scores_above`, `onnx_nms` and
  `onnx_box_iou`, bit for bit as numpy computes them.
- `tools/gen_image_ref.py` and `tools/gen_detect_ref.py`, and the tests `image_exact` and
  `detect` that compare against their Pillow, OpenCV and numpy outputs.

## [0.1.0]

### Added

- First release: sessions, introspection, tensors, providers, `<onnx/image>` and
  `<onnx/raw>`.
