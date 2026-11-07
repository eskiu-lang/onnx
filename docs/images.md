# Images

`<onnx/image>`: preparing images for a model's input, including resizers that match Pillow and OpenCV byte for byte. Every function and type is in the [API reference](api.md).

`<onnx/image>` works on 8-bit RGBA pixels (what most decoders give), and returns new
buffers to `free`:

```eskiu
import <onnx>;
import <onnx/image>;
import <gpu/image>;                  // PNG and JPEG decoding, from the gpu package

Bitmap img = image_load("cat.jpg")?;
defer img.free();
int w = (int)img.width;
int h = (int)img.height;
int side = w < h ? w : h;
*uint8 square = onnx_image_crop(img.rgba, w, h, (w - side) / 2, (h - side) / 2, side, side, ONNX_BORDER_CLAMP);
defer free(square);
*uint8 small = onnx_image_resize_area(square, side, side, 224, 224);
defer free(small);
OnnxImageFormat f = onnx_format_imagenet();
OnnxTensor x = onnx_image_tensor(small, 224, 224, &f)?;   // [1, 3, 224, 224]
```

| Function | |
|---|---|
| `onnx_image_crop` | a region that may reach outside the image, which is filled with zeros, the edge pixels or the image reflected about its edge (`ONNX_BORDER_ZERO`, `_CLAMP`, `_REFLECT`) |
| `onnx_image_resize` | bilinear with half-pixel centers, as ONNX `Resize` (linear, `half_pixel`); OpenCV's fixed-point `INTER_LINEAR` can differ by one or two levels (`onnx_image_resize_cv2` does not) |
| `onnx_image_resize_area` | the average of the pixels each output pixel covers, the same as OpenCV's `INTER_AREA` when shrinking |
| `onnx_image_tensor` | a new `[1, 3, h, w]` (or `[1, h, w, 3]`) float tensor |
| `onnx_image_into` | writes into item `i` of a float or float16 batch tensor |
| `onnx_image_resize_pillow` | Pillow's `Image.resize`, byte for byte, with `ONNX_PILLOW_BILINEAR`, `_BICUBIC`, `_LANCZOS`, `_BOX` or `_HAMMING` |
| `onnx_image_resize_pillow_box` | the same over a float source region, as `resize(size, filter, box=...)` |
| `onnx_image_crop_pillow` | Pillow's `Image.crop` with a float box: corners rounded half to even, zeros outside |
| `onnx_image_resize_cv2` | OpenCV's `cv2.resize` with `INTER_LINEAR` on 8-bit images, byte for byte |
| `onnx_minifasnet_box` | MiniFASNet's crop rule (Minivision's `_get_new_box`) as an `OnnxRect` |
| `onnx_minifasnet_crop` | MiniFASNet's input patch: that crop resized by `onnx_image_resize_cv2` |
| `onnx_round_even` | Python's `round()` |

`OnnxImageFormat` says how a pixel becomes a value: `(pixel * scale - mean[c]) / std[c]`,
with `nhwc` for the layout and `bgr` for the channel order (mean and std follow the
tensor's channel order). `onnx_format_unit()` gives 0..1, `onnx_format_imagenet()` the
ImageNet normalization, `onnx_format_raw()` 0..255.

## Matching Pillow and OpenCV

A model trained on images that Pillow or OpenCV resized scores differently when its input
is off by a level here and there, so `onnx_image_resize_pillow` and `onnx_image_resize_cv2`
repeat those libraries' integer arithmetic rather than an ideal filter. Both are checked
byte for byte against the libraries (`tests/image_exact.esk`, with outputs from
`tools/gen_image_ref.py`): Pillow 11.3 and 13.0 for every filter, enlarging and shrinking,
odd sizes and boxes; OpenCV 5.0 for `INTER_LINEAR`.

```eskiu
// face_liveness: image.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
*uint8 x = onnx_image_resize_pillow(img.rgba, w, h, 512, 512, ONNX_PILLOW_BILINEAR);
// a crop with a float box, then resized: image.crop(box).resize((224, 224), BILINEAR)
int cw = 0;
int ch = 0;
*uint8 face = onnx_image_crop_pillow(img.rgba, w, h, x1, y1, x2, y2, &cw, &ch);
*uint8 small = onnx_image_resize_pillow(face, cw, ch, 224, 224, ONNX_PILLOW_BILINEAR);
```

- **Pillow.** Pillow 11 and 12 always run the horizontal pass first. Pillow 13 runs the
  vertical one first when the height shrinks by more than twice as many pixels as the
  width, which changes some pixels; add `ONNX_PILLOW_13` to the filter
  (`ONNX_PILLOW_BILINEAR | ONNX_PILLOW_13`) to match it. The four channels are resampled
  alike, as Pillow resizes an RGB image; for an RGBA image with transparency Pillow
  premultiplies the alpha first, which these functions do not. Pillow 13's `MKS2013` and
  `MKS2021` filters are not included.
- **OpenCV.** `onnx_image_resize_cv2` is OpenCV's own code for 8-bit `INTER_LINEAR` (the
  default of `cv2.resize`), which every build runs except where a CPU-specific library
  replaces it: arm64 builds with KleidiCV (such as the opencv-python 5.0 wheel for macOS
  arm64) shrink both sides of an image with up to 3 channels by 1 to 3 times with a
  faster approximation that can differ by a level, and Carotene (in arm builds) resizes 1
  and 4 channel images with its own code. A 3-channel image enlarged, shrunk along one
  side or by more than 3 times matches on every build. Float images are not covered:
  OpenCV's float path uses fused multiply-adds for some pixels and not others, depending
  on the CPU.

MiniFASNet (Minivision's Silent-Face-Anti-Spoofing) crops the face box `[x, y, w, h]` scaled
by 2.7 or 4.0 about its center, with the scale lowered to fit the image and the crop
shifted inward at the borders, and resizes it with `cv2.resize` to 80x80; its weights take
BGR floats in 0..255:

```eskiu
*uint8 patch = onnx_minifasnet_crop(img.rgba, w, h, bx, by, bw, bh, 2.7, 80, 80);
OnnxImageFormat bgr = OnnxImageFormat{bgr: true};
OnnxTensor t = onnx_image_tensor(patch, 80, 80, &bgr)?;
```

Decoding image files is left to other packages. The examples use `<gpu/image>` from
[gpu](https://github.com/eskiu-lang/gpu), a dependency of this package, which decodes PNG
and JPEG in pure Eskiu and needs no GPU (`image_load` gives `width`, `height` and `rgba`).
Pixels from anywhere else (a camera, a video frame, a raw file) work the same once they are
RGBA.
