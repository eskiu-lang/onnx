<p align="center">
  <img src="https://raw.githubusercontent.com/eskiu-lang/eskiu/main/assets/logo.png" alt="Eskiu, a kitten with a ball of yarn" width="140">
</p>

<h2 align="center">onnx</h2>
<p align="center">ONNX model inference for Eskiu, over ONNX Runtime.</p>
<p align="center"><sub>Part of the <a href="https://github.com/eskiu-lang">Eskiu</a> project</sub></p>
<p align="center">
  <a href="https://eskiu-lang.org">eskiu-lang.org</a> &nbsp;&middot;&nbsp;
  <a href="#modules">Modules</a> &nbsp;&middot;&nbsp;
  <a href="examples/">Examples</a> &nbsp;&middot;&nbsp;
  <a href="https://github.com/eskiu-lang/eskiu">The language</a>
</p>

<p align="center">
  <a href="https://github.com/eskiu-lang/onnx/actions/workflows/ci.yml"><img src="https://github.com/eskiu-lang/onnx/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/version-0.1.0-6448d4" alt="Version 0.1.0">
  <img src="https://img.shields.io/badge/ONNX%20Runtime-1.31-orange" alt="ONNX Runtime 1.31">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT"></a>
</p>

---

`onnx` is an Eskiu package for running ONNX models. It calls
[ONNX Runtime](https://onnxruntime.ai) through its C API, so a model exported from PyTorch,
TensorFlow or scikit-learn runs on the CPU everywhere, and on CoreML (macOS), CUDA, DirectML
(Windows) or WebGPU when the ONNX Runtime build has them.

- **Sessions.** Load a model from a file or from bytes, with thread counts, the graph
  optimization level and an execution provider.
- **Introspection.** The name, element type and shape of every input and output, with -1
  for a dynamic dimension.
- **Tensors.** `float`, `float16`, `int64`, `int32` and `uint8` (and every other numeric
  type) in CPU memory, created from Eskiu arrays and read back after a run.
- **Images.** `<onnx/image>` crops, resizes (bilinear or area) and turns RGBA pixels into a
  normalized NCHW or NHWC tensor, in RGB or BGR order. Its Pillow and OpenCV resizers give
  the same bytes as those libraries, for models trained on images they resized.
- **Detection.** `<onnx/detect>` makes SSD anchors, decodes boxes, filters scores and runs
  non-maximum suppression, bit for bit as the numpy and PyTorch code detectors ship with.
- **Errors as `Result`.** A missing file, a wrong input name, type or shape, a provider the
  build lacks: each comes back as an error message, never a crash.
- **The whole API underneath.** `<onnx/raw>` wraps all 428 functions of the `OrtApi` table,
  generated from `onnxruntime_c_api.h`, with every slot, struct layout and enum value
  checked against the C compiler.

## Inference

```eskiu
import <io>;
import <onnx>;

Result<int, string> run() {
    OnnxSession s = onnx_session("model.onnx", null)?;
    defer s.release();
    float[4] x = {1.0, -2.0, 0.5, 3.0};
    int64[2] shape = {1, 4};
    OnnxTensor input = onnx_tensor_f32(&x[0], &shape[0], 2)?;
    defer input.release();
    OnnxTensor y = s.run1(&input)?;
    defer y.release();
    *float out = y.f32();
    for (i in 0..y.len()) printf("%f\n", (double)out[i]);
    return Ok<int, string>(0);
}

Result<int, string> r = run();
if (!r.ok) printf("error: %s\n", r.error);
```

`run1` is for a model with one input, and returns its first output. `OnnxSession_run`
takes inputs by name, in any order, and the outputs wanted, by name (null for all of
them, in model order):

```eskiu
string[2] names = {"ids", "mask"};
OnnxTensor[2] inputs = {ids, mask};
string[1] wanted = {"logits"};
List<OnnxTensor> out = s.run(&names[0], &inputs[0], 2, &wanted[0], 1)?;
defer onnx_tensors_free(&out);
*float logits = out.data[0].f32();
```

Before the run the inputs are checked against the model: every name must be one of its
inputs, every input must be given once, and each tensor must have the input's element
type and fixed dimensions.

```
onnx: the model has no input named 'y' (inputs: x)
onnx: input 'x' has shape [1, 5], the model wants [1, 4]
onnx: input 'x' is int64, the model wants float
```

### Introspection

```eskiu
for (i in 0..s.input_count()) {
    *OnnxInfo info = s.input(i);
    String dims;
    onnx_shape_text(info.shape.data, info.shape.size, &dims);
    printf("%s: %s %s\n", info.name.data, onnx_type_name(info.type), dims.data);   // x: float [-1, 3]
    String_free(&dims);
}
```

`s.input_index("x")` and `s.output_index("y")` find an input or output by name.
`s.metadata("names", &out)` reads a custom metadata entry of the model into a `String`
(false when the model has none by that name).

### Tensors

`onnx_tensor(type, data, shape, rank)` makes a tensor of any numeric element type and copies
its elements from `data` (zeros when `data` is null); `onnx_tensor_f32`, `_f16`, `_i64`,
`_i32` and `_u8` take typed pointers. The buffer belongs to ONNX Runtime and goes with
`release()`. `onnx_tensor_view(type, data, shape, rank)` wraps the caller's buffer instead,
with no copy; the buffer must outlive the tensor. After a run, `t.f32()` (or `f16`, `i64`, `i32`, `u8`) is the data, null when the
tensor holds another type; `t.at(i)` reads any numeric element as a `double`; `t.dim(i)`,
`t.rank()` and `t.len()` give the shape; `t.argmax()` the index of the largest element.

`float16` elements are `uint16` bits. `onnx_f32_to_f16` and `onnx_f16_to_f32` convert them,
rounding to nearest even.

### Options and providers

```eskiu
OnnxOptions o = OnnxOptions{provider: ONNX_AUTO, threads: 4, optimization: ONNX_OPT_ALL};
OnnxSession s = onnx_session("model.onnx", &o)?;
printf("running on %s\n", onnx_provider_name(s.provider));
```

| `provider` | |
|---|---|
| `ONNX_CPU` | the default |
| `ONNX_AUTO` | CoreML on Apple, DirectML on Windows, NNAPI on Android, else CUDA, else the CPU |
| `ONNX_COREML` | Apple's Core ML; `coreml_units` picks `"ALL"` (default), `"CPUAndGPU"`, `"CPUAndNeuralEngine"` or `"CPUOnly"`, and `coreml_format` `"MLProgram"` (default, iOS 15 and macOS 12 on) or `"NeuralNetwork"` |
| `ONNX_CUDA` | NVIDIA GPUs, with `device` as the GPU index |
| `ONNX_DIRECTML` | DirectX 12 GPUs on Windows, with `device` as the adapter index |
| `ONNX_WEBGPU` | WebGPU, in builds that include it |
| `ONNX_NNAPI` | Android's NNAPI (GPU, NPU or DSP); `nnapi_flags` takes `ONNX_NNAPI_FP16`, `ONNX_NNAPI_NCHW`, `ONNX_NNAPI_CPU_DISABLED`, `ONNX_NNAPI_CPU_ONLY` |

A provider the ONNX Runtime build lacks, or one that cannot compile the model, is an error
(listing what the build has), unless `fallback` is set, in which case the session runs on the
CPU (and `s.provider` says so). Core ML first compiles the model as an MLProgram, which keeps
float32 where the model needs it; a model it cannot compile that way gets the older
NeuralNetwork format, which on the GPU and the Neural Engine computes in float16.
`onnx_has_provider(ONNX_CUDA)` asks first. Which providers a build has depends on how it was
built: the Homebrew and macOS release builds have CoreML (the release also WebGPU), the Android AAR
has NNAPI, the
Linux and Windows release archives only the CPU. CUDA needs the `onnxruntime-gpu` archives,
and DirectML the `Microsoft.ML.OnnxRuntime.DirectML` package.

`threads` and `inter_threads` set the intra-op and inter-op thread pools (0 keeps ONNX
Runtime's choice), `optimization` is `ONNX_OPT_NONE`, `_BASIC`, `_EXTENDED` or `_ALL`.
`config` passes ONNX Runtime's session configuration entries as comma-separated
`key=value` pairs (`"session.intra_op.allow_spinning=0"`); the environment variable
`ONNX_CONFIG` adds entries to every session, before the options' own.

## Images

`<onnx/image>` works on 8-bit RGBA pixels (what most decoders give), and returns new
buffers to `free`:

```eskiu
import <onnx>;
import <onnx/image>;
import <gpu/image>;                  // PNG and JPEG decoding, from the gpu package

Image img = image_load("cat.jpg")?;
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

### Matching Pillow and OpenCV

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

## Detection

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

## Examples

[`examples/classify.esk`](examples/classify.esk) prints the top classes of an image. Run
with no arguments, it uses a tiny model in `examples/data` that names the main color of an
image; with an ImageNet classifier from the ONNX model zoo it names objects:

```
$ eskiuc build
$ build/onnx --model mobilenetv2-12.onnx --labels synset.txt messi5.jpg
mobilenetv2-12.onnx on messi5.jpg (CPU), 224x224 input
 97.12%  n04254680 soccer ball
  2.14%  n04118538 rugby ball
  0.22%  n04409515 tennis ball
  0.19%  n02799071 baseball
  0.14%  n03942813 ping-pong ball
```

The percentages are the ones Python's onnxruntime gives for the same model, crop and resize.
The model is [mobilenetv2-12.onnx](https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet)
and the labels [synset.txt](https://github.com/onnx/models/blob/main/validated/vision/classification/synset.txt).

[`examples/minifasnet.esk`](examples/minifasnet.esk) is face anti-spoofing: it says whether a
face is a live person or a photo or screen held up to the camera. It takes a MiniFASNet
model, such as `best_model.onnx` from
[facenox/face-antispoof-onnx](https://github.com/facenox/face-antispoof-onnx) (Apache-2.0),
an image and the face's box from a face detector:

```
$ curl -LO https://raw.githubusercontent.com/facenox/face-antispoof-onnx/main/models/best/98.20/best_model.onnx
$ eskiuc examples/minifasnet.esk -o build/minifasnet
$ build/minifasnet best_model.onnx face.jpg 212 197 346 387
box 212 197 346 387, logits real 2.788116 spoof -2.788251
real (live 99.62%, CPU)
```

It follows the project's Python preprocessing (a square crop 1.5 times the box with
reflected borders, an area resize to 128x128, RGB in 0..1), and gives the same logits as
the Python code to six decimals for faces larger than the model input. Model files are not
part of this repository.

## Platforms

| Platform | ONNX Runtime | Checked |
|---|---|---|
| macOS arm64 | the `osx-arm64` release archive, downloaded by `eskiuc build` | CI |
| Linux x86-64 and arm64 | the `linux-x64` and `linux-aarch64` release archives, downloaded | CI (x86-64) |
| Windows x86-64 (MinGW) | the `win-x64` release archive, downloaded | CI |
| iOS (device and simulator) | the `onnxruntime-c` CocoaPods archive (1.30) | iOS simulator |
| Android arm64 and x86_64 | the `onnxruntime-android` AAR | Android emulator (arm64) |

ONNX Runtime ships for 64-bit targets only, so building `<onnx>` for a 32-bit target or for
WebAssembly stops with an `#error` that says so. The package works with ONNX Runtime 1.16 or
newer: it uses the library's own API version when that is older than the 31 it was
generated from, and `ort_api_version()` says which one it got.

**iOS.** ONNX Runtime comes as a static framework in
`https://download.onnxruntime.ai/pod-archive-onnxruntime-c-1.30.0.zip`
(`onnxruntime.xcframework`, with `ios-arm64` and `ios-arm64_x86_64-simulator` slices). Point
`-L` at a directory with a `libonnxruntime.a` link to the slice's `onnxruntime` binary and
add the frameworks it uses:

```bash
eskiuc app.esk --target arm64-apple-ios15.1-simulator -o app -L ort-ios \
    --link-arg=-lc++ --link-arg=-framework --link-arg=CoreML --link-arg=-framework --link-arg=Foundation \
    --link-arg=-framework --link-arg=UIKit --link-arg=-framework --link-arg=Network
```

For an app, build the Eskiu code as a static library (`-o libapp.a`) and let Xcode link it
with the `onnxruntime-c` pod. The CPU and Core ML providers work there.

**Android.** The `onnxruntime-android` AAR from Maven Central is a zip whose
`jni/<abi>/libonnxruntime.so` is the library: `-L` that directory and ship the `.so` with
the app (`LD_LIBRARY_PATH` when running a test binary through `adb`). On an emulator running
on an Apple M4, ONNX Runtime's Arm kernels see SME in the reported CPU features but cannot
run it in the virtual machine; set `ONNX_CONFIG=mlas.disable_kleidiai=1` there.

## Install

In an Eskiu package (Eskiu 0.12 or newer):

```bash
eskiuc add onnx --git https://github.com/eskiu-lang/onnx --rev v0.1.0
eskiuc build
```

There is nothing else to install. The package's `eskiu.toml` names ONNX Runtime 1.31's own
release archives for macOS arm64, Linux (x86-64 and arm64) and Windows (x86-64 and arm64):
the first build that links it (`eskiuc build`, `run`, `test`) downloads the one for your
machine, checks its SHA-256 and keeps it in `~/.eskiu/native`, and every build links
against it, with no `-L`. On Windows the DLL is copied next to the program, ahead of an older
`onnxruntime.dll` that `System32` may hold.

The package works with ONNX Runtime 1.16 or newer. To use another build (Homebrew's, a GPU
build, another platform), pass its library directory with `-L` and, on Linux and macOS,
`--link-arg=-Wl,-rpath,DIR`; a platform the manifest does not list links against the
system's library this way.

## Modules

| Module | What it has |
|---|---|
| `<onnx>` | `onnx_session`, `onnx_session_bytes`, `OnnxSession` (`run`, `run1`, `input`, `output`), `OnnxTensor`, `OnnxOptions`, providers, float16 conversions |
| `<onnx/image>` | `onnx_image_crop`, `onnx_image_resize`, `onnx_image_resize_area`, `onnx_image_tensor`, `onnx_image_into`, `OnnxImageFormat`, `onnx_image_resize_pillow`, `onnx_image_resize_pillow_box`, `onnx_image_crop_pillow`, `onnx_image_resize_cv2`, `onnx_minifasnet_box`, `onnx_minifasnet_crop`, `OnnxRect` |
| `<onnx/detect>` | `OnnxAnchor`, `OnnxBox`, `OnnxBoxCoder`, `onnx_anchors_grid`, `onnx_boxes_decode`, `onnx_scores_above`, `onnx_nms` |
| `<onnx/raw>` | the ONNX Runtime C API, generated from `onnxruntime_c_api.h` |

Each object has a `release()` to pair with `defer`. `OnnxSession.handle` and
`OnnxTensor.value` are the `<onnx/raw>` handles, so a program can drop down to the C API
for anything the idiomatic layer does not cover (IO binding, run options, custom ops,
model metadata).

### The raw API

The C API is one exported function, `OrtGetApiBase`, and a table of function pointers
(`OrtApi`) called by position. `<onnx/raw>` has one Eskiu function per table entry,
`ort_Name`, with the C signature: `ort_CreateEnv(ORT_LOGGING_LEVEL_WARNING, "app", &env)`
calls `OrtApi::CreateEnv`. `ort_api()` is the table (null when the library is older than
API version 31) and `ort_version_string()` the library's version. Enums are Eskiu enums,
the C API's structs are Eskiu structs, and the handles are opaque structs used by pointer.

Eskiu calls a function value as `code(env, args...)`, so each wrapper builds one whose code
is the table entry and whose env is the C function's first argument. That works for any
function whose first argument is a pointer, an integer or an enum, which is all 428 of them
in 1.31. `uint8`, `int8`, `int16`, `uint16` and `bool` arguments are passed widened to 32
bits, as C callers do.

## Contributing

Bug reports are welcome; please include the program, the `eskiuc --version` output and the
ONNX Runtime version. The tests and the layout check run with:

```bash
eskiuc test
python3 tools/gen.py /opt/homebrew/opt/onnxruntime/include        # regenerates src/raw.esk
python3 -I tools/gen_models.py                                    # regenerates the test models
python3 -I tools/gen_image_ref.py pillow                          # Pillow outputs (run with Pillow 11 and 13)
python3 -I tools/gen_image_ref.py cv2                             # OpenCV outputs
python3 -I tools/gen_detect_ref.py                                # numpy detection reference
tools/layout_check.sh /opt/homebrew/opt/onnxruntime/include /opt/homebrew/lib
```

The test models are made with the `onnx` Python package and their expected outputs come
from Python's onnxruntime. `tools/layout_check.sh` compiles a C program and an Eskiu program
that print every struct's size and field offsets, every `OrtApi` slot and every enum value,
and compares their output. CI runs it and the tests on Linux, macOS and Windows.

## License

MIT. See [LICENSE](LICENSE).
