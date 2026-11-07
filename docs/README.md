# onnx documentation

Guides and reference for onnx. The [README](../README.md) has the overview.

<!-- pages -->
- [Detection](detection.md)
- [Images](images.md)
- [Inference](inference.md)
- [API reference](api.md): every public function and type
<!-- /pages -->

## Examples

[`examples/classify.esk`](../examples/classify.esk) prints the top classes of an image. Run
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

[`examples/minifasnet.esk`](../examples/minifasnet.esk) is face anti-spoofing: it says whether a
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

## Installation details

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
