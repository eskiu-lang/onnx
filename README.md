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
  normalized NCHW or NHWC tensor, in RGB or BGR order.
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
| `onnx_image_resize` | bilinear with half-pixel centers, as ONNX `Resize` (linear, `half_pixel`); OpenCV's fixed-point `INTER_LINEAR` can differ by one or two levels |
| `onnx_image_resize_area` | the average of the pixels each output pixel covers, the same as OpenCV's `INTER_AREA` when shrinking |
| `onnx_image_tensor` | a new `[1, 3, h, w]` (or `[1, h, w, 3]`) float tensor |
| `onnx_image_into` | writes into item `i` of a float or float16 batch tensor |

`OnnxImageFormat` says how a pixel becomes a value: `(pixel * scale - mean[c]) / std[c]`,
with `nhwc` for the layout and `bgr` for the channel order (mean and std follow the
tensor's channel order). `onnx_format_unit()` gives 0..1, `onnx_format_imagenet()` the
ImageNet normalization, `onnx_format_raw()` 0..255.

Decoding image files is left to other packages. The examples use `<gpu/image>` from
[gpu](https://github.com/eskiu-lang/gpu), a dependency of this package, which decodes PNG
and JPEG in pure Eskiu and needs no GPU (`image_load` gives `width`, `height` and `rgba`).
Pixels from anywhere else (a camera, a video frame, a raw file) work the same once they are
RGBA.

## Examples

[`examples/classify.esk`](examples/classify.esk) prints the top classes of an image. Run
with no arguments, it uses a tiny model in `examples/data` that names the main color of an
image; with an ImageNet classifier from the ONNX model zoo it names objects:

```
$ eskiuc build -L/opt/homebrew/lib
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
$ eskiuc -L/opt/homebrew/lib examples/minifasnet.esk -o build/minifasnet
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
| macOS arm64 | Homebrew or the `osx-arm64` release archive | CI |
| Linux x86-64 | the `linux-x64` release archive | CI |
| Windows x86-64 (MinGW) | the `win-x64` release archive | CI |
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

The package needs ONNX Runtime 1.16 or newer; the instructions below use 1.31:

```bash
brew install onnxruntime               # macOS
```

On Linux and Windows, ONNX Runtime comes from its
[release archives](https://github.com/microsoft/onnxruntime/releases) (`onnxruntime-linux-x64-1.31.0.tgz`,
`onnxruntime-linux-aarch64-1.31.0.tgz`, `onnxruntime-win-x64-1.31.0.zip`). Then, in an
Eskiu package (Eskiu 0.12 or newer):

```bash
eskiuc add onnx --git https://github.com/eskiu-lang/onnx --rev v0.1.0
eskiuc build -L/opt/homebrew/lib       # the directory that holds libonnxruntime
```

With a release archive, pass its `lib` directory to `-L` and, on Linux and macOS, to the
runtime loader too: `--link-arg=-Wl,-rpath,DIR`, or `LD_LIBRARY_PATH` on Linux. On Windows
(MinGW), link against `onnxruntime.dll` itself (`-L` a directory that holds it) and put it
next to the executable: Windows looks in `System32` before `PATH`, and `System32` may hold
an older `onnxruntime.dll` that does not have API version 31.

## Modules

| Module | What it has |
|---|---|
| `<onnx>` | `onnx_session`, `onnx_session_bytes`, `OnnxSession` (`run`, `run1`, `input`, `output`), `OnnxTensor`, `OnnxOptions`, providers, float16 conversions |
| `<onnx/image>` | `onnx_image_crop`, `onnx_image_resize`, `onnx_image_resize_area`, `onnx_image_tensor`, `onnx_image_into`, `OnnxImageFormat` |
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
eskiuc test -L/opt/homebrew/lib
python3 tools/gen.py /opt/homebrew/opt/onnxruntime/include        # regenerates src/raw.esk
python3 -I tools/gen_models.py                                    # regenerates the test models
tools/layout_check.sh /opt/homebrew/opt/onnxruntime/include /opt/homebrew/lib
```

The test models are made with the `onnx` Python package and their expected outputs come
from Python's onnxruntime. `tools/layout_check.sh` compiles a C program and an Eskiu program
that print every struct's size and field offsets, every `OrtApi` slot and every enum value,
and compares their output. CI runs it and the tests on Linux, macOS and Windows.

## License

MIT. See [LICENSE](LICENSE).
