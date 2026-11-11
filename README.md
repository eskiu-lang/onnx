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
  <img src="https://img.shields.io/badge/version-0.2.0-6448d4" alt="Version 0.2.0">
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

## Documentation

- [Introspection](docs/inference.md#introspection)
- [Tensors](docs/inference.md#tensors)
- [Options and providers](docs/inference.md#options-and-providers)
- [Images: preparing inputs, Pillow and OpenCV parity](docs/images.md)
- [Detection: SSD anchors, box decoding, NMS](docs/detection.md)
- [Examples](docs/README.md#examples)
- [Platforms](docs/README.md#platforms)
- [Modules](docs/README.md#modules)
- [Contributing](docs/README.md#contributing)
- [API reference](docs/api.md): every public function and type

## Install

In an Eskiu package (Eskiu 0.15.1 or newer, for gpu 0.6.0):

```bash
eskiuc add onnx --git https://github.com/eskiu-lang/onnx --rev v0.2.0
eskiuc build
```

The first build downloads ONNX Runtime 1.31 for your platform and checks its SHA-256; nothing else to install. See [installation details](docs/README.md#installation-details).

## License

MIT. See [LICENSE](LICENSE).
