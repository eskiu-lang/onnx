# Inference

Sessions, tensors and providers in `<onnx>`, past the first example in the [README](../README.md#inference). Every function and type is in the [API reference](api.md).

## Introspection

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

## Tensors

`onnx_tensor(type, data, shape, rank)` makes a tensor of any numeric element type and copies
its elements from `data` (zeros when `data` is null); `onnx_tensor_f32`, `_f16`, `_i64`,
`_i32` and `_u8` take typed pointers. The buffer belongs to ONNX Runtime and goes with
`release()`. `onnx_tensor_view(type, data, shape, rank)` wraps the caller's buffer instead,
with no copy; the buffer must outlive the tensor. After a run, `t.f32()` (or `f16`, `i64`, `i32`, `u8`) is the data, null when the
tensor holds another type; `t.at(i)` reads any numeric element as a `double`; `t.dim(i)`,
`t.rank()` and `t.len()` give the shape; `t.argmax()` the index of the largest element.

`float16` elements are `uint16` bits. `onnx_f32_to_f16` and `onnx_f16_to_f32` convert them,
rounding to nearest even.

## Options and providers

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
