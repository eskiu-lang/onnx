#!/usr/bin/env python3
"""Generate the test models in tests/data and their expected outputs.

Usage: python3 -I tools/gen_models.py
Needs the onnx, onnxruntime and numpy Python packages. Each model is small and built
from fixed seeds, so running this again gives the same files. Inputs and outputs are
raw little-endian arrays (.bin) that the tests read with fs_read_all.
"""
import os
import struct
import zlib

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "data")
EXAMPLES = os.path.join(ROOT, "examples", "data")


def save(model, path):
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


def model_of(nodes, inputs, outputs, inits=(), opset=17, name="g"):
    g = helper.make_graph(nodes, name, inputs, outputs, list(inits))
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)], producer_name="eskiu-onnx")


def raw(name, a):
    a.tofile(os.path.join(DATA, name))


def run(path, feeds):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return s.run(None, feeds)


def mlp():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((4, 3)).astype(np.float32)
    b = rng.standard_normal(3).astype(np.float32)
    m = model_of(
        [helper.make_node("MatMul", ["x", "w"], ["h"]), helper.make_node("Add", ["h", "b"], ["z"]),
         helper.make_node("Relu", ["z"], ["y"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
        [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")])
    path = os.path.join(DATA, "mlp.onnx")
    save(m, path)
    x = np.array([[1.0, -2.0, 0.5, 3.0]], dtype=np.float32)
    raw("mlp_x.bin", x)
    raw("mlp_y.bin", run(path, {"x": x})[0])


def conv():
    rng = np.random.default_rng(2)
    w = rng.standard_normal((2, 1, 3, 3)).astype(np.float32)
    b = np.array([0.1, -0.2], dtype=np.float32)
    m = model_of(
        [helper.make_node("Conv", ["x", "w", "b"], ["c"], pads=[1, 1, 1, 1]),
         helper.make_node("MaxPool", ["c"], ["y"], kernel_shape=[2, 2], strides=[2, 2])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 6, 6])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 3])],
        [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")])
    path = os.path.join(DATA, "conv.onnx")
    save(m, path)
    x = rng.standard_normal((1, 1, 6, 6)).astype(np.float32)
    raw("conv_x.bin", x)
    raw("conv_y.bin", run(path, {"x": x})[0])


def ints():
    # ids int64[N] -> y = ids * 3 + 1 (int64) and n = Cast(ids, int32) (int32);
    # pix uint8[4] -> f = pix / 255 (float).
    m = model_of(
        [helper.make_node("Mul", ["ids", "three"], ["t"]), helper.make_node("Add", ["t", "one"], ["y"]),
         helper.make_node("Cast", ["ids"], ["n"], to=TensorProto.INT32),
         helper.make_node("Cast", ["pix"], ["pf"], to=TensorProto.FLOAT),
         helper.make_node("Div", ["pf", "k255"], ["f"])],
        [helper.make_tensor_value_info("ids", TensorProto.INT64, ["n"]),
         helper.make_tensor_value_info("pix", TensorProto.UINT8, [4])],
        [helper.make_tensor_value_info("y", TensorProto.INT64, ["n"]),
         helper.make_tensor_value_info("n", TensorProto.INT32, ["n"]),
         helper.make_tensor_value_info("f", TensorProto.FLOAT, [4])],
        [numpy_helper.from_array(np.array(3, dtype=np.int64), "three"),
         numpy_helper.from_array(np.array(1, dtype=np.int64), "one"),
         numpy_helper.from_array(np.array(255, dtype=np.float32), "k255")])
    save(m, os.path.join(DATA, "ints.onnx"))


def dynamic():
    # x float[batch, 3] -> y = Softmax(x, axis 1), z = ReduceSum(x, axis 1) [batch]
    m = model_of(
        [helper.make_node("Softmax", ["x"], ["y"], axis=1),
         helper.make_node("ReduceSum", ["x", "axes"], ["z"], keepdims=0)],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3]),
         helper.make_tensor_value_info("z", TensorProto.FLOAT, ["batch"])],
        [numpy_helper.from_array(np.array([1], dtype=np.int64), "axes")])
    path = os.path.join(DATA, "dynamic.onnx")
    save(m, path)
    x = np.arange(15, dtype=np.float32).reshape(5, 3) / 4.0 - 1.0
    raw("dynamic_x.bin", x)
    y, z = run(path, {"x": x})
    raw("dynamic_y.bin", y)
    raw("dynamic_z.bin", z)


def multi():
    # a, b float[2, 2] -> sum = a + b, prod = a @ b, neg = -a
    m = model_of(
        [helper.make_node("Add", ["a", "b"], ["sum"]), helper.make_node("MatMul", ["a", "b"], ["prod"]),
         helper.make_node("Neg", ["a"], ["neg"])],
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 2]),
         helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 2])],
        [helper.make_tensor_value_info("sum", TensorProto.FLOAT, [2, 2]),
         helper.make_tensor_value_info("prod", TensorProto.FLOAT, [2, 2]),
         helper.make_tensor_value_info("neg", TensorProto.FLOAT, [2, 2])])
    helper.set_model_props(m, {"names": "{0: 'person', 1: 'bus'}"})
    save(m, os.path.join(DATA, "multi.onnx"))


def half():
    # x float16[2, 3] -> y = x * 1.5 + 0.25 (float16), s = Cast(x, float)
    m = model_of(
        [helper.make_node("Mul", ["x", "k"], ["t"]), helper.make_node("Add", ["t", "c"], ["y"]),
         helper.make_node("Cast", ["x"], ["s"], to=TensorProto.FLOAT)],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [2, 3]),
         helper.make_tensor_value_info("s", TensorProto.FLOAT, [2, 3])],
        [numpy_helper.from_array(np.array(1.5, dtype=np.float16), "k"),
         numpy_helper.from_array(np.array(0.25, dtype=np.float16), "c")])
    path = os.path.join(DATA, "half.onnx")
    save(m, path)
    x = np.array([[0.5, -1.25, 3.0], [65504.0 / 4, 1e-4, -0.0]], dtype=np.float16)
    raw("half_x.bin", x)
    raw("half_y.bin", run(path, {"x": x})[0])


def resize():
    # x float[1, C, H, W], sizes int64[4] -> bilinear resize with half-pixel centers,
    # the reference for <onnx/image>'s image_resize.
    m = model_of(
        [helper.make_node("Resize", ["x", "", "", "sizes"], ["y"], mode="linear",
                          coordinate_transformation_mode="half_pixel")],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, "c", "h", "w"]),
         helper.make_tensor_value_info("sizes", TensorProto.INT64, [4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, "c", "oh", "ow"])], opset=18)
    save(m, os.path.join(DATA, "resize.onnx"))


def colors():
    # A tiny image classifier for examples/classify.esk: x float[1, 3, 32, 32] normalized
    # with the ImageNet mean and std -> logits for red, green, blue, gray from the mean
    # color of the image. GlobalAveragePool, then a 1x1 Conv as the linear layer.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    # Undo the normalization so the classifier sees colors in 0..1.
    scale = std.reshape(1, 3, 1, 1)
    shift = mean.reshape(1, 3, 1, 1)
    w = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2], [0, 0, 0]], dtype=np.float32) * 8.0
    b = np.array([0, 0, 0, 1.0], dtype=np.float32)
    m = model_of(
        [helper.make_node("Mul", ["image", "scale"], ["t"]), helper.make_node("Add", ["t", "shift"], ["c"]),
         helper.make_node("GlobalAveragePool", ["c"], ["p"]),
         helper.make_node("Conv", ["p", "w", "b"], ["l"]),
         helper.make_node("Flatten", ["l"], ["logits"]),
         helper.make_node("Softmax", ["logits"], ["probs"], axis=1)],
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, [1, 4])],
        [numpy_helper.from_array(scale, "scale"), numpy_helper.from_array(shift, "shift"),
         numpy_helper.from_array(w.reshape(4, 3, 1, 1), "w"), numpy_helper.from_array(b, "b")])
    save(m, os.path.join(EXAMPLES, "colors.onnx"))
    with open(os.path.join(EXAMPLES, "colors.txt"), "w") as f:
        f.write("red\ngreen\nblue\ngray\n")


def png(path, rgb):
    h, w, _ = rgb.shape
    rows = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
                chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def sample():
    # A 64x48 sky-blue gradient with a small orange square, for examples/classify.esk.
    y, x = np.mgrid[0:48, 0:64]
    rgb = np.stack([40 + x // 2, 90 + y, 200 + x // 3], axis=-1)
    rgb[10:20, 40:52] = [240, 140, 30]
    png(os.path.join(EXAMPLES, "sample.png"), np.clip(rgb, 0, 255).astype(np.uint8))


def main():
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(EXAMPLES, exist_ok=True)
    mlp()
    conv()
    ints()
    dynamic()
    multi()
    half()
    resize()
    colors()
    sample()
    print("models written to tests/data and examples/data")


main()
