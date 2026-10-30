#!/usr/bin/env python3
"""Rescale biased FP16 linear/convolution ops for a QDQ scale-floor workaround.

For y = op(x, W) + b, rewrite as y = (op(scale*x, W) + scale*b) / scale.
For exported MatMul+Add linears, the Add bias is copied before scaling so the
original parameter tensor remains untouched. This is algebraically equivalent
apart from ordinary FP16 rounding. It is an experimental QAIRT conversion
workaround, not a device-compatibility claim.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import external_data_helper, helper, numpy_helper


def tensor_type(model: onnx.ModelProto, name: str) -> int | None:
    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        if value.name == name and value.type.HasField("tensor_type"):
            return value.type.tensor_type.elem_type
    for tensor in model.graph.initializer:
        if tensor.name == name:
            return tensor.data_type
    return None


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value).strip("_") or "op"


def fp16_scaled_copy(tensor: onnx.TensorProto, base_dir: Path, name: str, scale: float) -> onnx.TensorProto:
    if tensor.data_type != onnx.TensorProto.FLOAT16:
        raise ValueError(f"expected FP16 tensor, got {onnx.TensorProto.DataType.Name(tensor.data_type)}: {tensor.name}")
    if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data:
        if not any(item.key == "location" and item.value for item in tensor.external_data):
            raise ValueError(f"external tensor has no data location: {tensor.name}")
        external_data_helper.load_external_data_for_tensor(tensor, str(base_dir))
    values = numpy_helper.to_array(tensor).astype(np.float32) * scale
    if not np.isfinite(values).all() or np.max(np.abs(values), initial=0) > 65504:
        raise OverflowError(f"scaled FP16 values would overflow: {tensor.name}")
    return numpy_helper.from_array(values.astype(np.float16), name=name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="ONNX graph; external tensor files remain in the same directory")
    parser.add_argument("--scale", type=float, default=4.0, help="positive power-of-two scale (default: 4)")
    args = parser.parse_args()
    model_path = args.model.resolve()
    scale = float(args.scale)
    if scale <= 0 or not float(np.log2(scale)).is_integer():
        raise SystemExit("--scale must be a positive power of two")
    if not model_path.is_file():
        raise SystemExit(f"ONNX graph not found: {model_path}")

    model = onnx.load_model(model_path, load_external_data=False)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    scale_in_name = "qnn_biased_op_scale_in_fp16"
    scale_out_name = "qnn_biased_op_scale_out_fp16"
    scale_in = np.array(scale, dtype=np.float16)
    scale_out = np.array(1.0 / scale, dtype=np.float16)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    uses = Counter(value for node in model.graph.node for value in node.input)
    users: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for value in node.input:
            users[value].append(node)

    direct: dict[str, tuple[str, str, str, str]] = {}
    pairs_by_matmul: dict[str, dict[str, str]] = {}
    pairs_by_add: dict[str, dict[str, str]] = {}
    skipped: list[str] = []

    # First cover operators with a native bias input: Conv and Gemm.
    for index, node in enumerate(model.graph.node):
        if node.op_type not in {"Conv", "Gemm"} or len(node.input) < 3 or not node.input[2]:
            continue
        if not node.name:
            skipped.append(f"unnamed {node.op_type} at node index {index}")
            continue
        if node.output[0].startswith("qnn_") or node.input[0].startswith("qnn_"):
            continue  # already handled by an earlier explicit pass
        if tensor_type(inferred, node.input[0]) != onnx.TensorProto.FLOAT16:
            continue
        bias = initializers.get(node.input[2])
        if bias is None or bias.data_type != onnx.TensorProto.FLOAT16:
            continue
        tag = safe_name(node.name)
        bias_name = f"qnn_scaled_bias_{tag}"
        copied_bias = fp16_scaled_copy(bias, model_path.parent, bias_name, scale)
        model.graph.initializer.append(copied_bias)
        direct[node.name] = (
            node.input[0], node.output[0],
            f"qnn_{tag}_input_scaled", f"qnn_{tag}_output_scaled",
        )
        node.input[2] = bias_name

    # Cover linears exported as MatMul followed directly by a bias Add. AI Hub's
    # quantizer fuses this pattern to a Gemm, whose bias scale can hit the FP16
    # export floor even though the source graph contains no Gemm node.
    for index, matmul in enumerate(model.graph.node):
        if matmul.op_type != "MatMul" or len(matmul.input) != 2 or len(matmul.output) != 1:
            continue
        if not matmul.name or matmul.input[0].startswith("qnn_") or matmul.output[0].startswith("qnn_"):
            continue
        if tensor_type(inferred, matmul.input[0]) != onnx.TensorProto.FLOAT16:
            continue
        consumers = users[matmul.output[0]]
        if len(consumers) != 1 or consumers[0].op_type != "Add" or len(consumers[0].input) != 2:
            continue
        add = consumers[0]
        if not add.name or len(add.output) != 1 or add.output[0].startswith("qnn_"):
            continue
        bias_name = next((value for value in add.input if value != matmul.output[0]), None)
        bias = initializers.get(bias_name) if bias_name else None
        if bias is None or bias.data_type != onnx.TensorProto.FLOAT16:
            continue
        tag = safe_name(matmul.name)
        scaled_bias_name = f"qnn_scaled_bias_{tag}_add"
        model.graph.initializer.append(fp16_scaled_copy(bias, model_path.parent, scaled_bias_name, scale))
        pairs_by_matmul[matmul.name] = {
            "old_matmul_output": matmul.output[0],
            "matmul_input": matmul.input[0],
            "scaled_matmul_output": f"qnn_{tag}_matmul_scaled",
            "scaled_input": f"qnn_{tag}_input_scaled",
            "add_name": add.name,
            "old_add_output": add.output[0],
            "scaled_add_output": f"qnn_{tag}_add_scaled",
            "old_bias": bias_name,
            "scaled_bias": scaled_bias_name,
            "tag": tag,
        }
        pairs_by_add[add.name] = pairs_by_matmul[matmul.name]

    if not direct and not pairs_by_matmul:
        raise SystemExit("No eligible unscaled biased FP16 Conv/Gemm or MatMul+Add operators found")
    if scale_in_name not in initializers:
        model.graph.initializer.append(numpy_helper.from_array(scale_in, name=scale_in_name))
    if scale_out_name not in initializers:
        model.graph.initializer.append(numpy_helper.from_array(scale_out, name=scale_out_name))

    rewritten = []
    direct_nodes = []
    pair_mats = []
    pair_adds = []
    for node in model.graph.node:
        if node.name in direct:
            original_input, original_output, input_scaled, output_scaled = direct[node.name]
            tag = safe_name(node.name)
            node.input[0] = input_scaled
            node.output[0] = output_scaled
            rewritten.append(helper.make_node("Mul", [original_input, scale_in_name], [input_scaled], name=f"QNN_{tag}_InputRescale"))
            rewritten.append(node)
            rewritten.append(helper.make_node("Mul", [output_scaled, scale_out_name], [original_output], name=f"QNN_{tag}_OutputUnscale"))
            direct_nodes.append(node)
            continue
        if node.name in pairs_by_matmul:
            record = pairs_by_matmul[node.name]
            original_input = record["matmul_input"]
            input_scaled = record["scaled_input"]
            node.input[0] = input_scaled
            node.output[0] = record["scaled_matmul_output"]
            rewritten.append(helper.make_node("Mul", [original_input, scale_in_name], [input_scaled], name=f"QNN_{record['tag']}_InputRescale"))
            rewritten.append(node)
            pair_mats.append(node)
            continue
        if node.name in pairs_by_add:
            record = pairs_by_add[node.name]
            node.input[:] = [record["scaled_matmul_output"] if x == record["old_matmul_output"] else record["scaled_bias"] if x == record["old_bias"] else x for x in node.input]
            node.output[0] = record["scaled_add_output"]
            rewritten.append(node)
            rewritten.append(helper.make_node("Mul", [record["scaled_add_output"], scale_out_name], [record["old_add_output"]], name=f"QNN_{record['tag']}_OutputUnscale"))
            pair_adds.append(node)
            continue
        rewritten.append(node)
    model.graph.ClearField("node")
    model.graph.node.extend(rewritten)

    onnx.save_model(model, str(model_path))
    onnx.checker.check_model(str(model_path))
    checked = onnx.load_model(model_path, load_external_data=False)
    onnx.shape_inference.infer_shapes(checked, strict_mode=False)
    counts = Counter(node.op_type for node in direct_nodes)
    print(f"RESCALED_DIRECT {len(direct_nodes)}: {dict(counts)}")
    print(f"RESCALED_MATMUL_ADD_PAIRS {len(pair_mats)}")
    if pair_mats:
        print("FIRST_PAIR", pair_mats[0].name, "->", pair_adds[0].name)
        if len(pair_mats) > 23:
            print("PAIR_23", pair_mats[23].name, "->", pair_adds[23].name)
    print("ONNX checker and shape inference: PASS")
    print("Skipped:", skipped if skipped else "none")


if __name__ == "__main__":
    main()
