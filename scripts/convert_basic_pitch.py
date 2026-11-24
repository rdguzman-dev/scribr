"""Convert Basic Pitch's ONNX trunk into a NumPy weight archive.

The Basic Pitch wheel ships an ONNX export of the ICASSP 2022 model. This
script resolves the trunk and head weights by walking the graph, checks
their shapes against the architecture in `basic_pitch/models.py`, and
writes the arrays the torch port loads.

The export folds every BatchNorm into the preceding convolution, so the
converted weights include the BatchNorm affine and there are no separate
running statistics to map. The frontend's scalar BatchNorm is the one
remaining affine; its scale and shift are written as separate arrays.

Run with the ONNX artifact from `basic-pitch==0.4.0`:

```bash
uv run python -m scripts.convert_basic_pitch \
  --onnx /tmp/bp-artifact/basic_pitch/saved_models/icassp_2022/nmp.onnx
```
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "scribr"
    / "deep_learning"
    / "weights"
    / "basic_pitch_trunk.npz"
)


@dataclass(frozen=True)
class LayerSpec:
    """Expected identity of one convolution in the export.

    Attributes:
        weight: Initializer name of the kernel.
        shape: Kernel shape in ONNX `(out, in, height, width)` order.
        strides: Spatial strides.
        pads: ONNX padding as `(top, left, bottom, right)`.
        node_suffix: Suffix of the exported node name. The exporter mangles
            names by concatenating several scopes, so the suffix is the part
            that actually identifies the layer.
    """

    weight: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    pads: tuple[int, ...]
    node_suffix: str


# The first `Conv2D(32, (5, 5))` in `models.py` feeds a variable that is
# overwritten before use, so it is absent from the export. The contour
# trunk therefore starts at `conv2` (8 filters, 3x39) on the harmonic
# stack, then `conv3` reduces it to one contour map, and `conv4` turns
# that map into the 32-channel note feature. The exported BatchNorms are
# folded into the preceding convolutions.
MANIFEST: dict[str, LayerSpec] = {
    "conv2": LayerSpec(
        weight="const_fold_opt__727",
        shape=(8, 8, 3, 39),
        strides=(1, 1),
        pads=(1, 19, 1, 19),
        node_suffix="conv2d_1/Conv2D1",
    ),
    "conv3": LayerSpec(
        weight="const_fold_opt__710",
        shape=(1, 8, 5, 5),
        strides=(1, 1),
        pads=(2, 2, 2, 2),
        node_suffix="contours-reduced/BiasAdd/ReadVariableOp",
    ),
    "conv4": LayerSpec(
        weight="const_fold_opt__738",
        shape=(32, 1, 7, 7),
        strides=(1, 3),
        pads=(3, 2, 3, 2),
        node_suffix="conv2d_2/BiasAdd/ReadVariableOp",
    ),
    "note": LayerSpec(
        weight="const_fold_opt__702",
        shape=(1, 32, 7, 3),
        strides=(1, 1),
        pads=(3, 1, 3, 1),
        node_suffix="conv2d_3/BiasAdd/ReadVariableOp",
    ),
    "onset": LayerSpec(
        weight="const_fold_opt__707",
        shape=(32, 8, 5, 5),
        strides=(1, 3),
        pads=(2, 1, 2, 1),
        node_suffix="conv2d_4/Conv2D1",
    ),
    "onset_final": LayerSpec(
        weight="const_fold_opt__680",
        shape=(1, 33, 3, 3),
        strides=(1, 1),
        pads=(1, 1, 1, 1),
        node_suffix="conv2d_5/BiasAdd/ReadVariableOp",
    ),
}


@dataclass(frozen=True)
class GraphTensors:
    """Tensor names and resolved convolutions used by the port and tests.

    Attributes:
        pre_stack: Normalized log CQT before the frontend BatchNorm and
            harmonic stacking.
        harmonic_stack: The `(batch, frames, 264, 8)` stack fed to the
            trunk, before the transpose into ONNX's NCHW order.
        trunk_output: The 32-channel feature after `conv4`.
        note_output: The pretrained note head after its sigmoid.
        onset_output: The pretrained onset head after its sigmoid.
        frontend_scale: Scale of the folded frontend BatchNorm.
        frontend_shift: Shift of the folded frontend BatchNorm.
        layers: Resolved Conv nodes keyed by the names in `MANIFEST`.
    """

    pre_stack: str
    harmonic_stack: str
    trunk_output: str
    note_output: str
    onset_output: str
    frontend_scale: float
    frontend_shift: float
    layers: dict[str, onnx.NodeProto]


def load_graph(path: str | Path) -> onnx.ModelProto:
    """Load an ONNX model from `path`."""
    return onnx.load(str(path))


def _attribute(node: onnx.NodeProto, name: str) -> Any:
    for attribute in node.attribute:
        if attribute.name == name:
            return helper.get_attribute_value(attribute)

    return None


def _producer_map(
    graph: onnx.GraphProto,
) -> dict[str, onnx.NodeProto]:
    producers: dict[str, onnx.NodeProto] = {}

    for node in graph.node:
        for output in node.output:
            producers[output] = node

    return producers


def _consumer_map(
    graph: onnx.GraphProto,
) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)

    for node in graph.node:
        for name in node.input:
            if name:
                consumers[name].append(node)

    return consumers


def _only_consumer(
    consumers: dict[str, list[onnx.NodeProto]],
    tensor: str,
    op_type: str,
) -> onnx.NodeProto:
    matches = [
        node
        for node in consumers[tensor]
        if node.op_type == op_type
    ]

    if len(matches) != 1:
        raise ValueError(
            f"expected one {op_type} consumer of {tensor}, "
            f"found {len(matches)}"
        )

    return matches[0]


def _traces_back_to(
    producers: dict[str, onnx.NodeProto],
    tensor: str,
    source: str,
    allowed_ops: set[str],
) -> bool:
    """Return whether `tensor` is `source` through only reshaping ops."""
    current = tensor

    while current in producers:
        node = producers[current]

        if node.output[0] == source:
            return True

        if node.op_type not in allowed_ops:
            return False

        inputs = [name for name in node.input if name in producers]

        if len(inputs) != 1:
            return False

        current = inputs[0]

    return current == source


def _find_conv(
    graph: onnx.GraphProto,
    initializers: dict[str, np.ndarray],
    spec: LayerSpec,
) -> onnx.NodeProto:
    matches = [
        node
        for node in graph.node
        if node.op_type == "Conv"
        and node.input[1] in initializers
        and initializers[node.input[1]].shape == spec.shape
    ]

    if len(matches) != 1:
        raise ValueError(
            f"expected one Conv with kernel {spec.shape}, "
            f"found {len(matches)}"
        )

    node = matches[0]

    if node.input[1] != spec.weight:
        raise ValueError(
            f"Conv with kernel {spec.shape} uses weight "
            f"{node.input[1]!r}, expected {spec.weight!r}"
        )

    if not node.name.endswith(spec.node_suffix):
        raise ValueError(
            f"Conv with kernel {spec.shape} has node name "
            f"{node.name!r}, expected suffix {spec.node_suffix!r}"
        )

    if tuple(_attribute(node, "strides") or ()) != spec.strides:
        raise ValueError(f"unexpected strides on {node.name!r}")

    if tuple(_attribute(node, "pads") or ()) != spec.pads:
        raise ValueError(f"unexpected padding on {node.name!r}")

    return node


def _trace_frontend(
    graph: onnx.GraphProto,
    producers: dict[str, onnx.NodeProto],
    initializers: dict[str, np.ndarray],
    harmonic_stack: str,
) -> tuple[str, float, float]:
    """Return the pre-stack tensor and the folded BatchNorm affine."""
    stack_slice = producers[harmonic_stack]
    concat = producers[stack_slice.input[0]]
    batch_norm = next(
        name
        for name in concat.input
        if producers[name].op_type != "Pad"
    )
    add = producers[batch_norm]
    multiply = producers[next(
        name for name in add.input if name in producers
    )]
    unsqueeze = producers[next(
        name for name in multiply.input if name in producers
    )]

    scale_name = next(
        name for name in multiply.input if name in initializers
    )
    shift_name = next(
        name for name in add.input if name in initializers
    )
    scale = np.asarray(initializers[scale_name]).reshape(-1)
    shift = np.asarray(initializers[shift_name]).reshape(-1)

    if scale.shape != (1,) or shift.shape != (1,):
        raise ValueError("frontend BatchNorm affine must be scalar")

    return unsqueeze.input[0], float(scale[0]), float(shift[0])


def find_graph_tensors(graph: onnx.ModelProto) -> GraphTensors:
    """Resolve the trunk, heads, and frontend affine by graph traversal.

    Names come from the graph, not from substring matching. The exported
    node names are mangled and several layers share prefixes, so the
    convolutions are identified by kernel shape and then checked against
    `MANIFEST`.

    Args:
        graph: Loaded Basic Pitch ONNX model.

    Returns:
        Tensor names, the frontend affine, and the resolved Conv nodes.

    Raises:
        ValueError: If the graph does not have the expected architecture.
    """
    initializers = {
        tensor.name: numpy_helper.to_array(tensor)
        for tensor in graph.graph.initializer
    }
    producers = _producer_map(graph.graph)
    consumers = _consumer_map(graph.graph)
    layers = {
        name: _find_conv(graph.graph, initializers, spec)
        for name, spec in MANIFEST.items()
    }

    conv2 = layers["conv2"]
    conv3 = layers["conv3"]
    conv4 = layers["conv4"]

    relu_after_conv2 = _only_consumer(
        consumers, conv2.output[0], "Relu"
    )
    sigmoid_after_conv3 = _only_consumer(
        consumers, conv3.output[0], "Sigmoid"
    )
    relu_after_conv4 = _only_consumer(
        consumers, conv4.output[0], "Relu"
    )

    if conv3.input[0] != relu_after_conv2.output[0]:
        raise ValueError("conv3 does not consume the conv2 activation")

    if not _traces_back_to(
        producers,
        conv4.input[0],
        sigmoid_after_conv3.output[0],
        {"Reshape", "Unsqueeze", "Squeeze", "Transpose", "ExpandDims"},
    ):
        raise ValueError("conv4 does not consume the conv3 sigmoid")

    note = layers["note"]
    note_sigmoid = _only_consumer(consumers, note.output[0], "Sigmoid")

    if note.input[0] != relu_after_conv4.output[0]:
        raise ValueError("note head does not consume the trunk output")

    onset = layers["onset"]
    onset_relu = _only_consumer(consumers, onset.output[0], "Relu")
    onset_final = layers["onset_final"]
    onset_sigmoid = _only_consumer(
        consumers, onset_final.output[0], "Sigmoid"
    )
    onset_concat = producers[onset_final.input[0]]

    if onset_concat.op_type != "Concat":
        raise ValueError("onset head does not consume a concatenation")

    if {
        note_sigmoid.output[0],
        onset_relu.output[0],
    } - set(onset_concat.input):
        raise ValueError("onset concatenation is missing a branch")

    transpose = producers[conv2.input[0]]
    harmonic_stack = transpose.input[0]
    pre_stack, scale, shift = _trace_frontend(
        graph.graph,
        producers,
        initializers,
        harmonic_stack,
    )

    return GraphTensors(
        pre_stack=pre_stack,
        harmonic_stack=harmonic_stack,
        trunk_output=relu_after_conv4.output[0],
        note_output=note_sigmoid.output[0],
        onset_output=onset_sigmoid.output[0],
        frontend_scale=scale,
        frontend_shift=shift,
        layers=layers,
    )


def extract_weights(
    graph: onnx.ModelProto,
    tensors: GraphTensors,
) -> dict[str, np.ndarray]:
    """Return the trunk and frontend arrays to write to the archive."""
    initializers = {
        tensor.name: numpy_helper.to_array(tensor)
        for tensor in graph.graph.initializer
    }
    weights: dict[str, np.ndarray] = {}

    for name in ("conv2", "conv3", "conv4"):
        node = tensors.layers[name]
        weights[f"{name}.weight"] = np.ascontiguousarray(
            initializers[node.input[1]], dtype=np.float32
        )
        weights[f"{name}.bias"] = np.ascontiguousarray(
            initializers[node.input[2]], dtype=np.float32
        )

    weights["frontend.scale"] = np.array(
        [tensors.frontend_scale],
        dtype=np.float32,
    )
    weights["frontend.shift"] = np.array(
        [tensors.frontend_shift],
        dtype=np.float32,
    )

    return weights


def write_archive(
    weights: dict[str, np.ndarray],
    output: str | Path,
) -> Path:
    """Write the weight arrays to `output` and return its path."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **weights)

    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx",
        type=Path,
        required=True,
        help="nmp.onnx from basic-pitch==0.4.0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output .npz path (default: {DEFAULT_OUTPUT})",
    )

    return parser


def main() -> None:
    """Resolve the weights, report the manifest, and write the archive."""
    args = _parser().parse_args()
    graph = load_graph(args.onnx)
    tensors = find_graph_tensors(graph)
    weights = extract_weights(graph, tensors)

    print(f"Pre-stack tensor: {tensors.pre_stack}")
    print(f"Harmonic stack: {tensors.harmonic_stack}")
    print(f"Trunk output: {tensors.trunk_output}")
    print(
        "Frontend batch norm: "
        f"scale={tensors.frontend_scale:.6f} "
        f"shift={tensors.frontend_shift:.6f}"
    )

    for name, spec in MANIFEST.items():
        node = tensors.layers[name]
        print(
            f"{name}: node={node.name} weight={spec.weight} "
            f"shape={spec.shape}"
        )

    path = write_archive(weights, args.output)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
