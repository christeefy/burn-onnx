#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "onnxruntime>=1.22.0",
#   "numpy",
#   "torch",
# ]
# ///

import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnx

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import get_artifacts_dir

MODEL_NAME = "squeezenet"
ONNX_MODEL_ZOO_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.0-12.onnx"


def download_model(output_path):
    """Download SqueezeNet 1.0 ONNX model from ONNX Model Zoo."""
    print("Downloading SqueezeNet 1.0 (opset 12) model from ONNX Model Zoo...")
    print(f"  Source: {ONNX_MODEL_ZOO_URL}")

    try:
        urllib.request.urlretrieve(ONNX_MODEL_ZOO_URL, output_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {e}")

    if not output_path.exists():
        raise FileNotFoundError(f"Failed to download ONNX file to {output_path}")

    print(f"  Model downloaded to: {output_path}")


def prepare_model(input_path, output_path):
    """Apply shape inference to the model."""
    print("\nPreparing model with shape inference...")
    print(f"  Input: {input_path}")

    model = onnx.load(input_path)

    # Apply shape inference
    print("  Applying shape inference...")
    model = onnx.shape_inference.infer_shapes(model)

    onnx.save(model, output_path)
    print(f"  Model saved to: {output_path}")


def generate_test_data(model_path, output_dir):
    """Generate test input/output data and save as PyTorch tensors."""
    import onnxruntime as ort
    import torch

    print("\nGenerating test data...")

    model = onnx.load(model_path)

    print(f"  Model opset version: {model.opset_import[0].version}")
    print("  Inputs:")
    for input_info in model.graph.input:
        shape = []
        for dim in input_info.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                shape.append("dynamic")
        print(f"    - {input_info.name}: shape={shape}")

    print("  Outputs:")
    for output_info in model.graph.output:
        shape = []
        for dim in output_info.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                shape.append("dynamic")
        print(f"    - {output_info.name}: shape={shape}")

    # Generate random input image (ImageNet-like normalization)
    np.random.seed(42)
    test_input = np.random.rand(1, 3, 224, 224).astype(np.float32)

    # Get input name from model
    input_name = model.graph.input[0].name

    # Run inference through ONNX Runtime
    print("\n  Running inference through ONNX Runtime...")
    session = ort.InferenceSession(model_path)
    outputs = session.run(None, {input_name: test_input})

    output_names = [o.name for o in session.get_outputs()]
    print(f"  Output names: {output_names}")
    for i, name in enumerate(output_names):
        print(f"    - {name}: shape={outputs[i].shape}, dtype={outputs[i].dtype}")

    # Save test data as PyTorch tensors
    test_data = {}
    test_data[input_name] = torch.from_numpy(test_input)

    for i, name in enumerate(output_names):
        test_data[name] = torch.from_numpy(outputs[i])

    test_data_path = Path(output_dir) / "test_data.pt"
    torch.save(test_data, test_data_path)
    print(f"\n  Test data saved to: {test_data_path}")


def save_model_info(model_path, output_dir):
    """Save model structure information to a text file."""
    print("\nSaving model information...")

    model = onnx.load(model_path)

    info_path = Path(output_dir) / "model-python.txt"
    with open(info_path, "w") as f:
        f.write("SqueezeNet 1.0 Model Information\n")
        f.write("=" * 60 + "\n\n")

        f.write("Inputs:\n")
        for input_info in model.graph.input:
            f.write(f"  - {input_info.name}\n")
            shape = []
            for dim in input_info.type.tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(dim.dim_value)
                else:
                    shape.append("dynamic")
            f.write(f"    Shape: {shape}\n")
            f.write(
                f"    Type: {onnx.TensorProto.DataType.Name(input_info.type.tensor_type.elem_type)}\n"
            )

        f.write("\nOutputs:\n")
        for output_info in model.graph.output:
            f.write(f"  - {output_info.name}\n")
            shape = []
            for dim in output_info.type.tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(dim.dim_value)
                else:
                    shape.append("dynamic")
            f.write(f"    Shape: {shape}\n")
            f.write(
                f"    Type: {onnx.TensorProto.DataType.Name(output_info.type.tensor_type.elem_type)}\n"
            )

        f.write("\nModel Statistics:\n")
        f.write(f"  Opset version: {model.opset_import[0].version}\n")
        f.write(f"  Number of nodes: {len(model.graph.node)}\n")
        f.write(f"  Number of initializers: {len(model.graph.initializer)}\n")

        node_types = {}
        for node in model.graph.node:
            node_types[node.op_type] = node_types.get(node.op_type, 0) + 1

        f.write("\nNode types:\n")
        for op_type, count in sorted(node_types.items()):
            f.write(f"  {op_type}: {count}\n")

    print(f"  Model info saved to: {info_path}")


def main():
    print("=" * 60)
    print("SqueezeNet 1.0 Model Preparation Tool")
    print("=" * 60)

    artifacts_dir = get_artifacts_dir(MODEL_NAME)

    original_path = artifacts_dir / "squeezenet1.0-12.onnx"
    prepared_path = artifacts_dir / "squeezenet.onnx"
    test_data_path = artifacts_dir / "test_data.pt"
    model_info_path = artifacts_dir / "model-python.txt"

    # Check if all files already exist
    if prepared_path.exists() and test_data_path.exists() and model_info_path.exists():
        print("\n  All files already exist:")
        print(f"  Model: {prepared_path}")
        print(f"  Test data: {test_data_path}")
        print(f"  Model info: {model_info_path}")
        print("\nNothing to do!")
        return

    # Step 1: Download if needed
    if not original_path.exists() and not prepared_path.exists():
        print("\nStep 1: Downloading model...")
        download_model(original_path)
    else:
        print("\nStep 1: Model already downloaded, skipping...")

    # Step 2: Prepare (apply shape inference)
    if not prepared_path.exists():
        print("\nStep 2: Preparing model (shape inference)...")
        prepare_model(original_path, prepared_path)
        # Clean up original if we prepared from it
        if original_path.exists() and prepared_path.exists():
            original_path.unlink()
    else:
        print("\nStep 2: Model already prepared, skipping...")

    # Step 3: Generate test data
    if not test_data_path.exists():
        print("\nStep 3: Generating test data...")
        generate_test_data(prepared_path, artifacts_dir)
    else:
        print("\nStep 3: Test data already exists, skipping...")

    # Step 4: Save model info
    if not model_info_path.exists():
        print("\nStep 4: Saving model information...")
        save_model_info(prepared_path, artifacts_dir)
    else:
        print("\nStep 4: Model info already exists, skipping...")

    print("\n" + "=" * 60)
    print("  SqueezeNet 1.0 model preparation completed!")
    print(f"  Model: {prepared_path}")
    print(f"  Test data: {test_data_path}")
    print(f"  Model info: {model_info_path}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n  Error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
