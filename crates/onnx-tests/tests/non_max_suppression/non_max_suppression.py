#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
# ]
# ///

# used to generate model: onnx-tests/tests/non_max_suppression/non_max_suppression.onnx

import onnx
from onnx import helper


def build_model():
    """
    Build a NonMaxSuppression ONNX model.

    Inputs:
        boxes
        scores
        max_output_boxes_per_class
        iou_threshold
        score_threshold
    """

    nodes = [
        helper.make_node(
            "NonMaxSuppression",
            inputs=[
                "boxes",
                "scores",
                "max_output_boxes_per_class",
                "iou_threshold",
                "score_threshold",
            ],
            outputs=["selected_indices"],
            name="/NonMaxSuppression",
            center_point_box=0,
        )
    ]

    BATCHES = 1
    SPATIAL_DIMENSION = 16
    NUM_CLASSES = 100
    inputs = [
        helper.make_tensor_value_info(
            "boxes",
            elem_type=onnx.TensorProto.FLOAT,
            shape=(BATCHES, SPATIAL_DIMENSION, 4),
        ),
        helper.make_tensor_value_info(
            "scores",
            elem_type=onnx.TensorProto.FLOAT,
            shape=(BATCHES, NUM_CLASSES, SPATIAL_DIMENSION),
        ),
        helper.make_tensor_value_info(
            "max_output_boxes_per_class", elem_type=onnx.TensorProto.INT64, shape=(1,)
        ),
        helper.make_tensor_value_info(
            "iou_threshold", elem_type=onnx.TensorProto.FLOAT, shape=(1,)
        ),
        helper.make_tensor_value_info(
            "score_threshold", elem_type=onnx.TensorProto.FLOAT, shape=(1,)
        ),
    ]

    NUM_SELECTED_INDICES = 3
    outputs = [
        helper.make_tensor_value_info(
            "selected_indices",
            elem_type=onnx.TensorProto.INT64,
            shape=(NUM_SELECTED_INDICES, 3),
        ),
    ]

    model = helper.make_model(
        graph=helper.make_graph(
            nodes=nodes, name="main_graph", inputs=inputs, outputs=outputs
        ),
        opset_imports=[helper.make_operatorsetid("", 11)],
    )

    onnx.checker.check_model(model)
    onnx.save(model, "nms.onnx")
    print("Finished exporting model to nms.onnx")


if __name__ == "__main__":
    build_model()
