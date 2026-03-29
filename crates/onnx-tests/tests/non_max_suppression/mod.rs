use crate::include_models;
include_models!(nms);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn non_max_suppression() {
        let device = Default::default();
        let model: non_max_suppression::Model<TestBackend> =
            non_max_suppression::Model::new(&device);

        // TODO: Create input tensors matching what your .py script uses
        // boxes:   Tensor<TestBackend, 3> shape [num_batches, num_boxes, 4]
        // scores:  Tensor<TestBackend, 3> shape [num_batches, num_classes, num_boxes]
        // max_output_boxes_per_class: i64
        // iou_threshold: f32
        // score_threshold: f32
        let boxes = Tensor::<TestBackend, 3>::from_floats(floats, device);
        let scores = Tensor::<TestBackend, 3>::from_floats(floats, device);
        let max_output_boxes_per_class = 5;
        let iou_threshold = 0.5;
        let score_threshold = 0.8;

        // TODO: Call model.forward(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
        let output = model.forward(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold);
        let expected = TensorData::from(...);

        output.to_data().assert_eq(&expected, True);
    }
}
