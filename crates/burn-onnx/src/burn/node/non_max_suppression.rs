use onnx_ir::non_max_suppression::{BoxFormat, NonMaxSuppressionNode};

use super::prelude::*;

impl NodeCodegen for NonMaxSuppressionNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        if matches!(self.config.center_point_box, BoxFormat::Center) {
            panic!("Burn does not yet support center-box point positioning")
        }

        // TODO: Generate default values only when needed. Do not allocated unnecessarily. This would entail using a match statement, and allocating only in the None branch.
        let max_output_boxes_per_class_default =
            Argument::from_const_i64("max_output_boxes_per_class", 0);
        let iou_threshold_default = Argument::from_const_i64("iou_threshold", 0);
        let score_threshold_default = Argument::from_const_i64("score_threshold", 0);

        let boxes_arg = self.inputs.first().expect("Missing `boxes` input");
        let scores_arg = self.inputs.get(1).expect("Missing `scores` input");
        let max_output_boxes_per_class_arg = self
            .inputs
            .get(2)
            .unwrap_or(&max_output_boxes_per_class_default);
        let iou_threshold_arg = self.inputs.get(3).unwrap_or(&iou_threshold_default);
        let score_threshold_arg = self.inputs.get(4).unwrap_or(&score_threshold_default);
        let output = arg_to_ident(
            self.outputs
                .first()
                .expect("Missing `selected_indices` output"),
        );

        let boxes = scope.arg(boxes_arg);
        let scores = scope.arg(scores_arg);
        let max_output_boxes_per_class = scope.arg(max_output_boxes_per_class_arg);
        let iou_threshold = scope.arg(iou_threshold_arg);
        let score_threshold = scope.arg(score_threshold_arg);

        match (
            &boxes_arg.ty,
            &scores_arg.ty,
            &max_output_boxes_per_class_arg.ty,
            &iou_threshold_arg.ty,
            &score_threshold_arg.ty,
        ) {
            (
                ArgType::Tensor(_),
                ArgType::Tensor(_),
                ArgType::ScalarNative(_),
                ArgType::ScalarNative(_),
                ArgType::ScalarNative(_),
            ) => {
                quote! {
                    let [_, num_classes, _] = #boxes.shape().dims();
                    let options = burn::vision::ops::NmsOptions {
                        iou_threshold: #iou_threshold,
                        score_threshold: #score_threshold,
                        max_output_boxes: #max_output_boxes_per_class * num_classes,
                    };

                    let #output = burn::vision::backends::cpu::nms(  // TODO: This is private
                        #boxes, // TODO[Q]: The burn impl. treates this as [x1, y1, x2, y2] whilst the ONNX spec is [y1, x1, y2, x2]
                        #scores,
                        options,
                    );
                }
            }
            _ => {
                panic!("Unsupported argument types for NonMaxSuppression in burn-onnx");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use super::*;
    use insta::assert_snapshot;
    use onnx_ir::non_max_suppression::{NonMaxSuppressionConfig, NonMaxSuppressionNodeBuilder};

    #[test]
    fn test_nms_codegen() {
        let config = NonMaxSuppressionConfig::new(BoxFormat::Corner);
        let node = NonMaxSuppressionNodeBuilder::new("nms")
            .input_tensor("boxes", 3, DType::F64)
            .input_tensor("scores", 3, DType::F64)
            .input_scalar("max_output_boxes_per_class", DType::I64)
            .input_scalar("iou_threshold", DType::F64)
            .input_scalar("score_threshold", DType::F64)
            .output_tensor("selected_indices", 3, DType::I64)
            .config(config)
            .build();

        let code = codegen_forward_default(&node);
        assert_snapshot!(
            code, @r"
        pub fn forward(
            &self,
            boxes: Tensor<B, 3>,
            scores: Tensor<B, 3>,
            max_output_boxes_per_class: i64,
            iou_threshold: f64,
            score_threshold: f64,
        ) -> Tensor<B, 3, Int> {
            let [_, num_classes, _] = boxes.shape().dims();
            let options = burn::vision::ops::NmsOptions {
                iou_threshold: iou_threshold,
                score_threshold: score_threshold,
                max_output_boxes: max_output_boxes_per_class * num_classes,
            };
            let selected_indices = burn::vision::backends::cpu::nms(boxes, scores, options);
            selected_indices
        }
        "
        );
    }
}
