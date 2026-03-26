use std::arch::aarch64::int16x8x4_t;

use burn_tensor::DType;
use onnx_ir_derive::NodeBuilder;

use crate::{
    ArgType, Argument, Node, RawNode, TensorType,
    processor::{InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError},
};

/// How a bounding box's format should be specified
#[derive(Debug, Clone)]
pub enum BoxFormat {
    // TODO: Improve docs
    // (y1, x1, y2, x2) where
    Corner,
    // (x_center, y_center, width, height)
    Center,
}

#[derive(Debug, Clone, new)]
pub struct NonMaxSuppressionConfig {
    pub center_point_box: BoxFormat,
}

impl TryFrom<i64> for NonMaxSuppressionConfig {
    type Error = ProcessError;

    fn try_from(value: i64) -> Result<Self, Self::Error> {
        let box_format = match value {
            0 => BoxFormat::Corner,
            1 => BoxFormat::Center,
            _ => {
                return Err(ProcessError::InvalidInputCount {
                    expected: 0,
                    // TODO: Add comment or change impl. to handle positive and negative case
                    actual: usize::try_from(value).unwrap_or(usize::MAX),
                });
            }
        };

        Ok(Self {
            center_point_box: box_format,
        })
    }
}

#[derive(Debug, Clone, NodeBuilder)]
pub struct NonMaxSuppressionNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: NonMaxSuppressionConfig,
}

pub(crate) struct NonMaxSuppressionProcessor;

impl NodeProcessor for NonMaxSuppressionProcessor {
    type Config = NonMaxSuppressionConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 10,
            max_opset: None,
            inputs: InputSpec::Range(2, 5),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        _node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Per the spec, the output type is heterogenous
        // (i.e. independent on the input).
        _node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: DType::I64,
            rank: 3,
            static_shape: None,
        });

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError>
    where
        Self: Sized,
    {
        node.attrs
            .get("center_point_box")
            .map(|value| value.clone().into_i64())
            .unwrap_or(0)
            .try_into()
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node
    where
        Self: Sized,
    {
        let config = self
            .extract_config(&builder, opset)
            .expect("config extraction failed");

        Node::NonMaxSuppression(NonMaxSuppressionNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}
