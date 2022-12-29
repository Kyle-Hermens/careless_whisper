use crate::model;
use crate::model::residual::ResidualAttentionBlock;
use crate::model::{WhisperConv1D, WhisperLayerNorm};
use tch::nn::Module;
use tch::Tensor;

#[derive(Debug)]
pub struct AudioEncoder {
    conv1: WhisperConv1D,
    conv2: WhisperConv1D,
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln_post: WhisperLayerNorm,
}

impl AudioEncoder {
    pub fn new(n_mels: i64, n_ctx: i64, n_state: i64, n_head: i64, n_layer: i64) -> Self {
        let mut blocks = Vec::with_capacity(n_layer as usize);
        blocks.fill_with(|| ResidualAttentionBlock::new(n_state, n_head, false));
        AudioEncoder {
            positional_embedding: model::sinusoids(n_ctx as f64, n_state),
            conv1: WhisperConv1D::new(n_mels, n_state, 3, 1, 1),
            conv2: WhisperConv1D::new(n_state, n_state, 3, 2, 1),
            blocks,
            ln_post: WhisperLayerNorm::new(n_state),
        }
    }
}

impl Module for AudioEncoder {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut x = self.conv1.forward(xs).gelu("none");
        x = self.conv2.forward(&x).gelu("none");
        x = x.permute(&[0, 2, 1]);
        //TODO: put assert on x.shape here
        let kind = x.kind();
        let embed = self.positional_embedding.copy();
        x = (x + embed).to_dtype(kind, false, false);

        for block in &self.blocks {
            x = block.forward(&x);
        }
        self.ln_post.forward(&x)
    }
}
