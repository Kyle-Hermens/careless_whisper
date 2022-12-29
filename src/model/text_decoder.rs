use crate::model::residual::ResidualAttentionBlock;
use crate::model::WhisperLayerNorm;
use std::collections::HashMap;
use tch::nn::{Embedding, EmbeddingConfig, Module, VarStore};
use tch::{nn, Device, IndexOp, Kind, Tensor, TensorIndexer};

#[derive(Debug)]
pub struct TextDecoder {
    token_embedding: Embedding,
    positional_embedding: Tensor, //TODO: check if we need/can make parameter
    blocks: Vec<ResidualAttentionBlock>, //TODO: check if ModuleList is necessary
    ln: WhisperLayerNorm,
    mask: Tensor, //TODO: check register buffer
}

impl TextDecoder {
    pub fn _forward(
        &self,
        x: &Tensor,
        xa: &Tensor,
        kv_cache: &mut Option<HashMap<String, Tensor>>,
    ) -> Tensor {
        //TODO: clean this up
        let offset = kv_cache
            .iter()
            .filter_map(|map| map.values().next().map(|t| t.size()[1]))
            .next()
            .unwrap_or(0_i64);
        //TODO: verify indexing and shape call;
        let shape = x.size();
        let mut x = self.token_embedding.forward(&x)
            + self.positional_embedding.i((
                TensorIndexer::Select(offset),
                TensorIndexer::Select(offset + shape[shape.len() - 1]),
            ));

        x = x.to_dtype(xa.kind(), false, false);

        for block in &self.blocks {
            x = block._forward(&x, Some(xa), Some(&self.mask), kv_cache);
        }
        x = self.ln.forward(&x);
        x.matmul(
            &self
                .token_embedding
                .ws
                .to_dtype(x.kind(), false, false)
                .transpose(0, 1),
        )
        .to_dtype(Kind::Float, false, false)
    }

    pub fn new(n_vocab: i64, n_ctx: i64, n_state: i64, n_head: i64, n_layer: i64) -> Self {
        let mut blocks = Vec::with_capacity(n_layer as usize);
        blocks.fill_with(|| ResidualAttentionBlock::new(n_state, n_head, true));
        TextDecoder {
            //TODO: usual VarStore warning
            token_embedding: nn::embedding(
                VarStore::new(Device::cuda_if_available()).root(),
                n_vocab,
                n_ctx,
                EmbeddingConfig::default(),
            ),
            //TODO: doublecheck kind
            positional_embedding: Tensor::empty(
                &[n_vocab, n_state],
                (Kind::Double, Device::cuda_if_available()),
            ),
            blocks,
            ln: WhisperLayerNorm::new(n_state),
            //TODO: doublecheck kind
            //TODO: register_buffer for mask
            mask: Tensor::empty(&[n_ctx, n_ctx], (Kind::Double, Device::cuda_if_available()))
                .fill_(f64::NEG_INFINITY)
                .triu_(1),
        }
    }
}
