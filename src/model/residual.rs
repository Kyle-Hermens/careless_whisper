use crate::model::multi_head::MultiHeadAttention;
use crate::model::{WhisperLayerNorm, WhisperLinear};
use std::collections::HashMap;
use tch::nn::{Module, Sequential};
use tch::{nn, Tensor};

#[derive(Debug)]
pub struct ResidualAttentionBlock {
    attn: MultiHeadAttention,
    attn_ln: WhisperLayerNorm,
    cross_atn: Option<MultiHeadAttention>,
    cross_attn_ln: Option<WhisperLayerNorm>,
    mlp: Sequential,
    mlp_ln: WhisperLayerNorm,
}

impl ResidualAttentionBlock {
    pub(crate) fn new(n_state: i64, n_head: i64, cross_attention: bool) -> Self {
        let n_mlp = n_state * 4;
        let cross_atn = if cross_attention {
            Some(MultiHeadAttention::new(n_state, n_head))
        } else {
            None
        };
        let cross_attn_ln = if cross_attention {
            Some(WhisperLayerNorm::new(n_state))
        } else {
            None
        };
        ResidualAttentionBlock {
            attn: MultiHeadAttention::new(n_state, n_head),
            attn_ln: WhisperLayerNorm::new(n_state),
            cross_atn,
            cross_attn_ln,
            mlp: nn::seq()
                .add(WhisperLinear::new(n_state, n_mlp, true))
                .add_fn(|xs| xs.gelu("none")) //TODO: hopefully this is equivalent to applying GELU()
                .add(WhisperLinear::new(n_mlp, n_state, true)),
            mlp_ln: WhisperLayerNorm::new(n_state),
        }
    }

    //TODO: switch to wrapper for map
    pub fn _forward(
        &self,
        x: &Tensor,
        xa: Option<&Tensor>,
        mask: Option<&Tensor>,
        kv_cache: &mut Option<HashMap<String, Tensor>>,
    ) -> Tensor {
        //TODO: make sure Tensor addition order doesn't matter
        let mut x = self
            .attn
            .forward(self.attn_ln.forward(x), None, mask, kv_cache)
            + x;
        if let (Some(ref cross), Some(ref cross_ln)) = (&self.cross_atn, &self.cross_attn_ln) {
            x = cross.forward(cross_ln.forward(&x), xa, None, kv_cache) + x;
        }
        self.mlp.forward(&self.mlp_ln.forward(&x)) + x
    }
}

impl Module for ResidualAttentionBlock {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self._forward(xs, None, None, &mut None)
    }
}
