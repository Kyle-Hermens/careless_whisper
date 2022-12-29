use crate::model::WhisperLinear;
use std::collections::HashMap;
use tch::nn::Module;
use tch::Tensor;

#[derive(Debug)]
pub struct MultiHeadAttention {
    n_head: i64,
    query: WhisperLinear,
    key: WhisperLinear,
    value: WhisperLinear,
    out: WhisperLinear,
}

impl MultiHeadAttention {
    pub(crate) fn new(n_state: i64, n_head: i64) -> Self {
        MultiHeadAttention {
            n_head,
            query: WhisperLinear::new(n_state, n_state, true),
            key: WhisperLinear::new(n_state, n_state, false),
            value: WhisperLinear::new(n_state, n_state, true),
            out: WhisperLinear::new(n_state, n_state, true),
        }
    }

    pub fn forward(
        &self,
        xs: Tensor,
        xa: Option<&Tensor>,
        mask: Option<&Tensor>,
        kv_cache: &mut Option<HashMap<String, Tensor>>,
    ) -> Tensor {
        //TODO: make args references
        //TODO: make proper HashMap of Tensors and not use Strings
        let q = self.query.linear.forward(&xs);
        // let (k, v) = if kv_cache.is_none() || xa.is_none() ||
        // match (kv_cache, xa) {
        //     (Some(map), Some(xa_tensor)) if !map.contains_key(&self.key.to_string()) => {}
        // }
        let k;
        let v;
        let s: String = todo!();
        //TODO: use allclose with a newtype struct for Eq implementation

        if kv_cache.is_none()
            || xa.is_none()
            || kv_cache.filter(|map| map.contains_key(&s)).is_none()
        {
            let k = self.key.forward(&xa.unwrap_or(todo!()));
            let v = self.value.forward(&xa.unwrap_or(todo!()));
        } else {
            //TODO: use a wrapper type that uses f_equal
            k = todo!();
            v = todo!();
            // k = kv_cache
            //     .as_ref()
            //     .expect("Couldn't  unwrap kv")
            //     .get(&self.key.linear.ws)
            //     .map(|key| {
            //         let out = Tensor::default();
            //         key.clone(&out)
            //     })
            //     .expect("No key present");
            // v = kv_cache
            //     .as_ref()
            //     .expect("Couldn't  unwrap kv")
            //     .get(&self.value.linear.ws)
            //     .map(|val| {
            //         let out = Tensor::default();
            //         val.clone(&out)
            //     })
            //     .expect("No key present");
        }

        let wv = self.qkv_attention(&q, &k, &v, mask);
        self.out.forward(&wv)
    }

    fn qkv_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Tensor {
        todo!()
    }
}
