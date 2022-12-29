mod audio_encoder;
mod multi_head;
mod residual;
mod tests;
mod text_decoder;

use crate::model::audio_encoder::AudioEncoder;
use crate::model::residual::ResidualAttentionBlock;
use crate::model::text_decoder::TextDecoder;
use std::any::Any;
use std::collections::{BTreeMap, HashMap};
use std::ops::Mul;
use tch::nn::{
    conv1d, layer_norm, linear, Conv1D, ConvConfigND, Embedding, LayerNorm, LayerNormConfig,
    LinearConfig, Module, Sequential, VarStore,
};
use tch::{nn, Device, IndexOp, Kind, Tensor, TensorIndexer};

#[derive(Debug)]
struct WhisperLinear {
    linear: nn::Linear,
}

impl Module for WhisperLinear {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let weight = self.linear.ws.to_dtype(xs.kind(), false, false);
        let bias = self
            .linear
            .bs
            .as_ref()
            .map(|bias| bias.to_dtype(xs.kind(), false, false));
        xs.linear(&weight, bias)
    }
}
impl WhisperLinear {
    fn new(int_features: i64, out_features: i64, bias: bool) -> Self {
        let mut linear_config = LinearConfig::default();
        linear_config.bias = bias;
        WhisperLinear {
            linear: nn::linear(
                //TODO: make one shared varstore, probably
                VarStore::new(Device::cuda_if_available()).root(), //TODO: fix device usage
                int_features,
                out_features,
                linear_config,
            ),
        }
    }
}
//Linear bias is true by default

#[derive(Debug)]
struct WhisperLayerNorm {
    layer_norm: LayerNorm,
}

impl Module for WhisperLayerNorm {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.layer_norm
            .forward(&xs.to_dtype(Kind::Float, false, false))
            .to_dtype(xs.kind(), false, false)
    }
}

impl WhisperLayerNorm {
    fn new(n_state: i64) -> Self {
        WhisperLayerNorm {
            layer_norm: layer_norm(
                //TODO: check which varstore to use.
                VarStore::new(Device::cuda_if_available()).root(),
                vec![n_state],
                LayerNormConfig::default(),
            ),
        }
    }
}

#[derive(Debug)]
struct WhisperConv1D {
    conv1d: Conv1D,
}

impl WhisperConv1D {
    fn forward(&self, x: &Tensor) -> Tensor {
        //TODO change weight and bias datatype
        self.conv1d.forward(x)
    }
}

impl WhisperConv1D {
    fn new(
        in_channels: i64,
        out_channels: i64,
        kernel_size: i64,
        stride: i64,
        padding: i64,
    ) -> Self {
        //A Tensor
        //Might need to call conv and then create a Conv1D via conv1d
        // let item = tch::Tensor::f_conv1d()
        let mut config = ConvConfigND::default();
        config.stride = stride;
        config.padding = padding;

        let conv1d = conv1d(
            VarStore::new(Device::cuda_if_available()).root(),
            in_channels,
            out_channels,
            kernel_size,
            config,
        );

        WhisperConv1D { conv1d }
    }
}

fn sinusoids(length: f64, channels: i64) -> Tensor {
    let max_timescale = 10000;
    assert_eq!(channels % 2, 0);
    let log_timescale_increment = (max_timescale as f64).ln() / (channels / 2 - 1) as f64;
    let inv_timescales = tch::Tensor::arange(
        (channels / 2) as f64,
        (Kind::Double, Device::cuda_if_available()), //TODO: needs checking
    )
    .mul(-log_timescale_increment)
    .exp();
    let scaled_time = Tensor::arange(length, (Kind::Int64, Device::cuda_if_available()))
        .i((.., TensorIndexer::InsertNewAxis))
        * inv_timescales.i((TensorIndexer::InsertNewAxis, ..));

    Tensor::cat(&[scaled_time.sin(), scaled_time.cos()], 1)
}
#[derive(Debug)]
struct ModelDimensions {
    n_mels: i64,
    n_audio_ctx: i64,
    n_audio_state: i64,
    n_audio_head: i64,
    n_audio_layer: i64,
    n_vocab: i64,
    n_text_ctx: i64,
    n_text_state: i64,
    n_text_head: i64,
    n_text_layer: i64,
}

#[derive(Debug)]
struct Whisper {
    dims: ModelDimensions,
    encoder: AudioEncoder,
    decoder: TextDecoder,
}

impl Whisper {
    pub fn new(dims: ModelDimensions) -> Self {
        Whisper {
            encoder: AudioEncoder::new(
                dims.n_mels,
                dims.n_audio_ctx,
                dims.n_audio_state,
                dims.n_audio_head,
                dims.n_audio_layer,
            ),
            decoder: TextDecoder::new(
                dims.n_vocab,
                dims.n_text_ctx,
                dims.n_text_state,
                dims.n_text_head,
                dims.n_text_layer,
            ),
            dims,
        }
    }

    pub fn embed_audio(&self, mel: &Tensor) -> Tensor {
        self.encoder.forward(mel)
    }

    pub fn logits(&self, tokens: Tensor, audio_features: Tensor) -> Tensor {
        //TODO: may need to implicitly pass in kv_cache
        self.decoder._forward(&tokens, &audio_features, &mut None)
    }

    //TODO: why does this actually return a Dictionary of string to tensor?
    pub fn forward(&self, mel: &Tensor, tokens: &Tensor) -> HashMap<String, Tensor> {
        //TODO: pass in kv_cache?
        let result = self
            .decoder
            ._forward(tokens, &self.encoder.forward(&mel), &mut None);
        todo!()
    }

    pub fn is_multilingual(&self) -> bool {
        self.dims.n_vocab == 51865
    }
    pub fn device(&self) -> Device {
        todo!()
    }

    pub fn install_kv_cache_hooks(&self, cache: Option<HashMap<String, Tensor>>) {
        todo!()
    }
}
