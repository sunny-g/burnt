use crate::{module_fn, Error, LoadFromSafeTensors, SafeTensorsConfig};
use burn::{
    config::Config,
    module::{Module, ModuleMapper, ModuleVisitor, Param, ParamId, RunningState},
    nn::{self, cache::TensorCache, Embedding, LayerNorm, Linear, ReLU},
};
use burn_tensor::{activation::sigmoid, backend::Backend, Float, Numeric, Tensor, TensorKind};
use rayon::prelude::*;
use safetensors::SafeTensors;
use std::{
    borrow::Cow,
    io::{stdout, Write},
};
use tokenizers::{Encoding, Tokenizer};

const LN_OVERRIDES: [(&str, &str); 2] = [("gamma", "weight"), ("beta", "bias")];

///
#[derive(Debug, Module)]
pub struct Mix<B: Backend> {
    inner: Param<Tensor<B, 1>>,
}

impl<B: Backend> Mix<B> {
    pub fn new_with(name: impl AsRef<str>, val: Tensor<B, 1>) -> Self {
        let inner = Param::new(name.as_ref().into(), val);
        Self { inner }
    }

    /// (x * mix) + (last_x * -mix)
    ///
    /// # Shapes
    ///
    /// - x: `[..., any, d_model]`
    /// - last_x: `[..., any, d_model]`
    /// - output: `[..., any, d_model]`
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>, last_x: Tensor<B, D>) -> Tensor<B, D> {
        let mix = self.inner.val();
        (x * mix.clone().unsqueeze()) + (last_x * mix.add_scalar(-1).unsqueeze())
    }
}

impl<B: Backend> From<Param<Tensor<B, 1>>> for Mix<B> {
    fn from(inner: Param<Tensor<B, 1>>) -> Self {
        Self { inner }
    }
}

/// Corresponds to:
/// 1. blocks.N.att.time_mix_[kvr]
/// 2. blocks.N.ffn.time_mix_[kr]
impl<'a, B: Backend> LoadFromSafeTensors<B> for MixRecord<B> {
    fn load<'data>(config: SafeTensorsConfig<B>, st: &SafeTensors<'data>) -> Result<Self, Error> {
        let id = config.name().into();
        config.load(st).map(|tensor| Self {
            inner: Param::new(id, tensor),
        })
    }
}

/*
 * Attention
 */

// ? cache??
#[derive(Debug, Module)]
pub struct AttentionTime<B: Backend> {
    first: Param<Tensor<B, 1>>,
    decay: Param<Tensor<B, 1>>,
    mix_k: Mix<B>,
    mix_v: Mix<B>,
    mix_r: Mix<B>,
}

impl<'a, B: Backend> AttentionTime<B> {
    pub fn new(n_embd: usize) -> Self {
        Self {
            first: Param::new("time_first".into(), Tensor::zeros([n_embd])),
            decay: Param::new("time_decay".into(), Tensor::zeros([n_embd])),
            mix_k: Mix::new_with("mix_k", Tensor::zeros([n_embd])),
            mix_v: Mix::new_with("mix_v", Tensor::zeros([n_embd])),
            mix_r: Mix::new_with("mix_r", Tensor::zeros([n_embd])),
        }
    }
}

/// Corresponds to:
/// 1. blocks.N.time_[first,decay]
/// 2. blocks.N.time_mix_[kvr]
impl<'a, B: Backend> LoadFromSafeTensors<B> for AttentionTimeRecord<B> {
    fn load<'data>(config: SafeTensorsConfig<B>, st: &SafeTensors<'data>) -> Result<Self, Error> {
        Ok(Self {
            first: config.scoped("time_first").load(st)?,
            // todo: Time decay can be precomputed to simplify inference.
            // todo: neg exp?
            decay: config.scoped("time_decay").load(st)?,
            mix_k: config.scoped("time_mix_k").load(st)?,
            mix_v: config.scoped("time_mix_v").load(st)?,
            mix_r: config.scoped("time_mix_r").load(st)?,
        })
    }
}

///
#[derive(Debug, Module)]
pub struct Attention<B: Backend> {
    time: AttentionTime<B>,
    key: Linear<B>,
    value: Linear<B>,
    receptance: Linear<B>,
    output: Linear<B>,
}

impl<'a, B: Backend> Attention<B> {
    pub fn new(n_embd: usize) -> Self {
        Self {
            time: AttentionTime::new(n_embd),
            key: nn::LinearConfig::new(n_embd, n_embd).init(),
            value: nn::LinearConfig::new(n_embd, n_embd).init(),
            receptance: nn::LinearConfig::new(n_embd, n_embd).init(),
            output: nn::LinearConfig::new(n_embd, n_embd).init(),
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn forward<const D: usize>(
        &self,
        x: Tensor<B, D>,
        state: &mut State<B, D>,
    ) -> Tensor<B, D> {
        let (r, k, v) = self.forward_rkv(x, state);

        let pp = state.p.value();
        let aa = state.a.value();
        let bb = state.b.value();
        let ww = self.time.first.val().unsqueeze().add(k.clone());

        let qq = pp.clone().maximum(ww.clone());
        let wkv = {
            let e1 = (pp.clone() - qq.clone()).exp();
            let e2 = (ww - qq).exp();
            (e1.clone() * aa.clone()) + (e2.clone() * v.clone()) / (e1 * bb.clone() + e2)
        };

        let ww = pp + self.time.decay.val().unsqueeze();
        let qq = ww.clone().maximum(k.clone());
        let e1 = (ww - qq.clone()).exp();
        let e2 = (k.clone() - qq.clone()).exp();

        state.a.update((e1.clone() * aa) + (e2.clone() * v));
        state.b.update((e1 * bb) + e2);
        state.p.update(qq);

        self.output.forward(r * wkv)
    }

    fn forward_rkv<const D: usize>(
        &self,
        x: Tensor<B, D>,
        state: &mut State<B, D>,
    ) -> (Tensor<B, D>, Tensor<B, D>, Tensor<B, D>) {
        let last_x = state.x.value();

        let xr = self.time.mix_r.forward(x.clone(), last_x.clone());
        let xk = self.time.mix_k.forward(x.clone(), last_x.clone());
        let xv = self.time.mix_v.forward(x.clone(), last_x);

        let sr = sigmoid(self.receptance.forward(xr));
        let k = self.key.forward(xk);
        let v = self.value.forward(xv);

        state.x.update(x.clone());

        (sr, k, v)
    }
}

#[cfg(feature = "torch")]
impl Attention<burn_tch::TchBackend> {
    #[cfg(feature = "cuda")]
    pub fn forward<const D: usize>(
        &self,
        x: Tensor<B, D>,
        state: &mut State<B, D>,
    ) -> Tensor<B, D> {
        let (r, k, v) = self.forward_rkv(x, state);
    }
}

/// Corresponds to:
/// 1. blocks.N.att.[key,value,output,receptance].weight
/// 3. Keys described in AttTime.
impl<'a, B: Backend> LoadFromSafeTensors<B> for AttentionRecord<B> {
    fn load<'data>(config: SafeTensorsConfig<B>, st: &SafeTensors<'data>) -> Result<Self, Error> {
        Ok(Self {
            key: config.scoped("key").load(st)?,
            value: config.scoped("value").load(st)?,
            output: config.scoped("output").load(st)?,
            receptance: config.scoped("receptance").load(st)?,
            time: config.load(st)?,
        })
    }
}

/*
 * FFN
 */

#[derive(Debug, Module)]
pub struct FeedForwardNetworkTime<B: Backend> {
    mix_k: Mix<B>,
    mix_r: Mix<B>,
}

impl<B: Backend> FeedForwardNetworkTime<B> {
    pub fn new(n_embd: usize) -> Self {
        Self {
            mix_k: Mix::new_with("mix_k", Tensor::zeros([n_embd])),
            mix_r: Mix::new_with("mix_r", Tensor::zeros([n_embd])),
        }
    }
}

/// Corresponds to:
/// 1. blocks.N.ffn.time_mix_[kr]
impl<'a, B: Backend> LoadFromSafeTensors<B> for FeedForwardNetworkTimeRecord<B> {
    fn load<'data>(config: SafeTensorsConfig<B>, st: &SafeTensors<'data>) -> Result<Self, Error> {
        Ok(Self {
            mix_k: config.scoped("time_mix_k").load(st)?,
            mix_r: config.scoped("time_mix_r").load(st)?,
        })
    }
}

#[derive(Debug, Module)]
pub struct FeedForwardNetwork<B: Backend> {
    time: FeedForwardNetworkTime<B>,
    key: Linear<B>,
    value: Linear<B>,
    receptance: Linear<B>,
}

impl<'a, B: Backend> FeedForwardNetwork<B> {
    pub fn new(n_embd: usize) -> Self {
        Self {
            time: FeedForwardNetworkTime::new(n_embd),
            key: nn::LinearConfig::new(n_embd, n_embd).init(),
            value: nn::LinearConfig::new(n_embd, n_embd).init(),
            receptance: nn::LinearConfig::new(n_embd, n_embd).init(),
        }
    }

    pub fn new_with(n_embd: usize, record: FeedForwardNetworkRecord<B>) -> Self {
        Self {
            time: FeedForwardNetworkTime::new(n_embd).load_record(record.time),
            key: nn::LinearConfig::new(n_embd, n_embd).init_with(record.key),
            value: nn::LinearConfig::new(n_embd, n_embd).init_with(record.value),
            receptance: nn::LinearConfig::new(n_embd, n_embd).init_with(record.receptance),
        }
    }

    pub fn forward<const D: usize>(
        &self,
        x: Tensor<B, D>,
        state: &mut State<B, D>,
    ) -> Tensor<B, D> {
        let last_x = state.x.value();

        let xr = self.time.mix_r.forward(x.clone(), last_x.clone());
        let xk = self.time.mix_k.forward(x.clone(), last_x);

        let mut k = self.key.forward(xk);
        k = ReLU::new().forward(k).powf(2.0);
        let kv = self.value.forward(k);
        let r = sigmoid(self.receptance.forward(xr));

        r * kv
    }
}

/// Corresponds to:
/// 1. blocks.N.ffn.[key,value,receptance].weight
/// 3. Keys described in FeedForwardNetworkTime.
impl<'a, B: Backend> LoadFromSafeTensors<B> for FeedForwardNetworkRecord<B> {
    fn load<'data>(config: SafeTensorsConfig<B>, st: &SafeTensors<'data>) -> Result<Self, Error> {
        Ok(Self {
            key: config.scoped("key").load(st)?,
            value: config.scoped("value").load(st)?,
            receptance: config.scoped("receptance").load(st)?,
            time: config.load(st)?,
        })
    }
}

/*
 * Block
 */

///
#[derive(Debug, Module)]
pub struct Block<B: Backend> {
    ln_tm: LayerNorm<B>,
    ln_cm: LayerNorm<B>,
    att: Attention<B>,
    ffn: FeedForwardNetwork<B>,
}

impl<'a, B: Backend> Block<B> {
    pub fn new(config: &ModelConfig) -> Self {
        Self {
            ln_tm: nn::LayerNormConfig::new(config.n_embd).init(),
            ln_cm: nn::LayerNormConfig::new(config.n_embd).init(),
            att: Attention::new(config.n_embd),
            ffn: FeedForwardNetwork::new(config.n_embd),
        }
    }

    pub fn new_with(config: &ModelConfig, record: BlockRecord<B>) -> Self {
        Self::new(config).load_record(record)
    }

    pub fn forward<const D: usize>(
        &self,
        mut x: Tensor<B, D>,
        state: &mut State<B, D>,
    ) -> Tensor<B, D> {
        let x_tm_norm = self.ln_tm.forward(x.clone());
        x = x + self.att.forward(x_tm_norm, state);

        let x_cm_norm = self.ln_cm.forward(x.clone());
        x = x + self.ffn.forward(x_cm_norm, state);
        x
    }
}

impl<'a, B: Backend> LoadFromSafeTensors<B> for BlockRecord<B> {
    fn load<'data>(config: SafeTensorsConfig<B>, st: &SafeTensors<'data>) -> Result<Self, Error> {
        Ok(Self {
            ln_tm: config.with_scoped_overrides("ln1", LN_OVERRIDES).load(st)?,
            ln_cm: config.with_scoped_overrides("ln2", LN_OVERRIDES).load(st)?,
            att: config.scoped("att").load(st)?,
            ffn: config.scoped("ffn").load(st)?,
        })
    }
}

#[derive(Default)]
pub struct BlockCache<B: Backend> {
    /// time mixing state
    pub cm_last_x: TensorCache<B, 1>,
    pub tm_last_x: TensorCache<B, 1>,
    pub tm_num: TensorCache<B, 1>,
    pub tm_den: TensorCache<B, 1>,
}

/*
 * Model
 */

///
#[derive(Debug, Default, Config, Module)]
pub struct ModelConfig {
    #[config(default = 50277)]
    vocab_size: usize,
    n_layers: usize,
    n_embd: usize,
    ctx_len: usize,
    #[config(default = true)]
    use_cache: bool,
    #[config(default = false)]
    output_hidden_states: bool,
    #[config(default = false)]
    scale_attn_weights: bool,
}

impl ModelConfig {
    // pub fn init<B: Backend>(&self) -> Model<B> {
    //     Model {
    //         config: self.clone(),
    //         emb: nn::EmbeddingConfig::new(self.vocab_size, self.n_embd).init(),
    //         ln0: nn::LayerNormConfig::new(self.n_embd).init(),
    //         blocks: (0..self.n_layers).map(|_| Block::new(self)).collect(),
    //         ln_out: nn::LayerNormConfig::new(self.n_embd).init(),
    //         head: nn::LinearConfig::new(self.n_embd, self.vocab_size).init(),
    //     }
    // }

    // pub fn init_with<B: Backend>(&self, record: ModelRecord<B>) -> Model<B> {
    //     Model {
    //         config: self.clone(),
    //         emb: nn::EmbeddingConfig::new(self.vocab_size, self.n_embd).init_with(record.emb),
    //         ln0: nn::LayerNormConfig::new(self.n_embd).init_with(record.ln0),
    //         blocks: (0..self.n_layers)
    //             .map(|_| Block::new_with(self, record.blocks[i]))
    //             .collect(),
    //         ln_out: nn::LayerNormConfig::new(self.n_embd).init_with(record.ln_out),
    //         head: nn::LinearConfig::new(self.n_embd, self.vocab_size).init_with(record.head),
    //     }
    // }

    // #[cfg(feature = "std")]
    // pub fn init_from_safetensors<B: Backend>(
    //     &self,
    //     path: impl AsRef<std::path::Path>,
    //     device: Option<B::Device>,
    // ) -> Result<Model<B>, Error> {
    //     SafeTensorsConfig::default()
    //         .with_device(device.unwrap_or_default())
    //         .load_from_file(path)
    //         .map(|rec| self.init_with(rec))
    // }
}

///
#[derive(Debug, Module)]
pub struct Model<B: Backend> {
    config: ModelConfig,
    emb: Embedding<B>,
    ln0: LayerNorm<B>,
    blocks: Vec<Block<B>>,
    ln_out: LayerNorm<B>,
    head: Linear<B>,
}

/// Container for the model output.
#[derive(Debug, Clone)]
pub struct ModelOutput<B: Backend, const D: usize> {
    /// Hidden state of the last layer of the decoder, or logits for a custom head
    /// module after the decoder (e.g. vocabulary logits for language modeling tasks)
    output: Tensor<B, D>,
    // /// Cached attention layers keys and values if the model is used for generation
    // cache: Option<Vec<Option<LayerState>>>,
    // /// Hidden states for all intermediate layers
    // all_hidden_states: Option<Vec<Tensor>>,
    // /// Attention weights for all intermediate layers
    // all_attentions: Option<Vec<Tensor>>,
}

impl<B: Backend> Model<B> {
    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    pub fn input_embeddings<const D: usize>(&self, token: usize) -> Tensor<B, D> {
        // struct Visitor<'a, B: Backend> {
        //     token: usize,
        //     ret: &'a mut Option<Tensor<B, 1>>,
        // }
        // impl<'a, B: Backend> ModuleVisitor<B> for Visitor<'a, B> {
        //     fn visit<const D: usize>(&mut self, _id: &ParamId, tensor: &Tensor<B, D>) {
        //         // self.ret.insert(tensor.index([0..1, self.token]));
        //     }
        // }

        // let mut ret = None;
        // let mut visitor = Visitor {
        //     token,
        //     ret: &mut ret,
        // };
        // self.emb.visit(&mut visitor);
        // ret.unwrap()

        // module_fn!()
        todo!()
    }

    pub fn forward<const D: usize>(
        &self,
        token: usize,
        state: &mut State<B, D>,
    ) -> ModelOutput<B, D> {
        // todo: get embeddings for token
        let mut x = self.input_embeddings(token);
        x = self.ln0.forward(x);
        x = self
            .blocks
            .iter()
            .fold(x, |x, layer| layer.forward(x, state));
        x = self.ln_out.forward(x);
        x = self.head.forward(x);

        ModelOutput { output: x }
    }
}

/// emb.weight
/// head.weight
/// ln_out.[weight,bias]
impl<'a, B: Backend> LoadFromSafeTensors<B> for ModelRecord<B> {
    fn load<'data>(config: SafeTensorsConfig<B>, st: &SafeTensors<'data>) -> Result<Self, Error> {
        let emb = config.scoped("emb").load(st)?;
        let ln0 = config
            .with_scoped_overrides("blocks.0.ln0", LN_OVERRIDES)
            .load(st)?;
        let head = config.scoped("head").load(st)?;
        let ln_out = config
            .with_scoped_overrides("ln_out", LN_OVERRIDES)
            .load(st)?;

        let num_blocks = 1 + st
            .names()
            .iter()
            .filter_map(|name| name.strip_prefix("blocks.")?.split('.').next())
            .map(|s| s.parse::<usize>().expect("failed to parse block num"))
            .max()
            .expect("should have at least one block");

        println!("found {} blocks", num_blocks);

        let blocks = (0..num_blocks)
            .into_par_iter()
            .map(|i| {
                println!(".");
                stdout().flush().ok();

                config.scoped(format!("blocks.{}", i)).load(&st)
            })
            .collect::<Result<Vec<_>, Error>>()?;

        println!("loaded {} blocks", num_blocks);

        let mut model = Self {
            config: Default::default(), // todo
            emb,
            head,
            ln0,
            blocks,
            ln_out,
            // vocab_size: emb.weight().size()[0],
            // embed_size: emb.weight().size()[1],
            // n_embed,
        };

        // FIXME: do we need to do this?
        // apply layer norm to embeddings for each token
        // model.emb = self.ln0.
        // model = (0..Self::VOCAB_SIZE).fold(model, |mut model, token| {
        //     model.emb = module_fn!(
        //         map=model.emb,
        //         id=ParamId::from("emb.weight"),
        //         args=(usize, &'a LN<B>),
        //         init=|| (token, &model.ln0),
        //         fn=|&mut (token, ln0), w: Tensor<B, D>| {
        //             if token == 0 {
        //                 println!("embedding dims {:?}", w.dims());
        //                 stdout().flush().ok();
        //             }
        //
        //             let n_embed = w.dims()[1];
        //             let idx = [token..token + 1, 0..n_embed];
        //             let idxemb = w.clone().index(idx.clone());
        //             w.index_assign(idx, LN::<B>::forward(ln0, idxemb))
        //         }
        //     );
        //     model
        // });

        println!("finished normalizing embeddings");

        Ok(model)
    }
}

///
#[derive(Debug, Module)]
pub struct State<B: Backend, const D: usize> {
    // last_probs: TensorCache<B, 1>,
    // blocks: Vec<BlockCache<B>>,
    x: RunningState<Tensor<B, D>>,
    a: RunningState<Tensor<B, D>>,
    b: RunningState<Tensor<B, D>>,
    p: RunningState<Tensor<B, D>>,
}

impl<B: Backend, const D: usize> State<B, D> {
    // pub fn new(vocab_size: usize, n_blocks: usize, embed_size: usize) -> Self {
    //     Self {
    //         blocks: vec![BlockCache::new(embed_size); n_blocks],
    //         last_probs: Tensor::zeros([vocab_size]),
    //     }
    // }
}

/*
 *
 */

// ///
// pub struct Context<B: Backend> {
//     pub model: Model<B>,
//     pub state: ModelCache<B>,
//     pub last_probs: Tensor<B, 1>,
//     pub tokenizer: Tokenizer,
//     pub sampler: Box<dyn FnMut(&Tensor<B, 1>) -> Option<usize>>,
// }
// impl<B: Backend> Context<B> {
//     pub fn new(
//         model: Model<B>,
//         tokenizer: Tokenizer,
//         sampler: impl FnMut(&Tensor<B, 1>) -> Option<usize> + 'static,
//     ) -> Self {
//         let state = vec![BlockCache::new(model.embed_size); model.num_blocks];
//         let last_probs = Tensor::zeros([model.vocab_size]);
//         Self {
//             model,
//             state,
//             last_probs,
//             tokenizer,
//             sampler: Box::new(sampler),
//         }
//     }
// }

// pub struct Response<'a, B: Backend> {
//     pub toks: Encoding,
//     pub context: &'a Context<B>,
// }

// impl<'a, B: Backend> Response<'a, B> {
//     pub fn try_new(prompt: impl AsRef<str>, context: &'a Context<B>) -> Result<Self, Err> {
//         let toks = context
//             .tokenizer
//             .encode(prompt.as_ref(), false)
//             .map_err(|e| anyhow::anyhow!(e))?;

//         Ok(Self { toks, context })
//     }

//     // pub fn into_iter(self) -> impl Iterator<Item = String> {
//     //     // let mut context = self.context.clone();
//     //     // let mut last_probs = context.last_probs.clone();
//     //     // let mut state = context.state.clone();
//     //     // let mut toks = self.toks;
//     //     // let mut sampler = context.sampler;
//     //     // let mut tokenizer = context.tokenizer;
//     //     // std::iter::from_fn(move || {
//     //     //     let tokid = (sampler)(&last_probs)?;
//     //     //     if tokid == 0 {
//     //     //         return None;
//     //     //     }

//     //     //     if let Ok(next) = tokenizer.decode(vec![tokid as u32], false) {}
//     //     //     last_probs = context.model.evaluate(tokid, &mut state);
//     //     //     Some(next)
//     //     // })

//     //     let Self { toks, context } = self;
//     //     toks.get_ids().into_iter().map(|tokid| {
//     //         context.
//     //     })
//     // }
// }

// impl<'a, B: Backend> Iterator for Response<'a, B> {
//     type Item = String;
//     fn next(&mut self) -> Option<Result<Self::Item, Err> {
//         let tokid = (self.sampler)(&self.last_probs)?;
//         if tokid == 0 {
//             return None;
//         }

//         if let Ok(next) = self.tokenizer.decode(vec![tokid as u32], false) {}
//         self.last_probs = self.model.evaluate(&mut self.state, tokid);
//         Some(next)
//     }
// }
