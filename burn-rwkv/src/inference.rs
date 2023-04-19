use crate::{Model, ModelConfig};
use burn_core::{config::Config, module::Module};
use burn_tensor::backend::Backend;
use tokenizers::Tokenizer;

#[derive(Debug, Config, Module)]
pub struct GeneratorConfig {
    model_config: ModelConfig,

    /// Temperature setting. Values higher than 1 will improve originality at the risk of reducing relevance (default: 1.0)
    #[config(default = 1.0)]
    pub temperature: f32,
    /// Top_k values for sampling tokens. Value higher than 0 will enable the feature (default: 0)
    pub top_k: i64,
    /// Top_p value for [Nucleus sampling, Holtzman et al.](http://arxiv.org/abs/1904.09751). Keep top tokens until cumulative probability reaches top_p (default: 0.85)
    #[config(default = 0.85)]
    pub top_p: f32,
    ///
    #[config(default = false)]
    pub forever: bool,
    #[config(default = 42)]
    pub seed: u64,
    pub max_tokens: Option<usize>,

    #[config(default = false)]
    pub quantized: bool,

    /// Minimum sequence length (default: 0)
    pub min_length: i64,
    /// Maximum sequence length (default: 20)
    pub max_length: Option<i64>,
    /// Sampling flag. If true, will perform top-k and/or nucleus sampling on generated tokens, otherwise greedy (deterministic) decoding (default: true)
    pub do_sample: bool,
    /// Early stopping flag indicating if the beam search should stop as soon as `num_beam` hypotheses have been generated (default: false)
    pub early_stopping: bool,
    /// Number of beams for beam search (default: 5)
    pub num_beams: i64,
}

/// # Language generation model based on the GPT-J architecture
#[derive(Debug, Module)]
pub struct Generator<B: Backend> {
    model: Model<B>,
    // tokenizer: Tokenizer,
    vocab_size: usize,
    // var_store: nn::VarStore,
    generate_config: GeneratorConfig,
    bos_token_id: Option<usize>,
    eos_token_ids: Option<Vec<usize>>,
    // pad_token_id: Option<usize>,
    is_encoder_decoder: bool,
    decoder_start_id: Option<usize>,
    max_position_embeddings: usize,
}
