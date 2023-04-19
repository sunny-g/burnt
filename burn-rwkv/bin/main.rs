extern crate burn_rwkv;

use burn_core::{module::Module, tensor::backend::Backend};
use burn_ndarray::{NdArrayBackend, NdArrayDevice};
use burn_rwkv::{Error, Model, ModelConfig, SafeTensorsConfig};
use clap::Parser;
use std::{io, path::PathBuf};
use tokenizers::Tokenizer;

type Rwkv = Model<NdArrayBackend<f32>>;

///
#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    tokenizer: PathBuf,

    #[arg(short, long)]
    model: PathBuf,
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    let tokenizer = Tokenizer::from_file(&args.tokenizer).map_err(Error::Tokenizers)?;
    println!(
        "loaded tokenizer; vocab size: w/ & w/o added tokens {} {}",
        tokenizer.get_vocab_size(true),
        tokenizer.get_vocab_size(false)
    );

    let config = ModelConfig::default();
    let model: Rwkv = config.init_from_safetensors(&args.model, Some(NdArrayDevice::default()))?;
    println!("loaded model with {} layers", model.num_layers());

    Ok(())
}
