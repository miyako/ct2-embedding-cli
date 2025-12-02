use clap::Parser;
use tokenizers::Tokenizer;
use std::path::PathBuf;

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("embedding_bridge.h");

        type EmbeddingModel;

        // rust::Str in C++ maps to &str in Rust
        fn new_embedding_model(model_path: &str, device: &str) -> UniquePtr<EmbeddingModel>;
        
        // rust::Vec<T> in C++ maps to Vec<T> in Rust (by value for return)
        // const rust::Vec<T>& in C++ maps to &Vec<T> in Rust
        fn encode(
            self: &EmbeddingModel, 
            input_ids: &Vec<u32>, 
            lengths: &Vec<usize>
        ) -> Vec<f32>;
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Portable CTranslate2 Embedding Generator")]
struct Args {
    #[arg(short, long)]
    model: String,

    #[arg(short, long)]
    text: String,

    #[arg(short, long, default_value = "cpu")]
    device: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let tokenizer_path = PathBuf::from(&args.model).join("tokenizer.json");
    if !tokenizer_path.exists() {
        anyhow::bail!("tokenizer.json not found in model directory");
    }

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    println!("Loading model from {}...", args.model);
    let model = ffi::new_embedding_model(&args.model, &args.device);

    if model.is_null() {
        anyhow::bail!("Failed to initialize C++ model.");
    }

    let encoding = tokenizer.encode(args.text, true)
        .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
    
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    let len = ids.len();

    println!("Input IDs: {:?}", ids);

    let lengths = vec![len]; 
    let output_flat = model.encode(&ids, &lengths);

    println!("Generated Embedding Vector (Size: {}):", output_flat.len());
    println!("{:?}", &output_flat[..std::cmp::min(10, output_flat.len())]);
    
    Ok(())
}