use clap::Parser;
use tokenizers::Tokenizer;
use std::path::PathBuf;
use serde::Serialize;
use std::io::{self, Read};

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

// Define the structure for the JSON output
#[derive(Serialize, Debug)] // <--- NEW: Derive Serialize to allow JSON conversion
struct EmbeddingOutput {
    text: String,
    ids: Vec<u32>,
    embeddings: Vec<f32>,
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Portable CTranslate2 Embedding Generator")]
struct Args {
    #[arg(short, long)]
    model: String,

    #[arg(short, long)]
    text: Option<String>,

    #[arg(short, long, default_value = "cpu")]
    device: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let input_text = if let Some(text) = args.text {
        text
    } else {
        eprintln!("Reading input from stdin (pipe mode)...");
        let mut buffer = String::new();
        // Read all available data from standard input
        io::stdin().read_to_string(&mut buffer)
            .map_err(|e| anyhow::anyhow!("Failed to read from stdin: {}", e))?;
    
        // Trim leading/trailing whitespace (e.g., trailing newline) and return the string.
        buffer.trim().to_string()
    };

    if input_text.is_empty() {
        eprintln!("Warning: Input text is empty. Exiting.");
        return Ok(());
    }
    
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

    let encoding = tokenizer.encode(input_text.clone(), true)
        .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
    
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    let lengths: Vec<usize> = vec![ids.len()];

    // eprintln!("Input IDs: {:?}", &ids[..cmp::min(10, ids.len())]); 

    let output_flat = model.encode(&ids, &lengths);

    let output = EmbeddingOutput {
        text: input_text,
        ids: ids,
        embeddings: output_flat,
    };
    
    let json_output = serde_json::to_string_pretty(&output)?;
    
    println!("{}", json_output);
    
    // println!("Generated Embedding Vector (Size: {}): \n{}", output_flat.len(), output_flat.iter().map(|f| format!("{:.6}", f)).collect::<Vec<String>>().join(", "));
    
    Ok(())
}