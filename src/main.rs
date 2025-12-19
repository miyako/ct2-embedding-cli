use axum::{
    extract::{DefaultBodyLimit, State}, 
    http::StatusCode, 
    routing::post, 
    Json, Router
};
use clap::Parser;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tower::ServiceBuilder; // Added for middleware layering

// --- Constants for Limits ---
const MAX_PAYLOAD_SIZE: usize = 10 * 1024 * 1024; // 2MB
const MAX_CONCURRENT_REQUESTS: usize = 100;      // Max simultaneous model calls

// ... (ffi bridge, structs, and ModelContext remain the same) ...

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("embedding_bridge.h");
        type EmbeddingModel;
        fn new_embedding_model(model_path: &str, device: &str) -> UniquePtr<EmbeddingModel>;
        fn encode(
            self: &EmbeddingModel, 
            input_ids: &Vec<u32>, 
            lengths: &Vec<usize>
        ) -> Vec<f32>;
    }
}

unsafe impl Send for ffi::EmbeddingModel {}
unsafe impl Sync for ffi::EmbeddingModel {}

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum InputData {
    Single(String),
    Batch(Vec<String>),
}

#[derive(serde::Deserialize)]
struct EmbeddingRequest {
    input: InputData,
    #[serde(default = "default_model")]
    model: String,
}

fn default_model() -> String { "default".to_string() }

#[derive(serde::Serialize)]
struct EmbeddingOutput {
    model: String,
    object: String,
    usage: Usage,
    data: Vec<EmbeddingData>,
}

#[derive(serde::Serialize)]
struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}

#[derive(serde::Serialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
    object: String,
}

pub struct ModelContext {
    tokenizer: Tokenizer,
    model: cxx::UniquePtr<ffi::EmbeddingModel>,
}

unsafe impl Send for ModelContext {}
unsafe impl Sync for ModelContext {}

pub struct AppState {
    pub name: String,
    pub ctx: Arc<ModelContext>,
}

type SharedState = Arc<AppState>;

// --- Logic ---

async fn embedding_handler(
    State(state): State<SharedState>,
    Json(payload): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingOutput>, (StatusCode, String)> {
    
    let ctx = Arc::clone(&state.ctx);
    let model_name = state.name.clone();

    let result = tokio::task::spawn_blocking(move || {
        let inputs = match payload.input {
            InputData::Single(s) => vec![s],
            InputData::Batch(v) => v,
        };

        // 1. Domain Logic Limit: Check batch size
        if inputs.is_empty() {
            return Err((StatusCode::BAD_REQUEST, "Input is empty".to_string()));
        }

        // 2. Parallel Tokenization
        let encodings: Vec<_> = inputs
            .into_par_iter()
            .map(|text| {
                ctx.tokenizer.encode(text, true)
                    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut flat_ids = Vec::new();
        let mut lengths = Vec::new();
        let mut total_tokens = 0;

        for enc in &encodings {
            let ids = enc.get_ids();
            flat_ids.extend_from_slice(ids);
            lengths.push(ids.len());
            total_tokens += ids.len();
        }

        // 3. Batch Inference (C++)
        let all_embeddings_flat = ctx.model.encode(&flat_ids, &lengths);
        
        let num_inputs = lengths.len();
        if num_inputs == 0 {
             return Err((StatusCode::INTERNAL_SERVER_ERROR, "Model returned no data".to_string()));
        }
        let dim = all_embeddings_flat.len() / num_inputs;
        
        let data = all_embeddings_flat
            .chunks_exact(dim)
            .enumerate()
            .map(|(i, emb)| EmbeddingData {
                embedding: emb.to_vec(),
                index: i,
                object: "embedding".to_string(),
            })
            .collect();

        Ok(EmbeddingOutput {
            model: model_name,
            object: "list".to_string(),
            usage: Usage {
                prompt_tokens: total_tokens,
                total_tokens,
            },
            data,
        })
    }).await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    result.map(Json)
}

// --- Main ---

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    model: String,
    #[arg(short, long, default_value = "cpu")]
    device: String,
    #[arg(long, default_value_t = false)]
    server: bool,
    #[arg(long, default_value_t = 3000)]
    port: u16,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let tokenizer_path = PathBuf::from(&args.model).join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    println!("Loading model from {} on {}...", args.model, args.device);
    let model = ffi::new_embedding_model(&args.model, &args.device);
    if model.is_null() {
        anyhow::bail!("Failed to initialize C++ model.");
    }

    let model_name = Path::new(&args.model)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model")
        .to_string();

    let model_ctx = Arc::new(ModelContext { tokenizer, model });

    if args.server {
        let shared_state = Arc::new(AppState {
            name: model_name,
            ctx: model_ctx,
        });

        // Define our middleware stack
        let middleware = ServiceBuilder::new()
            // Limit 1: Global Concurrency (Backpressure)
            // Only 10 requests will be processed at a time; others wait in queue
            .layer(tower::limit::ConcurrencyLimitLayer::new(MAX_CONCURRENT_REQUESTS))
            // Limit 2: Request Body Size (Security)
            // Reject any request larger than 2MB immediately
            .layer(DefaultBodyLimit::max(MAX_PAYLOAD_SIZE));

        let app = Router::new()
            .route("/v1/embeddings", post(embedding_handler))
            .layer(middleware) // Apply limits to all routes
            .with_state(shared_state);

        let addr = std::net::SocketAddr::from(([0, 0, 0, 0], args.port));
        let listener = tokio::net::TcpListener::bind(addr).await?;
        println!("Server listening on http://0.0.0.0:{}", args.port);
        axum::serve(listener, app).await?;
    } else {
        println!("CLI mode: Please implement logic or use --server");
    }

    Ok(())
}