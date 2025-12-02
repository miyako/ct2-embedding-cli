use clap::Parser;
use tokenizers::Tokenizer;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use std::io::{self, Read};
use std::sync::Arc;
use tokio::sync::Mutex; // Async Mutex for the web server
use axum::{
    extract::State,
    routing::post,
    Json,
    Router,
    http::StatusCode,
};
use std::net::SocketAddr;

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

// --- KEY FIX: Mark the C++ type as Thread Safe ---
unsafe impl Send for ffi::EmbeddingModel {}
unsafe impl Sync for ffi::EmbeddingModel {}

// --- Data Structures ---

#[derive(Serialize, Debug)]
struct EmbeddingOutput {
    text: String,
    ids: Vec<u32>,
    embeddings: Vec<f32>,
}

#[derive(Deserialize, Debug)]
struct EmbeddingRequest {
    text: String,
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

    /// Run in HTTP server mode
    #[arg(long, default_value_t = false)]
    server: bool,

    /// Port to listen on (only used in server mode)
    #[arg(long, default_value_t = 3000)]
    port: u16,
}

// Container to hold the heavy resources (Model + Tokenizer).
// We wrap the C++ model in a wrapper because we can't easily clone UniquePtr.
struct ModelContext {
    tokenizer: Tokenizer,
    model: cxx::UniquePtr<ffi::EmbeddingModel>,
}

// Since cxx::UniquePtr is Send but not Sync (usually), and we are sharing it
// across async threads, we will wrap the Context in a Mutex.
type SharedState = Arc<Mutex<ModelContext>>;

// --- Logic ---

// Helper function to perform the actual logic, used by both CLI and Server
fn compute_embeddings(
    ctx: &ModelContext, 
    text: String
) -> anyhow::Result<EmbeddingOutput> {
    if text.is_empty() {
        anyhow::bail!("Input text is empty");
    }

    // 1. Tokenize
    let encoding = ctx.tokenizer.encode(text.clone(), true)
        .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
    
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    let lengths: Vec<usize> = vec![ids.len()];

    // 2. Inference (C++)
    // Note: This blocks the thread. In a high-throughput async environment,
    // you might want to use spawn_blocking, but for simplicity here we call directly.
    let output_flat = ctx.model.encode(&ids, &lengths);

    // 3. Construct Output
    Ok(EmbeddingOutput {
        text,
        ids,
        embeddings: output_flat,
    })
}

// --- HTTP Handlers ---

async fn embedding_handler(
    State(state): State<SharedState>,
    Json(payload): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingOutput>, (StatusCode, String)> {
    // Acquire lock on the model
    let ctx = state.lock().await;

    // Run computation
    match compute_embeddings(&ctx, payload.text) {
        Ok(output) => Ok(Json(output)),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }
}

// --- Main ---

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // 1. Load Tokenizer
    let tokenizer_path = PathBuf::from(&args.model).join("tokenizer.json");
    if !tokenizer_path.exists() {
        anyhow::bail!("tokenizer.json not found in model directory");
    }

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // 2. Load C++ Model
    println!("Loading model from {} on {}...", args.model, args.device);
    let model = ffi::new_embedding_model(&args.model, &args.device);

    if model.is_null() {
        anyhow::bail!("Failed to initialize C++ model.");
    }

    // 3. Prepare State
    let ctx = ModelContext { tokenizer, model };

    // 4. Branch based on mode
    if args.server {
        // --- SERVER MODE ---
        let port = args.port;
        println!("Starting HTTP server on http://0.0.0.0:{}", port);
        
        let shared_state = Arc::new(Mutex::new(ctx));

        let app = Router::new()
            .route("/embeddings", post(embedding_handler))
            .with_state(shared_state);

        let addr = SocketAddr::from(([0, 0, 0, 0], port));
        let listener = tokio::net::TcpListener::bind(addr).await?;
        
        axum::serve(listener, app).await?;
    
    } else {
        // --- CLI MODE ---
        let input_text = if let Some(text) = args.text {
            text
        } else {
            // Pipe mode
            let mut buffer = String::new();
            if atty::is(atty::Stream::Stdin) {
                // Determine if we should prompt or expect pipe, 
                // but usually if no args and no pipe, we might error or wait.
                // For this snippet, strict pipe reading:
                eprintln!("Reading input from stdin (pipe mode)...");
            }
            io::stdin().read_to_string(&mut buffer)
                .map_err(|e| anyhow::anyhow!("Failed to read from stdin: {}", e))?;
            buffer.trim().to_string()
        };

        let output = compute_embeddings(&ctx, input_text)?;
        let json_output = serde_json::to_string_pretty(&output)?;
        println!("{}", json_output);
    }

    Ok(())
}