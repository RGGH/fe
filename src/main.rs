use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Start timing
    let start = Instant::now();

    // With default InitOptions
    let _model = TextEmbedding::try_new(Default::default())?;

    // With custom InitOptions
    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
    )?;

    let documents = vec![
        "passage: Hello, World!",
        "query: Hello, World!",
        "passage: This is an example passage.",
        "passage: This is also an example passage.",
    ];

    // Generate embeddings with the default batch size, 256
    let embeddings = model.embed(documents, None)?;

    println!("Embeddings length: {}", embeddings.len()); // -> Embeddings length: 4
    println!("Embedding dimension: {}", embeddings[0].len()); // -> Embedding dimension: 384
    println!("{:?}", embeddings);

    // Stop timing and calculate the duration
    let duration = start.elapsed();

    // Log the time taken
    println!("Time taken: {:?}", duration); // approx 500ms

    Ok(())
}
