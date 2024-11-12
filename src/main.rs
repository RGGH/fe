use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use rayon::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Start timing
    let start = Instant::now();

    // With custom InitOptions
    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
    )?;

    let documents = vec![
        "passage: This is an example passage.",
        "fastembed-rs is licensed under Apache 2.0",
        "query: Hello, World!",
        "passage: This is an example passage.",
    ];

    // Use rayon to parallelize embedding generation
    let embeddings: Vec<Vec<f32>> = documents
        .par_iter()
        .map(|document| model.embed(vec![document.to_string()], None)) // Generate embedding for each document
        .filter_map(|result| result.ok()) // Filter out any errors (or handle them appropriately)
        .flat_map(|embedding| embedding) // Flatten the Vec<Vec<f32>> into a single Vec<f32>
        .collect();

    println!("Embeddings length: {}", embeddings.len()); // -> Embeddings length: 4
    println!("Embedding dimension: {}", embeddings[0].len()); // -> Embedding dimension: 384

    // Stop timing and calculate the duration
    let duration = start.elapsed();
    println!("Time taken: {:?}", duration); // 235ms - takes less than half the time of sequential!

    Ok(())
}
