use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, PointStruct, SearchParamsBuilder, SearchPointsBuilder, SearchResponse, UpsertPointsBuilder, VectorParamsBuilder
};
use qdrant_client::Payload;
use qdrant_client::Qdrant;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    let start_time = std::time::Instant::now();
    // Read the documents from a file
    let file_path = "documents.txt"; // Specify the file path here
    let documents = read_documents_from_file(file_path)?;

    // Initialize the embedding model with custom options
    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
    )?;

    // Generate embeddings
    let embeddings = model.embed(documents.clone(), None)?;

    // Connect to the Qdrant client
    let client = Qdrant::from_url("http://localhost:6334").build()?;

    let collection_name = "test"; // Consistent collection name

    // Check if the collection exists, create it if it doesn't
    if !client.collection_exists(collection_name).await? {
        let vector_size = embeddings[0].len() as u64; // Get the embedding vector size
        client
            .create_collection(
                CreateCollectionBuilder::new(collection_name)
                    .vectors_config(VectorParamsBuilder::new(vector_size, Distance::Cosine)),
            )
            .await?;
    } else {
        println!("Collection `{}` already exists!", collection_name);
    }

    // Prepare points with embeddings and corresponding documents as payload
    let points: Vec<PointStruct> = embeddings
        .into_iter()
        .enumerate()
        .filter(|(id, _)| !documents[*id].is_empty())
        .map(|(id, vector)| {
            let payload: Payload = serde_json::json!({ "document": documents[id] })
                .try_into()
                .unwrap();
            PointStruct::new(id as u64, vector, payload)
        })
        .collect();

    // Upsert the points into the collection
    client
        .upsert_points(UpsertPointsBuilder::new(collection_name, points))
        .await?;

    let time_elapsed = std::time::Instant::now() - start_time;
    println!("{:?}", time_elapsed);

    let search_result = search_qdrant("grass", model, client, collection_name);
    // Print the search result for debugging
    println!("{:#?}", search_result.await);

    Ok(())
}

async fn search_qdrant(input: &str, model:TextEmbedding,client:Qdrant,collection_name:&str)->SearchResponse{
    let embeddings = model.embed(vec![input.to_string()], None).unwrap();
    let vector = embeddings[0].clone(); // Get the first vector from the embeddings
    let search_result = client
        .search_points(
            SearchPointsBuilder::new(collection_name, vector, 3)
                .with_payload(true) // Include payload in the results
                .params(SearchParamsBuilder::default().exact(true)), // Optional search params
            ).await.unwrap();
    search_result
}

// Function to read documents from a file
fn read_documents_from_file<P>(path: P) -> Result<Vec<String>>
where
    P: AsRef<Path>,
{
    let file = File::open(path).context("Failed to open file")?;
    let reader = io::BufReader::new(file);
    let documents: Vec<String> = reader
        .lines()
        .collect::<Result<_, _>>()
        .context("Failed to read lines from file")?;
    Ok(documents)
}
