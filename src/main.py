import time
import numpy as np
import asyncio
from fastembed import TextEmbedding
from qdrant_client import AsyncQdrantClient, models

documents = [
    "passage: Hello, World!",
    "query: Hello, World!",
    "passage: This is an example passage.",
    "fastembed is supported by and maintained by Qdrant."
]

async def main():
    # Timing the model initialization
    start_time = time.perf_counter()
    embedding_model = TextEmbedding()
    end_time = time.perf_counter()
    print(f"Model initialization took: {1000 * (end_time - start_time):.2f}ms")

    # Timing the embedding generation
    start_time = time.perf_counter()
    embeddings = list(embedding_model.embed(documents))  # Convert the generator to a list
    end_time = time.perf_counter()
    print(f"Embedding generation took: {1000 * (end_time - start_time):.2f}ms")

    # Timing the Qdrant client connection
    start_time = time.perf_counter()
    client = AsyncQdrantClient(url="http://localhost:6333")
    end_time = time.perf_counter()
    print(f"Qdrant client connection took: {(end_time - start_time) * 1e6:.2f}µs")

    collection_name = "test"

    # Try to create the collection
    start_time = time.perf_counter()
    try:
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=len(embeddings[0]), distance=models.Distance.COSINE)
        )
        print(f"Collection `{collection_name}` created!")
    except Exception as e:
        print(f"Collection `{collection_name}` already exists or error occurred: {str(e)}")
    end_time = time.perf_counter()
    print(f"Collection creation took: {1000 * (end_time - start_time):.2f}ms")

    # Prepare points to be upserted
    start_time = time.perf_counter()
    points = [
        models.PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={"document": documents[i]},
        )
        for i in range(len(documents))
    ]
    # Upsert points into the collection
    await client.upsert(
        collection_name=collection_name,
        points=points
    )
    end_time = time.perf_counter()
    print(f"Upsert operation took: {1000 * (end_time - start_time):.2f}ms")

    print("done!")

# Run the main function
asyncio.run(main())

