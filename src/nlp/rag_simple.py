import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="RAG pipeline without LangChain.")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV with reviews and sentiment")
    parser.add_argument("--persist_dir", type=str, default="chroma_store", help="Vector DB directory")
    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    if "review_text" not in df.columns:
        raise ValueError("CSV must have a `review_text` column.")

    # Embeddings model
    print("Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Create Chroma client
    client = chromadb.PersistentClient(path=args.persist_dir)
    collection = client.get_or_create_collection(
        name="emotion_reviews",
        metadata={"hnsw:space": "cosine"}
    )

    # Insert documents into Chroma
    print("Embedding & storing...")
    for i, row in df.iterrows():
        text = row["review_text"]
        metadata = {
            "emotion": row.get("predicted_label", ""),
            "sentiment": row.get("sentiment_label", ""),
            "image_path": row.get("image_path", "")
        }

        embedding = embedder.encode(text).tolist()

        collection.upsert(
            ids=[str(i)],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[text]
        )

    print("Vector DB ready!")

    # Load LLM for summarization
    print("Loading summarizer model...")
    summarizer = pipeline("text2text-generation", model="google/flan-t5-small")

    print("\n📌 RAG System Ready — Ask a question (type 'exit' to quit):\n")

    while True:
        query = input("Your question: ").strip()
        if query.lower() == "exit":
            break

        query_embedding = embedder.encode(query).tolist()

        # Retrieve relevant docs
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=4
        )

        docs: List[str] = results["documents"][0]
        metas: List[Dict] = results["metadatas"][0]

        context = "\n".join(docs)

        prompt = f"Summarize what people are saying:\n\n{context}\n\nSummary:"
        summary = summarizer(prompt, max_length=100)[0]["generated_text"]

        print("\nAnswer:", summary)
        print("\nSources:")
        for meta, doc in zip(metas, docs):
            print(f" - {meta['emotion']} ({meta.get('sentiment','?')}): {doc}")
        print("")


if __name__ == "__main__":
    main()
