import argparse
import pandas as pd
from pathlib import Path

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub


def parse_args():
    parser = argparse.ArgumentParser(description="RAG pipeline for querying emotion-driven reviews.")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="CSV with review_text, emotion, sentiment columns"
    )
    parser.add_argument(
        "--persist_dir",
        type=str,
        default="chroma_store",
        help="Directory to persist vector DB"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_csv = Path(args.csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"CSV not found: {input_csv}")

    print(f"Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)

    if "review_text" not in df.columns:
        raise ValueError("CSV must contain review_text column!")
    
    # Small and fast instructor embeddings
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    documents = []
    for _, row in df.iterrows():
        text = row["review_text"]
        metadata = {
            "emotion": row.get("predicted_label", ""),
            "sentiment": row.get("sentiment_label", ""),
            "image_path": row.get("image_path", "")
        }
        documents.append(Document(page_content=text, metadata=metadata))

    print(f"Embedding and storing vector data... ({len(documents)} docs)")
    
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embed_model,
        persist_directory=args.persist_dir
    )
    vector_db.persist()

    print(f"Vector DB successfully stored in: {args.persist_dir}")

    # RAG Retrieval chain with a small HF model
    llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature": 0.2})

    retriever = vector_db.as_retriever(search_kwargs={"k": 4})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    print("\n📌 RAG system ready! Ask anything. Type 'exit' to quit.\n")

    while True:
        query = input("Your question: ").strip()
        if query.lower() == "exit":
            break

        response = qa_chain({"query": query})
        print("\nAnswer:", response["result"])
        print("\nSources:")
        for doc in response["source_documents"]:
            print(f" - {doc.metadata['emotion']} ({doc.metadata['sentiment']}): {doc.page_content}")
        print("")


if __name__ == "__main__":
    main()
