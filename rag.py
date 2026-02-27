
import argparse
import os
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings_model(type="local"):
    """
    Get the embedding model based on the specified type.
    type="local": Using HuggingFace local embedding (no key required)
    type="OpenAIEmbeddings": Using OpenAI embedding (requires OPENAI_API_KEY)
    """
    if type == "OpenAIEmbeddings":
        return OpenAIEmbeddings()
    else:
        print("Using HuggingFace local embedding model (sentence-transformers/all-MiniLM-L6-v2).")

        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_literature_rag_db(pdf_paths, persist_directory="./chroma_db", chunk_size=1000, chunk_overlap=200):
    documents = []
    print("1. Loading Files...")
    for path in pdf_paths:
        if not os.path.exists(path):
            print(f"Warning: File {path} does not exist, skipping.")
            continue
        try:
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Loading {path} failed: {e}")
    if not documents:
        print("Having no documents to process. Please check the file paths and try again.")
        return None
    print("2. Segmenting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(documents)
    print(f"3. Building vector database (total {len(splits)} Segments)...")
    # === Using Costum Embedding Model ===
    embedding_model = get_embeddings_model(type="local")
    try:
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
        print("File loaded and vector database built successfully!")
        return vectorstore
    except Exception as e:
        print(f"Failed to build vector database: {e}")
        return None

#Usage example:
#rag_db = build_literature_rag_db(['1107.4557v1.pdf', 'Literature Meets Data- A Synergistic Approach to Hypothesis Generation.pdf'])

def parse_args():
    parser = argparse.ArgumentParser("Build RAG vector database")
    parser.add_argument(
        "--pdf_dir",
        type=str,
        required=True,
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--persist_directory",
        type=str,
        default="./chroma_db",
        help="Where to store the Chroma DB"
    )

    parser.add_argument(
    "--chunk_size",
    type=int,
    default=1000,
    help="Chunk size for text splitting"
    )

    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=200,
        help="Chunk overlap for text splitting"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    pdf_paths = [
        os.path.join(args.pdf_dir, f)
        for f in os.listdir(args.pdf_dir)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_paths:
        raise RuntimeError(f"No PDFs found in {args.pdf_dir}")
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap

    build_literature_rag_db(
        pdf_paths,
        persist_directory=args.persist_directory,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


if __name__ == "__main__":
    main()