import time
import os
import sys
import logging
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def benchmark_model_loading():
    """Benchmarks the time taken to load LLM and Embedding models."""
    logger.info("Benchmarking Model Loading...")
    
    start_time = time.time()
    llm = Ollama(model="llama3.2:latest", request_timeout=360.0)
    end_time = time.time()
    logger.info(f"LLM Load Time: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
    end_time = time.time()
    logger.info(f"Embedding Model Load Time: {end_time - start_time:.4f} seconds")

def benchmark_indexing():
    """Benchmarks indexing a sample document."""
    logger.info("Benchmarking Indexing...")
    
    # Create a dummy file if it doesn't exist
    if not os.path.exists("data/benchmark_sample.txt"):
        os.makedirs("data", exist_ok=True)
        with open("data/benchmark_sample.txt", "w") as f:
            f.write("This is a sample document for benchmarking purposes. " * 100)

    start_time = time.time()
    documents = SimpleDirectoryReader(input_files=["data/benchmark_sample.txt"]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    end_time = time.time()
    logger.info(f"Indexing Time: {end_time - start_time:.4f} seconds")
    return index

def benchmark_query(index):
    """Benchmarks querying the index."""
    logger.info("Benchmarking Querying...")
    
    query_engine = index.as_query_engine()
    
    start_time = time.time()
    response = query_engine.query("What is this document about?")
    end_time = time.time()
    logger.info(f"Query Time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    logger.info("Starting Benchmark...")
    benchmark_model_loading()
    index = benchmark_indexing()
    benchmark_query(index)
    logger.info("Benchmark Complete.")
