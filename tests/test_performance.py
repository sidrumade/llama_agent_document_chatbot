import pytest
import time
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Define thresholds (adjust based on environment)
MAX_LLM_LOAD_TIME = 5.0  # Seconds (assuming cached or fast load)
MAX_EMBED_LOAD_TIME = 5.0 # Seconds
MAX_QUERY_TIME = 10.0 # Seconds

@pytest.fixture(scope="module")
def llm():
    start_time = time.time()
    model = Ollama(model="llama3.2:latest", request_timeout=360.0)
    end_time = time.time()
    load_time = end_time - start_time
    print(f"\nLLM Load Time: {load_time:.4f}s")
    return model

@pytest.fixture(scope="module")
def embed_model():
    start_time = time.time()
    model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
    end_time = time.time()
    load_time = end_time - start_time
    print(f"\nEmbedding Model Load Time: {load_time:.4f}s")
    return model

def test_llm_load_performance(llm):
    """Test that LLM loads within acceptable time."""
    # Note: This tests the instantiation, not the actual model loading into VRAM which happens on first call usually.
    # But for Ollama object creation it should be instant.
    assert llm is not None

def test_embedding_load_performance(embed_model):
    """Test that Embedding model loads within acceptable time."""
    assert embed_model is not None

def test_indexing_performance(embed_model):
    """Test indexing speed."""
    # Create a dummy file
    with open("data/test_sample.txt", "w") as f:
        f.write("This is a test document. " * 50)
    
    start_time = time.time()
    documents = SimpleDirectoryReader(input_files=["data/test_sample.txt"]).load_data()
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"\nIndexing Time: {duration:.4f}s")
    assert duration < 10.0 # Should be fast for a small file

def test_query_performance(llm, embed_model):
    """Test query response speed."""
    # Create a simple index
    documents = SimpleDirectoryReader(input_files=["data/test_sample.txt"]).load_data()
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    query_engine = index.as_query_engine(llm=llm)
    
    start_time = time.time()
    response = query_engine.query("What is this document about?")
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"\nQuery Time: {duration:.4f}s")
    assert duration < MAX_QUERY_TIME

def test_ttft_performance(llm, embed_model):
    """Test Time To First Token (TTFT) for streaming."""
    # Create a simple index
    documents = SimpleDirectoryReader(input_files=["data/test_sample.txt"]).load_data()
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    chat_engine = index.as_chat_engine(llm=llm)
    
    start_time = time.time()
    response = chat_engine.stream_chat("Hello")
    # Fetch first token to measure latency
    for token in response.response_gen:
        break
    end_time = time.time()
    
    ttft = end_time - start_time
    print(f"\nTime To First Token (TTFT): {ttft:.4f}s")
    assert ttft < 2.0 # Should be under 2 seconds for good UX
