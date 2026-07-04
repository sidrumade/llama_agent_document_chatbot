import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Mock streamlit before importing app
sys.modules["streamlit"] = MagicMock()
import streamlit as st

# Configure st.cache_resource to be a pass-through decorator
def cache_resource(func):
    return func
st.cache_resource = cache_resource

# Define a concrete subclassable mock for Workflow
class MockWorkflow:
    def __init__(self, *args, **kwargs):
        pass

mock_core_workflow = MagicMock()
mock_core_workflow.Workflow = MockWorkflow
mock_core_workflow.Event = MagicMock
mock_core_workflow.StartEvent = MagicMock
mock_core_workflow.StopEvent = MagicMock

# Mock other heavy dependencies
mocked_modules = {
    "llama_index.core": MagicMock(),
    "llama_index.core.llms": MagicMock(),
    "llama_index.core.node_parser": MagicMock(),
    "llama_index.core.storage.docstore": MagicMock(),
    "llama_index.core.response_synthesizers": MagicMock(),
    "llama_index.core.retrievers": MagicMock(),
    "llama_index.retrievers.bm25": MagicMock(),
    "llama_index.core.postprocessor": MagicMock(),
    "llama_index.core.vector_stores": MagicMock(),
    "llama_index.core.workflow": mock_core_workflow,
    "llama_index.core.schema": MagicMock(),
    "llama_index.core.callbacks": MagicMock(),
    "llama_index.core.callbacks.base_handler": MagicMock(),
    "llama_index.core.chat_engine": MagicMock(),
    "llama_index.core.chat_engine.types": MagicMock(),
    "llama_index.embeddings.huggingface": MagicMock(),
    "llama_index.llms.ollama": MagicMock()
}

for mod_name, mock_obj in mocked_modules.items():
    sys.modules[mod_name] = mock_obj

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import app logic and rag_engine
from app import load_config, load_llm, load_embedding_model
from rag_engine import enrich_document_metadata, IndexManager

def test_load_config_success():
    """Test loading a valid config file."""
    with patch("yaml.safe_load", return_value={"llm_model_name": "test-model"}) as mock_yaml:
        with patch("builtins.open"):
            config = load_config()
            assert config["llm_model_name"] == "test-model"

def test_load_config_file_not_found():
    """Test behavior when config file is missing."""
    st.stop.side_effect = SystemExit

    with patch("builtins.open", side_effect=FileNotFoundError):
        with pytest.raises(SystemExit): 
             load_config()
        
        st.error.assert_called()
        st.stop.assert_called()

def test_load_llm_ollama():
    """Test Ollama LLM loading function."""
    with patch("app.Ollama") as MockOllama:
        load_llm("ollama", "test-model")
        MockOllama.assert_called_once()


def test_load_embedding_model():
    """Test Embedding model loading function."""
    with patch("app.HuggingFaceEmbedding") as MockEmbed:
        load_embedding_model()
        MockEmbed.assert_called_once()

def test_enrich_document_metadata():
    """Test document metadata enrichment with headers and types."""
    mock_doc = MagicMock()
    mock_doc.text = "### SYSTEM POLICY\nThis is a sample document text."
    mock_doc.metadata = {"file_name": "system_policy_v2.txt"}
    
    enrich_document_metadata([mock_doc])
    
    assert "upload_timestamp" in mock_doc.metadata
    assert mock_doc.metadata["filename"] == "system_policy_v2.txt"
    assert mock_doc.metadata["document_type"] == "txt"
    assert mock_doc.metadata["section"] == "SYSTEM POLICY"

def test_get_indexed_filenames_multi():
    """Test extraction of indexed filenames in multi-index."""
    mock_index_dict = {
        "type": "multi",
        "summary": MagicMock()
    }
    doc_info_1 = MagicMock()
    doc_info_1.metadata = {"filename": "doc_a.pdf"}
    doc_info_2 = MagicMock()
    doc_info_2.metadata = {"filename": "doc_b.docx"}
    
    mock_index_dict["summary"].ref_doc_info = {
        "1": doc_info_1,
        "2": doc_info_2
    }
    
    filenames = IndexManager.get_indexed_filenames(mock_index_dict)
    assert set(filenames) == {"doc_a.pdf", "doc_b.docx"}

def test_configure_logging():
    """Test that configure_logging successfully configures file handler."""
    import tempfile
    import logging
    from rag_engine import configure_logging
    
    # Clear existing handlers to ensure clean configuration for this test
    root_logger = logging.getLogger()
    old_handlers = list(root_logger.handlers)
    for h in old_handlers:
        root_logger.removeHandler(h)
        
    # Use a temp file for logging
    with tempfile.NamedTemporaryFile(delete=False) as tmp_log:
        tmp_log_name = tmp_log.name
        
    try:
        configure_logging(tmp_log_name)
        
        # Check that we have a file handler
        handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(handlers) >= 1
        
        # Log a message and check it writes
        logger = logging.getLogger("test_func_logger")
        logger.setLevel(logging.INFO)
        logger.info("Functional test log message")
        
        # Flush and close handlers
        for h in handlers:
            h.flush()
            h.close()
            root_logger.removeHandler(h)
            
        with open(tmp_log_name, "r", encoding="utf-8") as f:
            content = f.read()
            assert "Functional test log message" in content
    finally:
        # Restore old handlers to not disrupt pytest's own reporting
        for h in old_handlers:
            root_logger.addHandler(h)
        if os.path.exists(tmp_log_name):
            os.unlink(tmp_log_name)

