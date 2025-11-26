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

# Mock other heavy dependencies
# We need to mock the modules structure so that "from X import Y" works
mock_llama_index_core = MagicMock()
sys.modules["llama_index.core"] = mock_llama_index_core
sys.modules["llama_index.core.callbacks"] = MagicMock()
sys.modules["llama_index.core.callbacks.base_handler"] = MagicMock()
sys.modules["llama_index.core.chat_engine.types"] = MagicMock()
sys.modules["llama_index.embeddings.huggingface"] = MagicMock()
sys.modules["llama_index.llms.ollama"] = MagicMock()

# Now import app logic
# We need to be careful because app.py runs code on import.
# We will test functions by importing them if possible, or by mocking the whole execution.
# Since app.py is a script, it's better to test the functions we defined.

# To test functions inside app.py, we might need to refactor app.py to be more modular 
# or use a trick to import it without running the main block. 
# For now, let's assume we can import it and the side effects (st.title, etc.) are handled by the mock.

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import load_config, load_llm, load_embedding_model

def test_load_config_success(tmp_path):
    """Test loading a valid config file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("llm_model_name: test-model\nembedding_model_name: test-embed")
    
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        # We can't easily patch open for the actual function call if we don't control the path in the function
        # But load_config hardcodes "config.yaml".
        # Let's mock yaml.safe_load instead.
        with patch("yaml.safe_load", return_value={"llm_model_name": "test-model"}) as mock_yaml:
            with patch("builtins.open"):
                config = load_config()
                assert config["llm_model_name"] == "test-model"

def test_load_config_file_not_found():
    """Test behavior when config file is missing."""
    # Configure st.stop to raise SystemExit to mimic real behavior
    st.stop.side_effect = SystemExit

    with patch("builtins.open", side_effect=FileNotFoundError):
        with pytest.raises(SystemExit): 
             load_config()
        
        # Verify st.error was called
        st.error.assert_called()
        st.stop.assert_called()

def test_load_llm():
    """Test LLM loading function."""
    with patch("app.Ollama") as MockOllama:
        load_llm()
        MockOllama.assert_called_once()

def test_load_embedding_model():
    """Test Embedding model loading function."""
    with patch("app.HuggingFaceEmbedding") as MockEmbed:
        load_embedding_model()
        MockEmbed.assert_called_once()
