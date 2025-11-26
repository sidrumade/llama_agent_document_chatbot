# Project Context: LlamaIndex Document Chatbot

## Overview
This project is a local, privacy-focused document chatbot built with **Streamlit** and **LlamaIndex**. It allows users to upload documents (PDF, TXT, DOCX), create vector embeddings locally using **HuggingFace** models, and chat with them using **Ollama** (e.g., Llama 3).

## Tech Stack
-   **Frontend**: Streamlit
-   **Orchestration**: LlamaIndex
-   **LLM**: Ollama (running locally, e.g., `llama3.2:latest`)
-   **Embeddings**: HuggingFace (`BAAI/bge-large-en-v1.5` by default)
-   **Vector Store**: Local storage (persisted in `indexes/`)

## Project Structure
-   **`app.py`**: The main application entry point.
    -   Initializes LlamaIndex settings (LLM, Embeddings).
    -   Manages Streamlit session state for chat history and loaded models.
    -   Handles file uploads to `data/`.
    -   Builds vector indexes from documents and saves them to `indexes/`.
    -   Loads existing indexes and runs the chat engine (`CONTEXT` mode with Streaming).
-   **`config.yaml`**: Configuration file to specify the LLM and embedding model names.
-   **`data/`**: Temporary storage for uploaded raw documents.
-   **`indexes/`**: Storage for persisted LlamaIndex vector stores.
-   **`tests/`**: Test suite directory.
    -   `test_performance.py`: Verifies TTFT and model loading speed.
    -   `test_functional.py`: Functional tests for app logic.
    -   `benchmark.py`: Standalone performance benchmark script.
-   **`requirements.txt`**: Python dependencies.

## Key Workflows
1.  **Initialization**: `app.py` loads config, connects to Ollama, and loads the embedding model. **Models are cached** using `st.cache_resource` to prevent reloading.
2.  **Ingestion**: User uploads files -> Saved to `data/`.
3.  **Indexing**: User selects files -> `SimpleDirectoryReader` loads data -> `VectorStoreIndex` creates embeddings -> Persisted to `indexes/<timestamp>`.
4.  **Chat**: User selects an index -> Index loaded -> `ChatEngine` initialized (Context Mode) -> User asks question -> **Streaming Response** displayed token-by-token.

## Environment & Setup
-   **Virtual Env**: Uses `uv` or standard `venv`.
-   **Ollama**: Must be running locally (`ollama serve`).
-   **Run**: `streamlit run app.py`
-   **Test**: `pytest`

## Developer Notes
-   The app uses a custom `TimingCallbackHandler` to log performance metrics.
-   **Performance**: Models are cached. Chat uses `stream_chat` for low latency (<2s TTFT).
-   `OLLAMA_HOST` environment variable can be used to point to a remote Ollama instance.
