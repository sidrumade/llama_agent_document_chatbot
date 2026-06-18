# Project Context: LlamaIndex Document Chatbot

## 🎯 Overview
This project is a privacy-focused document chatbot built with **Streamlit** and **LlamaIndex**. By default, it runs completely offline using local models via **Ollama** and **HuggingFace**, but it can also be configured to connect to Google's **Gemini** API. It allows users to upload documents (PDF, TXT, DOCX), create vector embeddings locally, persist them, and chat with them using a streaming agentic RAG workflow.

## 💻 Tech Stack
-   **Frontend & UI**: Streamlit
-   **Orchestration Framework**: LlamaIndex
-   **LLM Providers**: Ollama (e.g., `llama3.2:latest` for fully offline execution) or Google Gemini (online API)
-   **Embeddings Generation**: HuggingFace (`BAAI/bge-small-en-v1.5` by default)
-   **Reranker Model**: HuggingFace (`BAAI/bge-reranker-base` by default)
-   **Vector Database Store**: Local file-system vector store (persisted in `indexes/`)
-   **Testing Framework**: pytest (functional and performance tests)

## 📁 Project Structure
-   **`app.py`**: Main application code.
    -   Initializes LlamaIndex configurations.
    -   Leverages `@st.cache_resource` for high-performance memory caching of model instantiations.
    -   Monitors event-based callback performance via a custom `TimingCallbackHandler`.
    -   Saves uploaded documents to `data/` and builds indexes under `indexes/`.
    -   Manages the UI sidebar configuration and chat inputs.
-   **`config.yaml`**: Configuration file defining `llm_provider`, LLM models (synthesis & planner), embedding and reranker model names, and pipeline settings.
-   **`data/`**: Ingestion directory storing uploaded raw source documents.
-   **`indexes/`**: Vector database directories, separated by user-defined index names.
-   **`requirements.txt`**: Pinned Python package dependencies.
-   **`test_report.txt`**: Saved execution output of functional tests and coverage metrics.
-   **`tests/`**: Suite containing tests and benchmarks.
    -   `test_functional.py`: Runs logic checks against configurations and model loading mocks.
    -   `test_performance.py`: Measures model load times, indexing speeds, and Time-To-First-Token (TTFT < 2s).
    -   `benchmark.py`: Standalone script to verify local database and querying speeds.

## 🔄 Key Workflows
1.  **Model Ingest & Caching**: Streamlit loads the embedding models and Ollama client interfaces using caching decorators to prevent reload lag.
2.  **Document Reader**: Source texts are read from `data/` using LlamaIndex's `SimpleDirectoryReader`.
3.  **Embeddings Creation**: Chunks are processed into vector coordinates via HuggingFace's local embedding pipeline.
4.  **Index Persistence**: Vectors are written locally to their respective `indexes/` subfolder.
5.  **Agentic RAG Query Execution & Response Streaming**: When a query is submitted, the system runs `RefinedAgenticWorkflow`:
    -   **Planning**: The planner LLM (Ollama or Gemini) analyzes the query to select a retrieval strategy (`summary`, `hierarchical`, `sentence_window`, or `hybrid`), define metadata filters, and decompose the query into sub-queries.
    -   **Retrieval**: Fetches relevant nodes using the chosen strategy.
    -   **Post-processing**: Applies metadata expansion (for sentence window), cross-encoder reranking, contextual compression (extracting relevant sentences with the LLM), and long context reordering.
    -   **Synthesis**: Streams the final source-cited response to the Streamlit UI.
6.  **Granular Timing Logs**: The custom callback handler logs performance metrics (Embedding, LLM, Retrieval, and Synthesis durations) to standard output.

## ⚙️ Environment Setup
-   **Ollama Backend**: Must be running (`ollama serve`).
-   **Ollama Network Host**: Customize server/client connections using the `OLLAMA_HOST` env variable.
-   **Run Command**: `streamlit run app.py`
-   **Test Command**: `pytest`
-   **Benchmark Command**: `python tests/benchmark.py`

