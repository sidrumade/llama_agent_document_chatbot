# 📄 Local LlamaIndex Document Chatbot

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LlamaIndex](https://img.shields.io/badge/Orchestration-LlamaIndex-red.svg)](https://www.llamaindex.ai/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-orange.svg)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/Backend-Ollama-black.svg)](https://ollama.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A privacy-focused, premium local document chatbot built with **Streamlit** and **LlamaIndex**. This application allows users to upload documents (PDFs, TXT, DOCX), generate high-performance vector embeddings locally, persist search indexes, and engage in real-time, context-aware streaming chats powered by **Ollama** (e.g., Llama 3.2).

---

## 🏛️ System Architecture & Data Flow

Below is the visual workflow detailing how uploaded documents are loaded, embedded, and queried in a context-aware chat loop.

```mermaid
graph TD
    %% Styling Classes
    classDef user fill:#64b5f6,stroke:#1565c0,stroke-width:2px,color:#000;
    classDef storage fill:#ffb74d,stroke:#f57c00,stroke-width:2px,color:#000;
    classDef process fill:#81c784,stroke:#2e7d32,stroke-width:2px,color:#000;
    classDef cache fill:#ba68c8,stroke:#6a1b9a,stroke-width:2px,color:#000;
    classDef llm fill:#e0e0e0,stroke:#424242,stroke-width:2px,color:#000;

    %% Workflow Nodes
    User(["User"]) ::: user
    Files["Upload PDF, TXT, DOCX"] ::: user
    DataFolder[("data/ directory")] ::: storage
    Reader["SimpleDirectoryReader"] ::: process
    Embedder["HuggingFaceEmbedding<br/>BAAI/bge-large-en-v1.5"] ::: process
    VectorStore["VectorStoreIndex"] ::: process
    IndexFolder[("indexes/ directory")] ::: storage

    CachedEmbed["Cached Embedding Model"] ::: cache
    CachedLLM["Cached Ollama LLM<br/>llama3.2:latest"] ::: cache
    OllamaSrv["Ollama Service<br/>localhost:11434"] ::: llm
    
    UI["Streamlit Chat Interface"] ::: user
    ChatEngine["ChatEngine Context Mode"] ::: process
    Callback["TimingCallbackHandler"] ::: process
    Console["Performance Logs / stdout"] ::: storage

    %% Flow Connections
    User -->|Uploads| Files
    Files -->|Saved as raw files| DataFolder
    DataFolder -->|Read files| Reader
    Reader -->|Extract Text| Embedder
    CachedEmbed -.->|Generates Vectors| Embedder
    Embedder -->|Constructs| VectorStore
    VectorStore -->|Persists Index| IndexFolder

    User -->|Selects & Loads Index| UI
    IndexFolder -->|Loads VectorStore| UI
    UI -->|Ask Question| ChatEngine
    CachedLLM -.->|Instantiate LLM| ChatEngine
    OllamaSrv <-->|API Calls| CachedLLM
    
    ChatEngine -->|Retrieve Context| VectorStore
    ChatEngine -->|stream_chat| UI
    ChatEngine -->|Triggers Events| Callback
    Callback -->|Measure Latency| Console
```

---

## ✨ Features

-   **Multi-Format Document Upload**: Ingest `PDF`, `TXT`, and `DOCX` files.
-   **Local Vector Indexing**: Create robust vector stores with `LlamaIndex` using the high-accuracy `BAAI/bge-large-en-v1.5` Hugging Face embedding model.
-   **Index Directory Persistence**: Save built indexes into separate subdirectories under `indexes/` and easily load them on-demand via the sidebar.
-   **Session State-Caching**: High-performance resource caching (`st.cache_resource`) prevents reloading the embedding models and local Ollama instance on every click.
-   **Real-time Streaming Responses**: Provides low Time-To-First-Token (TTFT) interactions via `stream_chat`.
-   **Granular Performance Callback Logging**: Intercepts LlamaIndex orchestrations to measure exact timings for embedding generation, database retrievals, LLM generation, synthesis, and full queries.
-   **Network-capable Ollama Support**: Readily supports connecting to remote or network-shared Ollama instances via the `OLLAMA_HOST` variable.

---

## 🛠️ Prerequisites

Ensure you have the following installed on your machine:
-   **Python 3.8+** (Python 3.10 or 3.12 recommended)
-   **Ollama** installed and running on your local machine (or on your network).
-   Download the default Ollama instruct model by running:
    ```sh
    ollama pull llama3.2:latest
    ```
    *(Alternatively, you can pull any supported model, such as `llama3:8b-instruct-q4_K_M`, and update `config.yaml` accordingly).*

---

## 🚀 Installation & Setup

1.  **Clone the Repository**:
    ```sh
    git clone <your-repository-url>
    cd llama_agent_document_chatbot
    ```

2.  **Create and Activate a Virtual Environment**:
    -   **On Windows (PowerShell)**:
        ```powershell
        python -m venv .venv
        .venv\Scripts\Activate.ps1
        ```
    -   **On Linux/macOS**:
        ```sh
        python -m venv .venv
        source .venv/bin/activate
        ```

3.  **Install Required Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

---

## ⚙️ Configuration

The application parses environment and model configurations from two primary files:

### 1. `config.yaml`
Modify the configurations in `config.yaml` to substitute model endpoints and providers:
```yaml
llm_provider: "ollama"  # Set to "ollama" for offline local execution, or "gemini" for Google Gemini API
llm_model_name: "llama3.2:latest"
llm_planner_model_name: "llama3.2:latest"
embedding_model_name: "BAAI/bge-small-en-v1.5"
reranker_model_name: "BAAI/bge-reranker-base"
query_fusion_queries: 3
enable_context_compression: true
enable_auto_merging: true
```
-   `llm_provider`: The LLM service provider. Set to `"ollama"` to run completely offline/locally, or `"gemini"` to use the cloud Gemini service.
-   `llm_model_name`: The LLM model name to target for synthesizing final responses (e.g., `llama3.2:latest` for Ollama, or `models/gemini-2.5-flash` for Gemini).
-   `llm_planner_model_name`: The LLM model name used for the planning step of the agentic RAG workflow.
-   `embedding_model_name`: Hugging Face model identifier for generating vector embeddings locally.
-   `reranker_model_name`: Hugging Face model identifier for local cross-encoder reranking.
-   `query_fusion_queries`: Number of query variations generated by the planner LLM.
-   `enable_context_compression`: If true, uses the LLM to contextually compress/extract relevant sentences from retrieved chunks.
-   `enable_auto_merging`: If true, automatically merges retrieved child leaf nodes into larger parent nodes.

### 2. Network Ollama Configurations (`OLLAMA_HOST`)
By default, the Ollama client attempts connection to `http://localhost:11434`. To run Ollama on a remote server or make it accessible on a local area network:

#### A. Configure the Server Machine
Before launching the Ollama service, define the listener host.
-   **On Linux/macOS**:
    ```sh
    export OLLAMA_HOST=0.0.0.0:11434
    ollama serve
    ```
-   **On Windows (PowerShell)**:
    ```powershell
    $env:OLLAMA_HOST="0.0.0.0:11434"
    ollama serve
    ```

#### B. Configure the Client Machine (Running Streamlit)
Provide the remote server's IP address to the Streamlit app's environment context.
-   **On Linux/macOS**:
    ```sh
    export OLLAMA_HOST=http://<ollama-server-ip>:11434
    streamlit run app.py
    ```
-   **On Windows (PowerShell)**:
    ```powershell
    $env:OLLAMA_HOST="http://<ollama-server-ip>:11434"
    streamlit run app.py
    ```

---

## 💻 Running the Application

1.  **Start Streamlit**:
    ```sh
    streamlit run app.py
    ```
2.  **Open browser** and navigate to `http://localhost:8501`.

---

## 📖 How to Use

1.  **Sidebar - Upload Documents**:
    -   Drag and drop PDF, TXT, or DOCX files into the file uploader. 
    -   Files are instantly written to the local `data/` directory.
2.  **Sidebar - Build Index**:
    -   Select which of the saved files to include in the current index.
    -   Enter an optional name for the index (defaults to the current timestamp if blank).
    -   Click **Build Index**. This reads the files, generates embeddings, and saves the vectors to `indexes/<index_name>`.
3.  **Sidebar - Load Index**:
    -   Select the desired index from the dropdown of available stored indexes.
    -   Click **Load Index** to initialize the Vector Store index in your active session.
4.  **Main Chat Area - Conversation**:
    -   Ask questions in the chat input. 
    -   The query engine will perform similarity retrieval against the index, feed the relevant text context to the Ollama LLM, and stream the generated answer token-by-token.

---

## 🧑‍💻 Developer Architecture Guide

### ⚡ Caching Mechanisms
To deliver a responsive user experience (UX) and prevent expensive memory leaks, models are decorated with `@st.cache_resource`. This ensures the Hugging Face embedding model (`sentence-transformers`) and the Ollama API interfaces are loaded exactly **once** per app lifecycle:
```python
@st.cache_resource
def load_llm():
    return Ollama(model=LLAMA_MODEL_NAME, request_timeout=360.0, context_window=8000)

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbedding(model_name=HUGGINGFACE_EMBEDDING_MODEL_NAME)
```

### ⏱️ Performance Callback Tracking
The application registers a custom `TimingCallbackHandler` subclass of LlamaIndex's `BaseCallbackHandler`. This monitors events and prints precise timing diagnostics directly to the terminal stdout:
*   `CBEventType.EMBEDDING`: Time taken to run document text chunks through the Hugging Face transformer model.
*   `CBEventType.LLM`: Total time spent waiting for response generation from the Ollama backend API.
*   `CBEventType.RETRIEVE`: Vector database similarity lookup duration.
*   `CBEventType.SYNTHESIZE`: LlamaIndex combining retrieved context node results.
*   `CBEventType.QUERY`: Complete query pipeline execution duration.

Example console performance prints:
```
2026-05-28 20:26:00 - INFO - Embedding model loaded in 4.12 seconds.
2026-05-28 20:26:05 - INFO - Embedding generation took: 0.85 seconds
2026-05-28 20:26:08 - INFO - Retrieval took: 0.12 seconds
2026-05-28 20:26:12 - INFO - LLM call took: 3.42 seconds
```

---

## 🧪 Testing & Benchmarking Suite

The project includes functional, integration, and performance benchmarking scripts inside the `tests/` directory.

### 1. Functional Tests
Verify individual module methods, config loaders, and model setups using mocked interfaces.
```sh
pytest tests/test_functional.py
```

### 2. Performance & UX Tests
Validate real-time performance thresholds. These tests measure Time To First Token (TTFT) and model loading speeds to ensure strict adherence to UX constraints.
```sh
pytest tests/test_performance.py -s
```
*Key Performance KPIs verified:*
-   **LLM Load Time**: Asserts instantiation speeds.
-   **Embedding Indexing Time**: Asserts quick vector creation for small text volumes (<10s threshold).
-   **Time to First Token (TTFT)**: Ensures the streaming interface begins rendering text in **under 2.0 seconds** (crucial for responsive chat UX).

### 3. Standalone Benchmark Tool
Run a standalone performance sweep that builds a test index, indexes mock files, and processes query benchmarks, outputting detailed timing statistics to the terminal.
```sh
python tests/benchmark.py
```

---

## 📁 Project Structure

```
.
├── app.py                  # Streamlit application entrypoint & session-state coordinator
├── config.yaml             # Configuration declarations for LLM & Embedding models
├── requirements.txt        # Pinned Python package dependencies
├── test_report.txt         # Detailed pytest results and code coverage metrics
├── data/                   # Directory storing uploaded raw text/PDF/DOCX documents
├── indexes/                # Vector store database partitions, grouped by index name
└── tests/                  # Robust testing suite
    ├── benchmark.py        # Standalone speed performance script
    ├── test_functional.py  # Mocked streamlit & system logic assertions
    └── test_performance.py # UX latency, TTFT, and processing threshold validations
```

