# LlamaIndex Document Chatbot

This is a Streamlit web application that allows you to chat with your documents. You can upload PDF, TXT, or DOCX files, build a search index, and then ask questions to a chatbot that uses the content of your documents to answer.

## Features

-   **File Upload:** Upload multiple documents (PDF, TXT, DOCX).
-   **Index Building:** Create a [LlamaIndex](https://www.llamaindex.ai/) vector store index from your uploaded documents.
-   **Index Selection:** Choose from previously built indexes.
-   **Chat Interface:** A simple chat interface to ask questions about your documents.
-   **Ollama Integration:** Uses Ollama for generating responses. The model is configurable via `config.yaml`.
-   **Hugging Face Embeddings:** Uses `BAAI/bge-large-en-v1.5` for generating embeddings, also configurable in `config.yaml`.

## Prerequisites

Before you begin, ensure you have the following installed:

-   [Python 3.8+](https://www.python.org/downloads/)
-   [Ollama](https://ollama.ai/) installed and running.
-   The default model pulled in Ollama. You can pull it by running:
    ```sh
    ollama pull llama3:8b-instruct-q4_K_M
    ```

## Installation

1.  **Clone the repository:**
    ```sh
    git clone <your-repository-url>
    cd llama_agent_document_chatbot
    ```

2.  **Create a virtual environment and activate it:**
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Configuration

This application uses a `config.yaml` file to manage model settings. You can edit this file to change the models used by the application.

-   `llm_model_name`: The name of the Ollama model to use for chat.
-   `embedding_model_name`: The name of the Hugging Face model to use for document embeddings.

## How to Run

1.  **Run the Streamlit application:**
    ```sh
    streamlit run app.py
    ```

2.  **Open your web browser** and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Running Ollama on a Network

By default, Ollama runs on `http://localhost:11434`. If you want to run Ollama on a different machine or make it accessible from other devices on your network, you need to configure the `OLLAMA_HOST` environment variable.

### 1. Configure Ollama Server

Before starting the Ollama server, set the `OLLAMA_HOST` environment variable to the desired IP address and port.

**On Linux/macOS:**
```sh
export OLLAMA_HOST=0.0.0.0:11434
ollama serve
```

**On Windows (PowerShell):**
```powershell
$env:OLLAMA_HOST="0.0.0.0:11434"
ollama serve
```
-   `0.0.0.0` makes Ollama accessible from any IP address on your network. You can replace this with a specific IP address if needed (e.g., `192.168.1.100:11434`).
-   Ensure your firewall allows connections to the specified port (e.g., `11434`).

### 2. Configure the Client (`app.py`)

The `app.py` application will automatically pick up the `OLLAMA_HOST` environment variable if it's set in the environment where `streamlit run app.py` is executed.

**If `app.py` is running on the same machine as Ollama:**
No additional configuration is needed for `app.py` if you've set `OLLAMA_HOST` for the Ollama server.

**If `app.py` is running on a different machine than Ollama:**
You need to set the `OLLAMA_HOST` environment variable on the machine running `app.py` to point to the Ollama server's network address.

**On Linux/macOS (on the client machine):**
```sh
export OLLAMA_HOST=http://<ollama-server-ip>:11434
streamlit run app.py
```

**On Windows (PowerShell on the client machine):**
```powershell
$env:OLLAMA_HOST="http://<ollama-server-ip>:11434"
streamlit run app.py
```
Replace `<ollama-server-ip>` with the actual IP address of the machine running your Ollama server.

## How to Use

1.  **Upload Documents:**
    -   In the sidebar, use the file uploader to select one or more documents.

2.  **Build an Index:**
    -   After uploading, select the files you want to include in the index.
    -   (Optional) Provide a name for your index. If you don't, a timestamp will be used as the name.
    -   Click the "Build Index" button.

3.  **Load an Index:**
    -   Once you have at least one index built, you can select it from the "Choose an index" dropdown.
    -   Click the "Load Index" button.

4.  **Chat with your documents:**
    -   Once an index is loaded, you can start asking questions in the chat input field.

## Project Structure

```
.
├── app.py              # The main Streamlit application file
├── config.yaml         # Configuration file for model names
├── requirements.txt    # Python dependencies
├── data/               # Directory to store uploaded documents
└── indexes/            # Directory to store the generated indexes
```
