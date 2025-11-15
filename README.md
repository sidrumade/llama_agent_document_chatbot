# LlamaIndex Document Chatbot

This is a Streamlit web application that allows you to chat with your documents. You can upload PDF, TXT, or DOCX files, build a search index, and then ask questions to a chatbot that uses the content of your documents to answer.

## Features

-   **File Upload:** Upload multiple documents (PDF, TXT, DOCX).
-   **Index Building:** Create a [LlamaIndex](https://www.llamaindex.ai/) vector store index from your uploaded documents.
-   **Index Selection:** Choose from previously built indexes.
-   **Chat Interface:** A simple chat interface to ask questions about your documents.
-   **Ollama Integration:** Uses Ollama with the Llama 3.2 model for generating responses.
-   **Hugging Face Embeddings:** Uses `BAAI/bge-large-en-v1.5` for generating embeddings.

## Prerequisites

Before you begin, ensure you have the following installed:

-   [Python 3.8+](https://www.python.org/downloads/)
-   [Ollama](https://ollama.ai/) installed and running.
-   The `llama3.2` model pulled in Ollama. You can pull it by running:
    ```sh
    ollama pull llama3.2
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

## How to Run

1.  **Run the Streamlit application:**
    ```sh
    streamlit run app.py
    ```

2.  **Open your web browser** and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

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
├── requirements.txt    # Python dependencies
├── data/               # Directory to store uploaded documents
└── indexes/            # Directory to store the generated indexes
```
