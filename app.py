import streamlit as st
import os
from datetime import datetime
import time
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from llama_index.core.chat_engine.types import ChatMode


# Load environment variables
load_dotenv()

class AppSettings(BaseSettings):
    LLAMA_MODEL_NAME: str = "llama3.2:latest"
    HUGGINGFACE_EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"

settings = AppSettings()

st.title("📄 LlamaIndex Chatbot")

# Ensure data and indexes directories exist
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("indexes"):
    os.makedirs("indexes")

with st.sidebar:
    st.header("Model Status")
    # Function to handle model loading with progress bars
    def initialize_models():
        if "llm" not in st.session_state:
            st.write("Initializing models...")
            
            progress_bar = st.progress(0, text="Loading LLM...")
            try:
                st.session_state.llm = Ollama(
                    model=settings.LLAMA_MODEL_NAME,
                    request_timeout=360.0,
                    context_window=8000,
                )
                Settings.llm = st.session_state.llm
                progress_bar.progress(50, text="LLM loaded. Loading embedding model...")
            except Exception as e:
                st.error(f"Failed to load LLM: {e}")
                st.stop()

            try:
                st.session_state.embed_model = HuggingFaceEmbedding(model_name=settings.HUGGINGFACE_EMBEDDING_MODEL_NAME)
                Settings.embed_model = st.session_state.embed_model
                progress_bar.progress(100, text="All models loaded successfully!")
                time.sleep(2) # Give user time to read the success message
                progress_bar.empty()
                st.rerun() # Rerun to clear the progress bar and messages
            except Exception as e:
                st.error(f"Failed to load embedding model: {e}")
                st.stop()

    # Initialize models if they are not in session state
    if "llm" not in st.session_state or "embed_model" not in st.session_state:
        initialize_models()
    else:
        st.markdown("Status: <span style='color:green'>●</span> Models Loaded", unsafe_allow_html=True)

    st.header("Index Status")
    # Index loaded indicator
    if "loaded_index" in st.session_state:
        st.markdown("Status: <span style='color:green'>●</span> Index Loaded", unsafe_allow_html=True)
    else:
        st.markdown("Status: <span style='color:red'>●</span> No Index Loaded", unsafe_allow_html=True)

    with st.expander("Upload & Build Index", expanded=True):
        st.header("Upload Documents")
        uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=["pdf", "txt", "docx"])
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join("data", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Saved {uploaded_file.name} to data/")

        st.header("Build Index")
        
        # File selection for indexing
        files_in_data_dir = [f for f in os.listdir("data") if os.path.isfile(os.path.join("data", f))]
        selected_files_for_indexing = st.multiselect("Select files to index:", files_in_data_dir)

        index_name_input = st.text_input("Enter index name (optional, defaults to timestamp):")
        if st.button("Build Index"):
            if not selected_files_for_indexing:
                st.warning("Please select at least one file to build an index.")
            else:
                with st.spinner("Building index... This may take a while!"):
                    input_files = [os.path.join("data", f) for f in selected_files_for_indexing]
                    documents = SimpleDirectoryReader(input_files=input_files).load_data()
                    
                    if not documents:
                        st.warning("Could not load any documents from the selected files.")
                    else:
                        index = VectorStoreIndex.from_documents(documents)
                        
                        if not index_name_input:
                            index_name_input = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        index_dir = os.path.join("indexes", index_name_input)
                        index.storage_context.persist(persist_dir=index_dir)
                        st.success(f"Index '{index_name_input}' built and saved to '{index_dir}'")

    st.header("Select Index")
    available_indexes = [d for d in os.listdir("indexes") if os.path.isdir(os.path.join("indexes", d))]
    
    if not available_indexes:
        st.warning("No indexes found. Please build an index first.")
    else:
        selected_index_name = st.selectbox("Choose an index", available_indexes)
        if st.button("Load Index"):
            with st.spinner(f"Loading index '{selected_index_name}'..."):
                try:
                    storage_context = StorageContext.from_defaults(persist_dir=os.path.join("indexes", selected_index_name))
                    index = load_index_from_storage(storage_context)
                    st.session_state.loaded_index = index
                    st.success(f"Index '{selected_index_name}' loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading index: {e}")

    st.header("Chat History")
    if st.button("Clear Chat"):
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]
        st.success("Chat history cleared!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Inform user if models are not loaded
if "llm" not in st.session_state:
    st.info("Models are initializing in the sidebar. Please wait...")

# Chat input and response generation
models_loaded = "llm" in st.session_state
if prompt := st.chat_input(
    "Ask a question about the documents...", 
    disabled=not models_loaded
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# Generate assistant response if the last message is from the user
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    if "loaded_index" not in st.session_state:
        with st.chat_message("assistant"):
            st.warning("Please load an index first from the sidebar.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time() # Start timer
                chat_engine = st.session_state.loaded_index.as_chat_engine(
                    chat_mode=ChatMode.CONDENSE_QUESTION,
                    verbose=True,
                    llm=st.session_state.llm,
                )
                response = chat_engine.chat(st.session_state.messages[-1]["content"])
                end_time = time.time() # End timer
                response_time = round(end_time - start_time, 2) # Calculate response time

                full_response_content = f"{response.response}\n\n(Response time: {response_time} seconds)"
                st.session_state.messages.append({"role": "assistant", "content": full_response_content})
                st.rerun()
