import streamlit as st
import os
from datetime import datetime
import time
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.chat_engine import SimpleChatEngine

st.title("📄 LlamaIndex Chatbot")

# Ensure data and indexes directories exist
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("indexes"):
    os.makedirs("indexes")

# Initialize LLM and Embedding Model
@st.cache_resource
def load_llm():
    return Ollama(
        model="llama3.2:latest",
        request_timeout=360.0,
        context_window=8000,
    )

@st.cache_resource
def load_embed_model():
    return HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")

Settings.llm = load_llm()
Settings.embed_model = load_embed_model()

with st.sidebar:
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
    
    # Index loaded indicator
    if "loaded_index" in st.session_state:
        st.markdown("Status: <span style='color:green'>●</span> Index Loaded", unsafe_allow_html=True)
    else:
        st.markdown("Status: <span style='color:red'>●</span> No Index Loaded", unsafe_allow_html=True)

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

# Chat input and response generation
if prompt := st.chat_input("Ask a question about the documents..."):
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
                    chat_mode="condense_question",
                    verbose=True,
                    llm=Settings.llm,
                )
                response = chat_engine.chat(st.session_state.messages[-1]["content"])
                end_time = time.time() # End timer
                response_time = round(end_time - start_time, 2) # Calculate response time

                full_response_content = f"{response.response}\n\n(Response time: {response_time} seconds)"
                st.session_state.messages.append({"role": "assistant", "content": full_response_content})
                st.rerun()
