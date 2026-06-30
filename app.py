import streamlit as st
import os
import time
import logging
import yaml
from datetime import datetime
import asyncio
import nest_asyncio
import httpx
from dotenv import load_dotenv

# Apply nest_asyncio to support nested event loops in Streamlit
nest_asyncio.apply()

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.callbacks import CallbackManager, CBEventType
from llama_index.core.callbacks.base_handler import BaseCallbackHandler

# Import from our new RAG engine
from rag_engine import IndexManager, RefinedAgenticWorkflow, PlanStepEvent, RetrievalStepEvent, PostprocessStepEvent, TextChunkEvent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Loading ---
def load_config():
    """Loads configuration from config.yaml."""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error("config.yaml not found. Please make sure it exists.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading config.yaml: {e}")
        st.stop()

config = load_config()
LLM_MODEL_NAME = config.get("llm_model_name", "llama3.2:latest")
LLM_PLANNER_MODEL_NAME = config.get("llm_planner_model_name", "llama3.2:latest")
HUGGINGFACE_EMBEDDING_MODEL_NAME = config.get("embedding_model_name", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL_NAME = config.get("reranker_model_name", "BAAI/bge-reranker-base")

# Helper to construct resolved Ollama URL from config or env
def get_ollama_base_url():
    # 1. Read from config.yaml
    host = config.get("ollama_host", None)
    port = config.get("ollama_port", None)
    
    # 2. Fallback to environment variables if not in config
    if not host:
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").strip()
    
    host = str(host).strip()
    
    # If host is just a port, e.g., "11434"
    if host.isdigit():
        port = int(host)
        host = "127.0.0.1"
        
    protocol = "http://"
    if host.startswith("http://"):
        host = host[7:]
    elif host.startswith("https://"):
        host = host[8:]
        protocol = "https://"
        
    # Replace 0.0.0.0 with 127.0.0.1 for Windows compatibility
    if "0.0.0.0" in host:
        host = host.replace("0.0.0.0", "127.0.0.1")
        
    if ":" in host:
        final_host = host
    else:
        if port:
            final_host = f"{host}:{port}"
        else:
            final_host = f"{host}:11434"
            
    return f"{protocol}{final_host}"

# Helper to check connection status
def check_ollama_connection(base_url: str) -> tuple[bool, str]:
    try:
        url = base_url.rstrip("/") + "/api/tags"
        response = httpx.get(url, timeout=5.0)
        if response.status_code == 200:
            models_data = response.json()
            models_list = [m["name"] for m in models_data.get("models", [])]
            models_str = ", ".join(models_list) if models_list else "None"
            return True, f"Connected! Available models: {models_str}"
        else:
            return False, f"Failed. HTTP Status Code: {response.status_code}"
    except Exception as e:
        return False, f"Connection failed: {type(e).__name__}: {e}"

# --- End Configuration Loading ---

# Define the custom callback handler for timing
class TimingCallbackHandler(BaseCallbackHandler):
    def __init__(self, event_starts_to_ignore=[], event_types_to_ignore=[]):
        super().__init__(event_starts_to_ignore, event_types_to_ignore)
        self.timings = {}

    def on_event_start(self, event_type, payload=None, event_id: str = "", parent_id: str = "", **kwargs):
        self.timings[event_id] = time.time()

    def on_event_end(self, event_type, payload=None, event_id: str = "", parent_id: str = "", **kwargs):
        if event_id in self.timings:
            duration = time.time() - self.timings.pop(event_id)
            if event_type == CBEventType.EMBEDDING:
                logging.info(f"Embedding generation took: {duration:.2f} seconds")
            elif event_type == CBEventType.LLM:
                logging.info(f"LLM call took: {duration:.2f} seconds")
            elif event_type == CBEventType.QUERY:
                logging.info(f"Query processing took: {duration:.2f} seconds")
            elif event_type == CBEventType.RETRIEVE:
                logging.info(f"Retrieval took: {duration:.2f} seconds")
            elif event_type == CBEventType.SYNTHESIZE:
                logging.info(f"Synthesis took: {duration:.2f} seconds")

    def start_trace(self, trace_id: str | None = None) -> None:
        pass

    def end_trace(self, trace_id: str | None = None, trace_map: dict[str, list[str]] | None = None) -> None:
        pass

# Set up the callback manager
timing_handler = TimingCallbackHandler()
callback_manager = CallbackManager([timing_handler])
Settings.callback_manager = callback_manager

st.title("📄 LlamaIndex Chatbot")

# Ensure data and indexes directories exist
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("indexes"):
    os.makedirs("indexes")

with st.sidebar:
    st.header("Model Status")

    # Ollama Connection Configuration UI
    st.subheader("Ollama Connection")
    resolved_host = get_ollama_base_url()
    st.caption(f"Host: `{resolved_host}`")
    
    if st.button("🔌 Check Ollama Connection", key="check_ollama_conn"):
        with st.spinner("Pinging Ollama..."):
            success, msg = check_ollama_connection(resolved_host)
            if success:
                st.success(msg)
            else:
                st.error(msg)

    # Function to handle model loading with progress bars
    @st.cache_resource
    def load_llm(provider, model_name, api_key=None):
        ollama_host = get_ollama_base_url()
        logging.info(f"Loading LLM (ollama - {model_name})...")
        logging.info(f"Initializing Ollama client with base_url: {ollama_host}")
        return Ollama(
            model=model_name,
            base_url=ollama_host,
            request_timeout=360.0,
            context_window=8000,
        )

    @st.cache_resource
    def load_embedding_model():
        logging.info("Loading embedding model...")
        return HuggingFaceEmbedding(model_name=HUGGINGFACE_EMBEDDING_MODEL_NAME)

    @st.cache_resource
    def load_reranker(model_name):
        logging.info(f"Loading reranker model {model_name}...")
        from llama_index.core.postprocessor import SentenceTransformerRerank
        return SentenceTransformerRerank(
            model=model_name,
            top_n=5
        )

    def initialize_models():
        logging.info("Initializing models...")
        start_time = time.time()
        
        needs_loading = (
            "llm" not in st.session_state or 
            "planner_llm" not in st.session_state or 
            "embed_model" not in st.session_state or 
            "reranker" not in st.session_state
        )
        
        if needs_loading:
            st.write("Initializing models...")
            progress_bar = st.progress(0, text="Loading Synthesis LLM...")
            
            try:
                llm_start_time = time.time()
                # Load primary synthesis LLM
                st.session_state.llm = load_llm("ollama", LLM_MODEL_NAME)
                Settings.llm = st.session_state.llm
                
                # Load planning LLM
                st.session_state.planner_llm = load_llm("ollama", LLM_PLANNER_MODEL_NAME)
                
                llm_end_time = time.time()
                logging.info(f"LLMs loaded in {llm_end_time - llm_start_time:.2f} seconds.")
                progress_bar.progress(33, text="LLMs loaded. Loading embedding model...")
            except Exception as e:
                logging.error(f"Failed to load LLM: {e}")
                st.error(f"Failed to load LLM: {e}")
                st.stop()

            try:
                embed_start_time = time.time()
                st.session_state.embed_model = load_embedding_model()
                Settings.embed_model = st.session_state.embed_model
                embed_end_time = time.time()
                logging.info(f"Embedding model loaded in {embed_end_time - embed_start_time:.2f} seconds.")
                progress_bar.progress(66, text="Embedding model loaded. Loading reranker...")
            except Exception as e:
                logging.error(f"Failed to load embedding model: {e}")
                st.error(f"Failed to load embedding model: {e}")
                st.stop()

            try:
                reranker_start_time = time.time()
                st.session_state.reranker = load_reranker(RERANKER_MODEL_NAME)
                reranker_end_time = time.time()
                logging.info(f"Reranker loaded in {reranker_end_time - reranker_start_time:.2f} seconds.")
                progress_bar.progress(100, text="All models loaded successfully!")
                time.sleep(1) 
                progress_bar.empty()
                st.rerun()
            except Exception as e:
                logging.error(f"Failed to load reranker: {e}")
                st.error(f"Failed to load reranker: {e}")
                st.stop()
        
        end_time = time.time()
        logging.info(f"Model initialization finished in {end_time - start_time:.2f} seconds.")

    # Initialize models if they are not in session state
    if "llm" not in st.session_state or "embed_model" not in st.session_state or "reranker" not in st.session_state:
        initialize_models()
    else:
        st.markdown("Status: <span style='color:green'>●</span> Models Loaded", unsafe_allow_html=True)

    st.header("Retrieval Settings")
    query_fusion_queries = st.slider(
        "Query Fusion Variations",
        min_value=1,
        max_value=5,
        value=config.get("query_fusion_queries", 3),
        help="Number of query variations to generate for hybrid fusion search."
    )
    enable_auto_merging = st.toggle(
        "Enable Auto-Merging Context",
        value=config.get("enable_auto_merging", True),
        help="Automatically merge retrieved child chunks into parent sections when beneficial."
    )
    enable_context_compression = st.toggle(
        "Enable Context Compression",
        value=config.get("enable_context_compression", True),
        help="Use the LLM to extract only relevant sentences from retrieved chunks."
    )

    st.header("Index Status")
    # Index loaded indicator
    if "loaded_index_dict" in st.session_state:
        idx_type = st.session_state.loaded_index_dict.get("type", "legacy")
        st.markdown(f"Status: <span style='color:green'>●</span> Index Loaded ({idx_type.capitalize()})", unsafe_allow_html=True)
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
        files_in_data_dir = [f for f in os.listdir("data") if os.path.isfile(os.path.join("data", f))]
        selected_files_for_indexing = st.multiselect("Select files to index:", files_in_data_dir)

        index_name_input = st.text_input("Enter index name (optional, defaults to timestamp):")
        if st.button("Build Index"):
            if not selected_files_for_indexing:
                st.warning("Please select at least one file to build an index.")
            elif "llm" not in st.session_state:
                st.warning("Please configure and load models before building an index.")
            else:
                with st.spinner("Building multi-layer index... This may take a while as summaries are generated!"):
                    logging.info("Building multi-layer index...")
                    start_time = time.time()
                    input_files = [os.path.join("data", f) for f in selected_files_for_indexing]
                    documents = SimpleDirectoryReader(input_files=input_files).load_data()
                    
                    if not documents:
                        st.warning("Could not load any documents from the selected files.")
                    else:
                        if not index_name_input:
                            index_name_input = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        index_dict = IndexManager.build_multi_index(
                            documents,
                            index_name_input,
                            st.session_state.llm,
                            st.session_state.embed_model
                        )
                        st.session_state.loaded_index_dict = index_dict
                        
                        # Preserve legacy reference for backward compatibility
                        st.session_state.loaded_index = index_dict["hierarchical"]
                        
                        end_time = time.time()
                        logging.info(f"Multi-index '{index_name_input}' built and saved in {end_time - start_time:.2f} seconds.")
                        st.success(f"Multi-index '{index_name_input}' built successfully!")
                        st.rerun()

    st.header("Select Index")
    available_indexes = [d for d in os.listdir("indexes") if os.path.isdir(os.path.join("indexes", d))]
    
    if not available_indexes:
        st.warning("No indexes found. Please build an index first.")
    else:
        selected_index_name = st.selectbox("Choose an index", available_indexes)
        if st.button("Load Index"):
            with st.spinner(f"Loading index '{selected_index_name}'..."):
                logging.info(f"Loading index '{selected_index_name}'...")
                start_time = time.time()
                try:
                    index_dict = IndexManager.load_index(selected_index_name, st.session_state.embed_model)
                    st.session_state.loaded_index_dict = index_dict
                    
                    # Backward compatibility fallback
                    if index_dict["type"] == "multi":
                        st.session_state.loaded_index = index_dict["hierarchical"]
                    else:
                        st.session_state.loaded_index = index_dict["index"]
                        
                    end_time = time.time()
                    logging.info(f"Index '{selected_index_name}' loaded in {end_time - start_time:.2f} seconds.")
                    st.success(f"Index '{selected_index_name}' loaded successfully!")
                    st.rerun()
                except Exception as e:
                    logging.error(f"Error loading index: {e}")
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
        if "citations" in message:
            with st.expander("📚 Sources Cited", expanded=False):
                for cite in message["citations"]:
                    st.markdown(
                        f"**[{cite['index']}] {cite['filename']}** (Page {cite['page_number']}, Section: `{cite['section']}`) — *Relevance: {cite['confidence']}%*"
                    )

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
    if "loaded_index_dict" not in st.session_state:
        with st.chat_message("assistant"):
            st.warning("Please load an index first from the sidebar.")
    else:
        with st.chat_message("assistant"):
            # Execute Agentic RAG Workflow
            logging.info("Executing Agentic RAG Workflow...")
            start_time = time.time()
            
            wf_settings = {
                "query_fusion_queries": query_fusion_queries,
                "enable_auto_merging": enable_auto_merging,
                "enable_context_compression": enable_context_compression
            }
            
            # Setup workflow
            workflow = RefinedAgenticWorkflow(
                index_dict=st.session_state.loaded_index_dict,
                llm=st.session_state.llm,
                embed_model=st.session_state.embed_model,
                reranker=st.session_state.reranker,
                settings=wf_settings
            )
            
            # Inject planner LLM
            workflow.llm = st.session_state.planner_llm
            
            # Streaming Output Block
            message_placeholder = st.empty()
            state = {"full_response": "", "citations_data": []}
            
            async def run_workflow_stream():
                user_query = st.session_state.messages[-1]["content"]
                handler = workflow.run(query=user_query)
                async for event in handler.stream_events():
                    if isinstance(event, PlanStepEvent):
                        logging.info(f"Query Plan: {event.reasoning}")
                        logging.info(f"- Strategy: {event.strategy}")
                        logging.info(f"- Sub-queries: {event.sub_queries}")
                        if event.metadata_filters:
                            logging.info(f"- Filters: {event.metadata_filters}")
                    elif isinstance(event, RetrievalStepEvent):
                        logging.info(f"Retrieval: Found {event.num_nodes} chunks using {event.strategy} search.")
                    elif isinstance(event, PostprocessStepEvent):
                        logging.info(f"Processing: {event.msg}")
                    elif isinstance(event, TextChunkEvent):
                        state["full_response"] += event.text
                        message_placeholder.markdown(state["full_response"] + "▌")
                        
                # Retrieve final response block
                result = await handler
                state["citations_data"] = result.get("citations", [])
                # The final response generation yields using TextChunkEvent, but we assign final result
                return result.get("response", "")

            try:
                # Run the asynchronous loop synchronously in Streamlit
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                final_txt = loop.run_until_complete(run_workflow_stream())
                
                # Render final text without cursor
                message_placeholder.markdown(final_txt)
                
                # Store final text and citations in chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": final_txt,
                    "citations": state["citations_data"]
                })
                
                if state["citations_data"]:
                    with st.expander("📚 Sources Cited", expanded=False):
                        for cite in state["citations_data"]:
                            st.markdown(
                                f"**[{cite['index']}] {cite['filename']}** (Page {cite['page_number']}, Section: `{cite['section']}`) — *Relevance: {cite['confidence']}%*"
                            )
            except Exception as e:
                logging.error(f"Error executing agentic workflow: {e}", exc_info=True)
                st.error(f"Error executing agentic workflow: {e}")
                st.stop()
            
            end_time = time.time()
            logging.info(f"Total response generated in {end_time - start_time:.2f} seconds.")
