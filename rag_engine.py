import os
import time
import logging
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

import streamlit as st
import yaml
from llama_index.core.llms import ChatMessage

from llama_index.core import (
    VectorStoreIndex,
    DocumentSummaryIndex,
    StorageContext,
    Settings,
    load_index_from_storage
)
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SentenceWindowNodeParser,
    get_leaf_nodes
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core.retrievers import AutoMergingRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import (
    SentenceTransformerRerank,
    LongContextReorder,
    MetadataReplacementPostProcessor
)
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.core.workflow import (
    Workflow,
    step,
    Event,
    StartEvent,
    StopEvent,
    Context
)
from llama_index.core.schema import NodeWithScore, Document

# Setup logging
def configure_logging(log_file: str = "logs/chatbot.log"):
    """Configures the root logger to write logs to both standard output and a file."""
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Avoid duplicating handlers if already set up
    has_file_handler = False
    has_stream_handler = False
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            has_file_handler = True
        elif isinstance(handler, logging.StreamHandler):
            has_stream_handler = True
            
    if not has_file_handler:
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Failed to create log file {log_file}: {e}")
            
    if not has_stream_handler:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)

# Setup initial logger for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Metadata Enrichment ---
def enrich_document_metadata(documents: List[Document]):
    """Enriches loaded documents with standardized metadata fields."""
    start_time = time.time()
    logger.info(f"Enriching metadata for {len(documents)} documents...")
    for doc in documents:
        # Set upload timestamp
        if "upload_timestamp" not in doc.metadata:
            doc.metadata["upload_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Set standardized filename
        if "filename" not in doc.metadata:
            doc.metadata["filename"] = doc.metadata.get("file_name", "unknown")
            
        # Set standardized document type
        if "document_type" not in doc.metadata:
            if "file_type" in doc.metadata:
                doc.metadata["document_type"] = doc.metadata["file_type"]
            else:
                _, ext = os.path.splitext(doc.metadata.get("file_name", ""))
                doc.metadata["document_type"] = ext.lstrip(".").lower()
                
        # Set standardized page number
        if "page_number" not in doc.metadata:
            if "page_label" in doc.metadata:
                doc.metadata["page_number"] = str(doc.metadata["page_label"])
            else:
                doc.metadata["page_number"] = "1"
                
        # Simple heuristic to extract section if text is available
        if "section" not in doc.metadata or doc.metadata["section"] == "General":
            doc.metadata["section"] = "General"
            if doc.text:
                lines = doc.text.split("\n")
                for line in lines[:5]:
                    trimmed = line.strip()
                    if trimmed.startswith(("#", "##", "###")) or (0 < len(trimmed) < 40 and trimmed.isupper()):
                        doc.metadata["section"] = trimmed.lstrip("# ").strip()
                        break
    logger.info(f"Metadata enrichment completed in {time.time() - start_time:.4f} seconds.")

# --- Index Manager ---
class IndexManager:
    """Manages the creation, persistence, and loading of the multi-layer indices."""
    
    @staticmethod
    def build_multi_index(documents: List[Document], index_name: str, llm: Any, embed_model: Any) -> Dict[str, Any]:
        """Builds hierarchical, sentence-window, and document summary indexes."""
        start_total = time.time()
        logger.info(f"Starting to build multi-index '{index_name}'...")
        enrich_document_metadata(documents)
        
        base_dir = os.path.join("indexes", index_name)
        os.makedirs(base_dir, exist_ok=True)
        
        # 1. Hierarchical (Auto-Merging) Index
        logger.info("Building Hierarchical Index...")
        start_hier = time.time()
        hier_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
        hier_nodes = hier_parser.get_nodes_from_documents(documents)
        leaf_nodes = get_leaf_nodes(hier_nodes)
        
        docstore = SimpleDocumentStore()
        docstore.add_documents(hier_nodes)
        
        hier_storage_context = StorageContext.from_defaults()
        hier_storage_context.docstore = docstore
        
        hier_index = VectorStoreIndex(
            leaf_nodes,
            storage_context=hier_storage_context,
            embed_model=embed_model
        )
        hier_dir = os.path.join(base_dir, "hierarchical")
        hier_index.storage_context.persist(persist_dir=hier_dir)
        logger.info(f"Hierarchical Index built and persisted in {time.time() - start_hier:.2f} seconds.")
        
        # 2. Sentence Window Index
        logger.info("Building Sentence Window Index...")
        start_sw = time.time()
        sw_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text"
        )
        sw_nodes = sw_parser.get_nodes_from_documents(documents)
        
        sw_index = VectorStoreIndex(
            sw_nodes,
            embed_model=embed_model
        )
        sw_dir = os.path.join(base_dir, "sentence_window")
        sw_index.storage_context.persist(persist_dir=sw_dir)
        logger.info(f"Sentence Window Index built and persisted in {time.time() - start_sw:.2f} seconds.")
        
        # 3. Document Summary Index
        logger.info("Building Document Summary Index...")
        start_summary = time.time()
        response_synthesizer = get_response_synthesizer(
            llm=llm,
            response_mode=ResponseMode.TREE_SUMMARIZE
        )
        
        summary_index = DocumentSummaryIndex.from_documents(
            documents,
            llm=llm,
            embed_model=embed_model,
            response_synthesizer=response_synthesizer,
            show_progress=True
        )
        summary_dir = os.path.join(base_dir, "summary")
        summary_index.storage_context.persist(persist_dir=summary_dir)
        logger.info(f"Document Summary Index built and persisted in {time.time() - start_summary:.2f} seconds.")
        
        logger.info(f"Total time to build and persist multi-index '{index_name}': {time.time() - start_total:.2f} seconds.")
        return {
            "type": "multi",
            "hierarchical": hier_index,
            "sentence_window": sw_index,
            "summary": summary_index
        }

    @staticmethod
    def load_index(index_name: str, embed_model: Any) -> Dict[str, Any]:
        """Loads index files with fallback support for legacy indexes."""
        start_total = time.time()
        base_dir = os.path.join("indexes", index_name)
        hier_dir = os.path.join(base_dir, "hierarchical")
        sw_dir = os.path.join(base_dir, "sentence_window")
        summary_dir = os.path.join(base_dir, "summary")
        
        if os.path.exists(hier_dir) and os.path.exists(sw_dir) and os.path.exists(summary_dir):
            logger.info(f"Loading Multi-Index '{index_name}'...")
            
            start_hier = time.time()
            hier_storage_context = StorageContext.from_defaults(persist_dir=hier_dir)
            hier_index = load_index_from_storage(hier_storage_context, embed_model=embed_model)
            logger.info(f"Hierarchical Index loaded in {time.time() - start_hier:.2f} seconds.")
            
            start_sw = time.time()
            sw_storage_context = StorageContext.from_defaults(persist_dir=sw_dir)
            sw_index = load_index_from_storage(sw_storage_context, embed_model=embed_model)
            logger.info(f"Sentence Window Index loaded in {time.time() - start_sw:.2f} seconds.")
            
            start_summary = time.time()
            summary_storage_context = StorageContext.from_defaults(persist_dir=summary_dir)
            summary_index = load_index_from_storage(summary_storage_context, embed_model=embed_model)
            logger.info(f"Document Summary Index loaded in {time.time() - start_summary:.2f} seconds.")
            
            logger.info(f"Total time to load Multi-Index '{index_name}': {time.time() - start_total:.2f} seconds.")
            return {
                "type": "multi",
                "hierarchical": hier_index,
                "sentence_window": sw_index,
                "summary": summary_index
            }
        else:
            logger.info(f"Loading Legacy Index '{index_name}'...")
            storage_context = StorageContext.from_defaults(persist_dir=base_dir)
            index = load_index_from_storage(storage_context, embed_model=embed_model)
            logger.info(f"Total time to load Legacy Index '{index_name}': {time.time() - start_total:.2f} seconds.")
            return {
                "type": "legacy",
                "index": index
            }

    @staticmethod
    def get_indexed_filenames(index_dict: Dict[str, Any]) -> List[str]:
        """Extracts unique filenames indexed in the system."""
        filenames = set()
        if index_dict["type"] == "multi":
            summary_index = index_dict["summary"]
            for doc_id, doc_info in summary_index.ref_doc_info.items():
                metadata = doc_info.metadata
                if "filename" in metadata:
                    filenames.add(metadata["filename"])
        else:
            legacy_index = index_dict["index"]
            for doc_id, node in legacy_index.docstore.docs.items():
                metadata = node.metadata
                if "filename" in metadata:
                    filenames.add(metadata["filename"])
                elif "file_name" in metadata:
                    filenames.add(metadata["file_name"])
        return list(filenames)

# --- Property Graph Interface Stub ---
class PropertyGraphManagerStub:
    """Interface stub prepared for Phase 10: Property Graph Index (Neo4j)."""
    def __init__(self, neo4j_url: Optional[str] = None, neo4j_username: Optional[str] = None, neo4j_password: Optional[str] = None):
        self.url = neo4j_url
        self.username = neo4j_username
        self.password = neo4j_password
        
    def build_graph_index(self, documents: List[Document], embed_model: Any, llm: Any):
        logger.info("PropertyGraphIndex build requested. Graph Database connection details provided.")
        raise NotImplementedError("Property Graph Index is prepared but not fully implemented in this phase.")

    def get_graph_retriever(self, index: Any):
        raise NotImplementedError("Graph retrieval is prepared but not fully implemented in this phase.")

# --- LlamaIndex Workflows Events ---
class PlanStepEvent(Event):
    strategy: str
    metadata_filters: List[Dict[str, Any]]
    sub_queries: List[str]
    reasoning: str

class RetrievalStepEvent(Event):
    num_nodes: int
    strategy: str

class PostprocessStepEvent(Event):
    msg: str

class TextChunkEvent(Event):
    text: str

class StepTimingEvent(Event):
    step_name: str
    duration: float

# Let's define the Events for payload passing:
class PlannedPlanEvent(Event):
    query: str
    strategy: str
    metadata_filters: List[Dict[str, Any]]
    sub_queries: List[str]
    reasoning: str

class RetrievedNodesEvent(Event):
    query: str
    strategy: str
    nodes: List[NodeWithScore]

class ProcessedContextEvent(Event):
    query: str
    nodes: List[NodeWithScore]
    context_str: str

class RefinedAgenticWorkflow(Workflow):
    """Linear workflow passing payload from step to step."""
    
    def __init__(
        self,
        index_dict: Dict[str, Any],
        llm: Any,
        embed_model: Any,
        reranker: Any,
        settings: Dict[str, Any],
        timeout: float = 120.0
    ):
        super().__init__(timeout=timeout)
        self.index_dict = index_dict
        self.llm = llm
        self.embed_model = embed_model
        self.reranker = reranker
        self.settings = settings

    def _clean_json_text(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def _build_metadata_filters(self, plan_filters: List[Dict[str, Any]]) -> Optional[MetadataFilters]:
        filters = []
        for f in plan_filters:
            key = f.get("key")
            val = f.get("value")
            op = f.get("operator", "==")
            
            operator = FilterOperator.EQ
            if op == "==":
                operator = FilterOperator.EQ
            elif op == "in":
                operator = FilterOperator.IN
            elif op == "contains" or op == "text_match":
                operator = FilterOperator.TEXT_MATCH
            
            filters.append(MetadataFilter(key=key, value=val, operator=operator))
        return MetadataFilters(filters=filters) if filters else None

    def _get_retriever(self, strategy: str, metadata_filters: Optional[MetadataFilters] = None, top_k: int = 10):
        num_queries = self.settings.get("query_fusion_queries", 3)
        
        if self.index_dict["type"] == "legacy":
            index = self.index_dict["index"]
            vector_retriever = index.as_retriever(similarity_top_k=top_k, filters=metadata_filters)
            nodes = list(index.docstore.docs.values())
            bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)
            return QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                llm=self.llm,
                similarity_top_k=top_k,
                num_queries=num_queries,
                mode="reciprocal_rerank",
                use_async=True
            )
            
        if strategy == "summary":
            summary_index = self.index_dict["summary"]
            return summary_index.as_retriever(similarity_top_k=2)
            
        elif strategy == "hierarchical":
            hier_index = self.index_dict["hierarchical"]
            storage_context = hier_index.storage_context
            leaf_nodes = get_leaf_nodes(list(storage_context.docstore.docs.values()))
            
            vector_retriever = hier_index.as_retriever(similarity_top_k=top_k, filters=metadata_filters)
            bm25_retriever = BM25Retriever.from_defaults(nodes=leaf_nodes, similarity_top_k=top_k)
            
            fusion_retriever = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                llm=self.llm,
                similarity_top_k=top_k,
                num_queries=num_queries,
                mode="reciprocal_rerank",
                use_async=True
            )
            
            if self.settings.get("enable_auto_merging", True):
                return AutoMergingRetriever(
                    fusion_retriever,
                    storage_context,
                    verbose=True
                )
            return fusion_retriever
            
        elif strategy == "sentence_window":
            sw_index = self.index_dict["sentence_window"]
            storage_context = sw_index.storage_context
            sw_nodes = list(storage_context.docstore.docs.values())
            
            vector_retriever = sw_index.as_retriever(similarity_top_k=top_k, filters=metadata_filters)
            bm25_retriever = BM25Retriever.from_defaults(nodes=sw_nodes, similarity_top_k=top_k)
            
            return QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                llm=self.llm,
                similarity_top_k=top_k,
                num_queries=num_queries,
                mode="reciprocal_rerank",
                use_async=True
            )
        else:
            return self._get_retriever("hierarchical", metadata_filters, top_k)

    @step
    async def plan_query(self, ctx: Context, ev: StartEvent) -> PlannedPlanEvent:
        """Executes LLM planning step."""
        start_time = time.time()
        query = getattr(ev, "query", None)
        if not query or not isinstance(query, str):
            logger.warning("Query not provided or is invalid. Defaulting to 'general query'.")
            query = "general query"
            
        available_files = IndexManager.get_indexed_filenames(self.index_dict)
        
        system_prompt = "You are an Agentic Query Planner for a document retrieval system. Your goal is to analyze the user's query and decide the best retrieval plan."
        
        user_prompt = f"""Available Files in the database:
{available_files}

User Query: "{query}"

You must output a JSON object with the following fields:
1. "retrieval_strategy": The best strategy for this query. Must be one of:
   - "summary": Use when the user asks for high-level summaries, directory indexes, overviews, or comparisons across documents.
   - "hierarchical": Use when the query is highly detailed, specific, and requires returning complete sections/paragraphs (auto-merging chunks).
   - "sentence_window": Use when the query needs extremely fine-grained/precise text context (sentence-level accuracy).
   - "hybrid": Default. Use for standard semantic and keyword searches.
2. "metadata_filters": A list of metadata filters to apply, or an empty list.
   Each filter should be a dictionary: {{"key": "filename" | "document_type" | "section" | "upload_timestamp", "value": string, "operator": "==" | "in" | "contains"}}
   Example: If user asks about "v1 policy", and "policy_v1.pdf" is in the available files, filter by: {{"key": "filename", "value": "policy_v1.pdf", "operator": "=="}}
3. "sub_queries": A list of decomposed sub-queries if this is a complex or multi-step/comparison query. If not complex, include only the original query.
   Example for "What changed between v1 and v2?": ["What is in v1?", "What is in v2?"]
4. "reasoning": A brief explanation of your planning decision.

Return ONLY the JSON block. Do not include markdown formatting or backticks outside the JSON itself.
JSON:
"""
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ]
        
        response = await self.llm.achat(messages)
        cleaned_response = self._clean_json_text(response.message.content)
        
        try:
            plan_data = json.loads(cleaned_response)
        except Exception as e:
            logger.error(f"Failed to parse planning response: {e}. Output was: {response.text}")
            plan_data = {
                "retrieval_strategy": "hybrid",
                "metadata_filters": [],
                "sub_queries": [query],
                "reasoning": "Plan generation failed; falling back to hybrid retrieval strategy."
            }
            
        strategy = plan_data.get("retrieval_strategy", "hybrid")
        metadata_filters = plan_data.get("metadata_filters", [])
        sub_queries = plan_data.get("sub_queries", [query])
        reasoning = plan_data.get("reasoning", "Executing default plan.")
        
        ctx.write_event_to_stream(PlanStepEvent(
            strategy=strategy,
            metadata_filters=metadata_filters,
            sub_queries=sub_queries,
            reasoning=reasoning
        ))
        
        duration = time.time() - start_time
        ctx.write_event_to_stream(StepTimingEvent(step_name="Query Planning", duration=duration))
        logger.info(f"Workflow Step [Query Planning] completed in {duration:.2f} seconds.")
        
        return PlannedPlanEvent(
            query=query,
            strategy=strategy,
            metadata_filters=metadata_filters,
            sub_queries=sub_queries,
            reasoning=reasoning
        )

    @step
    async def retrieve_context(self, ctx: Context, ev: PlannedPlanEvent) -> RetrievedNodesEvent:
        """Runs the query fusion / summary / hierarchical retrievers in parallel."""
        start_time = time.time()
        all_retrieved_nodes = []
        metadata_filters = self._build_metadata_filters(ev.metadata_filters)
        
        async def retrieve_for_query(sub_query: str):
            retriever = self._get_retriever(ev.strategy, metadata_filters)
            if ev.strategy == "summary":
                summary_nodes = await retriever.aretrieve(sub_query)
                relevant_files = []
                for n in summary_nodes:
                    filename = n.metadata.get("filename")
                    if filename:
                        relevant_files.append(filename)
                
                ctx.write_event_to_stream(PostprocessStepEvent(msg=f"Found relevant documents via summaries: {relevant_files}. Retrieving detailed chunks."))
                
                sub_filters = None
                if relevant_files:
                    from llama_index.core.vector_stores import FilterOperator
                    if len(relevant_files) == 1:
                        sub_filters = MetadataFilters(filters=[MetadataFilter(key="filename", value=relevant_files[0], operator=FilterOperator.EQ)])
                    else:
                        sub_filters = MetadataFilters(filters=[MetadataFilter(key="filename", value=relevant_files, operator=FilterOperator.IN)])
                
                detail_retriever = self._get_retriever("hierarchical", sub_filters)
                sub_nodes = await detail_retriever.aretrieve(sub_query)
                return summary_nodes + sub_nodes
            else:
                return await retriever.aretrieve(sub_query)

        tasks = [retrieve_for_query(sq) for sq in ev.sub_queries]
        results = await asyncio.gather(*tasks)
        for r in results:
            all_retrieved_nodes.extend(r)
                
        # Deduplicate
        unique_nodes = {}
        for n in all_retrieved_nodes:
            unique_nodes[n.node.node_id] = n
        deduped_nodes = list(unique_nodes.values())
        
        ctx.write_event_to_stream(RetrievalStepEvent(
            num_nodes=len(deduped_nodes),
            strategy=ev.strategy
        ))
        
        duration = time.time() - start_time
        ctx.write_event_to_stream(StepTimingEvent(step_name="Context Retrieval", duration=duration))
        logger.info(f"Workflow Step [Context Retrieval] completed in {duration:.2f} seconds. Retrieved {len(deduped_nodes)} nodes.")
        
        return RetrievedNodesEvent(
            query=ev.query,
            strategy=ev.strategy,
            nodes=deduped_nodes
        )

    @step
    async def postprocess_context(self, ctx: Context, ev: RetrievedNodesEvent) -> ProcessedContextEvent:
        """Applies sentence window replacement, reranking, long context reordering, and LLM extraction."""
        start_time = time.time()
        nodes = ev.nodes
        
        # 1. If sentence window, replace with full window text before reranking
        if ev.strategy == "sentence_window" and self.index_dict["type"] == "multi":
            ctx.write_event_to_stream(PostprocessStepEvent(msg="Expanding sentence nodes to window text."))
            window_postprocessor = MetadataReplacementPostProcessor(target_metadata_key="window")
            nodes = window_postprocessor.postprocess_nodes(nodes)
            
        # 2. Reranking
        if self.reranker and nodes:
            ctx.write_event_to_stream(PostprocessStepEvent(msg="Re-ranking retrieved chunks with cross-encoder..."))
            nodes = self.reranker.postprocess_nodes(nodes, query_str=ev.query)
            
        # 3. Contextual Compression (LLM-based context extraction)
        enable_compression = self.settings.get("enable_context_compression", True)
        if enable_compression and nodes:
            ctx.write_event_to_stream(PostprocessStepEvent(msg="Performing contextual compression on top documents..."))
            start_compression = time.time()
            try:
                # Compress only top 4 nodes to keep latency low
                nodes_to_compress = nodes[:4]
                other_nodes = nodes[4:]
                
                system_prompt = "You are an information extraction assistant. Given the document text and the user query, extract ONLY the sentences from the document text that are directly relevant to answering the query."
                
                async def compress_single_node(node_with_score):
                    node = node_with_score.node
                    user_prompt = f"""Do not rewrite or summarize. Extract the sentences word-for-word. If no sentences are relevant, reply with "No relevant information".

Query: "{ev.query}"
Document Text:
---
{node.text}
---

Relevant sentences:"""
                    messages = [
                        ChatMessage(role="system", content=system_prompt),
                        ChatMessage(role="user", content=user_prompt)
                    ]
                    response = await self.llm.achat(messages)
                    cleaned_txt = response.message.content.strip()
                    if cleaned_txt and cleaned_txt != "No relevant information":
                        node.text = cleaned_txt
                    return node_with_score

                tasks = [compress_single_node(n) for n in nodes_to_compress]
                compressed_nodes = await asyncio.gather(*tasks)
                nodes = list(compressed_nodes) + other_nodes
                logger.info(f"Contextual compression of {len(nodes_to_compress)} nodes completed in {time.time() - start_compression:.2f} seconds.")
            except Exception as ex:
                logger.error(f"Compression failed: {ex}. Continuing without compression.")
                
        # 4. Long Context Reorder (avoid lost-in-the-middle)
        if len(nodes) > 2:
            ctx.write_event_to_stream(PostprocessStepEvent(msg="Re-ordering context to place best documents at edges."))
            reorder = LongContextReorder()
            nodes = reorder.postprocess_nodes(nodes)
            
        # Format context string
        context_str = ""
        for idx, node_with_score in enumerate(nodes):
            node = node_with_score.node
            meta = node.metadata
            context_str += f"Source [{idx+1}]:\n"
            context_str += f"Filename: {meta.get('filename', 'unknown')}\n"
            context_str += f"Page: {meta.get('page_number', '1')}\n"
            context_str += f"Section: {meta.get('section', 'General')}\n"
            context_str += f"Text:\n{node.text}\n"
            context_str += f"---------------------\n\n"
            
        duration = time.time() - start_time
        ctx.write_event_to_stream(StepTimingEvent(step_name="Post-processing & Compression", duration=duration))
        logger.info(f"Workflow Step [Post-processing & Compression] completed in {duration:.2f} seconds.")
            
        return ProcessedContextEvent(
            query=ev.query,
            nodes=nodes,
            context_str=context_str
        )

    @step
    async def synthesize(self, ctx: Context, ev: ProcessedContextEvent) -> StopEvent:
        """Synthesizes the final source-cited response using streaming."""
        start_time = time.time()
        
        system_prompt = "You are an expert AI Assistant answering questions based on document context. Your goal is to answer the query accurately and cite your sources."
        
        user_prompt = f"""Context Information:
---------------------
{ev.context_str}
---------------------

User Query: {ev.query}

Instructions:
1. Answer the query clearly using only the provided context. If the context does not contain the answer, state that.
2. Every claim/fact must be cited using inline citations referencing the source number, e.g., [1], [2].
3. At the end of your response, add a section called "Sources Cited" with the following format for each unique source:
   [N] Filename: <filename>, Page: <page_number>, Section: <section> (Confidence: <score>%)
   (Calculate a confidence score based on similarity or relevance of that source to the query, typically between 80% and 99%).

Answer:
"""
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ]
        
        response_stream = await self.llm.astream_chat(messages)
        full_response = ""
        async for chunk in response_stream:
            full_response += chunk.delta
            ctx.write_event_to_stream(TextChunkEvent(text=chunk.delta))
            
        # Extract structured citations info
        citations = []
        for idx, node_with_score in enumerate(ev.nodes):
            meta = node_with_score.node.metadata
            score = round((node_with_score.score or 0.85) * 100, 1)
            citations.append({
                "index": idx + 1,
                "filename": meta.get("filename", "unknown"),
                "page_number": meta.get("page_number", "1"),
                "section": meta.get("section", "General"),
                "confidence": score
            })
            
        duration = time.time() - start_time
        ctx.write_event_to_stream(StepTimingEvent(step_name="Synthesis & Streaming", duration=duration))
        logger.info(f"Workflow Step [Synthesis & Streaming] completed in {duration:.2f} seconds.")
            
        return StopEvent(result={
            "response": full_response,
            "citations": citations,
            "nodes": ev.nodes
        })
