"""
Backend services - handles model loading and business logic.
Models are loaded ONCE at startup and reused for all requests.
"""
import asyncio
import os
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from urllib.parse import quote

import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    FieldCondition, Filter, MatchAny, MatchValue, Range,
    Prefetch, FusionQuery, Fusion, SparseVector,
    PointStruct, Distance, VectorParams,
)

# Sentinel value for "no expiration" - year 2100
NO_EXPIRATION_TIMESTAMP = 4102444800
from sentence_transformers import SentenceTransformer
from sqlalchemy.exc import IntegrityError

from utils.db_connection import Connection

logger = logging.getLogger("backend")


class DebugCollector:
    """Collects debug information for pipeline steps."""
    
    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self.start_time: float = time.time()
        self._step_start: float = 0
    
    def start_step(self):
        """Mark the start of a new step."""
        self._step_start = time.time()
    
    def add_step(self, step_name: str, description: str, data: Any = None, truncate_at: int = 2000):
        """Add a pipeline step with timing."""
        duration_ms = (time.time() - self._step_start) * 1000 if self._step_start else 0
        
        # Convert data to string and truncate if needed
        data_str = None
        if data is not None:
            if isinstance(data, str):
                data_str = data
            else:
                try:
                    data_str = json.dumps(data, indent=2, ensure_ascii=False, default=str)
                except Exception:
                    data_str = str(data)
            
            if len(data_str) > truncate_at:
                data_str = data_str[:truncate_at] + f"\n\n... [truncated, total {len(data_str)} chars]"
        
        self.steps.append({
            "step_number": len(self.steps) + 1,
            "step_name": step_name,
            "description": description,
            "data": data_str,
            "duration_ms": round(duration_ms, 2),
        })
    
    def get_total_duration_ms(self) -> float:
        """Get total duration since collector was created."""
        return round((time.time() - self.start_time) * 1000, 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for response."""
        return {
            "total_duration_ms": self.get_total_duration_ms(),
            "pipeline_steps": self.steps,
        }


class SearchService:
    """
    Handles vector and hybrid search operations.
    Loads models once at initialization.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.client: Optional[QdrantClient] = None
        self.dense_model: Optional[SentenceTransformer] = None
        self.sparse_model = None
        self.reranker = None
        self.hybrid_enabled = False
        self.reranking_enabled = False
        self.collection_name = os.getenv("QDRANT_COLLECTION", "documents")
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize models and connections. Call once at startup."""
        if self._initialized:
            return
            
        logger.info("Initializing SearchService...")
        
        # Load dense embedding model
        model_name = self.config["embeddings"]["embedding_model"]
        logger.info(f"Loading dense model: {model_name}")
        self.dense_model = SentenceTransformer(model_name)
        logger.info("Dense model loaded successfully")
        
        # Load sparse model if hybrid search enabled
        hybrid_config = self.config.get("hybrid_search", {})
        if hybrid_config.get("enabled", False):
            try:
                from fastembed import SparseTextEmbedding
                sparse_model_name = hybrid_config.get("sparse_model", "Qdrant/bm25")
                logger.info(f"Loading sparse model: {sparse_model_name}")
                self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name)
                self.hybrid_enabled = True
                logger.info("Sparse model loaded successfully - hybrid search enabled")
            except Exception as e:
                logger.warning(f"Failed to load sparse model: {e}")
                self.hybrid_enabled = False
        
        # Load reranker model if enabled
        rerank_config = self.config.get("reranking", {})
        if rerank_config.get("enabled", False):
            try:
                from sentence_transformers import CrossEncoder
                reranker_model_name = rerank_config.get("model", "BAAI/bge-reranker-v2-m3")
                logger.info(f"Loading reranker model: {reranker_model_name}")
                self.reranker = CrossEncoder(reranker_model_name, max_length=512)
                self.reranking_enabled = True
                logger.info("Reranker model loaded successfully - reranking enabled")
            except Exception as e:
                logger.warning(f"Failed to load reranker model: {e}")
                self.reranking_enabled = False
        
        # Connect to Qdrant
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        logger.info(f"Connecting to Qdrant at {host}:{port}")
        self.client = QdrantClient(host=host, port=port)
        logger.info("Connected to Qdrant successfully")
        
        self._initialized = True
        logger.info("SearchService initialization complete")
    
    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        try:
            if not self._initialized or not self.client:
                return False
            # Try a simple operation
            self.client.get_collections()
            return True
        except Exception:
            return False
    
    def _build_filter(
        self,
        files: List[str] = None,
        only_valid: bool = False,
    ) -> Optional[Filter]:
        """Build Qdrant filter from parameters."""
        conditions = []
        
        # Always filter for validated chunks
        conditions.append(FieldCondition(
            key="validated",
            match=MatchValue(value=True),
        ))
        
        if files:
            conditions.append(FieldCondition(
                key="title",
                match=MatchAny(any=files),
            ))
        
        if only_valid:
            # Document is valid if:
            # 1. is_valid = true AND
            # 2. valid_until >= current_timestamp (we use NO_EXPIRATION_TIMESTAMP for no expiration)
            conditions.append(FieldCondition(
                key="is_valid",
                match=MatchValue(value=True),
            ))
            
            # Check valid_until: must be in the future or NO_EXPIRATION_TIMESTAMP
            current_ts = int(datetime.now().timestamp())
            conditions.append(FieldCondition(
                key="valid_until",
                range=Range(gte=current_ts),
            ))
        
        return Filter(must=conditions) if conditions else None
    
    def _encode_sparse(self, text: str) -> SparseVector:
        """Encode text to sparse vector."""
        embeddings = list(self.sparse_model.embed([text]))[0]
        return SparseVector(
            indices=embeddings.indices.tolist(),
            values=embeddings.values.tolist()
        )
    
    def search(
        self,
        query: str,
        limit: int = 5,
        files: List[str] = None,
        only_valid: bool = False,
        ensure_diversity: bool = True,
        use_reranking: bool = True,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Perform search (hybrid if enabled, otherwise dense-only).
        Optionally reranks results using a cross-encoder for better relevance.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            files: Filter by document titles
            only_valid: Only search valid documents
            ensure_diversity: If True, ensures results from different documents
            use_reranking: If True and reranking is enabled, rerank results
            debug: If True, returns tuple of (results, debug_info) instead of just results
            
        Returns:
            List of search results, or tuple of (results, debug_info) if debug=True
        """
        if not self._initialized:
            raise RuntimeError("SearchService not initialized")
        
        debug_info = {} if debug else None
        
        query_filter = self._build_filter(files=files, only_valid=only_valid)
        query_prompt = self.config["embeddings"]["query_prompt"]
        
        # Fetch more results for reranking/diversity (reranking benefits from more candidates)
        if self.reranking_enabled and use_reranking:
            fetch_limit = limit * 4  # More candidates for reranking
        elif ensure_diversity:
            fetch_limit = limit * 3
        else:
            fetch_limit = limit
        
        if self.hybrid_enabled and self.sparse_model is not None:
            if debug:
                results, search_debug = self._hybrid_search_debug(query, query_prompt, fetch_limit, query_filter)
                debug_info["search_steps"] = search_debug
            else:
                results = self._hybrid_search(query, query_prompt, fetch_limit, query_filter)
        else:
            results = self._dense_search(query, query_prompt, fetch_limit, query_filter)
            if debug:
                debug_info["search_steps"] = {"type": "dense_only", "results_count": len(results)}
        
        # Apply reranking if enabled (before diversity to get best results first)
        if self.reranking_enabled and use_reranking and len(results) > 0:
            pre_rerank_results = results.copy() if debug else None
            # Rerank all results, then apply diversity/limit
            results = self._rerank(query, results, top_k=None)
            if debug:
                debug_info["reranking"] = {
                    "enabled": True,
                    "model": self.config.get("reranking", {}).get("model", "unknown"),
                    "input_count": len(pre_rerank_results),
                    "results_before": [
                        {"rank": i+1, "title": r["title"], "page": r["page_number"], "score": round(r["score"], 4), "text": r["text"]}
                        for i, r in enumerate(pre_rerank_results)
                    ],
                    "results_after": [
                        {"rank": i+1, "title": r["title"], "page": r["page_number"], "rerank_score": round(r.get("rerank_score", r["score"]), 4), "original_score": round(r.get("original_score", 0), 4), "text": r["text"]}
                        for i, r in enumerate(results)
                    ],
                }
        elif debug:
            debug_info["reranking"] = {"enabled": False}
        
        # Apply document diversity if requested
        if ensure_diversity and len(results) > limit:
            pre_diversity_count = len(results) if debug else None
            results = self._ensure_document_diversity(results, limit)
            if debug:
                debug_info["diversity"] = {
                    "applied": True,
                    "input_count": pre_diversity_count,
                    "output_count": len(results),
                }
        elif debug:
            debug_info["diversity"] = {"applied": False}
        
        final_results = results[:limit]
        
        if debug:
            return final_results, debug_info
        return final_results
    
    def _ensure_document_diversity(
        self,
        results: List[Dict[str, Any]],
        limit: int,
        max_per_document: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Select results ensuring diversity across documents.
        
        Prioritizes getting results from different documents, 
        while still respecting relevance scores.
        """
        diverse_results = []
        doc_counts = {}  # Track how many results we have per document
        
        # First pass: select top result from each document
        for result in results:
            doc_title = result.get("title", "unknown")
            if doc_title not in doc_counts:
                doc_counts[doc_title] = 0
            
            if doc_counts[doc_title] < max_per_document:
                diverse_results.append(result)
                doc_counts[doc_title] += 1
                
            if len(diverse_results) >= limit:
                break
        
        # If we don't have enough, add more from any document
        if len(diverse_results) < limit:
            for result in results:
                if result not in diverse_results:
                    diverse_results.append(result)
                    if len(diverse_results) >= limit:
                        break
        
        return diverse_results
    
    def _fetch_adjacent_chunks(
        self,
        title: str,
        page_number: int,
        exclude_text: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch chunks from adjacent pages (page-1, page, page+1) of the same document.
        
        Args:
            title: Document title to filter by
            page_number: Center page number
            exclude_text: Text to exclude (the original chunk) - first 100 chars used for matching
            
        Returns:
            List of adjacent chunks sorted by page number
        """
        # Build filter for same document and adjacent pages
        adjacent_pages = [max(1, page_number - 1), page_number, page_number + 1]
        
        conditions = [
            FieldCondition(key="validated", match=MatchValue(value=True)),
            FieldCondition(key="title", match=MatchValue(value=title)),
        ]
        
        # Scroll to get all chunks matching criteria
        try:
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=conditions),
                limit=50,  # Reasonable limit for a single document's adjacent pages
                with_payload=True,
            )
            
            # Filter to adjacent pages and format
            adjacent_chunks = []
            exclude_prefix = exclude_text[:100] if exclude_text else None
            
            for point in results:
                if not point.payload:
                    continue
                chunk_page = point.payload.get("page_number", 0)
                chunk_text = point.payload.get("text", "")
                
                # Only include chunks from adjacent pages
                if chunk_page not in adjacent_pages:
                    continue
                
                # Skip the original chunk (by matching text prefix)
                if exclude_prefix and chunk_text.startswith(exclude_prefix):
                    continue
                
                adjacent_chunks.append({
                    "text": chunk_text,
                    "title": point.payload.get("title", ""),
                    "page_number": chunk_page,
                    "score": 0,  # No score for adjacent chunks
                    "content_type": point.payload.get("content_type", ""),
                    "url": point.payload.get("url", ""),
                    "_is_adjacent": True,
                })
            
            # Sort by page number
            adjacent_chunks.sort(key=lambda x: x["page_number"])
            return adjacent_chunks
            
        except Exception as e:
            logger.warning(f"Failed to fetch adjacent chunks: {e}")
            return []
    
    def expand_context(
        self,
        results: List[Dict[str, Any]],
        max_adjacent_per_result: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Expand search results with adjacent chunks for fuller context.
        
        For each search result, fetches chunks from adjacent pages in the same document
        and combines them to provide more complete context.
        
        Args:
            results: Original search results
            max_adjacent_per_result: Maximum adjacent chunks to add per original result
            
        Returns:
            Expanded results with adjacent context merged
        """
        if not results:
            return results
        
        expanded = []
        seen_texts = set()  # Avoid duplicates
        
        for result in results:
            title = result.get("title", "")
            page = result.get("page_number", 0)
            original_text = result.get("text", "")
            
            # Add the original result
            if original_text[:100] not in seen_texts:
                result["_is_adjacent"] = False
                expanded.append(result)
                seen_texts.add(original_text[:100])
            
            # Fetch adjacent chunks
            if title and page > 0:
                adjacent = self._fetch_adjacent_chunks(
                    title=title,
                    page_number=page,
                    exclude_text=original_text,
                )
                
                # Add up to max_adjacent_per_result adjacent chunks
                added = 0
                for adj in adjacent:
                    adj_text = adj.get("text", "")
                    if adj_text[:100] not in seen_texts and added < max_adjacent_per_result:
                        expanded.append(adj)
                        seen_texts.add(adj_text[:100])
                        added += 1
        
        # Sort by title and page to keep context coherent
        expanded.sort(key=lambda x: (x.get("title", ""), x.get("page_number", 0)))
        
        logger.info(f"Context expansion: {len(results)} original → {len(expanded)} with adjacent")
        return expanded
    
    def _dense_search(
        self,
        query: str,
        query_prompt: str,
        limit: int,
        query_filter: Optional[Filter],
    ) -> List[Dict[str, Any]]:
        """Perform dense vector search."""
        logger.info("Performing dense search")
        
        dense_vector = self.dense_model.encode(
            query_prompt + query,
            normalize_embeddings=True
        ).tolist()
        
        # Use "dense" vector name for hybrid collections, None for single-vector
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vector,
            using="dense",  # Named vector for hybrid collection
            query_filter=query_filter,
            limit=limit,
            timeout=100,
        )
        
        return self._format_results(results.points)
    
    def _hybrid_search(
        self,
        query: str,
        query_prompt: str,
        limit: int,
        query_filter: Optional[Filter],
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search with RRF fusion."""
        logger.info("Performing hybrid search")
        
        # Encode dense vector
        dense_vector = self.dense_model.encode(
            query_prompt + query,
            normalize_embeddings=True
        ).tolist()
        
        # Encode sparse vector
        sparse_vector = self._encode_sparse(query)
        
        # Hybrid search with prefetch and RRF fusion
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using="dense",
                    filter=query_filter,
                    limit=limit * 2,
                ),
                Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    filter=query_filter,
                    limit=limit * 2,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            timeout=100,
        )
        
        return self._format_results(results.points)
    
    def _hybrid_search_debug(
        self,
        query: str,
        query_prompt: str,
        limit: int,
        query_filter: Optional[Filter],
    ) -> tuple:
        """
        Perform hybrid search with detailed debug information.
        
        Returns:
            Tuple of (results, debug_info) where debug_info contains:
            - dense_results: Top results from dense search alone
            - sparse_results: Top results from sparse search alone  
            - fused_results: Results after RRF fusion
            - sparse_tokens: The sparse vector tokens/weights
        """
        import time
        logger.info("Performing hybrid search with debug info")
        debug_info = {"type": "hybrid"}
        
        # Step 1: Encode dense vector
        t0 = time.time()
        dense_vector = self.dense_model.encode(
            query_prompt + query,
            normalize_embeddings=True
        ).tolist()
        debug_info["dense_encoding_ms"] = round((time.time() - t0) * 1000, 1)
        
        # Step 2: Encode sparse vector
        t0 = time.time()
        sparse_vector = self._encode_sparse(query)
        debug_info["sparse_encoding_ms"] = round((time.time() - t0) * 1000, 1)
        
        # Show top sparse tokens (most weighted terms)
        if sparse_vector.indices and sparse_vector.values:
            token_weights = list(zip(sparse_vector.indices, sparse_vector.values))
            token_weights.sort(key=lambda x: x[1], reverse=True)
            debug_info["sparse_tokens_count"] = len(sparse_vector.indices)
            debug_info["top_sparse_weights"] = [
                {"token_id": idx, "weight": round(w, 4)} 
                for idx, w in token_weights[:10]
            ]
        
        # Step 3: Run dense search separately (fetch same amount as main search)
        t0 = time.time()
        dense_results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vector,
            using="dense",
            query_filter=query_filter,
            limit=limit * 2,  # Fetch more to see full picture
            timeout=100,
        )
        debug_info["dense_search_ms"] = round((time.time() - t0) * 1000, 1)
        dense_formatted = self._format_results(dense_results.points)
        debug_info["dense_results"] = [
            {"rank": i+1, "title": r["title"], "page": r["page_number"], "score": round(r["score"], 4), "text": r["text"]}
            for i, r in enumerate(dense_formatted)
        ]
        debug_info["dense_total_count"] = len(dense_formatted)
        
        # Step 4: Run sparse search separately
        t0 = time.time()
        sparse_results = self.client.query_points(
            collection_name=self.collection_name,
            query=sparse_vector,
            using="sparse",
            query_filter=query_filter,
            limit=limit * 2,  # Fetch more to see full picture
            timeout=100,
        )
        debug_info["sparse_search_ms"] = round((time.time() - t0) * 1000, 1)
        sparse_formatted = self._format_results(sparse_results.points)
        debug_info["sparse_results"] = [
            {"rank": i+1, "title": r["title"], "page": r["page_number"], "score": round(r["score"], 4), "text": r["text"]}
            for i, r in enumerate(sparse_formatted)
        ]
        debug_info["sparse_total_count"] = len(sparse_formatted)
        
        # Step 5: Run hybrid fusion
        t0 = time.time()
        fused_results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using="dense",
                    filter=query_filter,
                    limit=limit * 2,
                ),
                Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    filter=query_filter,
                    limit=limit * 2,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit * 2,  # Fetch more to see full picture
            timeout=100,
        )
        debug_info["fusion_ms"] = round((time.time() - t0) * 1000, 1)
        fused_formatted = self._format_results(fused_results.points)
        debug_info["fused_results"] = [
            {"rank": i+1, "title": r["title"], "page": r["page_number"], "score": round(r["score"], 4), "text": r["text"]}
            for i, r in enumerate(fused_formatted)
        ]
        debug_info["fused_total_count"] = len(fused_formatted)
        
        # Add summary
        debug_info["summary"] = {
            "total_search_ms": debug_info["dense_search_ms"] + debug_info["sparse_search_ms"] + debug_info["fusion_ms"],
            "dense_unique": len(set(r["title"] + str(r["page_number"]) for r in dense_formatted[:10])),
            "sparse_unique": len(set(r["title"] + str(r["page_number"]) for r in sparse_formatted[:10])),
        }
        
        return fused_formatted, debug_info
    
    def _format_results(self, results) -> List[Dict[str, Any]]:
        """Format search results into a standard format."""
        formatted = []
        for point in results:
            if not point.payload:
                continue
            formatted.append({
                "text": point.payload.get("text", ""),
                "title": point.payload.get("title", ""),
                "page_number": point.payload.get("page_number", 0),
                "score": point.score,
                "content_type": point.payload.get("content_type", ""),
                "url": point.payload.get("url", ""),
            })
        return formatted
    
    def _rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results using a cross-encoder model.
        
        Cross-encoders are more accurate than bi-encoders because they
        can attend to both query and document simultaneously.
        
        Args:
            query: The search query
            results: List of search results to rerank
            top_k: Number of top results to return (None = return all, reranked)
            
        Returns:
            Reranked results with updated scores
        """
        if not results or not self.reranking_enabled or not self.reranker:
            return results
        
        # Get top_k from config if not specified
        if top_k is None:
            top_k = self.config.get("reranking", {}).get("top_k", len(results))
        
        logger.info(f"Reranking {len(results)} results for query: {query[:50]}...")
        
        # Prepare query-document pairs for cross-encoder
        pairs = [(query, result["text"]) for result in results]
        
        # Get reranking scores
        try:
            rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)
            
            # Add rerank scores to results
            for i, result in enumerate(results):
                result["original_score"] = result["score"]
                result["rerank_score"] = float(rerank_scores[i])
            
            # Sort by rerank score (descending)
            reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
            
            # Update score field to be the rerank score for downstream use
            for result in reranked:
                result["score"] = result["rerank_score"]
            
            logger.info(f"Reranking complete. Top result score: {reranked[0]['rerank_score']:.4f}")
            
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results[:top_k] if top_k else results


class ChatService:
    """
    Handles LLM chat operations.
    Returns structured JSON from LLM and formats it for display.
    """
    
    def __init__(self, config: dict, search_service: SearchService, prompt_service: 'PromptService' = None):
        self.config = config
        self.search_service = search_service
        self.prompt_service = prompt_service
        self.llm = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize LLM. Call once at startup."""
        if self._initialized:
            return
            
        import re
        from langchain_anthropic import ChatAnthropic
        from langchain_openai import ChatOpenAI
        
        logger.info("Initializing ChatService...")
        
        model_name = self.config["llm"]["model"]
        temperature = self.config["llm"].get("temperature", 0)
        api_key = os.getenv("LLM_API_KEY")
        
        if re.match("claude", model_name):
            self.llm = ChatAnthropic(
                model_name=model_name,
                temperature=temperature,
                api_key=api_key,
                max_retries=2,
            )
        elif re.match("gpt", model_name):
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=api_key,
                max_retries=2,
            )
        
        logger.info(f"LLM initialized: {model_name}")
        self._initialized = True
    
    def _build_context(self, results: List[Dict[str, Any]]) -> str:
        """Build context string from search results."""
        context = ""
        for result in results:
            context += f'Title: {result["title"]}\n'
            context += f'Page: {result["page_number"]}\n'
            if result.get("url"):
                context += f'URL: {result["url"]}\n'
            context += f'\n{result["text"]}\n'
            context += "*" * 15 + "\n"
        return context
    
    def _parse_llm_json(self, content: str, query: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response, handling potential formatting issues.
        
        Args:
            content: Raw LLM response content
            query: Original query (used as fallback term)
            
        Returns:
            Parsed term entry dictionary
        """
        import re
        
        logger.debug(f"Raw LLM response: {content[:500]}...")
        
        json_str = None
        
        # Try to extract JSON from markdown code blocks (greedy match)
        json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*\})\s*```', content)
        if json_match:
            json_str = json_match.group(1)
            logger.debug("Extracted JSON from code block")
        else:
            # Try to find raw JSON object in the content
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                json_str = json_match.group(0)
                logger.debug("Extracted raw JSON object")
            else:
                logger.warning("No JSON object found in LLM response")
                json_str = content.strip()
        
        try:
            parsed = json.loads(json_str)
            logger.info("Successfully parsed LLM JSON response")
            
            # Ensure required fields exist with defaults
            return {
                "term": parsed.get("term", query),
                "definitions": parsed.get("definitions", []),
                "related_terms": parsed.get("related_terms", []),
                "usage_evidence": parsed.get("usage_evidence", []),
                "see_also": parsed.get("see_also", []),
            }
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
            logger.warning(f"Attempted to parse: {json_str[:200] if json_str else 'None'}...")
            # Return empty structure on parse failure
            return {
                "term": query,
                "definitions": [],
                "related_terms": [],
                "usage_evidence": [],
                "see_also": [],
            }
    
    def _create_text_fragment_url(self, url: str, text: str) -> str:
        """
        Create a URL with text fragment for highlighting.
        
        Args:
            url: Base document URL
            text: Text to highlight (will be truncated for fragment)
            
        Returns:
            URL with text fragment appended
        """
        if not url or not text:
            return url
        
        # Take first ~50 chars for the text fragment to keep URL reasonable
        # Use a meaningful snippet, trying to end at word boundary
        snippet = text[:80] if len(text) <= 80 else text[:80].rsplit(' ', 1)[0]
        
        # URL-encode the text for the fragment
        encoded_text = quote(snippet, safe='')
        
        # Append text fragment
        return f"{url}#:~:text={encoded_text}"
    
    def _format_source_link(self, item: Dict[str, Any], text_for_highlight: str = "") -> str:
        """
        Format source information, with clickable link if URL is available.
        
        Args:
            item: Dict containing source, page, url, and optionally text
            text_for_highlight: Text to highlight in the linked document
            
        Returns:
            Formatted source string (with HTML link that opens in new tab if URL available)
        """
        source = item.get('source', 'Unknown')
        page = item.get('page', '?')
        url = item.get('url', '')
        
        if url:
            # Create link with text fragment that opens in new tab
            highlight_text = text_for_highlight or item.get('text', '')
            link_url = self._create_text_fragment_url(url, highlight_text)
            return f'(<a href="{link_url}" target="_blank">{source}, lk {page}</a>)'
        else:
            return f"({source}, lk {page})"
    
    def _is_low_confidence(self, confidence: str) -> bool:
        """Check if confidence level is low (inferred or expanded)."""
        return confidence in ("inferred", "expanded")
    
    def _get_confidence_label(self, confidence: str, language: str = "et") -> str:
        """Get human-readable label for confidence level (only for low confidence)."""
        if language == "et":
            labels = {
                "inferred": "tuletatud kontekstist",
                "expanded": "laiendatud otsing",
            }
        else:
            labels = {
                "inferred": "inferred from context",
                "expanded": "from expanded search",
            }
        return labels.get(confidence, "")
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using word overlap.
        Returns a value between 0 and 1.
        """
        t1 = text1.strip().lower()
        t2 = text2.strip().lower()
        
        if not t1 or not t2:
            return 0.0
        
        # Use word-based comparison for meaningful similarity
        words1 = set(t1.split())
        words2 = set(t2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _is_contained_in(self, shorter_text: str, longer_text: str) -> bool:
        """
        Check if the shorter text is meaningfully contained in the longer text.
        The shorter text must be a proper substring, not just sharing some words.
        """
        t1 = shorter_text.strip().lower()
        t2 = longer_text.strip().lower()
        
        # Direct substring check
        return t1 in t2
    
    def _deduplicate_definitions(self, definitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate definitions:
        - Remove shorter definitions that are fully contained in longer ones
        - Keep definitions from different sources even if similar (different sources = more authority)
        - Merge very similar definitions (>85% word overlap) keeping the longer one
        """
        if not definitions:
            return definitions
        
        # Sort ALL definitions by text length (longest first)
        defs_sorted = sorted(definitions, key=lambda x: len(x.get("text", "")), reverse=True)
        
        kept = []
        for defn in defs_sorted:
            text = defn.get("text", "").strip()
            if not text:
                continue
                
            is_duplicate = False
            
            for kept_defn in kept:
                kept_text = kept_defn.get("text", "").strip()
                
                # Case 1: This text is fully contained in a longer definition
                if self._is_contained_in(text, kept_text):
                    is_duplicate = True
                    logger.debug(f"Removing definition (substring): '{text[:50]}...' contained in '{kept_text[:50]}...'")
                    break
                
                # Case 2: Very high word overlap (>85%) - likely same definition with minor differences
                similarity = self._text_similarity(text, kept_text)
                if similarity > 0.85:
                    is_duplicate = True
                    logger.debug(f"Removing definition (similar {similarity:.0%}): '{text[:50]}...'")
                    break
            
            if not is_duplicate:
                kept.append(defn)
        
        logger.info(f"Definitions: {len(definitions)} → {len(kept)} after deduplication")
        return kept
    
    def _deduplicate_related_terms(self, related_terms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate related terms by term name (case-insensitive).
        Keep the one with higher confidence or more complete information.
        """
        if not related_terms:
            return related_terms
        
        # Group by normalized term name
        by_term = defaultdict(list)
        for rt in related_terms:
            term_key = rt.get("term", "").strip().lower()
            by_term[term_key].append(rt)
        
        deduplicated = []
        confidence_order = {"direct": 0, "strong": 1, "inferred": 2, "expanded": 3}
        
        for term_key, terms in by_term.items():
            # Sort by confidence (direct > strong > inferred > expanded)
            terms_sorted = sorted(terms, key=lambda x: confidence_order.get(x.get("confidence", "direct"), 99))
            # Keep the best one
            deduplicated.append(terms_sorted[0])
        
        logger.info(f"Related terms: {len(related_terms)} → {len(deduplicated)} after deduplication")
        return deduplicated
    
    def _deduplicate_usage_evidence(self, usage_evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate usage evidence:
        - Remove entries where the quoted text is fully contained in another
        - Keep longer/more complete quotes
        - Merge very similar quotes (>85% word overlap)
        """
        if not usage_evidence:
            return usage_evidence
        
        # Sort by text length (longest first)
        sorted_evidence = sorted(usage_evidence, key=lambda x: len(x.get("text", "")), reverse=True)
        
        kept = []
        for evidence in sorted_evidence:
            text = evidence.get("text", "").strip()
            if not text:
                continue
                
            is_duplicate = False
            for kept_ev in kept:
                kept_text = kept_ev.get("text", "").strip()
                
                # Case 1: This quote is fully contained in a longer quote
                if self._is_contained_in(text, kept_text):
                    is_duplicate = True
                    break
                
                # Case 2: Very high word overlap - likely same quote
                similarity = self._text_similarity(text, kept_text)
                if similarity > 0.85:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept.append(evidence)
        
        logger.info(f"Usage evidence: {len(usage_evidence)} → {len(kept)} after deduplication")
        return kept
    
    def _deduplicate_term_entry(self, term_entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deduplicate all components of a term entry.
        """
        term_entry["definitions"] = self._deduplicate_definitions(term_entry.get("definitions", []))
        term_entry["related_terms"] = self._deduplicate_related_terms(term_entry.get("related_terms", []))
        term_entry["usage_evidence"] = self._deduplicate_usage_evidence(term_entry.get("usage_evidence", []))
        # see_also is just a list of strings - dedupe by case-insensitive uniqueness
        see_also = term_entry.get("see_also", [])
        seen = set()
        unique_see_also = []
        for term in see_also:
            key = term.strip().lower()
            if key not in seen:
                seen.add(key)
                unique_see_also.append(term)
        term_entry["see_also"] = unique_see_also
        
        return term_entry
    
    def _calculate_match_quality(self, term_entry: Dict[str, Any], expanded_query_used: bool = False) -> Dict[str, Any]:
        """
        Calculate overall match quality summary.
        
        Returns:
            Dict with match quality metrics and optional warning
        """
        direct_count = 0
        inferred_count = 0
        total_count = 0
        
        # Count confidence levels across all extraction types
        for defn in term_entry.get("definitions", []):
            confidence = defn.get("confidence", "direct")
            total_count += 1
            if confidence in ("direct", "strong"):
                direct_count += 1
            else:
                inferred_count += 1
        
        for rel in term_entry.get("related_terms", []):
            confidence = rel.get("confidence", "direct")
            total_count += 1
            if confidence in ("direct", "strong"):
                direct_count += 1
            else:
                inferred_count += 1
        
        for usage in term_entry.get("usage_evidence", []):
            confidence = usage.get("confidence", "direct")
            total_count += 1
            if confidence in ("direct", "strong"):
                direct_count += 1
            else:
                inferred_count += 1
        
        # Determine overall confidence
        if total_count == 0:
            overall = "low"
            warning = "Otsest vastet ei leitud. Tulemused põhinevad seotud kontekstil."
        elif direct_count >= total_count * 0.7:
            overall = "high"
            warning = None
        elif direct_count >= total_count * 0.4:
            overall = "medium"
            warning = None  # Medium confidence is fine, no need to warn
        else:
            overall = "low"
            warning = "Enamik tulemusi on tuletatud kontekstist. Kontrolli allikaid."
        
        # Add expanded query info (not a warning, just info)
        if expanded_query_used and not warning:
            warning = "Kasutati laiendatud otsingut (sünonüümid/seotud terminid)."
        elif expanded_query_used and warning:
            warning = warning + " Kasutati laiendatud otsingut."
        
        return {
            "overall_confidence": overall,
            "direct_matches": direct_count,
            "inferred_matches": inferred_count,
            "expanded_query_used": expanded_query_used,
            "warning_message": warning,
        }
    
    def _format_term_entry(
        self, 
        term_entry: Dict[str, Any], 
        show_confidence: bool = True,
        output_categories: List[str] = None,
    ) -> str:
        """
        Format structured term entry into human-readable markdown text.
        
        Minimalist approach:
        - No badges or emojis for good matches (direct/strong)
        - Subtle text indicator only for low-confidence items (inferred/expanded)
        
        Args:
            term_entry: Parsed term entry dictionary
            show_confidence: Whether to show confidence indicators for low-confidence items
            output_categories: List of categories to include in output.
                               Options: "definitions", "related_terms", "usage_evidence"
                               If None or empty, includes all categories.
            
        Returns:
            Formatted markdown string with clickable source links
        """
        # Default to all categories if none specified
        if not output_categories:
            output_categories = ["definitions", "related_terms", "usage_evidence"]
        
        lines = []
        
        # Term header
        lines.append(f"**{term_entry['term']}**")
        lines.append("")
        
        # Match quality warning (if present) - without emojis
        match_quality = term_entry.get("match_quality", {})
        if match_quality and match_quality.get("warning_message"):
            lines.append(f"{match_quality['warning_message']}")
            lines.append("")
        
        # Definitions
        if "definitions" in output_categories and term_entry["definitions"]:
            lines.append("**Definitsioonid:**")
            for i, defn in enumerate(term_entry["definitions"], 1):
                text = defn.get('text', '')
                confidence = defn.get('confidence', 'direct')
                source_link = self._format_source_link(defn, text)
                
                # Only show indicator for low-confidence items
                if show_confidence and self._is_low_confidence(confidence):
                    label = self._get_confidence_label(confidence)
                    lines.append(f"  {i}. {text} {source_link} — *{label}*")
                else:
                    lines.append(f"  {i}. {text} {source_link}")
            lines.append("")
        
        # Related terms
        if "related_terms" in output_categories and term_entry["related_terms"]:
            lines.append("**Seotud terminid:**")
            for i, rel in enumerate(term_entry["related_terms"], 1):
                term = rel.get('term', '')
                relation = rel.get("relation_type", "related")
                confidence = rel.get('confidence', 'direct')
                source_link = self._format_source_link(rel, term)
                
                # Only show indicator for low-confidence items
                if show_confidence and self._is_low_confidence(confidence):
                    label = self._get_confidence_label(confidence)
                    lines.append(f"  {i}. **{term}** ({relation}) {source_link} — *{label}*")
                else:
                    lines.append(f"  {i}. **{term}** ({relation}) {source_link}")
            lines.append("")
        
        # Usage evidence
        if "usage_evidence" in output_categories and term_entry["usage_evidence"]:
            lines.append("**Kasutuskontekstid:**")
            for i, usage in enumerate(term_entry["usage_evidence"], 1):
                text = usage.get('text', '')
                context = usage.get('context', '')
                confidence = usage.get('confidence', 'direct')
                source_link = self._format_source_link(usage, text)
                
                # Only show indicator for low-confidence items
                label_suffix = ""
                if show_confidence and self._is_low_confidence(confidence):
                    label_suffix = f" — *{self._get_confidence_label(confidence)}*"
                
                # Format: optional context, then exact quote in italics
                if context:
                    lines.append(f"  {i}. {context} — *\"{text}\"* {source_link}{label_suffix}")
                else:
                    lines.append(f"  {i}. *\"{text}\"* {source_link}{label_suffix}")
            lines.append("")
        
        # See also - now handled by frontend with native Panel buttons
        # We just include the terms in the structured data (term_entry.see_also)
        # The frontend will render these as clickable buttons in footer_objects
        
        return "\n".join(lines)
    
    def chat(
        self,
        query: str,
        limit: int = 5,
        files: List[str] = None,
        only_valid: bool = False,
        prompt_type: str = "terminology_analysis",
        debug: bool = False,
        expand_context: bool = False,
        use_reranking: bool = True,
        output_categories: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a chat query: retrieve context and generate response.
        
        Args:
            query: User query/keyword
            limit: Number of context chunks to retrieve
            files: Filter by document titles
            only_valid: Only use valid documents
            prompt_type: Type of prompt to use from database
            debug: If True, collect and return debug information for each pipeline step
            expand_context: If True, expand results with adjacent chunks for fuller context
            use_reranking: If True and reranking is enabled, rerank results with cross-encoder
            output_categories: List of categories to include in output (definitions, related_terms, usage_evidence)
            
        Returns:
            Dict with response text, structured term entry, metadata, and optionally debug info
        """
        if not self._initialized:
            raise RuntimeError("ChatService not initialized")
        
        from langchain_core.prompts import ChatPromptTemplate
        
        # Initialize debug collector if debug mode is enabled
        debug_collector = DebugCollector() if debug else None
        
        if debug:
            logger.info(f"🔍 DEBUG MODE ENABLED for query: {query}")
        
        # Step 1: Query received
        if debug_collector:
            debug_collector.start_step()
            debug_collector.add_step(
                "Query Received",
                "Original user query received by the system",
                {"query": query}
            )
        
        # Step 2: Filters applied
        if debug_collector:
            debug_collector.start_step()
            filter_info = {
                "limit": limit,
                "files": files if files else "All documents",
                "only_valid": only_valid,
                "prompt_type": prompt_type,
                "expand_context": expand_context,
                "use_reranking": use_reranking,
            }
            debug_collector.add_step(
                "Filters Applied",
                "Search filters and parameters configured for this query",
                filter_info
            )
        
        # Step 3: Vector search
        if debug_collector:
            debug_collector.start_step()
        
        # Use debug search if debug mode is enabled
        search_debug_info = None
        if debug:
            results, search_debug_info = self.search_service.search(
                query=query,
                limit=limit,
                files=files,
                only_valid=only_valid,
                use_reranking=use_reranking,
                debug=True,
            )
        else:
            results = self.search_service.search(
                query=query,
                limit=limit,
                files=files,
                only_valid=only_valid,
                use_reranking=use_reranking,
            )
        
        # Step 3b: Context expansion with adjacent chunks
        original_count = len(results)
        if expand_context and results:
            if debug_collector:
                debug_collector.start_step()
            
            results = self.search_service.expand_context(results, max_adjacent_per_result=2)
            
            if debug_collector:
                debug_collector.add_step(
                    "Context Expansion",
                    f"Expanded context with adjacent chunks from same documents",
                    {
                        "original_chunks": original_count,
                        "expanded_chunks": len(results),
                        "adjacent_added": len(results) - original_count,
                    }
                )
        
        if debug_collector:
            search_type = "hybrid (dense + sparse)" if self.search_service.hybrid_enabled else "dense only"
            if self.search_service.reranking_enabled:
                search_type += " + reranking"
            
            # Include detailed search debug info if available
            search_step_data = {
                "search_type": search_type,
                "reranking_enabled": self.search_service.reranking_enabled,
                "reranker_model": self.config.get("reranking", {}).get("model", "N/A") if self.search_service.reranking_enabled else "N/A",
                "collection": self.search_service.collection_name,
                "results_count": len(results),
            }
            if search_debug_info:
                search_step_data["detailed_steps"] = search_debug_info
            
            debug_collector.add_step(
                "Vector Search",
                f"Performed {search_type} search in Qdrant database",
                search_step_data,
                truncate_at=500000  # Large limit to show all raw results
            )
        
        # Step 4: Search results
        if debug_collector:
            debug_collector.start_step()
            results_summary = [
                {
                    "title": r.get("title", ""),
                    "page": r.get("page_number", 0),
                    "score": round(r.get("score", 0), 4),
                    "text_preview": r.get("text", "")[:200] + "..." if len(r.get("text", "")) > 200 else r.get("text", ""),
                }
                for r in results
            ]
            debug_collector.add_step(
                "Search Results",
                f"Retrieved {len(results)} relevant document chunks",
                results_summary,
                truncate_at=4000
            )
        
        # Step 5: Build context
        if debug_collector:
            debug_collector.start_step()
        
        context = self._build_context(results)
        
        if debug_collector:
            debug_collector.add_step(
                "Context Built",
                "Formatted search results into context string for LLM",
                context,
                truncate_at=3000
            )
        
        # Step 6: Load system prompt
        if debug_collector:
            debug_collector.start_step()
        
        system_prompt = self._get_system_prompt(prompt_type)
        
        if debug_collector:
            debug_collector.add_step(
                "Prompt Loaded",
                f"Loaded system prompt: '{prompt_type}'",
                system_prompt,
                truncate_at=2000
            )
        
        # Step 7: Build full prompt
        if debug_collector:
            debug_collector.start_step()
        
        chat_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Keyword: {user_input}"),
            ("human", "Key sections:\n{retrieval_results}"),
        ])
        
        prompt = chat_template.format_messages(
            user_input=query,
            retrieval_results=context
        )
        
        if debug_collector:
            # Format prompt messages for display
            prompt_display = []
            for msg in prompt:
                content_preview = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
                prompt_display.append({
                    "role": msg.type,
                    "content_length": len(msg.content),
                    "content_preview": content_preview,
                })
            debug_collector.add_step(
                "Full Prompt Formed",
                "Complete prompt assembled with system prompt, user query, and context",
                prompt_display,
                truncate_at=3000
            )
        
        # Step 8: LLM invocation
        if debug_collector:
            debug_collector.start_step()
        
        try:
            response = self.llm.invoke(prompt)
            raw_llm_response = response.content
            
            if debug_collector:
                debug_collector.add_step(
                    "LLM Response",
                    f"Received response from {self.config['llm']['model']}",
                    raw_llm_response,
                    truncate_at=4000
                )
            
            # Step 9: Parse JSON
            if debug_collector:
                debug_collector.start_step()
            
            term_entry = self._parse_llm_json(response.content, query)
            
            if debug_collector:
                debug_collector.add_step(
                    "JSON Parsed",
                    "Extracted structured term entry from LLM response",
                    term_entry,
                    truncate_at=3000
                )
            
            # Step 10: Format response
            if debug_collector:
                debug_collector.start_step()
            
            formatted_response = self._format_term_entry(term_entry, output_categories=output_categories)
            
            if debug_collector:
                debug_collector.add_step(
                    "Response Formatted",
                    "Converted structured data to human-readable markdown",
                    formatted_response,
                    truncate_at=2000
                )
            
            result = {
                "response": formatted_response,
                "term_entry": term_entry,
                "context_used": len(results),
            }
            
            # Add debug info if enabled
            if debug_collector:
                debug_info = debug_collector.to_dict()
                debug_info["search_type"] = "hybrid" if self.search_service.hybrid_enabled else "dense"
                debug_info["model_used"] = self.config["llm"]["model"]
                debug_info["embedding_model"] = self.config["embeddings"]["embedding_model"]
                result["debug_info"] = debug_info
            
            return result
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            error_result = {
                "response": "Ilmnes viga, vabandust. Proovi uuesti.",
                "term_entry": None,
                "context_used": len(results),
            }
            
            if debug_collector:
                debug_collector.add_step(
                    "Error",
                    f"An error occurred during LLM processing",
                    {"error": str(e)}
                )
                debug_info = debug_collector.to_dict()
                debug_info["search_type"] = "hybrid" if self.search_service.hybrid_enabled else "dense"
                debug_info["model_used"] = self.config["llm"]["model"]
                debug_info["embedding_model"] = self.config["embeddings"]["embedding_model"]
                error_result["debug_info"] = debug_info
            
            return error_result
    
    def _get_system_prompt(self, prompt_type: str = "terminology_analysis") -> str:
        """Get the system prompt for the LLM from database."""
        if self.prompt_service:
            prompt_text = self.prompt_service.get_prompt_text(prompt_type)
            if prompt_text:
                # Escape curly braces for ChatPromptTemplate
                return prompt_text.replace("{", "{{").replace("}", "}}")
        
        raise RuntimeError(f"Prompt '{prompt_type}' not found in database. Please ensure prompts are initialized.")
    
    def _get_parallel_prompt(self, prompt_type: str) -> str:
        """Get a parallel extraction prompt from database."""
        if self.prompt_service:
            prompt_text = self.prompt_service.get_prompt_text(prompt_type)
            if prompt_text:
                # Escape curly braces for ChatPromptTemplate
                return prompt_text.replace("{", "{{").replace("}", "}}")
        
        raise RuntimeError(f"Prompt '{prompt_type}' not found in database. Please ensure prompts are initialized.")

    # =========================================================================
    # Query Expansion for enhanced search
    # =========================================================================
    
    def _expand_query_sync(self, query: str) -> Dict[str, Any]:
        """
        Expand a query with synonyms and related terms using LLM.
        Synchronous version for use in thread pool.
        
        Returns:
            Dict with 'original', 'expanded' (list of terms), and 'language'
        """
        from langchain_core.prompts import ChatPromptTemplate
        
        # Get expansion prompt from DB
        expansion_prompt = self._get_parallel_prompt("query_expansion")
        
        chat_template = ChatPromptTemplate.from_messages([
            ("system", expansion_prompt),
            ("human", "Keyword to expand: {query}"),
        ])
        
        prompt_messages = chat_template.format_messages(query=query)
        
        try:
            response = self.llm.invoke(prompt_messages)
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                parsed = json.loads(json_match.group(0))
                return {
                    "original": query,
                    "expanded": parsed.get("expanded", []),
                    "language": parsed.get("language", "unknown"),
                }
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
        
        # Return just original on failure
        return {"original": query, "expanded": [], "language": "unknown"}

    async def _expand_query_async(
        self,
        query: str,
        executor: ThreadPoolExecutor,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Async wrapper for query expansion.
        
        Returns:
            Tuple of (expansion_result, duration_ms)
        """
        start_time = time.time()
        loop = asyncio.get_event_loop()
        
        try:
            result = await loop.run_in_executor(
                executor,
                self._expand_query_sync,
                query
            )
            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"✓ Query expanded: {query} → +{len(result.get('expanded', []))} terms ({duration_ms:.0f}ms)")
            return result, duration_ms
        except Exception as e:
            logger.error(f"Query expansion error: {e}")
            duration_ms = (time.time() - start_time) * 1000
            return {"original": query, "expanded": [], "language": "unknown"}, duration_ms

    def _search_with_expansion(
        self,
        query: str,
        expanded_terms: List[str],
        limit: int,
        files: List[str] = None,
        only_valid: bool = False,
        use_reranking: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search with original query and expanded terms, then deduplicate.
        Reranking is done ONCE at the end on combined results for performance.
        
        IMPORTANT: Original query results are ALWAYS kept first.
        Expanded results only ADD to (not replace) original results.
        
        Returns:
            Combined results: all original + additional expanded (up to limit*2)
        """
        # Step 1: Search with original query WITHOUT reranking (we'll rerank at the end)
        original_results = self.search_service.search(
            query=query,
            limit=limit,
            files=files,
            only_valid=only_valid,
            use_reranking=False,  # Don't rerank yet
        )
        
        # Mark original results
        for r in original_results:
            r["_source"] = "original"
        
        logger.info(f"Original search found {len(original_results)} results")
        
        # Step 2: Search with each expanded term individually for additional results
        expanded_results = []
        if expanded_terms:
            seen_in_original = set()
            for r in original_results:
                # Use title + page as unique key to avoid same chunk
                key = (r.get("title", ""), r.get("page_number", 0), r.get("text", "")[:50])
                seen_in_original.add(key)
            
            # Search each expanded term (limit searches to avoid slowdown)
            for term in expanded_terms[:3]:  # Max 3 expanded terms
                try:
                    extra = self.search_service.search(
                        query=term,
                        limit=3,  # Small limit per term
                        files=files,
                        only_valid=only_valid,
                        use_reranking=False,  # Don't rerank yet
                    )
                    for r in extra:
                        key = (r.get("title", ""), r.get("page_number", 0), r.get("text", "")[:50])
                        if key not in seen_in_original:
                            r["_source"] = f"expanded:{term}"
                            expanded_results.append(r)
                            seen_in_original.add(key)
                except Exception as e:
                    logger.warning(f"Expanded search for '{term}' failed: {e}")
            
            logger.info(f"Expanded search found {len(expanded_results)} additional results")
        
        # Step 3: Combine all results
        combined = original_results.copy()
        combined.extend(expanded_results)
        
        # Step 4: Rerank ONCE on combined results using the ORIGINAL query
        # This ensures we rank by relevance to what the user actually asked for
        if use_reranking and self.search_service.reranking_enabled and len(combined) > 0:
            logger.info(f"Reranking {len(combined)} combined results (once at end)")
            combined = self.search_service._rerank(query, combined, top_k=None)
        
        # Limit total to reasonable size (1.5x original limit)
        max_total = int(limit * 1.5)
        return combined[:max_total]

    # =========================================================================
    # Parallel extraction methods
    # =========================================================================
    
    def _invoke_llm_sync(self, prompt_messages) -> str:
        """Synchronous LLM invocation for use in thread pool."""
        response = self.llm.invoke(prompt_messages)
        return response.content
    
    async def _extract_single(
        self,
        extraction_type: str,
        prompt_template: str,
        query: str,
        context: str,
        executor: ThreadPoolExecutor,
    ) -> Tuple[str, Dict[str, Any], float]:
        """
        Extract a single category (definitions/related_terms/usage_evidence).
        
        Returns:
            Tuple of (extraction_type, parsed_result, duration_ms)
        """
        from langchain_core.prompts import ChatPromptTemplate
        
        start_time = time.time()
        
        chat_template = ChatPromptTemplate.from_messages([
            ("system", prompt_template),
            ("human", "Keyword: {user_input}"),
            ("human", "Key sections:\n{retrieval_results}"),
        ])
        
        prompt_messages = chat_template.format_messages(
            user_input=query,
            retrieval_results=context
        )
        
        loop = asyncio.get_event_loop()
        
        try:
            # Run LLM call in thread pool to not block
            raw_response = await loop.run_in_executor(
                executor,
                self._invoke_llm_sync,
                prompt_messages
            )
            
            # Parse the JSON response
            parsed = self._parse_extraction_json(raw_response, extraction_type)
            duration_ms = (time.time() - start_time) * 1000
            
            logger.info(f"✓ {extraction_type} extraction completed in {duration_ms:.0f}ms")
            return (extraction_type, parsed, duration_ms, raw_response)
            
        except Exception as e:
            logger.error(f"✗ {extraction_type} extraction failed: {e}")
            duration_ms = (time.time() - start_time) * 1000
            return (extraction_type, {}, duration_ms, str(e))
    
    def _parse_extraction_json(self, content: str, extraction_type: str) -> Dict[str, Any]:
        """Parse JSON from a single extraction response."""
        import re
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*\})\s*```', content)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                json_str = json_match.group(0)
            else:
                return {}
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse {extraction_type} JSON")
            return {}
    
    async def chat_parallel(
        self,
        query: str,
        limit: int = 5,
        files: List[str] = None,
        only_valid: bool = False,
        debug: bool = False,
        expand_query: bool = False,
        expand_context: bool = False,
        use_reranking: bool = True,
        output_categories: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a chat query using PARALLEL extraction.
        
        Runs three specialized prompts in parallel:
        1. Definitions extraction
        2. Related terms extraction  
        3. Usage evidence extraction
        
        Optionally expands the query with synonyms/related terms first.
        Optionally expands context with adjacent chunks.
        
        Args:
            query: User query/keyword
            limit: Number of context chunks to retrieve
            files: Filter by document titles
            only_valid: Only use valid documents
            debug: If True, collect and return debug information
            expand_query: If True, expand query with LLM before searching
            expand_context: If True, expand results with adjacent chunks for fuller context
            use_reranking: If True and reranking is enabled, rerank results with cross-encoder
            output_categories: List of categories to include in output (definitions, related_terms, usage_evidence)
            
        Returns:
            Dict with response text, structured term entry, and optionally debug info
        """
        if not self._initialized:
            raise RuntimeError("ChatService not initialized")
        
        # Initialize debug collector if enabled
        debug_collector = DebugCollector() if debug else None
        
        if debug:
            logger.info(f"🔍 DEBUG MODE (PARALLEL) for query: {query}")
        
        mode_str = "parallel+expansion" if expand_query else "parallel"
        if expand_context:
            mode_str += "+context"
        if use_reranking:
            mode_str += "+reranking"
        logger.info(f"🚀 Starting {mode_str.upper()} extraction for: {query}")
        
        # Step 1: Query received
        if debug_collector:
            debug_collector.start_step()
            debug_collector.add_step(
                "Query Received",
                f"Original user query received ({mode_str} mode)",
                {"query": query, "mode": mode_str, "expand_query": expand_query, "expand_context": expand_context, "use_reranking": use_reranking}
            )
        
        # Step 2: Filters applied
        if debug_collector:
            debug_collector.start_step()
            debug_collector.add_step(
                "Filters Applied",
                "Search filters configured",
                {"limit": limit, "files": files or "All", "only_valid": only_valid, "expand_context": expand_context, "use_reranking": use_reranking}
            )
        
        # Step 3: Query expansion (if enabled)
        expanded_terms = []
        if expand_query:
            if debug_collector:
                debug_collector.start_step()
            
            with ThreadPoolExecutor(max_workers=1) as expansion_executor:
                expansion_result, expansion_duration = await self._expand_query_async(
                    query, expansion_executor
                )
                expanded_terms = expansion_result.get("expanded", [])
            
            if debug_collector:
                debug_collector.add_step(
                    "Query Expansion",
                    f"LLM expanded query with {len(expanded_terms)} additional terms",
                    {
                        "original": query,
                        "expanded_terms": expanded_terms,
                        "language": expansion_result.get("language", "unknown"),
                        "duration_ms": round(expansion_duration, 1),
                    }
                )
        
        # Step 4: Vector search (with or without expansion)
        if debug_collector:
            debug_collector.start_step()
        
        search_debug_info = None
        if expand_query and expanded_terms:
            # First, capture debug info for the original query if debug mode
            if debug:
                _, search_debug_info = self.search_service.search(
                    query=query,
                    limit=limit,
                    files=files,
                    only_valid=only_valid,
                    use_reranking=use_reranking,
                    debug=True,
                )
            
            # Use expanded search with deduplication
            results = self._search_with_expansion(
                query=query,
                expanded_terms=expanded_terms,
                limit=limit,
                files=files,
                only_valid=only_valid,
                use_reranking=use_reranking,
            )
            search_desc = f"Searched with original + {len(expanded_terms)} expanded terms"
        else:
            # Standard search (with debug if enabled)
            if debug:
                results, search_debug_info = self.search_service.search(
                    query=query,
                    limit=limit,
                    files=files,
                    only_valid=only_valid,
                    use_reranking=use_reranking,
                    debug=True,
                )
            else:
                results = self.search_service.search(
                    query=query,
                    limit=limit,
                    files=files,
                    only_valid=only_valid,
                    use_reranking=use_reranking,
                )
            search_desc = "Standard hybrid search"
        
        if debug_collector:
            if self.search_service.reranking_enabled and use_reranking:
                search_desc += " + reranking"
            search_info = {
                "search_type": search_desc,
                "reranking_enabled": self.search_service.reranking_enabled,
                "reranker_model": self.config.get("reranking", {}).get("model", "N/A") if self.search_service.reranking_enabled else "N/A",
                "results_count": len(results),
            }
            if expand_query:
                search_info["expanded_terms_used"] = expanded_terms
                # Show source distribution
                original_count = sum(1 for r in results if r.get("_source") == "original")
                expanded_count = sum(1 for r in results if r.get("_source") == "expanded")
                search_info["from_original_query"] = original_count
                search_info["from_expanded_query"] = expanded_count
            
            # Include detailed hybrid search debug info (for original query)
            if search_debug_info:
                search_info["detailed_steps"] = search_debug_info
            
            # Also include the final combined results with full text
            search_info["final_results"] = [
                {"rank": i+1, "title": r.get("title", ""), "page": r.get("page_number", 0), "score": round(r.get("score", 0), 4), "source": r.get("_source", "original"), "text": r.get("text", "")}
                for i, r in enumerate(results)
            ]
            
            debug_collector.add_step(
                "Vector Search",
                search_desc,
                search_info,
                truncate_at=500000  # Large limit to show all raw results
            )
        
        # Step 4b: Context expansion with adjacent chunks
        pre_expansion_count = len(results)
        if expand_context and results:
            if debug_collector:
                debug_collector.start_step()
            
            results = self.search_service.expand_context(results, max_adjacent_per_result=2)
            
            if debug_collector:
                debug_collector.add_step(
                    "Context Expansion",
                    "Expanded context with adjacent chunks from same documents",
                    {
                        "original_chunks": pre_expansion_count,
                        "expanded_chunks": len(results),
                        "adjacent_added": len(results) - pre_expansion_count,
                    }
                )
        
        # Step 5: Build context
        if debug_collector:
            debug_collector.start_step()
        
        context = self._build_context(results)
        
        if debug_collector:
            debug_collector.add_step(
                "Context Built",
                "Formatted context for LLM",
                context,
                truncate_at=2000
            )
        
        # Step 5: Parallel LLM extractions (only for selected categories)
        if debug_collector:
            debug_collector.start_step()
        
        # Default to all categories if none specified
        if not output_categories:
            output_categories = ["definitions", "related_terms", "usage_evidence"]
        
        # Load prompts only for selected categories
        prompts_loaded = {}
        if "definitions" in output_categories:
            prompts_loaded["definitions"] = self._get_parallel_prompt("definitions_extraction")
        if "related_terms" in output_categories:
            prompts_loaded["related_terms"] = self._get_parallel_prompt("related_terms_extraction")
        if "usage_evidence" in output_categories:
            prompts_loaded["usage_evidence"] = self._get_parallel_prompt("usage_evidence_extraction")
        # Always load see_also (it's a separate feature for navigation)
        prompts_loaded["see_also"] = self._get_parallel_prompt("see_also_extraction")
        
        if debug_collector:
            debug_collector.add_step(
                "Prompts Loaded",
                f"Loaded {len(prompts_loaded)} specialized prompts (based on selected categories)",
                {f"{k}_prompt_length": len(v) for k, v in prompts_loaded.items()}
            )
            debug_collector.start_step()
        
        # Create thread pool for parallel LLM calls
        num_tasks = len(prompts_loaded)
        with ThreadPoolExecutor(max_workers=num_tasks) as executor:
            # Launch only the selected extractions in parallel
            tasks = []
            for ext_type, prompt in prompts_loaded.items():
                tasks.append(
                    self._extract_single(
                        ext_type,
                        prompt,
                        query,
                        context,
                        executor
                    )
                )
            
            # Wait for all to complete
            extraction_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        if debug_collector:
            parallel_info = []
            for result in extraction_results:
                if isinstance(result, tuple):
                    ext_type, parsed, duration, raw = result
                    # Get items count based on extraction type
                    items = parsed.get(ext_type, parsed.get("definitions", parsed.get("related_terms", parsed.get("usage_evidence", parsed.get("see_also", [])))))
                    parallel_info.append({
                        "extraction": ext_type,
                        "duration_ms": round(duration, 1),
                        "items_found": len(items) if isinstance(items, list) else 0,
                        "raw_response": raw[:500] + "..." if len(str(raw)) > 500 else raw,
                    })
                else:
                    parallel_info.append({"error": str(result)})
            
            debug_collector.add_step(
                "Parallel LLM Extractions",
                f"Ran {len(prompts_loaded)} specialized prompts in parallel",
                parallel_info,
                truncate_at=4000
            )
        
        # Step 6: Combine results
        if debug_collector:
            debug_collector.start_step()
        
        # Combine all extraction results
        term_entry = {
            "term": query,
            "definitions": [],
            "related_terms": [],
            "usage_evidence": [],
            "see_also": [],
        }
        
        for result in extraction_results:
            if isinstance(result, Exception):
                logger.error(f"Extraction failed: {result}")
                continue
            
            ext_type, parsed, duration, raw = result
            
            if ext_type == "definitions":
                term_entry["definitions"] = parsed.get("definitions", [])
            elif ext_type == "related_terms":
                term_entry["related_terms"] = parsed.get("related_terms", [])
            elif ext_type == "usage_evidence":
                term_entry["usage_evidence"] = parsed.get("usage_evidence", [])
            elif ext_type == "see_also":
                term_entry["see_also"] = parsed.get("see_also", [])
        
        # Deduplicate results (remove near-duplicate definitions, terms, evidence)
        term_entry = self._deduplicate_term_entry(term_entry)
        
        # Calculate match quality for user transparency
        expanded_query_used = expand_query and len(expanded_terms) > 0
        match_quality = self._calculate_match_quality(term_entry, expanded_query_used)
        term_entry["match_quality"] = match_quality
        
        if debug_collector:
            debug_collector.add_step(
                "Results Combined",
                "Merged all extractions into term entry",
                {**term_entry, "match_quality": match_quality},
                truncate_at=3000
            )
        
        # Step 7: Format response
        if debug_collector:
            debug_collector.start_step()
        
        formatted_response = self._format_term_entry(term_entry, show_confidence=True, output_categories=output_categories)
        
        if debug_collector:
            debug_collector.add_step(
                "Response Formatted",
                "Converted to human-readable markdown",
                formatted_response,
                truncate_at=2000
            )
        
        logger.info(f"✅ Parallel extraction complete for: {query}")
        
        result = {
            "response": formatted_response,
            "term_entry": term_entry,
            "context_used": len(results),
        }
        
        if debug_collector:
            debug_info = debug_collector.to_dict()
            debug_info["search_type"] = "hybrid" if self.search_service.hybrid_enabled else "dense"
            debug_info["model_used"] = self.config["llm"]["model"]
            debug_info["embedding_model"] = self.config["embeddings"]["embedding_model"]
            debug_info["extraction_mode"] = "parallel"
            result["debug_info"] = debug_info
        
        return result


class PromptService:
    """
    Handles prompt management operations.
    Loads prompts from PostgreSQL and caches them.
    
    Prompts are organized into prompt sets. Each set contains multiple prompts
    that work together (e.g., for parallel extraction mode).
    """
    
    def __init__(self):
        self._sets_cache: Dict[int, Dict[str, Any]] = {}  # set_id -> set info
        self._prompts_cache: Dict[int, Dict[str, Dict[str, Any]]] = {}  # set_id -> {prompt_type -> prompt}
        self._default_set_id: Optional[int] = None
        self._initialized = False
    
    def _get_db_connection(self) -> Connection:
        """Create a database connection."""
        con = Connection(
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            db=os.getenv("PG_COLLECTION"),
        )
        con.establish_connection()
        return con
    
    def initialize(self) -> None:
        """Load all prompt sets and prompts into cache. Call once at startup."""
        if self._initialized:
            return
        
        logger.info("Initializing PromptService...")
        self._refresh_cache()
        self._initialized = True
        total_prompts = sum(len(prompts) for prompts in self._prompts_cache.values())
        logger.info(f"PromptService initialized with {len(self._sets_cache)} prompt sets, {total_prompts} prompts")
    
    def _refresh_cache(self) -> None:
        """Refresh the prompt sets and prompts cache from database."""
        con = self._get_db_connection()
        try:
            # Load prompt sets
            sets_result = con.execute_sql(
                "SELECT id, name, description, is_default, date_created, date_modified FROM prompt_sets ORDER BY name",
                [{}]
            )
            self._sets_cache = {}
            self._default_set_id = None
            
            if sets_result.get("data"):
                keys = list(sets_result["keys"])
                for row in sets_result["data"]:
                    row_dict = dict(zip(keys, row))
                    set_id = row_dict["id"]
                    self._sets_cache[set_id] = row_dict
                    if row_dict.get("is_default"):
                        self._default_set_id = set_id
            
            # Load prompts grouped by set
            prompts_result = con.execute_sql(
                "SELECT id, prompt_type, prompt_text, description, prompt_set_id, date_created, date_modified FROM prompts ORDER BY prompt_type",
                [{}]
            )
            self._prompts_cache = {set_id: {} for set_id in self._sets_cache.keys()}
            
            if prompts_result.get("data"):
                keys = list(prompts_result["keys"])
                for row in prompts_result["data"]:
                    row_dict = dict(zip(keys, row))
                    set_id = row_dict.get("prompt_set_id")
                    if set_id and set_id in self._prompts_cache:
                        self._prompts_cache[set_id][row_dict["prompt_type"]] = row_dict
        finally:
            con.close()
    
    # =========================================================================
    # Prompt Set Methods
    # =========================================================================
    
    def get_all_prompt_sets(self) -> List[Dict[str, Any]]:
        """Get all prompt sets."""
        return [
            {
                "id": s["id"],
                "name": s["name"],
                "description": s.get("description"),
                "is_default": s.get("is_default", False),
                "prompt_count": len(self._prompts_cache.get(s["id"], {})),
                "date_modified": str(s.get("date_modified")) if s.get("date_modified") else None,
            }
            for s in self._sets_cache.values()
        ]
    
    def get_prompt_set(self, set_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific prompt set with its prompts."""
        prompt_set = self._sets_cache.get(set_id)
        if not prompt_set:
            return None
        
        prompts = self._prompts_cache.get(set_id, {})
        return {
            "id": prompt_set["id"],
            "name": prompt_set["name"],
            "description": prompt_set.get("description"),
            "is_default": prompt_set.get("is_default", False),
            "date_created": str(prompt_set.get("date_created")) if prompt_set.get("date_created") else None,
            "date_modified": str(prompt_set.get("date_modified")) if prompt_set.get("date_modified") else None,
            "prompts": [
                {
                    "id": p["id"],
                    "prompt_type": p["prompt_type"],
                    "description": p.get("description"),
                    "date_modified": str(p.get("date_modified")) if p.get("date_modified") else None,
                }
                for p in prompts.values()
            ]
        }
    
    def get_default_set_id(self) -> Optional[int]:
        """Get the default prompt set ID."""
        return self._default_set_id
    
    def create_prompt_set(self, name: str, description: Optional[str] = None, is_default: bool = False) -> Dict[str, Any]:
        """Create a new prompt set."""
        con = self._get_db_connection()
        try:
            # If making this the default, unset other defaults first
            if is_default:
                con.execute_sql("UPDATE prompt_sets SET is_default = FALSE WHERE is_default = TRUE", [{}])
            
            result = con.execute_sql(
                """INSERT INTO prompt_sets (name, description, is_default)
                   VALUES (:name, :desc, :is_default)
                   RETURNING id, name, date_created""",
                [{"name": name, "desc": description, "is_default": is_default}]
            )
            con.commit()
            
            if result.get("data"):
                self._refresh_cache()
                return {
                    "success": True,
                    "message": "Prompt set created successfully",
                    "id": result["data"][0][0],
                    "name": name,
                }
            return {"success": False, "message": "Failed to create prompt set"}
        except Exception as e:
            if "unique" in str(e).lower():
                return {"success": False, "message": f"Prompt set '{name}' already exists"}
            return {"success": False, "message": str(e)}
        finally:
            con.close()
    
    def update_prompt_set(self, set_id: int, name: Optional[str] = None, description: Optional[str] = None, is_default: Optional[bool] = None) -> Dict[str, Any]:
        """Update a prompt set."""
        con = self._get_db_connection()
        try:
            updates = []
            params = {"set_id": set_id}
            
            if name is not None:
                updates.append("name = :name")
                params["name"] = name
            if description is not None:
                updates.append("description = :desc")
                params["desc"] = description
            if is_default is not None:
                if is_default:
                    # Unset other defaults first
                    con.execute_sql("UPDATE prompt_sets SET is_default = FALSE WHERE is_default = TRUE", [{}])
                updates.append("is_default = :is_default")
                params["is_default"] = is_default
            
            if not updates:
                return {"success": False, "message": "No updates provided"}
            
            updates.append("date_modified = CURRENT_TIMESTAMP")
            
            result = con.execute_sql(
                f"UPDATE prompt_sets SET {', '.join(updates)} WHERE id = :set_id RETURNING id",
                [params]
            )
            con.commit()
            
            if result.get("data"):
                self._refresh_cache()
                return {"success": True, "message": "Prompt set updated successfully"}
            return {"success": False, "message": "Prompt set not found"}
        finally:
            con.close()
    
    def delete_prompt_set(self, set_id: int) -> Dict[str, Any]:
        """Delete a prompt set and all its prompts."""
        con = self._get_db_connection()
        try:
            result = con.execute_sql(
                "DELETE FROM prompt_sets WHERE id = :set_id RETURNING id",
                [{"set_id": set_id}]
            )
            con.commit()
            
            if result.get("data"):
                self._refresh_cache()
                return {"success": True, "message": "Prompt set deleted successfully"}
            return {"success": False, "message": "Prompt set not found"}
        finally:
            con.close()
    
    def duplicate_prompt_set(self, set_id: int, new_name: str) -> Dict[str, Any]:
        """Duplicate a prompt set with all its prompts."""
        con = self._get_db_connection()
        try:
            # Get original set
            original_set = self._sets_cache.get(set_id)
            if not original_set:
                return {"success": False, "message": "Original prompt set not found"}
            
            # Create new set
            new_set_result = con.execute_sql(
                """INSERT INTO prompt_sets (name, description, is_default)
                   VALUES (:name, :desc, FALSE)
                   RETURNING id""",
                [{"name": new_name, "desc": original_set.get("description")}]
            )
            
            if not new_set_result.get("data"):
                return {"success": False, "message": "Failed to create new prompt set"}
            
            new_set_id = new_set_result["data"][0][0]
            
            # Copy all prompts from original set
            original_prompts = self._prompts_cache.get(set_id, {})
            for prompt in original_prompts.values():
                con.execute_sql(
                    """INSERT INTO prompts (prompt_type, prompt_text, description, prompt_set_id)
                       VALUES (:ptype, :text, :desc, :set_id)""",
                    [{"ptype": prompt["prompt_type"], "text": prompt["prompt_text"], 
                      "desc": prompt.get("description"), "set_id": new_set_id}]
                )
            
            con.commit()
            self._refresh_cache()
            
            return {
                "success": True,
                "message": f"Prompt set duplicated successfully with {len(original_prompts)} prompts",
                "id": new_set_id,
                "name": new_name,
            }
        except Exception as e:
            if "unique" in str(e).lower():
                return {"success": False, "message": f"Prompt set '{new_name}' already exists"}
            return {"success": False, "message": str(e)}
        finally:
            con.close()
    
    # =========================================================================
    # Prompt Methods (within a set)
    # =========================================================================
    
    def get_all_prompts(self, set_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all prompts, optionally filtered by set. For backward compatibility, returns all if no set specified."""
        if set_id is not None:
            prompts = self._prompts_cache.get(set_id, {})
            return [
                {
                    "id": p["id"],
                    "prompt_type": p["prompt_type"],
                    "description": p.get("description"),
                    "prompt_set_id": set_id,
                    "date_modified": str(p.get("date_modified")) if p.get("date_modified") else None,
                }
                for p in prompts.values()
            ]
        
        # Return all prompts from all sets (backward compatibility)
        all_prompts = []
        for sid, prompts in self._prompts_cache.items():
            for p in prompts.values():
                all_prompts.append({
                    "id": p["id"],
                    "prompt_type": p["prompt_type"],
                    "description": p.get("description"),
                    "prompt_set_id": sid,
                    "date_modified": str(p.get("date_modified")) if p.get("date_modified") else None,
                })
        return all_prompts
    
    def get_prompt(self, prompt_type: str, set_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get a specific prompt by type. Uses default set if set_id not specified."""
        target_set_id = set_id if set_id is not None else self._default_set_id
        if target_set_id is None:
            return None
        
        prompts = self._prompts_cache.get(target_set_id, {})
        prompt = prompts.get(prompt_type)
        
        if prompt:
            return {
                "id": prompt["id"],
                "prompt_type": prompt["prompt_type"],
                "prompt_text": prompt["prompt_text"],
                "description": prompt.get("description"),
                "prompt_set_id": target_set_id,
                "date_created": str(prompt.get("date_created")) if prompt.get("date_created") else None,
                "date_modified": str(prompt.get("date_modified")) if prompt.get("date_modified") else None,
            }
        return None
    
    def get_prompt_text(self, prompt_type: str, set_id: Optional[int] = None) -> Optional[str]:
        """Get just the prompt text for a given type. Uses default set if set_id not specified."""
        target_set_id = set_id if set_id is not None else self._default_set_id
        if target_set_id is None:
            return None
        
        prompts = self._prompts_cache.get(target_set_id, {})
        prompt = prompts.get(prompt_type)
        return prompt["prompt_text"] if prompt else None
    
    def update_prompt(self, prompt_type: str, prompt_text: str, description: Optional[str] = None, set_id: Optional[int] = None) -> Dict[str, Any]:
        """Update an existing prompt within a set."""
        target_set_id = set_id if set_id is not None else self._default_set_id
        if target_set_id is None:
            return {"success": False, "message": "No prompt set specified and no default set found"}
        
        con = self._get_db_connection()
        try:
            result = con.execute_sql(
                """UPDATE prompts 
                   SET prompt_text = :text, description = :desc, date_modified = CURRENT_TIMESTAMP 
                   WHERE prompt_type = :ptype AND prompt_set_id = :set_id
                   RETURNING id, prompt_type, date_modified""",
                [{"text": prompt_text, "desc": description, "ptype": prompt_type, "set_id": target_set_id}]
            )
            con.commit()
            
            if result.get("data"):
                self._refresh_cache()
                return {
                    "success": True,
                    "message": "Prompt updated successfully",
                    "prompt_type": prompt_type,
                    "prompt_set_id": target_set_id,
                    "date_modified": str(result["data"][0][2]) if result["data"][0][2] else None,
                }
            else:
                return {"success": False, "message": f"Prompt type '{prompt_type}' not found in set {target_set_id}"}
        finally:
            con.close()
    
    def create_prompt(self, prompt_type: str, prompt_text: str, description: Optional[str] = None, set_id: Optional[int] = None) -> Dict[str, Any]:
        """Create a new prompt within a set."""
        target_set_id = set_id if set_id is not None else self._default_set_id
        if target_set_id is None:
            return {"success": False, "message": "No prompt set specified and no default set found"}
        
        con = self._get_db_connection()
        try:
            result = con.execute_sql(
                """INSERT INTO prompts (prompt_type, prompt_text, description, prompt_set_id)
                   VALUES (:ptype, :text, :desc, :set_id)
                   RETURNING id, prompt_type, date_created""",
                [{"ptype": prompt_type, "text": prompt_text, "desc": description, "set_id": target_set_id}]
            )
            con.commit()
            
            if result.get("data"):
                self._refresh_cache()
                return {
                    "success": True,
                    "message": "Prompt created successfully",
                    "id": result["data"][0][0],
                    "prompt_type": prompt_type,
                    "prompt_set_id": target_set_id,
                }
            else:
                return {"success": False, "message": "Failed to create prompt"}
        except IntegrityError:
            return {"success": False, "message": f"Prompt type '{prompt_type}' already exists in this set"}
        finally:
            con.close()
    
    def delete_prompt(self, prompt_type: str, set_id: Optional[int] = None) -> Dict[str, Any]:
        """Delete a prompt from a set."""
        target_set_id = set_id if set_id is not None else self._default_set_id
        if target_set_id is None:
            return {"success": False, "message": "No prompt set specified and no default set found"}
        
        con = self._get_db_connection()
        try:
            result = con.execute_sql(
                "DELETE FROM prompts WHERE prompt_type = :ptype AND prompt_set_id = :set_id RETURNING id",
                [{"ptype": prompt_type, "set_id": target_set_id}]
            )
            con.commit()
            
            if result.get("data"):
                self._refresh_cache()
                return {"success": True, "message": "Prompt deleted successfully"}
            else:
                return {"success": False, "message": f"Prompt type '{prompt_type}' not found in set {target_set_id}"}
        finally:
            con.close()


class UploadService:
    """
    Handles document upload and processing.
    Reuses the SearchService's models for embeddings.
    """
    
    def __init__(self, config: dict, search_service: SearchService):
        self.config = config
        self.search_service = search_service
        self.collection_name = os.getenv("QDRANT_COLLECTION", "documents")
    
    def _get_db_connection(self) -> Connection:
        """Create a database connection."""
        con = Connection(
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            db=os.getenv("PG_COLLECTION"),
        )
        con.establish_connection()
        return con
    
    def _extract_text_from_pdf(self, pdf_bytes: bytes) -> List[Dict[str, Any]]:
        """Extract text content from PDF."""
        content = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text = page.get_text()
                # Basic text cleanup
                text = " ".join(text.split())
                content.append({
                    "page_number": page.number + 1,
                    "text": text,
                })
        return content
    
    def _chunk_text(self, text: str, max_tokens: int = 250) -> List[str]:
        """Split text into chunks. Simple word-based chunking."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word.split())
            if current_length + word_length > max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _create_embeddings_and_upload(
        self,
        document_metadata: Dict[str, Any],
        content: List[Dict[str, Any]],
        progress_callback: callable = None,
    ) -> int:
        """Create embeddings and upload to Qdrant. Returns number of chunks uploaded.
        
        Args:
            document_metadata: Metadata dict to store with each chunk
            content: List of page dicts with 'text' and 'page_number'
            progress_callback: Optional callback(stage, current, total, message) for progress
        """
        max_tokens = self.config["embeddings"]["max_tokens"]
        passage_prompt = self.config["embeddings"]["passage_prompt"]
        
        def report_progress(stage: str, current: int, total: int, message: str = ""):
            if progress_callback:
                progress_callback(stage, current, total, message)

        # Step 1: Collect all chunks first (for batch embedding)
        report_progress("chunking", 0, 100, "Teksti tükeldamine...")
        all_chunks = []
        for page_data in content:
            page_text = page_data["text"]
            if not page_text.strip():
                continue

            chunks = self._chunk_text(page_text, max_tokens)

            for chunk_text in chunks:
                if not chunk_text.strip():
                    continue

                all_chunks.append({
                    "text": chunk_text,
                    "page_number": page_data["page_number"],
                })

        if not all_chunks:
            return 0
        
        total_chunks = len(all_chunks)
        report_progress("chunking", 100, 100, f"{total_chunks} tekstilõiku leitud")

        logger.info(f"Creating embeddings for {total_chunks} chunks (batched)...")

        # Step 2: Batch encode all texts - with progress
        report_progress("embedding", 0, total_chunks, "Tihedate vektorite loomine...")
        texts_for_dense = [passage_prompt + c["text"] for c in all_chunks]
        
        # Encode in smaller batches to report progress
        embed_batch_size = 32
        dense_vectors = []
        for i in range(0, len(texts_for_dense), embed_batch_size):
            batch_texts = texts_for_dense[i:i + embed_batch_size]
            batch_vectors = self.search_service.dense_model.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=embed_batch_size,
            )
            dense_vectors.extend(batch_vectors)
            report_progress("embedding", min(i + embed_batch_size, total_chunks), total_chunks, 
                          f"Tihedate vektorite loomine ({min(i + embed_batch_size, total_chunks)}/{total_chunks})...")

        # Batch encode sparse vectors if hybrid enabled
        sparse_vectors = None
        if self.search_service.hybrid_enabled and self.search_service.sparse_model:
            report_progress("sparse", 0, total_chunks, "Hõredate vektorite loomine...")
            raw_texts = [c["text"] for c in all_chunks]
            sparse_embeddings = list(self.search_service.sparse_model.embed(raw_texts, batch_size=32))
            sparse_vectors = [
                SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
                for e in sparse_embeddings
            ]
            report_progress("sparse", total_chunks, total_chunks, "Hõredad vektorid loodud")
        
        # Step 3: Create points with pre-computed embeddings
        report_progress("points", 0, total_chunks, "Andmepunktide loomine...")
        points = []
        for i, chunk_data in enumerate(all_chunks):
            chunk_id = str(uuid.uuid4())
            
            payload = document_metadata.copy()
            payload["text"] = chunk_data["text"]
            payload["page_number"] = chunk_data["page_number"]
            payload["content_type"] = "text"
            payload["date_created"] = datetime.now().isoformat()
            payload["validated"] = True
            
            if sparse_vectors:
                vector = {
                    "dense": dense_vectors[i].tolist(),
                    "sparse": sparse_vectors[i],
                }
            else:
                vector = dense_vectors[i].tolist()
            
            points.append(PointStruct(
                id=chunk_id,
                vector=vector,
                payload=payload,
            ))
        
        # Step 4: Upload in batches with progress
        logger.info(f"Uploading {len(points)} chunks to Qdrant...")
        batch_size = 50  # Larger batches for upload
        uploaded = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.search_service.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=batch,
            )
            uploaded += len(batch)
            report_progress("upload", uploaded, len(points), 
                          f"Qdrant'i üleslaadimine ({uploaded}/{len(points)})...")
        
        report_progress("done", len(points), len(points), "Valmis!")
        return len(points)
    
    def upload_document(
        self,
        pdf_bytes: bytes,
        filename: str,
        metadata: Dict[str, Any],
        progress_callback: callable = None,
    ) -> Dict[str, Any]:
        """
        Upload a document: save metadata to DB, create embeddings, upload to Qdrant.
        
        Args:
            pdf_bytes: PDF file content
            filename: Original filename
            metadata: Document metadata dict
            progress_callback: Optional callback(stage, current, total, message) for progress
        
        Returns:
            Dict with status, message, and document_id
        """
        con = self._get_db_connection()
        
        def report_progress(stage: str, current: int, total: int, message: str = ""):
            if progress_callback:
                progress_callback(stage, current, total, message)
        
        try:
            report_progress("init", 0, 100, "Ettevalmistus...")
            
            # Ensure collection exists
            existing = [c.name for c in self.search_service.client.get_collections().collections]
            if self.collection_name not in existing:
                embedding_size = self.config["embeddings"]["embedding_size"]
                self.search_service.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
                )
                logger.info(f"Created collection: {self.collection_name}")
            
            report_progress("db", 5, 100, "Metaandmete salvestamine andmebaasi...")
            
            # Insert document metadata into PostgreSQL
            query = """
                INSERT INTO documents (
                    pdf_filename, json_filename, title, short_name, publication, year, author, 
                    languages, is_valid, current_state, url, document_type, is_translation, valid_until
                ) 
                VALUES (
                    :fname, :json_fname, :title, :short_name, :publication, :year, :author, 
                    :languages, :is_valid, :current_state, :url, :document_type, :is_translation, :valid_until
                )
                RETURNING documents.id
            """
            json_filename = filename.rsplit(".", 1)[0] + ".json"
            
            # Parse languages
            languages = metadata.get("languages", [])
            if isinstance(languages, list):
                languages = ",".join(languages)
            
            # Parse valid_until date
            valid_until_date = None
            valid_until_str = metadata.get("valid_until", "")
            if valid_until_str:
                try:
                    valid_until_date = datetime.strptime(valid_until_str, "%Y-%m-%d").date()
                except ValueError:
                    valid_until_date = None
            
            data = [{
                "fname": filename,
                "json_fname": json_filename,
                "title": metadata.get("title", ""),
                "short_name": metadata.get("short_name", ""),
                "publication": metadata.get("publication", ""),
                "year": metadata.get("publication_year", 2024),
                "author": metadata.get("author", ""),
                "languages": languages,
                "is_valid": metadata.get("is_valid", True),
                "url": metadata.get("url", ""),
                "current_state": "processing",
                "document_type": metadata.get("document_type", "other"),
                "is_translation": metadata.get("is_translation", False),
                "valid_until": valid_until_date,
            }]
            result = con.execute_sql(query, data)
            doc_id = result["data"][0][0]
            
            # Insert keywords
            keywords = metadata.get("keywords", [])
            if keywords:
                kw_query = "INSERT INTO keywords (keyword, document_id) VALUES (:kw, :doc_id)"
                kw_data = [{"kw": kw, "doc_id": doc_id} for kw in keywords]
                con.execute_sql(kw_query, kw_data)
            
            con.commit()
            logger.info(f"Document {doc_id} metadata saved to PostgreSQL")
            report_progress("db", 10, 100, "Metaandmed salvestatud")
            
            # Extract text from PDF
            report_progress("extract", 10, 100, "PDF-ist teksti ekstraheerimine...")
            content = self._extract_text_from_pdf(pdf_bytes)
            logger.info(f"Extracted {len(content)} pages from PDF")
            report_progress("extract", 15, 100, f"{len(content)} lehekülge ekstraheeritud")
            
            # Create embeddings and upload to Qdrant
            # Include ALL metadata fields for filtering and display
            # valid_until is stored as Unix timestamp for Qdrant range filtering
            # Use NO_EXPIRATION_TIMESTAMP for documents without expiration
            valid_until_str = metadata.get("valid_until", "")
            if valid_until_str:
                try:
                    valid_date = datetime.strptime(valid_until_str, "%Y-%m-%d")
                    valid_until_ts = int(valid_date.timestamp())
                except ValueError:
                    valid_until_ts = NO_EXPIRATION_TIMESTAMP
            else:
                valid_until_ts = NO_EXPIRATION_TIMESTAMP
            
            languages = metadata.get("languages", [])
            document_metadata = {
                "title": metadata.get("title", ""),
                "short_name": metadata.get("short_name", ""),
                "publication": metadata.get("publication", ""),
                "year": metadata.get("publication_year", 2024),
                "author": metadata.get("author", ""),
                "languages": languages if isinstance(languages, str) else ",".join(languages) if languages else "",
                "url": metadata.get("url", ""),
                "document_type": metadata.get("document_type", "other"),
                "is_translation": metadata.get("is_translation", False),
                "is_valid": metadata.get("is_valid", True),
                "valid_until": valid_until_ts,  # Unix timestamp for Qdrant filtering
                "keywords": metadata.get("keywords", []),
            }
            
            num_chunks = self._create_embeddings_and_upload(document_metadata, content, progress_callback)
            logger.info(f"Uploaded {num_chunks} chunks to Qdrant")
            
            # Update status to uploaded
            con.execute_sql(
                "UPDATE documents SET current_state = 'uploaded' WHERE id = :doc_id",
                [{"doc_id": doc_id}]
            )
            con.commit()
            con.close()
            
            return {
                "status": "success",
                "message": f"Document uploaded: {num_chunks} chunks created",
                "document_id": doc_id,
            }
            
        except IntegrityError:
            con.session.rollback()
            con.close()
            return {
                "status": "error",
                "message": "Document with this filename already exists",
                "document_id": None,
            }
        except Exception as e:
            logger.error(f"Upload error: {e}")
            con.session.rollback()
            con.close()
            return {
                "status": "error",
                "message": str(e),
                "document_id": None,
            }

    # =========================================================================
    # Document Management Methods
    # =========================================================================

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents with chunk counts."""
        con = self._get_db_connection()
        try:
            query = """
                SELECT d.id, d.title, d.short_name, d.document_type, d.author, 
                       d.year as publication_year, d.is_valid, d.valid_until,
                       d.current_state
                FROM documents d
                ORDER BY d.id DESC
            """
            result = con.execute_sql(query, [])
            documents = []
            
            for row in result["data"]:
                doc = {
                    "id": row[0],
                    "title": row[1] or "",
                    "short_name": row[2] or "",
                    "document_type": row[3] or "other",
                    "author": row[4] or "",
                    "publication_year": row[5],
                    "is_valid": row[6] if row[6] is not None else True,
                    "valid_until": row[7].isoformat() if row[7] else None,
                    "current_state": row[8] or "",
                    "chunk_count": 0,
                }
                
                # Get chunk count from Qdrant
                try:
                    count_result = self.search_service.client.count(
                        collection_name=self.collection_name,
                        count_filter=Filter(must=[
                            FieldCondition(key="title", match=MatchValue(value=doc["title"]))
                        ]),
                    )
                    doc["chunk_count"] = count_result.count
                except Exception:
                    pass
                
                documents.append(doc)
            
            con.close()
            return documents
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            con.close()
            return []

    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Get document details by ID."""
        con = self._get_db_connection()
        try:
            query = """
                SELECT d.id, d.title, d.short_name, d.document_type, d.author, 
                       d.publication, d.year as publication_year, d.url, d.languages,
                       d.is_translation, d.is_valid, d.valid_until, d.current_state,
                       d.date_created, d.date_modified
                FROM documents d
                WHERE d.id = :doc_id
            """
            result = con.execute_sql(query, [{"doc_id": doc_id}])
            
            if not result["data"]:
                con.close()
                return None
            
            row = result["data"][0]
            doc = {
                "id": row[0],
                "title": row[1] or "",
                "short_name": row[2] or "",
                "document_type": row[3] or "other",
                "author": row[4] or "",
                "publication": row[5] or "",
                "publication_year": row[6],
                "url": row[7] or "",
                "languages": row[8] or "",
                "is_translation": row[9] if row[9] is not None else False,
                "is_valid": row[10] if row[10] is not None else True,
                "valid_until": row[11].isoformat() if row[11] else None,
                "current_state": row[12] or "",
                "date_created": row[13].isoformat() if row[13] else None,
                "date_modified": row[14].isoformat() if row[14] else None,
                "chunk_count": 0,
                "keywords": [],
            }
            
            # Get keywords
            kw_result = con.execute_sql(
                "SELECT keyword FROM keywords WHERE document_id = :doc_id",
                [{"doc_id": doc_id}]
            )
            doc["keywords"] = [r[0] for r in kw_result["data"]]
            
            # Get chunk count from Qdrant
            try:
                count_result = self.search_service.client.count(
                    collection_name=self.collection_name,
                    count_filter=Filter(must=[
                        FieldCondition(key="title", match=MatchValue(value=doc["title"]))
                    ]),
                )
                doc["chunk_count"] = count_result.count
            except Exception:
                pass
            
            con.close()
            return doc
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            con.close()
            return None

    def update_document(self, doc_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update document metadata in both PostgreSQL and Qdrant."""
        con = self._get_db_connection()
        try:
            # First get current document to get title for Qdrant sync
            current = self.get_document(doc_id)
            if not current:
                return {"status": "error", "message": "Document not found"}
            
            # Build update query dynamically
            update_fields = []
            params = {"doc_id": doc_id}
            
            field_mapping = {
                "title": "title",
                "short_name": "short_name",
                "document_type": "document_type",
                "author": "author",
                "publication": "publication",
                "publication_year": "year",
                "url": "url",
                "languages": "languages",
                "is_translation": "is_translation",
                "is_valid": "is_valid",
            }
            
            for key, db_field in field_mapping.items():
                if key in updates and updates[key] is not None:
                    update_fields.append(f"{db_field} = :{key}")
                    params[key] = updates[key]
            
            # Handle valid_until separately (date conversion)
            if "valid_until" in updates:
                update_fields.append("valid_until = :valid_until")
                if updates["valid_until"]:
                    try:
                        params["valid_until"] = datetime.strptime(updates["valid_until"], "%Y-%m-%d").date()
                    except ValueError:
                        params["valid_until"] = None
                else:
                    params["valid_until"] = None
            
            if update_fields:
                update_fields.append("date_modified = NOW()")
                query = f"UPDATE documents SET {', '.join(update_fields)} WHERE id = :doc_id"
                con.execute_sql(query, [params])
            
            # Handle keywords update
            if "keywords" in updates:
                # Delete existing keywords
                con.execute_sql("DELETE FROM keywords WHERE document_id = :doc_id", [{"doc_id": doc_id}])
                # Insert new keywords
                if updates["keywords"]:
                    kw_data = [{"kw": kw, "doc_id": doc_id} for kw in updates["keywords"]]
                    con.execute_sql("INSERT INTO keywords (keyword, document_id) VALUES (:kw, :doc_id)", kw_data)
            
            con.commit()
            
            # Sync updates to Qdrant
            chunks_updated = self._sync_document_to_qdrant(current["title"], updates)
            
            con.close()
            return {
                "status": "success",
                "message": f"Document updated, {chunks_updated} chunks synced",
            }
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            con.session.rollback()
            con.close()
            return {"status": "error", "message": str(e)}

    def _sync_document_to_qdrant(self, title: str, updates: Dict[str, Any]) -> int:
        """Sync document metadata updates to Qdrant chunks."""
        try:
            # Find all chunks with this title
            results = self.search_service.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=[
                    FieldCondition(key="title", match=MatchValue(value=title))
                ]),
                limit=10000,
                with_payload=False,
            )[0]
            
            if not results:
                return 0
            
            # Build payload updates
            payload_updates = {}
            if "title" in updates:
                payload_updates["title"] = updates["title"]
            if "short_name" in updates:
                payload_updates["short_name"] = updates["short_name"] or ""
            if "author" in updates:
                payload_updates["author"] = updates["author"] or ""
            if "publication" in updates:
                payload_updates["publication"] = updates["publication"] or ""
            if "publication_year" in updates:
                payload_updates["year"] = updates["publication_year"]
            if "url" in updates:
                payload_updates["url"] = updates["url"] or ""
            if "languages" in updates:
                payload_updates["languages"] = updates["languages"] or ""
            if "document_type" in updates:
                payload_updates["document_type"] = updates["document_type"] or "other"
            if "is_translation" in updates:
                payload_updates["is_translation"] = updates["is_translation"]
            if "is_valid" in updates:
                payload_updates["is_valid"] = updates["is_valid"]
            if "valid_until" in updates:
                if updates["valid_until"]:
                    try:
                        valid_date = datetime.strptime(updates["valid_until"], "%Y-%m-%d")
                        payload_updates["valid_until"] = int(valid_date.timestamp())
                    except ValueError:
                        payload_updates["valid_until"] = NO_EXPIRATION_TIMESTAMP
                else:
                    payload_updates["valid_until"] = NO_EXPIRATION_TIMESTAMP
            if "keywords" in updates:
                payload_updates["keywords"] = updates["keywords"] or []
            
            if not payload_updates:
                return 0
            
            # Update all chunks
            point_ids = [point.id for point in results]
            
            self.search_service.client.set_payload(
                collection_name=self.collection_name,
                payload=payload_updates,
                points=point_ids,
            )
            
            return len(point_ids)
        except Exception as e:
            logger.error(f"Error syncing to Qdrant: {e}")
            return 0

    def delete_document(self, doc_id: int) -> Dict[str, Any]:
        """Delete document from PostgreSQL and Qdrant."""
        con = self._get_db_connection()
        try:
            # Get document title first for Qdrant deletion
            doc = self.get_document(doc_id)
            if not doc:
                return {"status": "error", "message": "Document not found", "chunks_deleted": 0}
            
            title = doc["title"]
            
            # Delete from Qdrant first
            chunks_deleted = self._delete_from_qdrant(title)
            
            # Delete keywords
            con.execute_sql("DELETE FROM keywords WHERE document_id = :doc_id", [{"doc_id": doc_id}])
            
            # Delete document
            con.execute_sql("DELETE FROM documents WHERE id = :doc_id", [{"doc_id": doc_id}])
            
            con.commit()
            con.close()
            
            return {
                "status": "success",
                "message": f"Document deleted",
                "chunks_deleted": chunks_deleted,
            }
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            con.session.rollback()
            con.close()
            return {"status": "error", "message": str(e), "chunks_deleted": 0}

    def _delete_from_qdrant(self, title: str) -> int:
        """Delete all chunks with given title from Qdrant."""
        try:
            # First count how many we'll delete
            count_result = self.search_service.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(must=[
                    FieldCondition(key="title", match=MatchValue(value=title))
                ]),
            )
            count = count_result.count
            
            # Delete chunks
            self.search_service.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(must=[
                    FieldCondition(key="title", match=MatchValue(value=title))
                ]),
            )
            
            return count
        except Exception as e:
            logger.error(f"Error deleting from Qdrant: {e}")
            return 0
