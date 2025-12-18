"""
FastAPI Backend for KVA Terminiprojekt.

This backend handles:
- Vector/hybrid search
- LLM chat with context retrieval
- Health checks

Models are loaded ONCE at startup for efficiency.
"""
import os
import sys
import json
import logging
import queue
import threading
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path for imports
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(_parent_dir, ".env"))

from fastapi import File, UploadFile, Form

from backend.schemas import (
    SearchRequest, SearchResponse, SearchResult,
    ChatRequest, ChatResponse, TermEntry, DebugInfo, PipelineStepDebug,
    HealthResponse,
    FiltersResponse, DocumentInfo, KeywordInfo,
    UploadMetadata, UploadResponse,
    DocumentDetail, DocumentListItem, DocumentsListResponse, DocumentUpdate, DocumentDeleteResponse,
    PromptSetInfo, PromptSetDetail, PromptSetsListResponse,
    PromptDetail, PromptUpdate,
)
from backend.services import SearchService, ChatService, UploadService, PromptService
from utils.db_connection import Connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("backend")

# Global services (initialized once at startup)
search_service: SearchService = None
chat_service: ChatService = None
upload_service: UploadService = None
prompt_service: PromptService = None


def load_config() -> dict:
    """Load configuration from file."""
    config_path = os.getenv('APP_CONFIG')
    if not config_path or not os.path.exists(config_path):
        config_path = os.path.join(_parent_dir, 'config', 'config.json')
    
    with open(config_path, 'r') as f:
        return json.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager - handles startup and shutdown.
    Models are loaded here ONCE at startup.
    """
    global search_service, chat_service, upload_service, prompt_service
    
    logger.info("=" * 50)
    logger.info("🚀 KVA Backend - Starting up...")
    logger.info("=" * 50)
    
    # Load configuration
    config = load_config()
    logger.info("Configuration loaded")
    
    # Initialize services
    logger.info("Initializing prompt service...")
    prompt_service = PromptService()
    prompt_service.initialize()
    
    logger.info("Initializing search service...")
    search_service = SearchService(config)
    search_service.initialize()
    
    logger.info("Initializing chat service...")
    chat_service = ChatService(config, search_service, prompt_service)
    chat_service.initialize()
    
    logger.info("Initializing upload service...")
    upload_service = UploadService(config, search_service)
    
    logger.info("=" * 50)
    logger.info("✅ Backend ready!")
    logger.info("=" * 50)
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="KVA Terminiprojekt API",
    description="API for terminology search and analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get services
def get_search_service() -> SearchService:
    if search_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return search_service


def get_chat_service() -> ChatService:
    if chat_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return chat_service


def get_upload_service() -> UploadService:
    if upload_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return upload_service


def get_prompt_service() -> PromptService:
    if prompt_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return prompt_service


# ============== API Routes ==============

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(service: SearchService = Depends(get_search_service)):
    """Check if the backend is healthy and ready to serve requests."""
    config = load_config()
    return HealthResponse(
        status="healthy" if service.is_healthy() else "unhealthy",
        models_loaded=service._initialized,
        qdrant_connected=service.is_healthy(),
        hybrid_search_enabled=service.hybrid_enabled,
        reranking_enabled=service.reranking_enabled,
        reranker_model=config.get("reranking", {}).get("model") if service.reranking_enabled else None,
    )


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(
    request: SearchRequest,
    service: SearchService = Depends(get_search_service),
):
    """
    Search for relevant document chunks.
    
    Uses hybrid search (dense + sparse vectors) if enabled,
    otherwise falls back to dense-only search.
    """
    try:
        filters = request.filters or {}
        results = service.search(
            query=request.query,
            limit=filters.limit if hasattr(filters, 'limit') else 10,
            files=filters.files if hasattr(filters, 'files') else None,
            only_valid=filters.only_valid if hasattr(filters, 'only_valid') else False,
        )
        
        return SearchResponse(
            results=[SearchResult(**r) for r in results],
            query=request.query,
            total=len(results),
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
):
    """
    Process a terminology query using RAG (Retrieval Augmented Generation) with parallel extraction mode.
    
    1. Retrieves relevant context from the document database
    2. Generates a comprehensive term entry using LLM with specialized parallel prompts
    3. Returns both formatted text and structured data
    
    Parameters:
    - debug=True: See the full pipeline with each step's input/output
    - expand_query=True: Expand query per-category with category-specific prompts (each category gets its own expansion and search)
    - expand_context=True: Expand retrieved chunks with adjacent paragraphs
    """
    try:
        filters = request.filters or {}
        limit = filters.limit if hasattr(filters, 'limit') else 10
        files = filters.files if hasattr(filters, 'files') else None
        only_valid = filters.only_valid if hasattr(filters, 'only_valid') else False
        
        # Always use parallel extraction mode
        mode = "PARALLEL"
        if request.expand_query:
            mode += "+EXPANSION"
        if request.expand_context:
            mode += "+CONTEXT"
        if request.use_reranking:
            mode += "+RERANKING"
        logger.info(f"Using {mode} extraction mode for query: {request.query}")
        
        result = await service.chat_parallel(
            query=request.query,
            limit=limit,
            files=files,
            only_valid=only_valid,
            debug=request.debug,
            expand_query=request.expand_query,
            expand_context=request.expand_context,
            use_reranking=request.use_reranking,
            early_parallelization=request.early_parallelization,
            output_categories=request.output_categories,
            prompt_set_id=request.prompt_set_id,
        )
        
        # Convert term_entry dict to Pydantic model if present
        term_entry = None
        if result.get("term_entry"):
            try:
                term_entry = TermEntry(**result["term_entry"])
            except Exception as e:
                logger.warning(f"Failed to parse term_entry: {e}")
        
        # Convert debug_info dict to Pydantic model if present
        debug_info = None
        if result.get("debug_info"):
            try:
                steps = [PipelineStepDebug(**step) for step in result["debug_info"]["pipeline_steps"]]
                debug_info = DebugInfo(
                    total_duration_ms=result["debug_info"]["total_duration_ms"],
                    pipeline_steps=steps,
                    search_type=result["debug_info"].get("search_type", "unknown"),
                    model_used=result["debug_info"].get("model_used", ""),
                    embedding_model=result["debug_info"].get("embedding_model", ""),
                    extraction_mode=result["debug_info"].get("extraction_mode", "single"),
                )
            except Exception as e:
                logger.warning(f"Failed to parse debug_info: {e}")
        
        return ChatResponse(
            response=result["response"],
            query=request.query,
            context_used=result["context_used"],
            term_entry=term_entry,
            debug_info=debug_info,
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/filters", response_model=FiltersResponse, tags=["Filters"])
async def get_filters():
    """
    Get available filter options (documents and keywords).
    """
    try:
        con = Connection(
            host=os.getenv("PG_HOST"),
            port=os.getenv("PG_PORT"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            db=os.getenv("PG_COLLECTION"),
        )
        con.establish_connection()
        
        # Get documents
        docs_df = con.statement_to_df(
            "SELECT id, title FROM documents WHERE current_state = 'uploaded' ORDER BY title"
        )
        documents = [
            DocumentInfo(id=row['id'], title=row['title'])
            for _, row in docs_df.iterrows()
        ]
        
        # Get keywords
        kw_df = con.statement_to_df(
            "SELECT DISTINCT keyword FROM keywords ORDER BY keyword"
        )
        keywords = [
            KeywordInfo(keyword=row['keyword'])
            for _, row in kw_df.iterrows()
        ]
        
        con.close()
        
        return FiltersResponse(documents=documents, keywords=keywords)
    except Exception as e:
        logger.error(f"Filters error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(..., description="PDF file to upload"),
    document_type: str = Form("other"),
    title: str = Form(""),
    short_name: str = Form(""),
    url: str = Form(""),
    author: str = Form(""),
    publication: str = Form(""),
    publication_year: int = Form(2024),
    languages: str = Form(""),  # Comma-separated
    is_translation: bool = Form(False),
    keywords: str = Form(""),   # Comma-separated
    is_valid: bool = Form(True),
    valid_until: str = Form(""),  # Date string YYYY-MM-DD or empty
    service: UploadService = Depends(get_upload_service),
):
    """
    Upload a PDF document with metadata.
    Creates embeddings and stores in Qdrant.
    """
    try:
        # Read file content
        pdf_bytes = await file.read()
        
        # Parse comma-separated fields
        lang_list = [l.strip() for l in languages.split(",") if l.strip()]
        kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
        
        metadata = {
            "document_type": document_type,
            "title": title,
            "short_name": short_name,
            "url": url,
            "author": author,
            "publication": publication,
            "publication_year": publication_year,
            "languages": lang_list,
            "is_translation": is_translation,
            "keywords": kw_list,
            "is_valid": is_valid,
            "valid_until": valid_until,
        }
        
        result = service.upload_document(
            pdf_bytes=pdf_bytes,
            filename=file.filename,
            metadata=metadata,
        )
        
        return UploadResponse(**result)
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload-stream", tags=["Documents"])
async def upload_document_stream(
    file: UploadFile = File(..., description="PDF file to upload"),
    document_type: str = Form("other"),
    title: str = Form(""),
    short_name: str = Form(""),
    url: str = Form(""),
    author: str = Form(""),
    publication: str = Form(""),
    publication_year: int = Form(2024),
    languages: str = Form(""),  # Comma-separated
    is_translation: bool = Form(False),
    keywords: str = Form(""),   # Comma-separated
    is_valid: bool = Form(True),
    valid_until: str = Form(""),  # Date string YYYY-MM-DD or empty
    service: UploadService = Depends(get_upload_service),
):
    """
    Upload a PDF document with SSE progress streaming.
    Returns Server-Sent Events with progress updates.
    """
    # Read file content first
    pdf_bytes = await file.read()
    filename = file.filename
    
    # Parse comma-separated fields
    lang_list = [l.strip() for l in languages.split(",") if l.strip()]
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
    
    metadata = {
        "document_type": document_type,
        "title": title,
        "short_name": short_name,
        "url": url,
        "author": author,
        "publication": publication,
        "publication_year": publication_year,
        "languages": lang_list,
        "is_translation": is_translation,
        "keywords": kw_list,
        "is_valid": is_valid,
        "valid_until": valid_until,
    }
    
    # Progress queue for communication between threads
    progress_queue = queue.Queue()
    result_holder = {"result": None, "error": None}
    
    def progress_callback(stage: str, current: int, total: int, message: str):
        """Send progress updates to the queue."""
        # Calculate overall percentage (0-100)
        # Stages: init(0-5), db(5-10), extract(10-15), chunking(15-20), 
        #         embedding(20-80), sparse(80-85), upload(85-100)
        stage_weights = {
            "init": (0, 5),
            "db": (5, 10),
            "extract": (10, 15),
            "chunking": (15, 20),
            "embedding": (20, 80),
            "sparse": (80, 85),
            "points": (85, 87),
            "upload": (87, 99),
            "done": (100, 100),
        }
        
        if stage in stage_weights:
            start, end = stage_weights[stage]
            if total > 0:
                stage_progress = current / total
            else:
                stage_progress = 1.0
            percentage = int(start + (end - start) * stage_progress)
        else:
            percentage = current
        
        progress_queue.put({
            "stage": stage,
            "percentage": min(percentage, 100),
            "message": message,
        })
    
    def upload_worker():
        """Worker thread that performs the upload."""
        try:
            result = service.upload_document(
                pdf_bytes=pdf_bytes,
                filename=filename,
                metadata=metadata,
                progress_callback=progress_callback,
            )
            result_holder["result"] = result
        except Exception as e:
            result_holder["error"] = str(e)
        finally:
            progress_queue.put(None)  # Signal completion
    
    def generate_sse():
        """Generator that yields SSE events."""
        # Start upload in background thread
        upload_thread = threading.Thread(target=upload_worker)
        upload_thread.start()
        
        try:
            while True:
                try:
                    progress = progress_queue.get(timeout=60)
                    if progress is None:
                        # Upload complete
                        break
                    
                    # Send SSE event
                    event_data = json.dumps(progress)
                    yield f"data: {event_data}\n\n"
                    
                except queue.Empty:
                    # Timeout - send keepalive
                    yield f"data: {json.dumps({'stage': 'keepalive', 'percentage': -1, 'message': 'Processing...'})}\n\n"
            
            # Wait for thread to finish
            upload_thread.join(timeout=5)
            
            # Send final result
            if result_holder["error"]:
                final = {
                    "stage": "error",
                    "percentage": 0,
                    "message": result_holder["error"],
                    "status": "error",
                }
            else:
                result = result_holder["result"]
                final = {
                    "stage": "complete",
                    "percentage": 100,
                    "message": result.get("message", "Valmis!"),
                    "status": result.get("status", "success"),
                    "document_id": result.get("document_id"),
                }
            
            yield f"data: {json.dumps(final)}\n\n"
            
        except Exception as e:
            logger.error(f"SSE error: {e}")
            yield f"data: {json.dumps({'stage': 'error', 'percentage': 0, 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


# ============== Document Management Routes ==============

@app.get("/documents", response_model=DocumentsListResponse, tags=["Documents"])
async def list_documents(service: UploadService = Depends(get_upload_service)):
    """List all documents with their metadata and chunk counts."""
    documents = service.list_documents()
    return DocumentsListResponse(
        documents=[DocumentListItem(**d) for d in documents],
        total=len(documents),
    )


@app.get("/documents/{doc_id}", response_model=DocumentDetail, tags=["Documents"])
async def get_document(
    doc_id: int,
    service: UploadService = Depends(get_upload_service),
):
    """Get detailed information about a specific document."""
    doc = service.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return DocumentDetail(**doc)


@app.put("/documents/{doc_id}", tags=["Documents"])
async def update_document(
    doc_id: int,
    updates: DocumentUpdate,
    service: UploadService = Depends(get_upload_service),
):
    """
    Update document metadata.
    Changes are synced to both PostgreSQL and Qdrant.
    """
    # Convert to dict, excluding None values
    update_dict = {k: v for k, v in updates.model_dump().items() if v is not None}
    
    if not update_dict:
        raise HTTPException(status_code=400, detail="No updates provided")
    
    result = service.update_document(doc_id, update_dict)
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result


@app.delete("/documents/{doc_id}", response_model=DocumentDeleteResponse, tags=["Documents"])
async def delete_document(
    doc_id: int,
    service: UploadService = Depends(get_upload_service),
):
    """
    Delete a document from both PostgreSQL and Qdrant.
    """
    result = service.delete_document(doc_id)
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return DocumentDeleteResponse(**result)


# ============== Prompt Management Routes (Simplified) ==============

@app.get("/prompt-sets", response_model=PromptSetsListResponse, tags=["Prompts"])
async def list_prompt_sets(service: PromptService = Depends(get_prompt_service)):
    """
    List all available prompt sets.
    Used to find the default prompt set ID.
    """
    prompt_sets = service.get_all_prompt_sets()
    return PromptSetsListResponse(prompt_sets=[PromptSetInfo(**s) for s in prompt_sets])


@app.get("/prompt-sets/{set_id}", response_model=PromptSetDetail, tags=["Prompts"])
async def get_prompt_set(
    set_id: int,
    service: PromptService = Depends(get_prompt_service),
):
    """
    Get a specific prompt set with its prompts.
    """
    prompt_set = service.get_prompt_set(set_id)
    if not prompt_set:
        raise HTTPException(status_code=404, detail=f"Prompt set {set_id} not found")
    return PromptSetDetail(**prompt_set)


@app.get("/prompt-sets/{set_id}/prompts/{prompt_type}", response_model=PromptDetail, tags=["Prompts"])
async def get_prompt(
    set_id: int,
    prompt_type: str,
    service: PromptService = Depends(get_prompt_service),
):
    """
    Get a specific prompt by type within a set.
    """
    prompt = service.get_prompt(prompt_type, set_id=set_id)
    if not prompt:
        raise HTTPException(status_code=404, detail=f"Prompt type '{prompt_type}' not found in set {set_id}")
    return PromptDetail(**prompt)


@app.put("/prompt-sets/{set_id}/prompts/{prompt_type}", tags=["Prompts"])
async def update_prompt(
    set_id: int,
    prompt_type: str,
    request: PromptUpdate,
    service: PromptService = Depends(get_prompt_service),
):
    """
    Update a prompt within a set.
    """
    result = service.update_prompt(
        prompt_type=prompt_type,
        prompt_text=request.prompt_text,
        description=request.description,
        set_id=set_id,
    )
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("message"))
    return result


# Run with: uvicorn backend.main:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
