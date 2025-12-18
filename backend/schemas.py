"""
Pydantic schemas for API request/response models.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


# ============================================================================
# Term Entry Structured Response Models
# ============================================================================

# Confidence levels for match quality transparency
# "direct" = exact term found and explicitly defined/used in source
# "strong" = term clearly discussed but definition may be implicit  
# "inferred" = information derived from context about related concepts
# "expanded" = found via query expansion (synonym/related term search)

class Definition(BaseModel):
    """A single definition with source information."""
    text: str = Field(..., description="The definition text")
    source: str = Field(..., description="Document title")
    page: int = Field(..., description="Page number")
    url: str = Field(default="", description="Document URL for linking")
    confidence: str = Field(default="direct", description="Match confidence: direct, strong, inferred, expanded")


class RelatedTerm(BaseModel):
    """A related term with its relationship type and source."""
    term: str = Field(..., description="The related term")
    relation_type: str = Field(..., description="Type of relation: synonym, broader, narrower, abbreviation, other")
    source: str = Field(..., description="Document title")
    page: int = Field(..., description="Page number")
    url: str = Field(default="", description="Document URL for linking")
    confidence: str = Field(default="direct", description="Match confidence: direct, strong, inferred, expanded")


class UsageEvidence(BaseModel):
    """A usage example/context with source information."""
    text: str = Field(..., description="The contextual paragraph")
    source: str = Field(..., description="Document title")
    page: int = Field(..., description="Page number")
    url: str = Field(default="", description="Document URL for linking")
    confidence: str = Field(default="direct", description="Match confidence: direct, strong, inferred, expanded")
    context: str = Field(default="", description="Brief context explaining the usage")


class MatchQuality(BaseModel):
    """Summary of overall match quality for user awareness."""
    overall_confidence: str = Field(..., description="Overall match quality: high, medium, low")
    direct_matches: int = Field(default=0, description="Number of direct/exact matches")
    inferred_matches: int = Field(default=0, description="Number of inferred/expanded matches")
    expanded_query_used: bool = Field(default=False, description="Whether query expansion was used")
    warning_message: Optional[str] = Field(default=None, description="Warning if results are mostly inferred")


class TermEntry(BaseModel):
    """Complete structured term entry from LLM analysis."""
    term: str = Field(..., description="The keyword/term being analyzed")
    definitions: List[Definition] = Field(default_factory=list, description="List of definitions found")
    related_terms: List[RelatedTerm] = Field(default_factory=list, description="List of related terms")
    usage_evidence: List[UsageEvidence] = Field(default_factory=list, description="Usage examples")
    see_also: List[str] = Field(default_factory=list, description="Terms for further exploration")
    match_quality: Optional[MatchQuality] = Field(default=None, description="Summary of result quality")


# ============================================================================
# Search and Filter Models
# ============================================================================

class SearchFilters(BaseModel):
    """Filters for search queries."""
    files: List[str] = Field(default_factory=list, description="Filter by document titles")
    limit: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    only_valid: bool = Field(default=False, description="Only search valid documents")


class SearchRequest(BaseModel):
    """Request body for search endpoint."""
    query: str = Field(..., min_length=1, description="Search query text")
    filters: Optional[SearchFilters] = None


class SearchResult(BaseModel):
    """A single search result."""
    text: str
    title: str
    page_number: int
    score: float
    content_type: Optional[str] = None


class SearchResponse(BaseModel):
    """Response from search endpoint."""
    results: List[SearchResult]
    query: str
    total: int


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    query: str = Field(..., min_length=1, description="User query/keyword")
    filters: Optional[SearchFilters] = None
    prompt_set_id: Optional[int] = Field(default=None, description="ID of prompt set to use (uses default if not specified)")
    debug: bool = Field(default=False, description="Enable debug mode to see full pipeline details")
    expand_query: bool = Field(default=True, description="Expand query per-category with category-specific prompts (definitions, related_terms, usage_evidence). Each category gets its own expansion and search for better results.")
    expand_context: bool = Field(default=False, description="Expand retrieved chunks with adjacent paragraphs for fuller context")
    use_reranking: bool = Field(default=True, description="Use cross-encoder reranking to improve search result relevance")
    early_parallelization: bool = Field(default=True, description="Enable early per-category query expansion/search before running parallel extractions")
    output_categories: List[str] = Field(
        default=["definitions", "related_terms", "usage_evidence"],
        description="Categories to include in output: definitions, related_terms, usage_evidence"
    )


class PipelineStepDebug(BaseModel):
    """Debug information for a single pipeline step."""
    step_number: int
    step_name: str
    description: str
    data: Optional[str] = Field(default=None, description="Step output data (may be truncated)")
    duration_ms: Optional[float] = Field(default=None, description="Time taken for this step in milliseconds")


class DebugInfo(BaseModel):
    """Complete debug information for a query pipeline."""
    total_duration_ms: float = Field(description="Total pipeline duration in milliseconds")
    pipeline_steps: List[PipelineStepDebug] = Field(default_factory=list)
    search_type: str = Field(description="Type of search used: 'hybrid' or 'dense'")
    model_used: str = Field(default="", description="LLM model used")
    embedding_model: str = Field(default="", description="Embedding model used")
    extraction_mode: str = Field(default="single", description="Extraction mode: 'single' or 'parallel'")


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    response: str = Field(..., description="Formatted response text for display")
    query: str
    context_used: int = Field(description="Number of context chunks used")
    term_entry: Optional[TermEntry] = Field(default=None, description="Structured term entry data")
    debug_info: Optional[DebugInfo] = Field(default=None, description="Debug information (only when debug=True)")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: bool
    qdrant_connected: bool
    hybrid_search_enabled: bool = False
    reranking_enabled: bool = False
    reranker_model: Optional[str] = None


class DocumentInfo(BaseModel):
    """Document information."""
    id: int
    title: str


class KeywordInfo(BaseModel):
    """Keyword information."""
    keyword: str


class FiltersResponse(BaseModel):
    """Available filters response."""
    documents: List[DocumentInfo]
    keywords: List[KeywordInfo]


class UploadMetadata(BaseModel):
    """Metadata for document upload."""
    publication: str = ""
    publication_year: int = 2024
    title: str = ""
    author: str = ""
    languages: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    url: str = ""
    is_valid: bool = True


class UploadResponse(BaseModel):
    """Response from upload endpoint."""
    status: str
    message: str
    document_id: Optional[int] = None


# ============================================================================
# Document Management Models
# ============================================================================

class DocumentDetail(BaseModel):
    """Full document details for viewing/editing."""
    id: int
    title: str
    short_name: Optional[str] = None
    document_type: Optional[str] = None
    author: Optional[str] = None
    publication: Optional[str] = None
    publication_year: Optional[int] = None
    url: Optional[str] = None
    languages: Optional[str] = None
    is_translation: bool = False
    keywords: List[str] = Field(default_factory=list)
    is_valid: bool = True
    valid_until: Optional[str] = None
    current_state: Optional[str] = None
    date_created: Optional[str] = None
    date_modified: Optional[str] = None
    chunk_count: int = 0


class DocumentListItem(BaseModel):
    """Document summary for list view."""
    id: int
    title: str
    short_name: Optional[str] = None
    document_type: Optional[str] = None
    author: Optional[str] = None
    publication_year: Optional[int] = None
    is_valid: bool = True
    valid_until: Optional[str] = None
    current_state: Optional[str] = None
    chunk_count: int = 0


class DocumentsListResponse(BaseModel):
    """Response listing all documents."""
    documents: List[DocumentListItem]
    total: int


class DocumentUpdate(BaseModel):
    """Request body for updating document metadata."""
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    short_name: Optional[str] = Field(None, max_length=100)
    document_type: Optional[str] = None
    author: Optional[str] = Field(None, max_length=255)
    publication: Optional[str] = Field(None, max_length=255)
    publication_year: Optional[int] = Field(None, ge=1900, le=2100)
    url: Optional[str] = Field(None, max_length=255)
    languages: Optional[str] = Field(None, max_length=255)
    is_translation: Optional[bool] = None
    keywords: Optional[List[str]] = None
    is_valid: Optional[bool] = None
    valid_until: Optional[str] = None  # Date string YYYY-MM-DD or empty to clear


class DocumentDeleteResponse(BaseModel):
    """Response from document delete endpoint."""
    status: str
    message: str
    chunks_deleted: int = 0


# ============================================================================
# Prompt Models (Simplified - single prompt set)
# ============================================================================

class PromptSetInfo(BaseModel):
    """Prompt set information for listing."""
    id: int
    name: str
    description: Optional[str] = None
    is_default: bool = False
    prompt_count: int = 0
    date_modified: Optional[str] = None


class PromptInfo(BaseModel):
    """Prompt information for listing."""
    id: int
    prompt_type: str
    description: Optional[str] = None
    date_modified: Optional[str] = None


class PromptSetDetail(BaseModel):
    """Full prompt set detail with its prompts."""
    id: int
    name: str
    description: Optional[str] = None
    is_default: bool = False
    date_created: Optional[str] = None
    date_modified: Optional[str] = None
    prompts: List[PromptInfo] = Field(default_factory=list)


class PromptDetail(BaseModel):
    """Full prompt detail including text."""
    id: int
    prompt_type: str
    prompt_text: str
    description: Optional[str] = None
    date_created: Optional[str] = None
    date_modified: Optional[str] = None


class PromptUpdate(BaseModel):
    """Request body for updating a prompt."""
    prompt_text: str = Field(..., min_length=10, max_length=10000, description="The new prompt text")
    description: Optional[str] = Field(None, max_length=500, description="Optional description")


class PromptSetsListResponse(BaseModel):
    """Response listing all available prompt sets."""
    prompt_sets: List[PromptSetInfo]
