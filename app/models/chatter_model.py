import asyncio
import json
import logging
from collections import defaultdict
from typing import Optional, Dict, Any, List
from urllib.parse import quote

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    Prefetch,
    FusionQuery,
    Fusion,
    SparseVector,
)
from sentence_transformers import SentenceTransformer
import re
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Configure logging
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)


class FilterFactory:
    """
    A class for creating and managing filters for the Qdrant search engine.

    apply_chunk_validity_filter = True
    apply_file_filter = False
    apply_document_validity_filter = False
    Attributes:
        apply_chunk_validity_filter (bool): Whether to apply a filter for chunk validity.
        apply_file_filter (bool): Whether to apply a filter for specific documents.
        apply_document_validity_filter (bool): Whether to apply a filter for document validity.
        target_files (list): A list of target file names for the file_filter.
        response_limit (int): The maximum number of search results to return.
        chunk_validity_filter (FieldCondition): A filter condition for chunk validity.
        document_validity_filter (FieldCondition): A filter condition for filtering out outdated documents.
        file_filter (FieldCondition): A filter condition for querying only specific files.
    """

    apply_chunk_validity_filter = True
    apply_file_filter = False
    apply_document_validity_filter = False

    target_files = []
    response_limit = 5

    chunk_validity_filter = FieldCondition(
        key="validated",
        match=MatchValue(
            value=True,
        ),
    )

    document_validity_filter = FieldCondition(
        key="is_valid",
        match=MatchValue(
            value=True,
        ),
    )

    file_filter = FieldCondition(
        key="title",
        match=MatchAny(any=target_files),
    )

    def apply_filters(
        self, files: list, response_limit: int, document_validity: bool
    ) -> None:
        """Coordinates modfication of filter settings."""
        self.update_file_filter_target_files(files)
        self.set_response_limit(response_limit)
        self.set_document_validity(document_validity)

    def update_file_filter_target_files(self, files: list) -> None:
        """Updates the target files for the file filter."""
        self.target_files = files
        if files:
            self.file_filter.match = MatchAny(any=self.target_files)
            self.apply_file_filter = True
        else:
            self.apply_file_filter = False

    def set_response_limit(self, response_limit: int) -> None:
        """Sets the maximum number of search results to return."""
        self.response_limit = response_limit

    def set_document_validity(self, document_validity: bool) -> None:
        """Sets whether to apply the document validity filter."""
        self.apply_document_validity_filter = document_validity

    def assemble_filter(self) -> Filter:
        """Assembles and returns the final filter based on the current settings."""
        conditions = []
        if self.apply_file_filter:
            conditions.append(self.file_filter)
        if self.apply_chunk_validity_filter:
            conditions.append(self.chunk_validity_filter)
        if self.apply_document_validity_filter:
            conditions.append(self.document_validity_filter)
        logger.info(f"Filter assembled: {Filter(must=conditions)}")
        return Filter(must=conditions)


class Retriever:
    """
    A class for interacting with a Qdrant vector search engine to retrieve relevant responses based on user input.
    Supports both pure vector search and hybrid search (vector + keyword/BM25).

    Attributes:
        model (SentenceTransformer): The pre-trained sentence transformer model for encoding text.
        sparse_model: The sparse embedding model for keyword-based search (BM25/SPLADE).
        collection_name (str): The name of the Qdrant collection to search.
        filterfactory (FilterFactory): An instance of the FilterFactory class for managing filters for search results.
        prompt (str): A prompt to prepend to the user input before encoding.
        host (str): The hostname or IP address of the Qdrant server.
        port (int): The port number of the Qdrant server.
        client (QdrantClient): The Qdrant client instance for interacting with the server.
        hybrid_enabled (bool): Whether hybrid search is enabled.
    """

    def __init__(
        self,
        model_name: str,
        collection_name: str,
        filterfactory: FilterFactory,
        prompt: str,
        hybrid_config: Optional[dict] = None,
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.collection_name = collection_name
        self.filterfactory = filterfactory
        self.prompt = prompt
        
        # Initialize hybrid search settings
        self.hybrid_enabled = False
        self.sparse_model = None
        
        if hybrid_config and hybrid_config.get("enabled", False):
            self._init_sparse_model(hybrid_config.get("sparse_model", "Qdrant/bm25"))
            self.hybrid_enabled = True
            logger.info(f"Hybrid search enabled with sparse model: {hybrid_config.get('sparse_model')}")

    def _init_sparse_model(self, sparse_model_name: str) -> None:
        """Initialize the sparse embedding model for keyword-based search."""
        try:
            from fastembed import SparseTextEmbedding
            self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name)
            logger.info(f"Sparse model {sparse_model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sparse model: {e}")
            self.hybrid_enabled = False

    def connect(self, host: str, port: int) -> None:
        """Connects to the Qdrant server using the provided host and port."""
        self.host = host
        self.port = port
        self.client = QdrantClient(host, port=port)

    async def retrieve_context(self, contents: str, user, instance) -> str:
        """A callback function for handling user input and generating LLM context information based on retrieved documents."""
        await asyncio.sleep(0.7)
        db_filter = self.filterfactory.assemble_filter()
        logger.info("Filter assembled")
        try:
            # Get similar documents using hybrid or pure vector search
            if self.hybrid_enabled and self.sparse_model is not None:
                try:
                    results = self.get_hybrid_similarities(self.prompt + contents, contents, db_filter)
                except Exception as hybrid_error:
                    # Fallback to dense-only search if hybrid fails
                    # (e.g., collection doesn't have sparse vectors configured)
                    logger.warning(f"Hybrid search failed, falling back to dense search: {hybrid_error}")
                    results = self.get_similarities(self.prompt + contents, db_filter)
            else:
                results = self.get_similarities(self.prompt + contents, db_filter)

            # Assemble response string
            response = ""
            for i in range(len(results["response_text"])):
                response += f'Title: {results["title"][i]}\n'
                response += f'Page: {results["page_no"][i]}\n'
                if results.get("url") and results["url"][i]:
                    response += f'URL: {results["url"][i]}\n'
                response += f'\n{results["response_text"][i]}\n{"*"*15}\n'

            return response
        except Exception as e:
            logger.info(e)
            return 'Response missing'

    def _encode_sparse(self, text: str) -> SparseVector:
        """
        Encodes text into a sparse vector using the BM25/SPLADE model.
        
        Args:
            text (str): The text to encode.
            
        Returns:
            SparseVector: The sparse vector representation.
        """
        embeddings = list(self.sparse_model.embed([text]))[0]
        return SparseVector(
            indices=embeddings.indices.tolist(),
            values=embeddings.values.tolist()
        )

    def get_hybrid_similarities(self, dense_text: str, sparse_text: str, query_filter: Filter) -> dict:
        """
        Performs hybrid search combining dense vector search and sparse keyword search using RRF fusion.

        Args:
            dense_text (str): The text with prompt prefix for dense embedding.
            sparse_text (str): The raw text for sparse embedding (without prompt).
            query_filter (Filter): A dictionary specifying the filter criteria for the search.

        Returns:
            dict: A dictionary containing lists of response texts, response types, scores, filenames, and page numbers.
        """
        logger.info("Performing hybrid search (dense + sparse)")
        
        # Encode dense vector
        dense_vector = list(
            self.model.encode(dense_text, normalize_embeddings=True).astype(float)
        )
        
        # Encode sparse vector
        sparse_vector = self._encode_sparse(sparse_text)
        
        # Perform hybrid search with prefetch and RRF fusion
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                # Dense vector search prefetch
                Prefetch(
                    query=dense_vector,
                    using="dense",
                    filter=query_filter,
                    limit=self.filterfactory.response_limit * 2,
                ),
                # Sparse vector search prefetch
                Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    filter=query_filter,
                    limit=self.filterfactory.response_limit * 2,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=self.filterfactory.response_limit,
            timeout=100,
        )
        logger.info("Hybrid search results retrieved")

        result_dict = defaultdict(list)

        for point in search_result.points:
            if not point.payload:
                continue
            result_dict["response_text"].append(point.payload["text"])
            result_dict["response_type"].append(point.payload["content_type"])
            result_dict["score"].append(point.score)
            result_dict["filename"].append(point.payload["filename"])
            result_dict["page_no"].append(point.payload["page_number"])
            result_dict["title"].append(point.payload["title"])
            result_dict["url"].append(point.payload.get("url", ""))

        return result_dict

    def get_similarities(self, text: str, query_filter: Filter) -> dict:
        """
        Searches for similar documents in the collection based on the given text and query filter.
        Falls back to this method when hybrid search is disabled or unavailable.

        Args:
            text (str): The text to search for similar documents.
            query_filter (Filter): A dictionary specifying the filter criteria for the search.

        Returns:
            dict: A dictionary containing lists of response texts, response types, scores, filenames, and page numbers.
        """

        logger.info("Searching for similar documents (dense vectors only)")
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=list(
                self.model.encode(text, normalize_embeddings=True).astype(float)
            ),
            query_filter=query_filter,
            limit=self.filterfactory.response_limit,
            timeout=100,
        )
        logger.info("Search results retrieved")

        result_dict = defaultdict(list)

        for point in search_result:
            if not point.payload:
                continue
            result_dict["response_text"].append(point.payload["text"])
            result_dict["response_type"].append(point.payload["content_type"])
            result_dict["score"].append(point.score)
            result_dict["filename"].append(point.payload["filename"])
            result_dict["page_no"].append(point.payload["page_number"])
            result_dict["title"].append(point.payload["title"])
            result_dict["url"].append(point.payload.get("url", ""))

        return result_dict


class LLMChat:
    """
    A class for interacting with large language models (LLMs) such as OpenAI's GPT or Anthropic's Claude.
    Returns structured JSON from LLM and formats it for display.

    Attributes:
        model_name (str): The name of the LLM model to use (e.g., "gpt-3.5", "claude-3.5-sonnet-latest").
        retriever (Retriever): A retriever instance for fetching contextual data based on user input.
        api_key (str): The API key for authenticating requests to the LLM provider.
        chat_template (ChatPromptTemplate): A predefined template for structuring chat messages sent to the LLM.
        temperature (float): Controls the randomness of the model's responses (default is 0).
        max_retries (int): The maximum number of retry attempts for failed API calls (default is 2).
        llm (ChatAnthropic or ChatOpenAI): The connected LLM client instance for generating responses.
    """

    # Updated prompt to request JSON output
    # NOTE: Curly braces are doubled to escape them from ChatPromptTemplate formatting
    SYSTEM_PROMPT = """**Role:** You are a terminologist searching for terminological information about a keyword.

**Objective:** You've collected key sections from various documents about the keyword. Your task is to analyze these sections and extract terminological information. ALWAYS use EXACT QUOTES. Focus on linguistic accuracy.

**Instructions:** Extract and organize the following information:

1. **Definitions** - Extract ALL definitions describing the term. Keep ORIGINAL WORDING. If a definition appears in multiple documents, note all sources.

2. **Related terms** - Identify:
   - Synonyms (spelling/form variations, precise synonyms)
   - Broader terms (keyword is a type/part/subcategory of this)
   - Narrower terms (this is a type/part/subcategory of keyword)
   - Abbreviations
   - Other related terms (frequently appearing in same contexts)

3. **Usage evidence** - Find paragraphs containing the keyword that show domain-specific applications or usage patterns. Must be coherent and include the keyword.

4. **See also** - Terms, abbreviations, or synonyms useful for further exploration.

**CRITICAL: You MUST respond with valid JSON only. No markdown, no explanations, just the JSON object.**

**Output Format (respond with this exact JSON structure):**
```json
{{
  "term": "the keyword being analyzed",
  "definitions": [
    {{"text": "definition text here", "source": "Document Title", "page": 1, "url": "document url if provided"}}
  ],
  "related_terms": [
    {{"term": "related term", "relation_type": "synonym|broader|narrower|abbreviation|other", "source": "Document Title", "page": 1, "url": "document url if provided"}}
  ],
  "usage_evidence": [
    {{"text": "contextual paragraph text", "source": "Document Title", "page": 1, "url": "document url if provided"}}
  ],
  "see_also": ["term1", "term2", "term3"]
}}
```

Remember:
- Use exact quotes from the source documents
- Include page numbers and URLs from the sources (if URL is provided in the key sections)
- relation_type must be one of: synonym, broader, narrower, abbreviation, other
- If no items found for a category, use an empty array []
- Response must be valid JSON only"""

    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Keyword: {user_input}"),
            ("human", "Key sections:\n{retrieval_results}"),
        ]
    )

    def __init__(
        self,
        model_name: str,
        retriever: Retriever,
        api_key,
        chat_template=None,
    ) -> None:
        self.retriever = retriever
        self.model_name = model_name
        self.temperature = 0
        self.max_retries = 2
        self.api_key = api_key
        self.llm = self.connect_language_model()
        self.chat_template = chat_template if chat_template else LLMChat.chat_template

    def connect_language_model(self):
        if re.match("claude", self.model_name):
            llm = ChatAnthropic(
                model_name=self.model_name,
                temperature=self.temperature,
                stop=None,
                base_url="https://api.anthropic.com",
                api_key=self.api_key,
                timeout=None,
                max_retries=self.max_retries,
            )

        elif re.match("gpt", self.model_name):
            llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=None,
                timeout=None,
                max_retries=self.max_retries,
                api_key=self.api_key,
            )

        return llm

    def _parse_llm_json(self, content: str, query: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response, handling potential formatting issues.
        
        Args:
            content: Raw LLM response content
            query: Original query (used as fallback term)
            
        Returns:
            Parsed term entry dictionary
        """
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
    
    def _format_term_entry(self, term_entry: Dict[str, Any]) -> str:
        """
        Format structured term entry into human-readable markdown text.
        
        Args:
            term_entry: Parsed term entry dictionary
            
        Returns:
            Formatted markdown string with clickable source links
        """
        lines = []
        
        # Term header
        lines.append(f"**{term_entry['term']}**")
        lines.append("")
        
        # Definitions
        if term_entry["definitions"]:
            lines.append("**Definitions:**")
            for i, defn in enumerate(term_entry["definitions"], 1):
                text = defn.get('text', '')
                source_link = self._format_source_link(defn, text)
                lines.append(f"  {i}. {text} {source_link}")
            lines.append("")
        
        # Related terms
        if term_entry["related_terms"]:
            lines.append("**Related Terms:**")
            for i, rel in enumerate(term_entry["related_terms"], 1):
                term = rel.get('term', '')
                relation = rel.get("relation_type", "related")
                source_link = self._format_source_link(rel, term)
                lines.append(f"  {i}. {term} ({relation}) {source_link}")
            lines.append("")
        
        # Usage evidence
        if term_entry["usage_evidence"]:
            lines.append("**Usage Evidence:**")
            for i, usage in enumerate(term_entry["usage_evidence"], 1):
                text = usage.get('text', '')
                source_link = self._format_source_link(usage, text)
                lines.append(f"  {i}. {text} {source_link}")
            lines.append("")
        
        # See also
        if term_entry["see_also"]:
            see_also_str = "; ".join(term_entry["see_also"])
            lines.append(f"**See Also:** {see_also_str}")
        
        return "\n".join(lines)

    async def chat_callback(
        self,
        contents: str,
    ):
        """A callback function for handling user input and generating responses."""
        await asyncio.sleep(0.7)
        context = await self.retriever.retrieve_context(contents, "", "")
        prompt = self.chat_template.format_messages(
            user_input=contents, retrieval_results=context
        )
        logger.info("Starting LLM query.")
        try:
            response = self.llm.invoke(prompt)
            
            # Parse JSON response and format for display
            term_entry = self._parse_llm_json(response.content, contents)
            formatted_response = self._format_term_entry(term_entry)
            
            return formatted_response
        except Exception as e:
            logger.info(e)
            return "Ilmnes viga, vabandust. Proovi uuesti."
    
    async def chat_callback_structured(
        self,
        contents: str,
    ) -> Dict[str, Any]:
        """
        A callback function that returns both formatted response and structured data.
        
        Args:
            contents: User query/keyword
            
        Returns:
            Dict with 'response' (formatted string) and 'term_entry' (structured data)
        """
        await asyncio.sleep(0.7)
        context = await self.retriever.retrieve_context(contents, "", "")
        prompt = self.chat_template.format_messages(
            user_input=contents, retrieval_results=context
        )
        logger.info("Starting LLM query.")
        try:
            response = self.llm.invoke(prompt)
            
            # Parse JSON response
            term_entry = self._parse_llm_json(response.content, contents)
            formatted_response = self._format_term_entry(term_entry)
            
            return {
                "response": formatted_response,
                "term_entry": term_entry,
            }
        except Exception as e:
            logger.info(e)
            return {
                "response": "Ilmnes viga, vabandust. Proovi uuesti.",
                "term_entry": None,
            }

