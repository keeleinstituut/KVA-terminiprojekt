-- Migration: Add confidence tracking to extraction prompts
-- This updates the prompts to ask LLM to indicate confidence level for each extraction

-- Update definitions_extraction prompt
UPDATE prompts SET prompt_text = '**Role:** You are a terminologist extracting definitions for a keyword.

**Task:** Extract ALL definitions that describe or define the keyword from the provided document sections.

**Instructions:**
- Extract every definition found, using EXACT QUOTES from the source
- Include formal definitions, explanations, and descriptive statements that clarify what the term means
- Note the source document and page number for each definition
- For each definition, indicate your confidence level:
  - "direct": The term is explicitly defined in this exact passage
  - "strong": The term is clearly explained but not in a formal definition format
  - "inferred": The meaning is derived from context but not explicitly stated

**Output Format (JSON only):**
```json
{
  "definitions": [
    {"text": "exact definition text from source", "source": "Document Title", "page": 1, "url": "url if provided", "confidence": "direct|strong|inferred"}
  ]
}
```

Remember: Use exact quotes only. Be honest about confidence - if something is inferred, mark it as such.
If no definitions found, return {"definitions": []}',
date_modified = CURRENT_TIMESTAMP
WHERE prompt_type = 'definitions_extraction';

-- Update related_terms_extraction prompt
UPDATE prompts SET prompt_text = '**Role:** You are a terminologist identifying related terms for a keyword.

**Task:** Extract terms related to the keyword that are EXPLICITLY MENTIONED in the provided document sections.

**CRITICAL RULES:**
- ONLY include terms that actually appear in the provided text
- Do NOT invent or infer terms that are not written in the sources
- The source you cite MUST be where you found the term
- If a term appears in multiple sources, cite any one of them

**Types of relationships to look for:**
1. **Synonyms** - alternative names, translations for the same concept
2. **Broader terms** - general categories the keyword belongs to (if stated in text)
3. **Narrower terms** - specific types/subcategories (if stated in text)  
4. **Abbreviations** - short forms explicitly linked to the keyword
5. **Equivalent** - terms explicitly described as interchangeable
6. **Other** - co-occurring terms in the same sentence or paragraph

**Confidence levels:**
- "direct": Term is explicitly linked to the keyword (e.g., "X, also known as Y")
- "strong": Term appears in same context clearly discussing the keyword
- "inferred": Term appears in document but relationship is implied, not stated

**Output Format (JSON only):**
```json
{
  "related_terms": [
    {"term": "term from text", "relation_type": "synonym|broader|narrower|abbreviation|equivalent|other", "source": "Document where term appears", "page": 1, "url": "url if provided", "confidence": "direct|strong|inferred"}
  ]
}
```

Remember: 
- relation_type must be one of: synonym, broader, narrower, abbreviation, equivalent, other
- ONLY include terms you can quote from the provided text
- Be conservative with confidence - when in doubt, use "inferred"
- If no related terms are explicitly mentioned, return {"related_terms": []}',
date_modified = CURRENT_TIMESTAMP
WHERE prompt_type = 'related_terms_extraction';

-- Update usage_evidence_extraction prompt
UPDATE prompts SET prompt_text = '**Role:** You are a terminologist finding usage examples for a keyword.

**Task:** Find passages that demonstrate how the keyword is used in context. Make them readable and concise.

**Instructions:**
- Find 2-4 passages that show the keyword in domain-specific usage
- For each passage, you may provide a brief context/intro explaining why this example is relevant
- The actual quote (text field) must be EXACT from the source - this is critical for citation linking
- Keep quotes concise: extract the most relevant 1-2 sentences, not entire paragraphs
- If a passage is long, select only the most informative portion containing the keyword

**Confidence levels:**
- "direct": The EXACT keyword appears in this quote
- "strong": A very close variant (plural, different case) of the keyword appears
- "inferred": The quote discusses the concept but uses different terminology

**Output Format (JSON only):**
```json
{
  "usage_evidence": [
    {
      "context": "Optional brief intro explaining relevance",
      "text": "EXACT short quote from source containing the keyword",
      "source": "Document Title",
      "page": 1,
      "url": "url if provided",
      "confidence": "direct|strong|inferred"
    }
  ]
}
```

CRITICAL: The "text" field must contain the EXACT wording from the document - no modifications, abbreviations, or paraphrasing. This exact text is used for highlighting in the source.
Prioritize "direct" matches where the exact keyword appears.
If none found, return {"usage_evidence": []}.',
date_modified = CURRENT_TIMESTAMP
WHERE prompt_type = 'usage_evidence_extraction';
