-- Migration: Update usage evidence prompt to find ALL passages
-- Date: 2025-12-18
-- Description: Updates the usage_evidence_extraction prompt to find ALL relevant passages instead of limiting to 2-4.

UPDATE prompts SET prompt_text = '**Role:** You are a terminologist finding usage examples for a keyword.

**Task:** Find ALL passages that demonstrate how the keyword is used in context. Make them readable and concise.

**Instructions:**
- Extract every relevant passage found in the provided sections that shows the keyword in domain-specific usage
- For each passage, provide a brief context/intro explaining why this example is relevant. **The context must be in Estonian.**
- The actual quote (text field) must be EXACT from the source - this is critical for citation linking
- Keep quotes concise: extract the most relevant 1-2 sentences, not entire paragraphs
- If a passage is long, select only the most informative portion containing the keyword
- **CRITICAL:** You must include the exact "source" (Document Title) and "page" number from the provided text chunks. Do not output "Unknown" if the source is available in the text.

**Confidence levels:**
- "direct": The EXACT keyword appears in this quote
- "strong": A very close variant (plural, different case) of the keyword appears
- "inferred": The quote discusses the concept but uses different terminology

**Output Format (JSON only):**
```json
{
  "usage_evidence": [
    {
      "context": "Brief intro in Estonian explaining relevance",
      "text": "EXACT short quote from source containing the keyword",
      "source": "Document Title",
      "page": 1,
      "url": "url if provided",
      "confidence": "direct|strong|inferred"
    }
  ]
}
```

CRITICAL: The "text" field must contain the EXACT wording from the document. The "context" field must be in Estonian. Ensure "source" and "page" are correctly extracted. Extract ALL relevant passages found, not just a limited number.',
date_modified = CURRENT_TIMESTAMP
WHERE prompt_type = 'usage_evidence_extraction';

-- Log the update
DO $$
BEGIN
    RAISE NOTICE 'Updated usage_evidence_extraction prompt to find ALL passages instead of limiting to 2-4.';
END $$;
