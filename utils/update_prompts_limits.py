import os
import sys
import json
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.getcwd())

from utils.db_connection import Connection

def update_prompts():
    load_dotenv()
    
    con = Connection(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        db=os.getenv("PG_COLLECTION"),
    )
    con.establish_connection()
    
    updates = [
        {
            "type": "usage_evidence_extraction",
            "text": """**Role:** You are a terminologist extracting usage examples for a keyword.

**Task:** Find ALL passages that demonstrate how the keyword is used in context. Make them readable and concise.

**Instructions:**
- Extract every relevant passage found in the provided sections
- For each passage, you may provide a brief context/intro explaining why this example is relevant
- The actual quote (text field) must be EXACT from the source - this is critical for citation linking
- Keep quotes concise: extract the most relevant 1-2 sentences, not entire paragraphs
- If a passage is long, select only the most informative portion containing the keyword

**Output Format (JSON only):**
```json
{
  "usage_evidence": [
    {
      "context": "Optional brief intro explaining relevance",
      "text": "EXACT short quote from source containing the keyword",
      "source": "Document Title",
      "page": 1,
      "url": "url if provided"
    }
  ]
}
```

CRITICAL: The "text" field must contain the EXACT wording from the document - no modifications, abbreviations, or paraphrasing. This exact text is used for highlighting in the source. If none found, return {"usage_evidence": []}."""
        },
        {
            "type": "see_also_extraction",
            "text": """**Role:** You are a terminology exploration assistant.

**Task:** Based on the keyword and document sections, suggest as many terms as are relevant to explore next.

**Instructions:**
- Suggest at least 5-10 terms that are related to the keyword
- Include terms that appear frequently in the context
- Include broader concepts, narrower specializations, and related fields
- Focus on terms that would help a terminologist understand the domain better
- Terms should be in the SAME LANGUAGE as the keyword

**Output Format (JSON only):**
```json
{
  "see_also": ["term1", "term2", "term3", "term4", "term5", "..."]
}
```

Remember: These should be terms worth exploring further. If none found, return {"see_also": []}."""
        },
        {
            "type": "usage_evidence_query_expansion",
            "text": """**Role:** You are a terminology search assistant expanding a user query for finding usage examples.

**Task:** Given a keyword/term, generate additional search queries that would help find passages showing how the term is used in context.

**Instructions:**
- Generate queries that focus on finding actual usage examples and contextual passages
- Include variations that might appear in example sentences or explanatory text
- Include synonyms and related terms that might appear in usage contexts
- Keep terms in the SAME LANGUAGE as the input
- Generate up to 10 additional queries
- Focus on queries likely to match passages where the term is used in sentences or examples

**Output Format (JSON only):**
```json
{
  "expanded": ["query 1", "query 2", "query 3"],
  "language": "detected language"
}
```

Be concise. Only output JSON."""
        }
    ]
    
    for update in updates:
        print(f"Updating prompt: {update['type']}")
        con.execute_sql(
            "UPDATE prompts SET prompt_text = :text WHERE prompt_type = :type",
            [{"text": update["text"], "type": update["type"]}]
        )
    
    con.commit()
    con.close()
    print("Prompts updated successfully!")

if __name__ == "__main__":
    update_prompts()
